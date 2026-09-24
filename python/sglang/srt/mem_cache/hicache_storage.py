from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Max pages per batched storage IO call.
STORAGE_BATCH_SIZE = 128
HICACHE_FILE_FORMAT_VERSION = 1


def hicache_format_fingerprint(format_spec: Optional[dict]) -> str:
    payload = {
        "schema": HICACHE_FILE_FORMAT_VERSION,
        "format": format_spec or {},
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


# Minimum alignment (bytes) for O_DIRECT segment I/O in HiCacheFile
# direct io_mode. 4 KiB is the safe lower bound every supported FS accepts.
DIRECT_IO_ALIGNMENT = 4096


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    # with dp-attention, tp_rank is attention-group-local; dp_rank disambiguates
    dp_rank: int = 0
    extra_config: Optional[dict] = None
    format_spec: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    DEEPSEEK_V4_C1 = "deepseek_v4_c1"
    DEEPSEEK_V4_C1_INDEXER = "deepseek_v4_c1_indexer"
    DEEPSEEK_V4_C1_INDEXER_SCALE = "deepseek_v4_c1_indexer_scale"
    DEEPSEEK_V4_C2 = "deepseek_v4_c2"
    DEEPSEEK_V4_C2_INDEXER = "deepseek_v4_c2_indexer"
    DEEPSEEK_V4_C2_INDEXER_SCALE = "deepseek_v4_c2_indexer_scale"
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    # FP4 indexer splits the indexer cache into separate payload/scale buffers,
    # so it needs a second pool alongside DEEPSEEK_V4_C4_INDEXER.
    DEEPSEEK_V4_C4_INDEXER_SCALE = "deepseek_v4_c4_indexer_scale"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    # fp8 unified_kv splits a row across a packed fp8 nope pool and a parallel
    # bf16 rope pool, so each compressed region mirrors to two host pools.
    DEEPSEEK_V4_C4_ROPE = "deepseek_v4_c4_rope"
    DEEPSEEK_V4_C128_ROPE = "deepseek_v4_c128_rope"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # Draft KV pool
    DRAFT = "draft"
    DRAFT_INDEXER = "draft_indexer"
    DRAFT_SWA = "draft_swa"

    def __str__(self) -> str:
        return self.value


class PoolTransferTarget(str, Enum):
    HOST = "host"
    DEVICE = "device"


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    device<->storage path: target_pool + target_indices
    nodes_to_load      : evicted nodes this transfer covers
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    target: PoolTransferTarget = PoolTransferTarget.HOST
    target_pool: Optional[Any] = None
    target_indices: Optional[torch.Tensor] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None
    indices_from_pool: Optional[PoolName] = None
    # Full IDs backing a dependent device allocation: resident tensors or
    # slices of the full rows allocated by this load, in transfer order.
    anchor_index_parts: Optional[List[torch.Tensor | slice]] = None


@dataclass(frozen=True)
class SidecarPoolSpec:
    """Pool whose transfer indices are reused from one real source pool."""

    pool_name: PoolName
    indices_from_pool: PoolName
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool."""

    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    # Pools with TRAILING_PAGES (SWA, Mamba state) only hold a window that ends on an
    # offloaded node boundary, so 5 can be restorable while 4 and 3 are not.
    # Each rank owns its own shard and may hold a different set, so reducing a
    # per-rank maximum would pick a length that is illegal on another rank; the
    # caller intersects these sets instead.
    restorable_prefix_pages: Optional[List[int]] = None

    @classmethod
    def empty(cls) -> PoolTransferResult:
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """Accumulate kv_hit_pages across batches (max = last successful batch)."""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, int]) -> None:
        """Record actual load/write success counts per extra pool.

        Every extra pool contributes a prefix that must be contiguous from the
        start, so count the leading run of successes
        """
        self.extra_pool_hit_pages.update(results)


def count_pool_hits(results: dict[str, List[bool]]) -> dict[str, int]:
    return {
        name: (rs.index(False) if False in rs else len(rs))
        for name, rs in results.items()
    }


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def register_mem_pool_device(self, mem_pool_device: Any):
        self.mem_pool_device = mem_pool_device

    def register_mem_device_pool_v2(self, device_pool: Any, device_pool_name: PoolName):
        if not hasattr(self, "registered_device_pools"):
            self.registered_device_pools = {}
        self.registered_device_pools[device_pool_name] = device_pool

    @property
    def supports_device_target(self) -> bool:
        """Capability: True when this backend can move complete cache pages
        directly between registered device (VRAM) pools and L3 storage, using
        ``PoolTransfer`` records with ``target=PoolTransferTarget.DEVICE``. A
        device-target transfer addresses registered VRAM slots through
        ``target_pool`` + ``target_indices`` and bypasses the host (L2) pool
        entirely. Default False keeps callers on the host-mediated paths.
        """
        return False

    @property
    def supports_zero_copy_page_io(self) -> bool:
        """Capability: True when this backend implements positional zero-copy
        page I/O -- ``batch_get_v1``/``batch_set_v1`` and
        ``batch_get_v2``/``batch_set_v2`` read and write directly into host
        pool memory via ``get_page_buffer_meta`` segments with no
        intermediate copy. Read-only; default False keeps callers on the
        copy-based paths.
        """
        return False

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Check which cache pages exist in storage, respecting per-pool hit policies.

        Longest-prefix semantics
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages.  The final ``final_pages`` is the minimum across all pools,
        so a missing auxiliary page shrinks the usable prefix.

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).

        Returns
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Read data from storage into the transfer target for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write data from the transfer target to storage for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class MetadataCache:
    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        # key -> monotonic timestamp
        self.cache: dict[str, float] = {}
        self.lock = threading.Lock()

    def add(self, key: str):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = time.monotonic()

    def remove(self, key: str):
        with self.lock:
            self.cache.pop(key, None)

    def contains(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False
            if self.ttl_seconds == -1.0:
                return True
            if time.monotonic() - self.cache[key] > self.ttl_seconds:
                del self.cache[key]
                return False
            return True

    def clear(self):
        with self.lock:
            self.cache.clear()


class HiCacheFile(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        self.config_suffix = f"_{model_name}"
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if enable_pp:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        # Under NSA context parallel each CP rank holds a disjoint slice of every
        # page, so give each rank its own file key to avoid a cross-rank write race.
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
        self.format_fingerprint = hicache_format_fingerprint(storage_config.format_spec)
        self.config_suffix += f"_fmt{self.format_fingerprint}"
        logger.info(
            "HiCacheFile format fingerprint=%s suffix=%s",
            self.format_fingerprint,
            self.config_suffix,
        )

        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # Metadata cache positive lookup toggle & TTL
        enable_cache_raw = None
        if storage_config.extra_config:
            enable_cache_raw = storage_config.extra_config.get("enable_metadata_cache")
        if enable_cache_raw is None:
            enable_cache_raw = (
                envs.SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE.get()
            )

        self.enable_metadata_cache = bool(enable_cache_raw)

        if self.enable_metadata_cache:
            ttl_raw = None
            if storage_config.extra_config:
                ttl_raw = storage_config.extra_config.get("metadata_ttl")
            if ttl_raw is None:
                ttl_raw = envs.SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL.get()

            self.metadata_ttl = float(ttl_raw) if ttl_raw is not None else 5.0
            self.metadata_cache = MetadataCache(self.metadata_ttl)
            self._scan_existing_files_to_metadata_cache()
        else:
            self.metadata_cache = None

        # All LRU / size accounting and disk eviction lives in the evictor so
        # this backend stays a thin raw-bytes store. Imported lazily: the storage
        # package __init__ pulls in the backend factory, which imports this
        # module, so a top-level import here would be circular.
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        self._evictor = LRUFileEvictor(
            self.file_path,
            self.config_suffix,
            tp_rank=tp_rank,
            is_mla_model=is_mla_model,
            extra_config=storage_config.extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
        )

        # Page I/O mode; extra_config takes precedence over the env default.
        io_mode_raw = None
        if storage_config.extra_config:
            io_mode_raw = storage_config.extra_config.get("io_mode")
        if io_mode_raw is None:
            io_mode_raw = envs.SGLANG_HICACHE_FILE_BACKEND_IO_MODE.get()
        if io_mode_raw not in ("buffered", "direct"):
            raise ValueError(
                "HiCacheFile io_mode must be 'buffered' or 'direct', got "
                f"{io_mode_raw!r}"
            )
        self.io_mode = io_mode_raw
        self.log_page_digests = envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.get()
        # Added to open(2) flags once the direct-mode O_DIRECT probe passes.
        self._o_direct = 0
        if self.io_mode == "direct":
            self._probe_direct_io()

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return os.path.join(
            self.file_path, f"{self._get_component_key(key, component_name)}.bin"
        )

    # ------------------------------------------------------------------
    # Direct (O_DIRECT) positional zero-copy page I/O
    # ------------------------------------------------------------------

    @property
    def supports_zero_copy_page_io(self) -> bool:
        # HiCacheFile reads/writes host pool pages in place only when O_DIRECT
        # segment I/O is active; buffered mode keeps the copy-based paths.
        return self.io_mode == "direct"

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        if self.io_mode == "direct":
            self._validate_direct_pool(mem_pool_host, "anchor KV")

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if self.io_mode == "direct":
            self._validate_direct_pool(host_pool, f"sidecar {host_pool_name}")
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def _probe_direct_io(self) -> None:
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError(
                "HiCacheFile io_mode='direct' requires os.O_DIRECT, which this "
                "platform does not provide."
            )
        probe_path = os.path.join(
            self.file_path,
            f".odirect_probe.{os.getpid()}.{threading.get_ident()}",
        )
        fd = None
        try:
            with open(probe_path, "wb") as probe_file:
                probe_file.truncate(DIRECT_IO_ALIGNMENT)
            raw = (ctypes.c_char * (2 * DIRECT_IO_ALIGNMENT))()
            off = (-ctypes.addressof(raw)) % DIRECT_IO_ALIGNMENT
            buf = memoryview(raw).cast("B")[off : off + DIRECT_IO_ALIGNMENT]
            fd = os.open(probe_path, os.O_RDONLY | os.O_DIRECT)
            if os.preadv(fd, [buf], 0) != DIRECT_IO_ALIGNMENT:
                raise IOError("short aligned O_DIRECT probe read")
            self._o_direct = os.O_DIRECT
        except OSError as e:
            raise RuntimeError(
                f"Filesystem backing {self.file_path!r} rejected an aligned "
                f"O_DIRECT probe ({e}); io_mode='direct' is strict and will "
                "not fall back to buffered I/O."
            ) from e
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(probe_path)
            except OSError:
                pass

    def _validate_direct_pool(self, host_pool, pool_desc: str) -> None:
        layout = getattr(host_pool, "layout", None)
        if layout not in ("page_first", "page_first_direct"):
            raise ValueError(
                f"HiCacheFile io_mode='direct': {pool_desc} pool needs a "
                f"page_first or page_first_direct layout, got {layout!r}."
            )
        if not callable(getattr(host_pool, "get_page_buffer_meta", None)):
            raise ValueError(
                f"HiCacheFile io_mode='direct': {pool_desc} pool lacks "
                "get_page_buffer_meta(); zero-copy segment I/O is impossible."
            )
        aligned = getattr(host_pool, "is_stride_page_aligned", None)
        if not callable(aligned) or not aligned(DIRECT_IO_ALIGNMENT):
            raise ValueError(
                f"HiCacheFile io_mode='direct': {pool_desc} pool page strides "
                f"are not {DIRECT_IO_ALIGNMENT}-byte aligned; O_DIRECT segment "
                "I/O would fail."
            )

    @staticmethod
    def _segment_view(ptr: int, size: int) -> memoryview:
        # Writable ctypes-backed view straight onto the host pool buffer.
        return memoryview((ctypes.c_char * size).from_address(ptr)).cast("B")

    def _check_direct_segments(self, segments: List[tuple]) -> None:
        for ptr, size in segments:
            if ptr % DIRECT_IO_ALIGNMENT or size % DIRECT_IO_ALIGNMENT:
                raise ValueError(
                    f"O_DIRECT segment unaligned (ptr=0x{ptr:x}, size={size}); "
                    f"{DIRECT_IO_ALIGNMENT}-byte alignment required."
                )

    def _page_segments(self, host_pool, page_indices) -> List[tuple]:
        """Ordered (ptr, size) segments for one logical page of *host_pool*."""
        ptrs, sizes = host_pool.get_page_buffer_meta(page_indices)
        if len(ptrs) != len(sizes):
            raise ValueError(
                f"get_page_buffer_meta returned {len(ptrs)} pointers vs "
                f"{len(sizes)} sizes"
            )
        segments = list(zip(ptrs, sizes))
        if self.io_mode == "direct":
            self._check_direct_segments(segments)
        return segments

    def _tensor_segments(self, tensor: torch.Tensor) -> List[tuple]:
        if not tensor.is_contiguous():
            raise ValueError(
                "HiCacheFile direct I/O requires a contiguous tensor (no copies)."
            )
        segments = [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]
        if self.io_mode == "direct":
            self._check_direct_segments(segments)
        return segments

    @staticmethod
    def _advance_iovecs(views: List[memoryview], n: int) -> None:
        while n:
            if views and len(views[0]) <= n:
                n -= len(views[0])
                views.pop(0)
            else:
                views[0] = views[0][n:]
                n = 0

    def _iovec_transfer(
        self, fd: int, offset: int, segments: List[tuple], write: bool
    ) -> None:
        """Complete a positional vector I/O of sum(size) bytes at *offset*.

        Aligned short I/O (O_DIRECT may return less than requested) resumes by
        advancing through the iovecs. A non-page-aligned short result or an
        EOF before completion is rejected.
        """
        views = [self._segment_view(ptr, size) for ptr, size in segments]
        total = sum(size for _, size in segments)
        done = 0
        direction = "write" if write else "read"
        while done < total:
            n = (os.pwritev if write else os.preadv)(fd, views, offset + done)
            if n is None or n <= 0:
                raise IOError(
                    f"{direction} stalled at offset {offset + done} "
                    f"({done}/{total} bytes complete)"
                )
            if self.io_mode == "direct" and n % DIRECT_IO_ALIGNMENT:
                raise IOError(
                    f"non-page-aligned short {direction} of {n} bytes at "
                    f"offset {offset + done} ({done}/{total} complete)"
                )
            done += n
            self._advance_iovecs(views, n)

    def _read_page_segments(self, storage_key: str, segments: List[tuple]) -> bool:
        """Read one page's segments from its file; True on an exact-length read."""
        suffixed = self._get_suffixed_key(storage_key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        expected = sum(size for _, size in segments)
        try:
            fd = os.open(tensor_path, os.O_RDONLY | self._o_direct)
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False
        try:
            size = os.fstat(fd).st_size
            if size != expected:
                logger.warning(
                    "HiCacheFile: wrong file size for %s: expected %d, got %d",
                    tensor_path,
                    expected,
                    size,
                )
                return False
            self._iovec_transfer(fd, 0, segments, write=False)
        except OSError as e:
            logger.warning(
                f"Failed to fetch {storage_key} from HiCacheFile storage: {e}"
            )
            return False
        finally:
            os.close(fd)
        self._evictor.touch(suffixed, tensor_path)
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return True

    def _write_page_segments(self, storage_key: str, segments: List[tuple]) -> bool:
        """Write one page's segments via a unique temp file + os.replace."""
        suffixed = self._get_suffixed_key(storage_key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        # Fast path: same key already on disk. Refresh recency and skip rewrite.
        if self.exists(storage_key):
            self._evictor.touch(suffixed, tensor_path)
            return True
        total = sum(size for _, size in segments)
        tmp_path = (
            f"{tensor_path}.tmp."
            f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )
        reserved = False
        fd = None
        try:
            if not self._evictor.reserve(suffixed, total, key=storage_key):
                logger.warning(
                    "HiCacheFile rejected storage write for key=%s bytes=%d",
                    storage_key,
                    total,
                )
                return False
            reserved = True
            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | self._o_direct,
                0o644,
            )
            os.ftruncate(fd, total)
            self._iovec_transfer(fd, 0, segments, write=True)
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {storage_key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                self._evictor.abort(suffixed)
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False

    def _batch_io_v1(
        self, keys: List[str], host_indices: torch.Tensor, write: bool
    ) -> List[bool]:
        """Per-page positional I/O on the KV anchor pool; per-page results."""
        pool = getattr(self, "mem_pool_host", None)
        op = "set" if write else "get"
        results: List[bool] = []
        for i, key in enumerate(keys):
            ok = False
            try:
                if pool is None or host_indices is None:
                    raise RuntimeError(
                        f"batch_{op}_v1 called before register_mem_pool_host"
                    )
                page_size = getattr(pool, "page_size", 1) or 1
                page_indices = host_indices[i * page_size : (i + 1) * page_size]
                if page_indices.numel() != page_size:
                    raise ValueError(
                        f"host_indices too short for page {i}: need {page_size}, "
                        f"got {page_indices.numel()}"
                    )
                page_offset = int(page_indices[0].item())
                segments = self._page_segments(pool, page_indices)
                if write:
                    self._log_host_page_digests(
                        "write", PoolName.KV, key, pool, page_offset
                    )
                    ok = self._write_page_segments(key, segments)
                else:
                    ok = self._read_page_segments(key, segments)
                    if ok:
                        self._log_host_page_digests(
                            "read", PoolName.KV, key, pool, page_offset
                        )
            except Exception as e:
                logger.error(f"HiCacheFile batch_{op}_v1 failed for {key}: {e}")
            results.append(ok)
        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._batch_io_v1(keys, host_indices, write=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._batch_io_v1(keys, host_indices, write=True)

    def _batch_io_v2_direct(
        self, transfers: List[PoolTransfer], write: bool
    ) -> dict[str, List[bool]]:
        """Direct-mode v2: positional segment I/O per PoolTransfer, keeping
        the existing component key scheme (KV bare key, sidecars key.<pool>)."""
        results: dict[str, List[bool]] = {}
        op = "batch_set_v2" if write else "batch_get_v2"
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices
            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue
            per_page: List[bool] = []
            for i, key in enumerate(keys):
                try:
                    page_offset = host_indices[i * page_size].item()
                    segments = self._page_segments(
                        host_pool,
                        host_indices[i * page_size : (i + 1) * page_size],
                    )
                    storage_key = self._log_key(transfer.name, key)
                    if write:
                        self._log_host_page_digests(
                            "write", transfer.name, key, host_pool, page_offset
                        )
                        ok = self._write_page_segments(storage_key, segments)
                    else:
                        ok = self._read_page_segments(storage_key, segments)
                        if ok:
                            self._log_host_page_digests(
                                "read", transfer.name, key, host_pool, page_offset
                            )
                except Exception as e:
                    logger.error(
                        f"HiCacheFile {op} failed for {transfer.name}/{key}: {e}"
                    )
                    ok = False
                per_page.append(ok)
            results[transfer.name] = per_page
        return results

    def _scan_existing_files_to_metadata_cache(self) -> None:
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if stem.endswith(self.config_suffix):
                self.metadata_cache.add(stem)

    def _log_storage_tensor_digest(
        self, direction: str, key: str, tensor: torch.Tensor
    ) -> None:
        if not self.log_page_digests:
            return
        flat_bytes = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
        logger.warning(
            "HiCache storage tensor digest direction=%s key=%s bytes=%d sha256=%s",
            direction,
            key,
            flat_bytes.numel(),
            hashlib.sha256(flat_bytes.numpy()).hexdigest(),
        )

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        if self.io_mode == "direct":
            segments = self._tensor_segments(target_location)
            return target_location if self._read_page_segments(key, segments) else None
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                actual = os.fstat(f.fileno()).st_size
                if actual != expected:
                    logger.warning(
                        "HiCacheFile: wrong file size for %s: expected %d, got %d",
                        tensor_path,
                        expected,
                        actual,
                    )
                    if self.metadata_cache is not None:
                        self.metadata_cache.remove(suffixed)
                    return None
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {suffixed}")
                self._log_storage_tensor_digest("read", key, target_location)
            self._evictor.touch(suffixed, tensor_path)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return target_location
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.io_mode == "direct":
            return self._write_page_segments(key, self._tensor_segments(value))
        suffixed = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        self._log_storage_tensor_digest("write", key, value)

        # Fast path: same key already on disk. Refresh recency and skip rewrite.
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            self._evictor.touch(suffixed, tensor_path)
            return True

        tmp_path = None
        reserved = False
        try:
            value_bytes = value.numel() * value.element_size()
            # Ask the evictor to admit + reserve disk space (evicting if needed).
            if not self._evictor.reserve(suffixed, value_bytes, key=key):
                return False
            reserved = True

            tmp_path = os.path.join(self.file_path, f".{uuid.uuid4().hex}.tmp")
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tmp_path)
            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                self._evictor.abort(suffixed)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        if self.metadata_cache is not None and self.metadata_cache.contains(key):
            return True
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if os.path.exists(tensor_path):
            if self.metadata_cache is not None:
                self.metadata_cache.add(key)
            return True
        return False

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        if self.metadata_cache is None:
            existing_files = set()
            with os.scandir(self.file_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name in target_files:
                        existing_files.add(entry.name)
            return existing_files

        existing_files = set()
        for filename in target_files:
            stem = filename[:-4]
            if self.metadata_cache.contains(stem):
                existing_files.add(filename)
            else:
                path = os.path.join(self.file_path, filename)
                if os.path.exists(path):
                    self.metadata_cache.add(stem)
                    existing_files.add(filename)
        return existing_files

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # Longest contiguous KV prefix present in storage.
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            elif kv_pages and transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                logger.warning(
                    "HiCache storage found %d KV pages but no trailing %s sidecar; "
                    "the prefix cannot be restored",
                    kv_pages,
                    name,
                )
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _log_host_page_digests(
        self, direction: str, pool_name: str, key: str, host_pool, page_offset: int
    ) -> None:
        if not self.log_page_digests:
            return
        get_components = getattr(host_pool, "get_debug_page_tensors", None)
        components = (
            get_components(page_offset)
            if get_components is not None
            else ((str(pool_name), host_pool.get_data_page(page_offset, flat=True)),)
        )
        storage_key = self._log_key(pool_name, key)
        for component, tensor in components:
            flat_bytes = tensor.detach().contiguous().view(torch.uint8)
            logger.warning(
                "HiCache page digest direction=%s pool=%s component=%s key=%s "
                "host_page=%d bytes=%d sha256=%s",
                direction,
                pool_name,
                component,
                storage_key,
                page_offset,
                flat_bytes.numel(),
                hashlib.sha256(flat_bytes.numpy()).hexdigest(),
            )

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        self._log_host_page_digests("read", pool_name, key, host_pool, page_offset)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes."""
        storage_key = self._log_key(pool_name, key)
        self._log_host_page_digests("write", pool_name, key, host_pool, page_offset)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        if self.io_mode == "direct":
            return self._batch_io_v2_direct(transfers, write=False)
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        if self.io_mode == "direct":
            return self._batch_io_v2_direct(transfers, write=True)
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self._evictor.clear()
            if self.metadata_cache is not None:
                self.metadata_cache.clear()
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
