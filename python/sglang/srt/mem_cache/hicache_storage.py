from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_format import (
    decode_page_header,
    encode_page_header,
    format_fingerprint,
    hicache_tensor_digest,
    page_file_size,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Max pages per batched storage IO call.
STORAGE_BATCH_SIZE = 128


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


@dataclass
class HiCacheStorageExtraInfo:
    # Page hashes preceding the call's keys, from the sequence start;
    # prefix_keys + keys is one contiguous chain for KV batches and sidecars.
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    AUXILIARY_SWA = "auxiliary_swa"
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
    """Destination of a host<->storage transfer.

    HOST   : the registered host (L2) pool pages — the default.
    DEVICE : a registered device staging arena; the transfer bypasses the
             host pool entirely (buffer_only device-target reads).
    """

    HOST = "host"
    DEVICE = "device"


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    BLOCK_GROUPS   : whole key groups at block boundaries from the span start
                     (a second form of the FULL pages, e.g. compressed blocks).
                     The pool reports the pages its leading complete groups
                     cover as its own verdict; it never gates the KV hit.
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"
    BLOCK_GROUPS = "block_groups"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    device<->storage path: target=DEVICE + device_indices (staging slots)
    nodes_to_load      : evicted nodes this transfer covers
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    target: PoolTransferTarget = PoolTransferTarget.HOST
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None
    indices_from_pool: Optional[PoolName] = None
    # BLOCK_GROUPS geometry: FULL pages per block and keys (pages) per group.
    block_pages: int = 0
    group_pages: int = 0


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

    def register_mem_device_pool_v2(self, device_pool, device_pool_name):
        """Register a device-side pool for device-target transfers.

        For the file backend this is a ``DeviceStagingPool`` mirroring the
        named host pool's page geometry; ``PoolTransfer`` records with
        ``target=PoolTransferTarget.DEVICE`` move whole pages between L3
        files and that arena, bypassing the host (L2) pool entirely.
        """
        if not hasattr(self, "registered_device_pools"):
            self.registered_device_pools = {}
        self.registered_device_pools[device_pool_name] = device_pool

    @property
    def supports_device_target(self) -> bool:
        """Capability: True when this backend can move complete cache pages
        directly between registered device pools and L3 storage through
        ``PoolTransfer`` records with ``target=PoolTransferTarget.DEVICE``.
        Default False keeps callers on the host-mediated paths.
        """
        return False

    @property
    def supports_zero_copy_page_io(self) -> bool:
        """Capability: True when this backend implements positional zero-copy
        page I/O — ``batch_get_v1``/``batch_set_v1`` and
        ``batch_get_v2``/``batch_set_v2`` read and write directly into host
        pool memory via ``get_page_buffer_meta`` segments with no
        intermediate copy. Read-only; default False keeps callers on the
        copy-based paths.
        """
        return False

    def touch_many(self, keys: List[str]) -> None:
        """Refresh storage recency for a batch of page keys. Backends without
        a recency index (or without persistent storage at all) ignore it."""
        return

    def maybe_sweep_orphans(self) -> None:
        """Periodic leftover-file sweep hook driven from the storage threads'
        idle paths. Backends without files ignore it."""
        return

    def on_host_pools_registered(self) -> None:
        """The initial host-pool registration burst is over: the anchor pool
        (``register_mem_pool_host``) and every host pool group entry
        (``register_mem_host_pool_v2``). Fires once per attach; pools
        registered later (dynamic sidecars) arrive through v2 alone and do
        not re-fire it. Backends that must finalize once the pool set is
        known -- e.g. register with a remote cache manager whose spec cannot
        be amended -- should do so here; raising fails the attach.
        """

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
        """Read data from storage into host memory for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.

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
    def __init__(self, ttl_seconds: float, max_entries: Optional[int] = None):
        self.ttl_seconds = ttl_seconds
        # 0 / None disables the cap. Without it, a key cached once and never
        # probed again lingers until process exit (entries are only dropped
        # on their own next lookup); purge() is driven from the orphan sweep.
        self.max_entries = max_entries or None
        # key -> monotonic timestamp, oldest-added first
        self.cache: OrderedDict[str, float] = OrderedDict()
        self._last_expiry_scan: float = float("-inf")
        self.lock = threading.Lock()

    def add(self, key: str):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = time.monotonic()
                if self.max_entries is not None and len(self.cache) > self.max_entries:
                    self._evict_over_limit_locked()

    def _evict_over_limit_locked(self):
        # The expiry scan is O(entries); at the cap, running it on every add
        # costs a full scan per write. Once per TTL window is enough — the
        # periodic purge() from the orphan sweep reclaims stragglers, and the
        # FIFO drop below still enforces the cap meanwhile.
        now = time.monotonic()
        if (
            self.ttl_seconds != -1.0
            and now - self._last_expiry_scan >= self.ttl_seconds
        ):
            self._last_expiry_scan = now
            self._remove_expired_locked(now)
        # Drop oldest-added beyond the cap; contains() reorders hits to the
        # end, so insertion order approximates recency.
        while len(self.cache) > self.max_entries:
            self.cache.popitem(last=False)

    def _remove_expired_locked(self, now: float) -> int:
        """Delete every entry older than the TTL; returns how many went."""
        if self.ttl_seconds == -1.0:
            return 0
        expired = [k for k, t in self.cache.items() if now - t > self.ttl_seconds]
        for k in expired:
            del self.cache[k]
        return len(expired)

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
            # Reorder to the end (recency) without extending the TTL.
            self.cache.move_to_end(key)
            return True

    def purge(self) -> int:
        """Drop every expired entry; returns how many were removed."""
        with self.lock:
            removed = self._remove_expired_locked(time.monotonic())
            self._last_expiry_scan = time.monotonic()
            return removed

    def clear(self):
        with self.lock:
            self.cache.clear()


class HiCacheFile(HiCacheStorage):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        extra_cfg = storage_config.extra_config or {}
        self.file_path = (
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get()
            or extra_cfg.get("file_storage_path")
            or file_path
        )

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
        shard_suffix = f"_{pp_size}_{pp_rank}" if enable_pp else ""
        # Under NSA context parallel each CP rank holds a disjoint slice of every
        # page, so give each rank its own file key to avoid a cross-rank write race.
        if attn_cp_size > 1:
            shard_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
        # MLA KV is replicated across TP ranks and shares one file per page;
        # TP-sharded state is keyed by tp_config_suffix instead.
        self.tp_config_suffix = f"_{model_name}_{tp_rank}_{tp_size}{shard_suffix}"
        self.config_suffix = (
            f"_{model_name}{shard_suffix}" if is_mla_model else self.tp_config_suffix
        )

        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # Page IO engine; extra_config takes precedence over the env default.
        # Imported lazily like LRUFileEvictor below: the storage package
        # __init__ pulls in the backend factory, which imports this module.
        from sglang.srt.mem_cache.storage.file.page_io import make_page_io

        io_engine_raw = extra_cfg.get("io_engine")
        if io_engine_raw is None:
            io_engine_raw = envs.SGLANG_HICACHE_FILE_BACKEND_IO_ENGINE.get()
        self.io_engine = make_page_io(self.file_path, io_engine_raw)

        self.log_page_digests = envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.get()

        verify_raw = extra_cfg.get("verify")
        if verify_raw is None:
            verify_raw = envs.SGLANG_HICACHE_FILE_VERIFY.get()
        if verify_raw not in ("full", "sampled", "off"):
            raise ValueError(
                f"HiCacheFile verify must be 'full', 'sampled', or 'off', "
                f"got {verify_raw!r}"
            )
        self.verify_mode = verify_raw
        # sampled: verify the crc of one page in every this-many reads.
        self._verify_sample_period = 64
        self._verified_read_count = 0

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
            max_entries_raw = None
            if storage_config.extra_config:
                max_entries_raw = storage_config.extra_config.get(
                    "metadata_cache_max_entries"
                )
            if max_entries_raw is None:
                max_entries_raw = (
                    envs.SGLANG_HICACHE_FILE_BACKEND_METADATA_CACHE_MAX_ENTRIES.get()
                )
            self.metadata_cache = MetadataCache(
                self.metadata_ttl, max_entries=max_entries_raw
            )
            self._scan_existing_files_to_metadata_cache()
        else:
            self.metadata_cache = None

        # All LRU / size accounting and disk eviction lives in the evictor so
        # this backend stays a thin raw-bytes store. Imported lazily: the storage
        # package __init__ pulls in the backend factory, which imports this
        # module, so a top-level import here would be circular.
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        # Each rank creates and evicts its TP-sharded files; rank 0 also owns
        # the shared MLA files.
        owned_suffixes = (self.tp_config_suffix,)
        if is_mla_model and tp_rank == 0:
            owned_suffixes += (self.config_suffix,)
        self._evictor = LRUFileEvictor(
            self.file_path,
            owned_suffixes=owned_suffixes,
            tp_rank=tp_rank,
            extra_config=storage_config.extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
            sibling_keys=self._group_sibling_keys,
        )
        if self._evictor.enabled:
            # Untracked leftover bytes count toward the free-space watermark.
            self._evictor.sweep_callback = self.sweep_orphans

        # Per-pool format fingerprints, filled as pools register. Files
        # carry the fingerprint in their header; a page written by another
        # geometry misses instead of loading corrupt.
        self._pool_fingerprints: dict[str, str] = {}
        # Device staging arenas per pool name (register_mem_device_pool_v2).
        self.registered_device_pools: dict = {}
        # Reusable aligned pinned bounce per staged pool, keyed by pool name.
        self._device_bounces: dict = {}
        self._device_bounce_lock = threading.Lock()

        grace_raw = extra_cfg.get("orphan_grace_seconds")
        if grace_raw is None:
            grace_raw = envs.SGLANG_HICACHE_FILE_BACKEND_ORPHAN_GRACE_SECONDS.get()
        self._orphan_grace_seconds = float(grace_raw)
        interval_raw = extra_cfg.get("orphan_sweep_interval_seconds")
        if interval_raw is None:
            interval_raw = (
                envs.SGLANG_HICACHE_FILE_BACKEND_ORPHAN_SWEEP_INTERVAL_SECONDS.get()
            )
        self._orphan_sweep_interval_seconds = float(interval_raw)
        self._next_orphan_sweep = time.monotonic() + self._orphan_sweep_interval_seconds

    def _get_suffixed_key(self, key: str) -> str:
        # Mamba state is TP-sharded even when the model's MLA KV is replicated.
        if key.endswith(f".{PoolName.MAMBA.value}"):
            return key + self.tp_config_suffix
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    # ------------------------------------------------------------------
    # Per-pool format fingerprints
    # ------------------------------------------------------------------

    def _register_pool_fingerprint(self, host_pool, pool_name: str) -> None:
        self._pool_fingerprints[pool_name] = format_fingerprint(
            host_pool.storage_format_spec()
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self._register_pool_fingerprint(mem_pool_host, PoolName.KV.value)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        self._register_pool_fingerprint(host_pool, str(host_pool_name))

    def _pool_fingerprint(self, pool_name: str) -> str:
        """Fingerprint of a registered pool's page format.

        Falls back to a constant for callers (and tests) that use the raw
        get/set API before any pool registered; the fallback is stable
        within a process, so raw-tensor round trips stay consistent.
        """
        fp = self._pool_fingerprints.get(pool_name)
        if fp is not None:
            return fp
        if pool_name == PoolName.KV.value:
            return format_fingerprint(None)
        raise ValueError(
            f"HiCacheFile has no format fingerprint for pool {pool_name!r}; "
            "the pool must register before its pages are read or written."
        )

    def _fingerprint_for_key(self, storage_key: str) -> str:
        """Fingerprint of the pool a component storage key addresses.

        Component keys are ``{key}`` for the KV payload and ``{key}.{pool}``
        for sidecars (see ``_log_key``); pool names contain no dots and the
        chain-hash keys are hex, so the trailing ``.{pool}`` is unambiguous.
        """
        for pool_name, fp in self._pool_fingerprints.items():
            if pool_name != PoolName.KV.value and storage_key.endswith(f".{pool_name}"):
                return fp
        return self._pool_fingerprint(PoolName.KV.value)

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
            if stem.endswith((self.config_suffix, self.tp_config_suffix)):
                self.metadata_cache.add(stem)

    # ------------------------------------------------------------------
    # Page file IO: header + padded payload through the IO engine
    # ------------------------------------------------------------------

    def _crc32_segments(self, segments) -> int:
        from sglang.srt.mem_cache.storage.file.page_io import segment_view

        crc = 0
        for ptr, size in segments:
            crc = zlib.crc32(segment_view(ptr, size), crc)
        return crc

    def _should_verify_crc(self) -> bool:
        if self.verify_mode == "off":
            return False
        if self.verify_mode == "full":
            return True
        self._verified_read_count += 1
        return self._verified_read_count % self._verify_sample_period == 0

    def _drop_page_file(self, suffixed: str, tensor_path: str, reason: str) -> None:
        """A page file that can never load: drop it from existence caches.

        The evictor's LRU entry is left in place; it is dropped as a
        no-op (already unlinked) the next time eviction reaches it.
        """
        logger.warning("HiCacheFile dropping page %s: %s", suffixed, reason)
        if self.metadata_cache is not None:
            self.metadata_cache.remove(suffixed)
        try:
            os.remove(tensor_path)
        except OSError:
            pass

    def _read_page_bytes(self, storage_key: str, fingerprint: str, segments) -> bool:
        """Read one page file into *segments*; True on an exact verified read.

        Any miss (absent, wrong fingerprint, wrong length, IO error, crc
        mismatch) returns False; unreadable files are unlinked so later
        existence checks stop reporting them.
        """
        suffixed = self._get_suffixed_key(storage_key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")
        header_raw = self.io_engine.read_header(tensor_path)
        if header_raw is None:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False
        try:
            header = decode_page_header(header_raw)
        except ValueError as e:
            self._drop_page_file(suffixed, tensor_path, str(e))
            return False
        expected = sum(size for _, size in segments)
        if header.fingerprint != fingerprint:
            self._drop_page_file(suffixed, tensor_path, "format fingerprint mismatch")
            return False
        if header.payload_len != expected:
            self._drop_page_file(suffixed, tensor_path, "wrong payload length")
            return False
        try:
            self.io_engine.read_payload(
                tensor_path, header.payload_len, header.payload_padded_len, segments
            )
        except (OSError, IOError) as e:
            logger.warning(
                "Failed to fetch %s from HiCacheFile storage: %s", storage_key, e
            )
            return False
        if self._should_verify_crc() and self._crc32_segments(segments) != (
            header.crc32 & 0xFFFFFFFF
        ):
            self._drop_page_file(suffixed, tensor_path, "crc mismatch")
            return False
        self._evictor.touch(suffixed, tensor_path)
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return True

    def _write_page_bytes(self, storage_key: str, fingerprint: str, segments) -> bool:
        """Write one page file atomically: reserve, tmp write, fsync, rename."""
        suffixed = self._get_suffixed_key(storage_key)
        tensor_path = os.path.join(self.file_path, f"{suffixed}.bin")

        # Fast path: same key already on disk. Refresh recency and skip rewrite.
        if self.exists(storage_key):
            self._evictor.touch(suffixed, tensor_path)
            return True

        payload_len = sum(size for _, size in segments)
        tmp_path = (
            f"{tensor_path}.tmp."
            f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
        )
        reserved = False
        try:
            if not self._evictor.reserve(
                suffixed, page_file_size(payload_len), key=storage_key
            ):
                logger.warning(
                    "HiCacheFile rejected storage write for key=%s bytes=%d",
                    storage_key,
                    payload_len,
                )
                return False
            reserved = True
            header = encode_page_header(
                fingerprint, payload_len, self._crc32_segments(segments)
            )
            self.io_engine.write_page_file(tmp_path, header, segments)
            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save page {storage_key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                self._evictor.abort(suffixed)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False

    # ------------------------------------------------------------------
    # Segment extraction (positional zero-copy page IO)
    # ------------------------------------------------------------------

    def _page_segments(self, host_pool, page_indices) -> Optional[List[tuple]]:
        """Ordered (ptr, size) segments for one page, or None when the pool
        must fall back to the whole-flat-page path (no segment metadata, or
        segments the engine cannot address in place)."""
        try:
            ptrs, sizes = host_pool.get_page_buffer_meta(page_indices)
        except NotImplementedError:
            return None
        if len(ptrs) != len(sizes):
            raise ValueError(
                f"get_page_buffer_meta returned {len(ptrs)} pointers vs "
                f"{len(sizes)} sizes"
            )
        return list(zip(ptrs, sizes))

    @staticmethod
    def _tensor_segments(tensor: torch.Tensor) -> List[tuple]:
        tensor = tensor.contiguous()
        return [(tensor.data_ptr(), tensor.numel() * tensor.element_size())]

    # ------------------------------------------------------------------
    # Raw tensor API (get/set)
    # ------------------------------------------------------------------

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        segments = self._tensor_segments(target_location)
        if not self._read_page_bytes(key, self._fingerprint_for_key(key), segments):
            return None
        if self.log_page_digests:
            logger.warning(
                "HiCache storage tensor digest direction=read key=%s bytes=%d "
                "sha256=%s",
                key,
                target_location.numel() * target_location.element_size(),
                hicache_tensor_digest(target_location),
            )
        return target_location

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
        if self.log_page_digests:
            logger.warning(
                "HiCache storage tensor digest direction=write key=%s bytes=%d "
                "sha256=%s",
                key,
                value.numel() * value.element_size(),
                hicache_tensor_digest(value),
            )
        return self._write_page_bytes(
            key, self._fingerprint_for_key(key), self._tensor_segments(value)
        )

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
        if os.path.isfile(tensor_path):
            if self.metadata_cache is not None:
                self.metadata_cache.add(key)
            return True
        return False

    # ------------------------------------------------------------------
    # Positional zero-copy page IO (v1, anchor KV pool)
    # ------------------------------------------------------------------

    @property
    def supports_zero_copy_page_io(self) -> bool:
        # Segment IO needs an engine that can address host pool pages in
        # place; the buffered engine would bounce every page anyway.
        return self.io_engine.name != "posix_buffered"

    @property
    def supports_device_target(self) -> bool:
        # Device-target reads work through a pinned bounce + async copy onto
        # the staging arena under every engine; the cuFile engine will read
        # straight into the arena rows.
        return True

    def _batch_io_v1(
        self, keys: List[str], host_indices: torch.Tensor, write: bool
    ) -> List[bool]:
        """Per-page positional IO on the KV anchor pool; per-page results."""
        pool = self.mem_pool_host
        op = "set" if write else "get"
        results: List[bool] = []
        for i, key in enumerate(keys):
            ok = False
            try:
                if pool is None or host_indices is None:
                    raise RuntimeError(
                        f"batch_{op}_v1 called before register_mem_pool_host"
                    )
                page_size = pool.page_size or 1
                page_indices = host_indices[i * page_size : (i + 1) * page_size]
                if page_indices.numel() != page_size:
                    raise ValueError(
                        f"host_indices too short for page {i}: need {page_size}, "
                        f"got {page_indices.numel()}"
                    )
                page_offset = int(page_indices[0].item())
                segments = self._page_segments(pool, page_indices)
                if segments is None:
                    raise ValueError(
                        "anchor pool has no segment metadata for zero-copy IO"
                    )
                fingerprint = self._pool_fingerprint(PoolName.KV.value)
                if write:
                    self._log_host_page_digests(
                        "write", PoolName.KV.value, key, pool, page_offset
                    )
                    ok = self._write_page_bytes(key, fingerprint, segments)
                else:
                    ok = self._read_page_bytes(key, fingerprint, segments)
                    if ok:
                        self._log_host_page_digests(
                            "read", PoolName.KV.value, key, pool, page_offset
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

    # ------------------------------------------------------------------
    # L3 recency refresh (page groups) and orphan janitor
    # ------------------------------------------------------------------

    def touch_many(self, keys: List[str]) -> None:
        """Bump L3 recency for whole page groups, KV payload and sidecars.

        Called once per insert with the batch of re-referenced page hashes;
        missing files are benign (the batched touch adopts nothing for them).
        """
        if not self._evictor.enabled:
            return
        pool_suffixes = [
            f".{pool}{self.tp_config_suffix}"
            for pool in self.registered_pools
            if str(pool) not in (PoolName.KV.value, "__default__")
        ]
        pairs = []
        for key in keys:
            kv_stem = key + self.config_suffix
            pairs.append((kv_stem, os.path.join(self.file_path, f"{kv_stem}.bin")))
            for pool in self.registered_pools:
                if str(pool) in (PoolName.KV.value, "__default__"):
                    continue
                # Mamba state keys are suffixed per rank even under MLA.
                suffix = (
                    self.tp_config_suffix
                    if str(pool) == PoolName.MAMBA.value
                    else self.config_suffix
                )
                stem = f"{key}.{pool}{suffix}"
                pairs.append((stem, os.path.join(self.file_path, f"{stem}.bin")))
        self._evictor.touch_many(pairs)

    def _group_sibling_keys(self, suffixed_key: str) -> List[str]:
        """Stems of every file storing part of the same page group.

        Page keys are hex hashes without dots, so a sidecar stem is
        unambiguously ``{key}.{pool}{suffix}`` while the KV payload stem is
        ``{key}{suffix}``.
        """
        for suffix in (self.config_suffix, self.tp_config_suffix):
            if not suffixed_key.endswith(suffix):
                continue
            stem = suffixed_key[: -len(suffix)]
            base, _, pool_tag = stem.rpartition(".")
            if not base:
                base, pool_tag = stem, ""
            siblings = [base + suffix]
            for pool in self.registered_pools:
                if str(pool) in (PoolName.KV.value, "__default__") or str(
                    pool
                ) == pool_tag:
                    continue
                siblings.append(f"{base}.{pool}{suffix}")
            return siblings
        return []

    # Temp/probe leftovers from crashed writes: positional page writes
    # ({suffixed}.bin.tmp.{pid}.{tid}.{uuid}) and the O_DIRECT capability
    # probe (.odirect_probe.{pid}.{tid}).
    _ORPHAN_TEMP_PATTERNS = (
        re.compile(r"\.bin\.tmp\.\d+\.\d+\.[0-9a-f]+$"),
        re.compile(r"^\.odirect_probe\.\d+\.\d+$"),
    )

    def sweep_orphans(self) -> int:
        """Delete leftover temp/probe files and unreadable/stale-format pages.

        Only files this instance could never read again are removed: temp
        and probe leftovers older than the grace window (a live writer may
        still own a young one), undecodable-header page files, and pages
        carrying this instance's model/rank identity in their stem but a
        format fingerprint none of its pools produced. Files of other
        ranks/models are never touched. Returns the bytes reclaimed.
        """
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return 0
        valid_fps = set(self._pool_fingerprints.values()) or None
        now = time.time()
        reclaimed = 0
        removed = 0
        for fn in names:
            path = os.path.join(self.file_path, fn)
            orphan = any(p.search(fn) for p in self._ORPHAN_TEMP_PATTERNS)
            if not orphan and fn.endswith(".bin"):
                stem = fn[:-4]
                if not stem.endswith((self.config_suffix, self.tp_config_suffix)):
                    continue  # another rank's or model's file
                orphan = self._is_stale_page_file(path, valid_fps)
            if not orphan:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if now - st.st_mtime < self._orphan_grace_seconds:
                continue
            try:
                os.remove(path)
            except OSError:
                continue
            reclaimed += st.st_size
            removed += 1
            if self.metadata_cache is not None:
                self.metadata_cache.remove(fn[:-4] if fn.endswith(".bin") else fn)
        if removed:
            logger.info(
                "HiCacheFile orphan sweep: removed %d leftover files, "
                "reclaimed %d bytes",
                removed,
                reclaimed,
            )
        if self.metadata_cache is not None:
            purged = self.metadata_cache.purge()
            if purged:
                logger.info(
                    "HiCacheFile metadata cache purged %d expired entries",
                    purged,
                )
        return reclaimed

    def _is_stale_page_file(self, path: str, valid_fps: Optional[Set[str]]) -> bool:
        """True when a page file can never load in this instance.

        An undecodable header (pre-format file or torn write) or a
        fingerprint none of the registered pools produced makes the page
        permanently unreadable; with no pools registered yet (direct
        construction before attach) the check stays conservative and only
        headerless files count.
        """
        header_raw = self.io_engine.read_header(path)
        if header_raw is None:
            return False  # gone already; not this sweep's business
        try:
            header = decode_page_header(header_raw)
        except ValueError:
            return True
        if valid_fps is not None:
            return header.fingerprint not in valid_fps
        return False

    def maybe_sweep_orphans(self) -> None:
        """Periodic orphan sweep hook for the storage threads' idle paths."""
        now = time.monotonic()
        if now < self._next_orphan_sweep:
            return
        self._next_orphan_sweep = now + self._orphan_sweep_interval_seconds
        self.sweep_orphans()

    # ------------------------------------------------------------------
    # Per-pool batch IO (v2)
    # ------------------------------------------------------------------

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
            return {
                filename
                for filename in target_files
                if os.path.isfile(os.path.join(self.file_path, filename))
            }

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
            dict(get_components(page_offset))
            if get_components is not None
            else {str(pool_name): host_pool.get_data_page(page_offset, flat=True)}
        )
        storage_key = self._log_key(pool_name, key)
        for component, tensor in components.items():
            logger.warning(
                "HiCache page digest direction=%s pool=%s component=%s key=%s "
                "host_page=%d bytes=%d sha256=%s",
                direction,
                pool_name,
                component,
                storage_key,
                page_offset,
                tensor.numel() * tensor.element_size(),
                hicache_tensor_digest(tensor),
            )

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        storage_key = self._log_key(pool_name, key)
        page_indices = torch.arange(
            page_offset, page_offset + (host_pool.page_size or 1), dtype=torch.int64
        )
        segments = (
            self._page_segments(host_pool, page_indices)
            if self.io_engine.name != "posix_buffered"
            else None
        )
        if segments is not None:
            ok = self._read_page_bytes(
                storage_key, self._pool_fingerprint(pool_name), segments
            )
        else:
            data_page = host_pool.get_dummy_flat_data_page()
            if not self._read_page_bytes(
                storage_key,
                self._pool_fingerprint(pool_name),
                self._tensor_segments(data_page),
            ):
                return False
            host_pool.set_from_flat_data_page(page_offset, data_page)
            ok = True
        if ok:
            self._log_host_page_digests("read", pool_name, key, host_pool, page_offset)
        return ok

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage."""
        storage_key = self._log_key(pool_name, key)
        self._log_host_page_digests("write", pool_name, key, host_pool, page_offset)
        page_indices = torch.arange(
            page_offset, page_offset + (host_pool.page_size or 1), dtype=torch.int64
        )
        segments = (
            self._page_segments(host_pool, page_indices)
            if self.io_engine.name != "posix_buffered"
            else None
        )
        if segments is not None:
            return self._write_page_bytes(
                storage_key, self._pool_fingerprint(pool_name), segments
            )
        # The flat page must stay alive until the positional write reads it:
        # segments are raw pointers, not owners.
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self._write_page_bytes(
            storage_key,
            self._pool_fingerprint(pool_name),
            self._tensor_segments(data_page),
        )

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        """Shared v2 driver: one PoolTransfer at a time, one op_fn call per page.

        A failing page logs and returns False for itself only, so a batch
        always publishes one result per page — the failure-ack contract the
        prefetch sync threads rely on (a mid-batch failure clamps the
        prefix without dropping the remaining pages' results).
        """
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = host_pool.page_size or 1
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

            per_page: List[bool] = []
            for i, key in enumerate(keys):
                try:
                    ok = op_fn(
                        transfer.name,
                        key,
                        host_pool,
                        host_indices[i * page_size].item(),
                    )
                except Exception as e:
                    logger.error(
                        f"HiCacheFile {op_fn.__name__} failed for "
                        f"{transfer.name}/{key}: {e}"
                    )
                    ok = False
                per_page.append(ok)
            results[transfer.name] = per_page
        return results

    def _device_bounce(self, pool_name: str, page_bytes: int) -> torch.Tensor:
        """One aligned pinned whole-page bounce per staged pool."""
        with self._device_bounce_lock:
            bounce = self._device_bounces.get(pool_name)
            if bounce is None or bounce.numel() < page_bytes:
                raw = torch.empty(
                    page_bytes + 4096, dtype=torch.uint8, pin_memory=True
                )
                offset = (-raw.data_ptr()) % 4096
                bounce = raw.narrow(0, offset, page_bytes)
                self._device_bounces[pool_name] = bounce
            return bounce

    def _device_read_pages(self, transfer: PoolTransfer) -> List[bool]:
        """Read pages of one transfer into its device staging arena.

        Each page is read positionally into an aligned pinned bounce, then
        copied asynchronously onto its arena row on the staging pool's
        copy stream; the scatter submission waits the recorded events.
        Per-page failures return False so the caller clamps at the first
        failed page exactly like the host path.
        """
        keys = transfer.keys or []
        staging = self.registered_device_pools.get(str(transfer.name))
        if staging is None:
            logger.error(
                "HiCacheFile device-target read for %s has no staging pool",
                transfer.name,
            )
            return [False] * len(keys)
        page_size = staging.page_size
        device_indices = transfer.device_indices
        if device_indices is None or device_indices.numel() != len(keys) * page_size:
            logger.error(
                "device-target indices length mismatch for %s: expected %s",
                transfer.name,
                len(keys) * page_size,
            )
            return [False] * len(keys)
        pool_name = str(transfer.name)
        bounce = self._device_bounce(pool_name, staging.page_bytes)
        segments = [(bounce.data_ptr(), staging.page_bytes)]
        fingerprint = self._pool_fingerprint(pool_name)
        per_page: List[bool] = []
        for i, key in enumerate(keys):
            slot = int(device_indices[i * page_size].item()) // page_size
            try:
                ok = self._read_page_bytes(
                    self._log_key(pool_name, key), fingerprint, segments
                )
                if ok:
                    staging.copy_page_from_host(slot, bounce)
                    if self.log_page_digests:
                        for component, tensor in staging.get_debug_page_tensors(slot):
                            logger.warning(
                                "HiCache page digest direction=read pool=%s "
                                "component=%s key=%s host_page=%d bytes=%d "
                                "sha256=%s",
                                pool_name,
                                component,
                                self._log_key(pool_name, key),
                                slot,
                                tensor.numel() * tensor.element_size(),
                                hicache_tensor_digest(tensor),
                            )
            except Exception as e:
                logger.error(
                    "HiCacheFile device-target read failed for %s/%s: %s",
                    pool_name,
                    key,
                    e,
                )
                ok = False
            per_page.append(ok)
        return per_page

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        host_transfers = []
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            if transfer.target == PoolTransferTarget.DEVICE:
                results[transfer.name] = self._device_read_pages(transfer)
            else:
                host_transfers.append(transfer)
        results.update(self._batch_io_v2(host_transfers, self._read_page))
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
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
