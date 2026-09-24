"""Host-side storage for the offloaded Qwen4-Exp PLE n-gram table.

``--ple-offload-embedding`` keeps the PLE table (47.7 GiB in fp8 for
Qwen3.8-Flash-Next) out of device memory and lets the Triton gather kernel read
rows straight from a host pointer. Two backends provide that pointer:

``pinned`` (default)
    ``torch.empty(..., pin_memory=True)``. On a discrete GPU this frees VRAM.

``file``
    A file-backed, shared ``mmap`` of a sparse file under
    ``--ple-offload-dir``. Meant for unified-memory parts (GB10 / DGX Spark and
    similar), where pinned host memory comes out of the *same* pool as the
    model weights and ``pinned`` therefore frees nothing: Qwen3.8-Flash-Next is
    126.0 GiB of weights on a 121.63 GiB box and does not boot with ``pinned``.
    The kernel dereferences the pageable pointer directly, which only works on
    devices that report ``cudaDevAttrPageableMemoryAccessUsesHostPageTables``;
    rows are paged in from storage on demand, the file is sparse, deterministic
    in name and reused across restarts, and gathers of prefill size hint the
    page cache (``posix_fadvise(WILLNEED)``) so page faults are served
    concurrently instead of one at a time. A background trimmer keeps the
    mapping's resident set under a budget, because faulting rows in maps whole
    page-cache folios and the table would otherwise creep towards full
    residency (see ``PleFileRssTrimmer``).

``safetensors`` direct (``SGLANG_QWEN4_PLE_SAFETENSORS``, see
``build_ple_safetensors_table`` below)
    No staging and no pinned memory at all: the gather-time reader maps the
    checkpoint's own ``*.ngram_embedding.shard_N.weight`` tensors to physical
    (file, byte offset) descriptors parsed once from
    ``model.safetensors.index.json`` and the safetensors headers, and copies
    requested rows to the device. Small batches fetch through a parallel
    ``pread`` pool, bulk gathers use two-dimensional shard views over the source
    ``mmap``, and the existing background trimmer bounds mapped residency while
    leaving hot data in the page cache.

This module has no Triton or CUDA-kernel imports so that its allocator,
prefetcher and safetensors descriptors can be unit-tested on CPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MADV_RANDOM = 1
_MADV_DONTNEED = 4
_PAGE_SHIFT = 12
# One MADV_DONTNEED call takes mmap_lock for its whole range; over the full
# 47.7 GiB table that is ~3.5 s during which every fault in the process --
# including the ones the gather kernel takes -- stalls. Trim in slices.
PLE_FILE_RSS_TRIM_CHUNK_BYTES = 1 << 30
# Below this many rows a gather is decode-sized (16 rows per token): the page
# faults are cheap and the host-side hint would cost more than it saves.
PLE_FILE_PREFETCH_MIN_ROWS = 2048


class PleFilePrefetcher:
    """Hint the page cache about the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults tens of thousands of
    4 KiB pages one at a time from inside the gather kernel. Advising them
    first (``posix_fadvise(WILLNEED)`` per distinct page, on one background
    thread) lets the block layer serve them concurrently. Measured on a GB10 /
    NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm: ~2,200-2,600).
    Decode-sized gathers are skipped; nothing runs during CUDA-graph capture.
    """

    def __init__(
        self,
        path: str,
        row_bytes: int,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
    ) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def pages_for_rows(row_ids: torch.Tensor, row_bytes: int) -> list[int]:
        start = row_ids.to(torch.int64) * row_bytes
        end = start + (row_bytes - 1)
        return (
            torch.cat([start >> _PAGE_SHIFT, end >> _PAGE_SHIFT])
            .unique(sorted=True)
            .tolist()
        )

    def _advise(self, pages: list[int]) -> None:
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError:
                return

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the hint for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # The .cpu() syncs the stream; acceptable for prefill chunks (~1 s) and
        # it is what lets the page set be computed without touching the kernel.
        row_ids = flat_ids.detach().cpu()
        if vocab_end is not None:
            # The file contains only this rank's vocabulary shard.
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids - vocab_start
        if row_ids.numel() == 0:
            return False
        pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        try:
            os.close(self._fd)
        except OSError:
            pass


class PleFileRssTrimmer:
    """Keep the mapped table's resident set under a budget.

    Every random row fault maps in a whole page-cache folio, so with large
    folios (Linux 6.x) the mapping's Rss climbs towards the table's full size
    while a generated token only reads a few KB of it: measured ~45 KB of Rss
    growth per token on a GB10. On a unified-memory part that is not a slow
    leak, it is a countdown -- the free-memory readings that size the KV pool
    come from the same pool the folios are accumulating in.

    ``MADV_RANDOM`` does not prevent it (it limits readahead I/O, not the
    mapping-in of folios already in cache) and ``posix_fadvise(DONTNEED)`` does
    not release them either. ``MADV_DONTNEED`` over the mapping does: the page
    table entries go, the pages stay in the page cache, and hot rows come back
    at minor-fault cost.

    Dropping entries under a running gather is the state this backend already
    handles: the file starts out entirely unfaulted and every cold row is
    faulted in from inside the kernel through the same host page tables. What
    must not happen is one ``madvise`` call over the whole table, so the trim
    is chunked (see ``PLE_FILE_RSS_TRIM_CHUNK_BYTES``) and runs on its own
    daemon thread -- decode replays a CUDA graph and executes no Python, so a
    hook in the gather would never fire in the phase that grows the table.
    """

    def __init__(
        self,
        addr: int,
        nbytes: int,
        budget_bytes: int,
        interval_s: float,
        chunk_bytes: int = PLE_FILE_RSS_TRIM_CHUNK_BYTES,
    ) -> None:
        self._addr = int(addr)
        self._nbytes = int(nbytes)
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ple-file-rss-trim", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def mapping_rss_bytes(self) -> Optional[int]:
        """Resident bytes of the VMAs backing the table, or None off Linux."""
        return _mapping_rss_bytes(self._addr, self._nbytes)

    def trim_once(self) -> int:
        """Drop the mapping's resident pages if over budget. Returns bytes freed."""
        before = self.mapping_rss_bytes()
        if before is None or before <= self._budget:
            return 0
        for offset in range(0, self._nbytes, self._chunk):
            if self._stop.is_set():
                break
            length = min(self._chunk, self._nbytes - offset)
            if not _madvise(self._addr + offset, length, _MADV_DONTNEED):
                return 0
            # Let the faults that queued behind mmap_lock through.
            self._stop.wait(0.005)
        after = self.mapping_rss_bytes()
        freed = before - after if after is not None else 0
        logger.info(
            "PLE table: trimmed resident set %.1f -> %.1f GiB (budget %.1f GiB)",
            before / 2**30,
            (after if after is not None else 0) / 2**30,
            self._budget / 2**30,
        )
        return max(freed, 0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.trim_once()
            except Exception as exc:  # advisory only; never fail a request
                logger.warning("PLE table: resident-set trim skipped (%s)", exc)

    def close(self) -> None:
        self._stop.set()


def allocate_ple_host_table(
    shape: Sequence[int],
    dtype: torch.dtype,
    backend: str = "pinned",
    table_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> torch.Tensor:
    """Return a host tensor of ``shape``/``dtype`` for the PLE table.

    For the file backend, ``table_dir`` should be private to one checkpoint
    (the server defaults it to ``$SGLANG_CACHE_DIR/ple/<model path>``): the
    file name only encodes shape, dtype and ``tag``, and every boot rewrites
    the whole table through the weight loader.
    """
    if backend not in PLE_OFFLOAD_BACKENDS:
        raise ValueError(
            f"unknown PLE offload backend {backend!r}; choose from {PLE_OFFLOAD_BACKENDS}"
        )
    if backend == "pinned":
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    table_dir = os.path.expanduser(table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logger.info(
        "PLE table: file-backed mmap %s (%.1f GiB, %s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _madvise_random(storage, nbytes)
    table = storage.view(dtype).view(*[int(d) for d in shape])
    table._sglang_ple_file_path = path  # consumed by PleFilePrefetcher
    return table


def make_ple_file_prefetcher(table: torch.Tensor) -> Optional[PleFilePrefetcher]:
    """A prefetcher for a table returned by ``allocate_ple_host_table(..., "file")``."""
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    row_bytes = (
        int(table.shape[-1]) * table.element_size()
        if table.dim() >= 2
        else table.element_size()
    )
    prefetcher = PleFilePrefetcher(path=path, row_bytes=row_bytes)
    logger.info(
        "PLE table: WILLNEED prefetch on for gathers of >= %d rows (row = %d B)",
        PLE_FILE_PREFETCH_MIN_ROWS,
        row_bytes,
    )
    return prefetcher


def make_ple_file_rss_trimmer(table: torch.Tensor) -> Optional[PleFileRssTrimmer]:
    """A started trimmer for a table from ``allocate_ple_host_table(..., "file")``.

    ``SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`` turns it off; it is also absent
    where the resident set cannot be read (no ``/proc/self/smaps``).
    """
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None:
        return None
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb <= 0:
        return None
    nbytes = table.numel() * table.element_size()
    if _mapping_rss_bytes(table.data_ptr(), nbytes) is None:
        logger.warning(
            "PLE table: resident-set trim off, /proc/self/smaps is not readable; "
            "the mapping will creep towards %.1f GiB resident",
            nbytes / 2**30,
        )
        return None
    trimmer = PleFileRssTrimmer(
        addr=table.data_ptr(),
        nbytes=nbytes,
        budget_bytes=int(budget_gb * 2**30),
        interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    trimmer.start()
    logger.info(
        "PLE table: resident set capped at %.1f GiB, checked every %.0f s",
        budget_gb,
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def check_file_backend_supported(device_index: int = 0) -> None:
    """Fail fast at load time instead of silently reading garbage in the kernel."""
    if envs.SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK.get():
        logger.warning(
            "PLE table: file backend device check skipped by "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK"
        )
        return
    supported = device_uses_host_page_tables(device_index)
    if supported is None:
        raise RuntimeError(
            "--ple-offload-backend file: could not query "
            "cudaDevAttrPageableMemoryAccessUsesHostPageTables. Set "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 only if you know the "
            "device reads pageable host memory through the host page tables."
        )
    if not supported:
        raise ValueError(
            "--ple-offload-backend file needs a device whose pageable host "
            "memory accesses go through the host page tables (unified-memory "
            "parts such as GB10). This device reports it does not; use "
            "--ple-offload-backend pinned."
        )


def default_ple_table_dir(model_path: str) -> str:
    """``$SGLANG_QWEN4_PLE_FILE_DIR/<model path>``, one directory per checkpoint."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path).rstrip("/")).strip("_")
    return os.path.join(envs.SGLANG_QWEN4_PLE_FILE_DIR.get(), safe or "model")


def ple_table_file_name(
    shape: Sequence[int], dtype: torch.dtype, tag: Optional[str] = None
) -> str:
    """Deterministic file name so the sparse table is reused across restarts.

    ``tag`` distinguishes tables of the same shape that must not share a file,
    e.g. the vocabulary shards of different tensor-parallel ranks.
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    elem = torch.empty(0, dtype=dtype).element_size()
    dims = "x".join(str(int(d)) for d in shape)
    suffix = f"_{tag}" if tag else ""
    return f"ple_table_{dims}_{str(dtype).replace('torch.', '')}_{numel * elem}B{suffix}.bin"


def device_uses_host_page_tables(device_index: int = 0) -> Optional[bool]:
    """Whether pageable host memory is directly addressable by the GPU.

    Returns None when the CUDA runtime library cannot be queried.
    """
    candidates = [ctypes.util.find_library("cudart")]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        candidates += sorted(
            os.path.join(torch_lib, f)
            for f in os.listdir(torch_lib)
            if f.startswith("libcudart.so")
        )
    try:
        import nvidia.cuda_runtime  # type: ignore

        nv_lib = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
        if os.path.isdir(nv_lib):
            candidates += sorted(
                os.path.join(nv_lib, f)
                for f in os.listdir(nv_lib)
                if f.startswith("libcudart.so")
            )
    except Exception:
        pass
    for name in [c for c in candidates if c]:
        try:
            cudart = ctypes.CDLL(name)
            value = ctypes.c_int()
            rc = cudart.cudaDeviceGetAttribute(
                ctypes.byref(value),
                ctypes.c_int(
                    _CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
                ),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                return bool(value.value)
        except OSError:
            continue
    return None


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


def _libc() -> Optional[ctypes.CDLL]:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
        except OSError:
            return None
    return _LIBC


def _madvise(addr: int, length: int, advice: int) -> bool:
    """``madvise(2)`` on our own mapping. Advisory: never affects correctness."""
    libc = _libc()
    if libc is None:
        return False
    try:
        rc = libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)
        )
    except Exception:
        return False
    if rc != 0:
        logger.warning(
            "PLE table: madvise(advice=%d) failed (errno %d)",
            advice,
            ctypes.get_errno(),
        )
        return False
    return True


def _mapping_rss_bytes(
    addr: int, nbytes: int, smaps_path: str = "/proc/self/smaps"
) -> Optional[int]:
    """Resident bytes of the VMAs overlapping ``[addr, addr + nbytes)``.

    Summed per mapping rather than taken from ``statm``/``smaps_rollup``: only
    the table's own residency should drive the trim, and on a unified-memory
    box the process RSS is dominated by everything else.
    """
    lo, hi = int(addr), int(addr) + int(nbytes)
    total = 0
    overlapping = False
    try:
        with open(smaps_path, "r") as f:
            for line in f:
                header = _SMAPS_HEADER.match(line)
                if header is not None:
                    start = int(header.group(1), 16)
                    end = int(header.group(2), 16)
                    overlapping = start < hi and end > lo
                elif overlapping:
                    rss = _SMAPS_RSS.match(line)
                    if rss is not None:
                        total += int(rss.group(1)) * 1024
    except OSError:
        return None
    return total


# ---------------------------------------------------------------------------
# Direct checkpoint backend (SGLANG_QWEN4_PLE_SAFETENSORS)
# ---------------------------------------------------------------------------

# The checkpoint stores the PLE n-gram table as ``<prefix>.ngram_embedding.
# shard_N.weight`` tensors; N is the sync-shard index and N * shard_size its
# first global row. The regex is deliberately layout-agnostic: within a
# safetensors file the physical data offsets follow the *lexical* order of the
# tensor names (shard 10 sits right after shard 1, and shard 2 can live in a
# later file after shard 127), so every row address comes from the parsed
# header, never from positional arithmetic over the file.
PLE_SHARD_TENSOR_RE = re.compile(r".*\.ngram_embedding\.shard_(\d+)\.weight$")

# Supported on-disk dtypes (safetensors header spelling) -> storage dtype.
_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
}

# Gathers up to this many rows are decode-sized: fetch them through a parallel
# pread pool (GIL released); larger gathers fancy-index per-file mmap views.
PLE_DIRECT_PREAD_MAX_ROWS = 64
PLE_DIRECT_PREAD_THREADS = 16


@dataclass(frozen=True)
class PleSafetensorsShard:
    """Validated numeric descriptor of one checkpoint n-gram shard."""

    shard_index: int
    row_start: int  # global row id of this shard's first row
    rows: int  # row count actually present in the tensor
    file_index: int  # index into PleSafetensorsTable.files
    offset: int  # byte offset of the tensor data inside the file


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Parse one safetensors header.

    Returns ``(entries, data_base)``; data_offsets in the entries are relative
    to ``data_base`` (the end of the header), which is where the numeric
    descriptors' absolute byte offsets come from."""
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (header_bytes,) = struct.unpack("<Q", prefix)
            file_size = os.fstat(f.fileno()).st_size
            if header_bytes == 0 or header_bytes > file_size - 8:
                raise ValueError(
                    f"{path}: invalid safetensors header size {header_bytes}"
                )
            raw = f.read(header_bytes)
    except OSError as exc:
        raise ValueError(f"{path}: cannot read safetensors header ({exc})") from exc
    if len(raw) != header_bytes:
        raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed safetensors header ({exc})") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header, 8 + header_bytes


def build_ple_safetensors_table(
    directory: str,
    *,
    shard_size: int,
    expected_dim: Optional[int] = None,
    expected_rows: Optional[int] = None,
) -> PleSafetensorsTable:
    """Parse a checkpoint into validated numeric shard descriptors, once.

    Uses ``model.safetensors.index.json`` when present to locate the PLE shard
    tensors (otherwise every ``*.safetensors`` file in the directory is
    scanned), then opens the header of each file holding one exactly once.
    Shard ``N`` describes global rows ``[N * shard_size, N * shard_size +
    rows)``; the row count and byte offset come from the tensor's own header
    entry, so multi-file layouts and the lexical physical order are handled
    without positional assumptions.

    The shard *set* must be complete: ids consecutive from 0 (a missing shard
    is a checkpoint error, not a zero-fill row) and, with ``expected_rows``
    (the embedding's global ``org_vocab_size``), every row up to it present --
    all shards full except a final tensor shortened exactly as
    ``ceil(expected_rows / shard_size)`` geometry implies. Gaps *within* the
    files are fine: PLE tensors may interleave with other tensors and span
    any file layout.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        raise ValueError(
            f"SGLANG_QWEN4_PLE_SAFETENSORS: {directory!r} is not a directory"
        )
    if int(shard_size) <= 0:
        raise ValueError(f"shard_size must be > 0, got {shard_size}")
    shard_size = int(shard_size)

    # name -> file name, from the index when it exists.
    wanted: dict[str, list[tuple[int, str]]] = {}  # path -> [(shard, tensor)]
    scanned_headers: dict[str, tuple[dict, int]] = {}
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{index_path}: unreadable index ({exc})") from exc
        for name, file_name in weight_map.items():
            match = PLE_SHARD_TENSOR_RE.match(name)
            if match is not None:
                path = os.path.join(directory, file_name)
                wanted.setdefault(path, []).append((int(match.group(1)), name))
        if not wanted:
            raise ValueError(
                f"{index_path}: no *.ngram_embedding.shard_N.weight tensors in "
                "the weight map; this checkpoint cannot serve the direct PLE "
                "table"
            )
    else:
        for file_name in sorted(os.listdir(directory)):
            if not file_name.endswith(".safetensors"):
                continue
            path = os.path.join(directory, file_name)
            header, data_base = _read_safetensors_header(path)
            matches = [
                (int(m.group(1)), name)
                for name in header
                if (m := PLE_SHARD_TENSOR_RE.match(name)) is not None
            ]
            if matches:
                wanted[path] = matches
                scanned_headers[path] = (header, data_base)
        if not wanted:
            raise ValueError(
                f"{directory}: no *.safetensors file holds an "
                "*.ngram_embedding.shard_N.weight tensor"
            )

    files: list[str] = []
    file_indices: dict[str, int] = {}
    by_shard: dict[int, PleSafetensorsShard] = {}
    dtype_str: Optional[str] = None
    dim: Optional[int] = None
    for path in sorted(wanted):
        matches = wanted[path]
        header, data_base = scanned_headers.get(path) or _read_safetensors_header(path)
        try:
            file_size = os.path.getsize(path)
        except OSError as exc:
            raise ValueError(f"{path}: unreadable shard file ({exc})") from exc
        for shard_index, name in matches:
            entry = header.get(name)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}: index lists {name!r} but the header omits it"
                )
            on_disk = entry.get("dtype")
            if on_disk not in _SAFETENSORS_DTYPES:
                raise ValueError(
                    f"{name}: unsupported PLE dtype {on_disk!r}; direct PLE "
                    f"table supports {sorted(_SAFETENSORS_DTYPES)}"
                )
            if dtype_str is None:
                dtype_str = on_disk
            elif dtype_str != on_disk:
                raise ValueError(
                    f"{name}: PLE dtype {on_disk!r} disagrees with the other "
                    f"PLE shards ({dtype_str!r})"
                )
            shape = entry.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(isinstance(d, int) and d > 0 for d in shape)
            ):
                raise ValueError(f"{name}: PLE tensor must be 2-D, got {shape!r}")
            rows, tensor_dim = int(shape[0]), int(shape[1])
            if dim is None:
                dim = tensor_dim
            elif dim != tensor_dim:
                raise ValueError(
                    f"{name}: PLE dim {tensor_dim} disagrees with the other "
                    f"PLE shards ({dim})"
                )
            if expected_dim is not None and tensor_dim != int(expected_dim):
                raise ValueError(
                    f"{name}: PLE dim {tensor_dim} does not match "
                    f"embedding_dim={expected_dim}"
                )
            offsets = entry.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(o, int) and o >= 0 for o in offsets)
            ):
                raise ValueError(f"{name}: invalid data_offsets {offsets!r}")
            begin, end = data_base + int(offsets[0]), data_base + int(offsets[1])
            itemsize = torch.empty(0, dtype=_SAFETENSORS_DTYPES[on_disk]).element_size()
            if (
                begin >= end
                or end > file_size
                or end - begin != rows * tensor_dim * itemsize
            ):
                raise ValueError(
                    f"{name}: data offsets [{begin}, {end}) do not cover "
                    f"{rows} x {tensor_dim} x {itemsize} bytes in {path} "
                    f"({file_size} bytes)"
                )
            row_start = shard_index * shard_size
            if row_start + rows > (shard_index + 1) * shard_size:
                raise ValueError(
                    f"{name}: {rows} rows at global row {row_start} overflow "
                    f"the {shard_size}-row shard slot"
                )
            if shard_index in by_shard:
                other = by_shard[shard_index]
                raise ValueError(
                    f"{name}: duplicate PLE shard {shard_index}, already "
                    f"described by {files[other.file_index]!r}"
                )
            if path not in file_indices:
                file_indices[path] = len(files)
                files.append(path)
            by_shard[shard_index] = PleSafetensorsShard(
                shard_index=shard_index,
                row_start=row_start,
                rows=rows,
                file_index=file_indices[path],
                offset=begin,
            )

    ordered = sorted(by_shard.values(), key=lambda s: s.shard_index)
    if [s.shard_index for s in ordered] != list(range(len(ordered))):
        missing = sorted(set(range(ordered[-1].shard_index + 1)) - set(by_shard))
        raise ValueError(
            f"{directory}: PLE shard ids must be consecutive starting at 0; "
            f"missing shard(s) {missing} (checkpoint holds {sorted(by_shard)})"
        )
    if expected_rows is not None:
        expected_rows = int(expected_rows)
        if expected_rows <= 0:
            raise ValueError(f"expected_rows must be > 0, got {expected_rows}")
        expected_shards = -(-expected_rows // shard_size)
        if len(ordered) != expected_shards:
            raise ValueError(
                f"{directory}: {len(ordered)} PLE shard(s) of {shard_size} "
                f"rows cannot cover org_vocab_size={expected_rows} "
                f"(expected {expected_shards} shards)"
            )
        for shard in ordered[:-1]:
            if shard.rows != shard_size:
                raise ValueError(
                    f"{directory}: PLE shard {shard.shard_index} holds "
                    f"{shard.rows} of {shard_size} rows at global row "
                    f"{shard.row_start}; org_vocab_size={expected_rows} "
                    "leaves no missing-row gaps before the final shard"
                )
        last = ordered[-1]
        if last.row_start + last.rows != expected_rows:
            raise ValueError(
                f"{directory}: PLE shards end at global row "
                f"{last.row_start + last.rows}, expected "
                f"org_vocab_size={expected_rows}"
            )
    shards = tuple(ordered)
    table = PleSafetensorsTable(
        files=files, shards=shards, dtype=_SAFETENSORS_DTYPES[dtype_str], dim=dim
    )
    logger.info(
        "PLE table: direct safetensors from %s (%d shards covering %d rows "
        "x %d, %s, in %d file(s))",
        directory,
        len(shards),
        table.num_rows,
        dim,
        dtype_str,
        len(files),
    )
    return table


class PleSafetensorsTable:
    """Row reads straight from the checkpoint's safetensors shards.

    Holds the validated numeric descriptors built by
    ``build_ple_safetensors_table`` plus lazily opened read fds and mmap views
    (one per shard file). Rows are addressed by *global* row id, which under
    TP is just the token id the gather was asked for, so no per-rank
    re-basing ever happens; ids a rank does not own are zero-filled by the
    caller. Not thread-safe for reads on purpose: gathers are serialized by
    the model runner.
    """

    def __init__(
        self,
        *,
        files: Sequence[str],
        shards: Sequence[PleSafetensorsShard],
        dtype: torch.dtype,
        dim: int,
    ) -> None:
        if not shards:
            raise ValueError("empty PLE shard descriptor set")
        self.files = tuple(files)
        self.shards = tuple(sorted(shards, key=lambda shard: shard.row_start))
        self.dtype = dtype
        self.dim = int(dim)
        self.row_bytes = self.dim * torch.empty(0, dtype=dtype).element_size()
        self.num_rows = int(self.shards[-1].row_start + self.shards[-1].rows)
        self._starts = np.array(
            [shard.row_start for shard in self.shards], dtype=np.int64
        )
        self._rows = np.array([shard.rows for shard in self.shards], dtype=np.int64)
        self._offsets = np.array(
            [shard.offset for shard in self.shards], dtype=np.int64
        )
        self._file_of = np.array(
            [shard.file_index for shard in self.shards], dtype=np.int64
        )
        self._fds: list[Optional[int]] = [None] * len(files)
        self._views: dict[int, np.memmap] = {}
        self._shard_views: dict[int, np.ndarray] = {}
        self._trimmers: dict[int, PleFileRssTrimmer] = {}
        self._pool: Optional[ThreadPoolExecutor] = None

    def read_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Rows as an ``(n, row_bytes)`` uint8 tensor, global row ids in.

        Row ids outside every checkpoint shard (table padding past the last
        saved shard) come back as zero bytes.
        """
        ids = rows.reshape(-1).to(torch.int64).numpy()
        pos, valid, byte_offsets = self._locate(ids)
        if ids.size <= PLE_DIRECT_PREAD_MAX_ROWS:
            return self._pread_fetch(ids.size, pos, valid, byte_offsets)
        return self._mmap_fetch(ids.size, pos, valid, byte_offsets)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for trimmer in self._trimmers.values():
            trimmer.close()
        self._trimmers.clear()
        for i, fd in enumerate(self._fds):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._fds[i] = None
        self._shard_views.clear()
        for view in self._views.values():
            view._mmap.close()
        self._views.clear()

    # -- internals ---------------------------------------------------------

    def _locate(self, ids: np.ndarray):
        """Global row ids -> (shard position, valid, byte offset), vectorized.

        The searchsorted-over-row-starts lookup handles multi-file and
        nonuniform shards and is indifferent to the physical order of the
        files and of the tensors inside them."""
        pos = np.searchsorted(self._starts, ids, side="right") - 1
        valid = pos >= 0
        pos = np.where(valid, pos, 0)
        local = ids - self._starts[pos]
        valid &= local < self._rows[pos]
        return pos, valid, self._offsets[pos] + local * self.row_bytes

    def _fd(self, file_index: int) -> int:
        fd = self._fds[file_index]
        if fd is None:
            fd = os.open(self.files[file_index], os.O_RDONLY)
            try:
                # Random row offsets: stop the readahead a sequential hint
                # would pull around every cold row.
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                pass
            self._fds[file_index] = fd
        return fd

    def _view(self, file_index: int) -> np.memmap:
        view = self._views.get(file_index)
        if view is None:
            view = np.memmap(self.files[file_index], dtype=np.uint8, mode="r")
            try:
                import mmap as mmap_module

                view._mmap.madvise(mmap_module.MADV_RANDOM)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "PLE table: direct madvise(MADV_RANDOM) on %s failed: %s",
                    self.files[file_index],
                    exc,
                )
            self._views[file_index] = view
            total_budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
            if (
                total_budget_gb > 0
                and _mapping_rss_bytes(view.ctypes.data, view.nbytes) is not None
            ):
                trimmer = PleFileRssTrimmer(
                    addr=view.ctypes.data,
                    nbytes=view.nbytes,
                    budget_bytes=int(total_budget_gb * 2**30 / len(self.files)),
                    interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
                )
                trimmer.start()
                self._trimmers[file_index] = trimmer
        return view

    def _shard_view(self, shard_position: int) -> np.ndarray:
        view = self._shard_views.get(shard_position)
        if view is None:
            shard = self.shards[shard_position]
            view = np.ndarray(
                shape=(shard.rows, self.row_bytes),
                dtype=np.uint8,
                buffer=self._view(shard.file_index),
                offset=shard.offset,
            )
            self._shard_views[shard_position] = view
        return view

    def _pread_fetch(self, n: int, pos, valid, byte_offsets) -> torch.Tensor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=PLE_DIRECT_PREAD_THREADS)
        out = bytearray(self.row_bytes * n)
        jobs = [
            (i, self._fd(int(self._file_of[p])), int(off))
            for i, (p, ok, off) in enumerate(zip(pos, valid, byte_offsets))
            if ok
        ]

        def _read(job):
            i, fd, off = job
            buf = os.pread(fd, self.row_bytes, off)
            if len(buf) != self.row_bytes:
                raise ValueError(
                    f"PLE table: short pread at {off}: {len(buf)} of "
                    f"{self.row_bytes} bytes"
                )
            return i, buf

        for i, buf in self._pool.map(_read, jobs):
            base = i * self.row_bytes
            out[base : base + self.row_bytes] = buf
        return torch.frombuffer(out, dtype=torch.uint8).reshape(n, self.row_bytes)

    def _mmap_fetch(self, n: int, pos, valid, byte_offsets) -> torch.Tensor:
        out = np.zeros((n, self.row_bytes), dtype=np.uint8)
        valid_indices = np.flatnonzero(valid)
        if not valid_indices.size:
            return torch.from_numpy(out)

        order = np.argsort(pos[valid_indices], kind="stable")
        ordered_indices = valid_indices[order]
        ordered_positions = pos[ordered_indices]
        boundaries = np.flatnonzero(np.diff(ordered_positions)) + 1
        for indices in np.split(ordered_indices, boundaries):
            shard_position = int(pos[indices[0]])
            local_rows = (
                byte_offsets[indices] - self._offsets[shard_position]
            ) // self.row_bytes
            out[indices] = self._shard_view(shard_position)[local_rows]
        return torch.from_numpy(out)
