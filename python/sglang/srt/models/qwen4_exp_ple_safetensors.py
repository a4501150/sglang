"""Qwen4-Exp PLE n-gram table served directly from checkpoint safetensors.

``--ple-offload-backend safetensors`` reads the PLE rows a gather needs
straight from the checkpoint's ``*.ngram_embedding.shard_N.weight``
tensors on storage, into a device staging arena, at gather time. The table
for Qwen3.8-Flash-Next is 95.4 GiB of bf16 -- too much for the ~89 GiB of
host RAM behind ``pinned``, and ``file`` needs a device that dereferences
pageable host memory, which discrete GPUs do not do. Nothing is staged or
allocated up front: only the safetensors headers are parsed, once, into
shard descriptors, and each gather touches only its own rows.

Gather pipeline (``Qwen4ExpSafetensorsEmbedding``):

1. ``torch.unique`` on the lookup ids (GPU, current stream) and one async
   D2H of the unique ids into a pinned buffer.
2. A single host worker thread maps unique ids to ``(file, byte offset)``
   via the descriptors, merges them into IO spans (4 KiB-aligned and
   merged in gds/p2pdma mode, exact and merged in compat mode), and reads
   the spans with a ``CuFileBatch`` into a preallocated registered device
   arena -- in arena-sized rounds when a gather needs more than fits.
   Without libcufile the same span plan is served by an ``O_DIRECT``
   ``preadv`` pool into a pinned host arena plus one async H2D per round.
3. A Triton kernel scatters arena rows through the unique-inverse mapping
   into the caller's output buffer (bf16; fp8 rows converted), zero for
   ids outside this rank's vocabulary shard.

Steps 2-3 run on the worker thread while the caller's thread keeps
launching layers; ``join_prefetch`` blocks until the scatter is enqueued
on the gather's stream. cuFile calls cannot be CUDA-graph captured: with
the breakable graph backend the prefetch runs inside an eager break
(replayed live every decode step), and the gather raises if it ever runs
inside a captured segment.

The descriptor parsing and span planning in this module are numpy/pure
and unit-testable on CPU; the cuFile/pread engines and the Triton
scatter need a GPU.
"""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
import json
import logging
import os
import queue
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import msgspec
import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    is_allocation_symmetric,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    is_capturing as breakable_capture_in_progress,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import cufile

logger = logging.getLogger(__name__)

PLE_SHARD_TENSOR_RE = re.compile(r".*\.ngram_embedding\.shard_(\d+)\.weight$")

# safetensors header dtype spelling -> storage dtype.
_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
}

_PAGE_BYTES = 4096
# Cap on spans (and so on per-round IOs) before the arena is reused; a
# 4096-token prefill of the 16-lookup table worst-cases one page per row.
_MAX_SPANS_PER_ROUND = 16384
_IO_TIMEOUT_S = 120.0
_PREAD_THREADS = 16


def _read_safetensors_header(path: str) -> Tuple[dict, int]:
    """Parse one safetensors header into ``(entries, data_base)``.

    ``data_offsets`` in the entries are relative to ``data_base``, the end
    of the header; the descriptors' byte offsets are absolute.
    """
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (header_bytes,) = struct.unpack("<Q", prefix)
            file_size = os.fstat(f.fileno()).st_size
            if header_bytes == 0 or header_bytes > file_size - 8:
                raise ValueError(f"{path}: invalid safetensors header size")
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


class PleSafetensorsShard(msgspec.Struct, frozen=True):
    """Validated numeric descriptor of one checkpoint n-gram shard."""

    shard_index: int
    row_start: int  # global row id of this shard's first row
    rows: int  # row count present in the tensor
    file_index: int  # index into PleSafetensorsTable.files
    offset: int  # byte offset of the tensor data inside the file


class PleSafetensorsTable:
    """Numeric shard descriptors for one PLE n-gram table.

    The checkpoint stores the table as ``<prefix>.ngram_embedding.
    shard_N.weight`` tensors; shard ``N`` covers global rows starting at
    ``N * shard_size``. Row addresses come from the parsed headers, never
    from positional arithmetic over the files: the physical order of the
    tensors inside a safetensors file is lexical, so shard 10 sits between
    shard 1 and shard 100. Locating a row is a numpy ``searchsorted`` over
    the shard row starts; gathers address rows by global id, which under TP
    is just the id the lookup produced (out-of-shard ids zero-fill).
    """

    def __init__(
        self,
        *,
        files: Tuple[str, ...],
        shards: Tuple[PleSafetensorsShard, ...],
        dtype: torch.dtype,
        dim: int,
    ) -> None:
        if not shards:
            raise ValueError("empty PLE shard descriptor set")
        self.files = files
        self.shards = shards
        self.dtype = dtype
        self.dim = int(dim)
        self.row_bytes = self.dim * torch.empty(0, dtype=dtype).element_size()
        self.num_rows = int(shards[-1].row_start + shards[-1].rows)
        self._starts = np.array([s.row_start for s in shards], dtype=np.int64)
        self._rows = np.array([s.rows for s in shards], dtype=np.int64)
        self._offsets = np.array([s.offset for s in shards], dtype=np.int64)
        self._file_of = np.array([s.file_index for s in shards], dtype=np.int64)

    @classmethod
    def build(
        cls,
        directory: str,
        *,
        expected_dim: int,
        expected_rows: int,
    ) -> PleSafetensorsTable:
        """Parse and validate a checkpoint's PLE shards, once.

        Uses ``model.safetensors.index.json`` when present (otherwise every
        ``*.safetensors`` file is scanned). The shard set must be complete --
        ids consecutive from 0, full shards except a possibly short final
        one, together covering at least ``expected_rows`` global rows -- and
        uniform in dtype and dim. Gaps within the files are fine: the
        tensors may interleave with other tensors and span any file layout.
        """
        directory = os.path.realpath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            raise ValueError(
                f"PLE safetensors backend: {directory!r} is not a directory"
            )
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"{index_path}: unreadable index ({exc})") from exc
            matches = sorted(
                (int(m.group(1)), name, fname)
                for name, fname in weight_map.items()
                if (m := PLE_SHARD_TENSOR_RE.match(name)) is not None
            )
            if not matches:
                raise ValueError(
                    f"{index_path}: no *.ngram_embedding.shard_N.weight tensors "
                    "in the weight map; the safetensors PLE backend cannot "
                    "serve this checkpoint"
                )
        else:
            matches = []
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(".safetensors"):
                    continue
                header, _ = _read_safetensors_header(os.path.join(directory, fname))
                for name in header:
                    m = PLE_SHARD_TENSOR_RE.match(name)
                    if m is not None:
                        matches.append((int(m.group(1)), name, fname))
            if not matches:
                raise ValueError(
                    f"{directory}: no *.safetensors file holds an "
                    "*.ngram_embedding.shard_N.weight tensor"
                )
            matches.sort()

        files: list[str] = []
        file_indices: dict[str, int] = {}
        # shard_index -> (rows, file_index, offset)
        parsed: dict[int, Tuple[int, int, int]] = {}
        dtype_str: Optional[str] = None
        dim: Optional[int] = None
        for shard_index, name, fname in matches:
            path = os.path.join(directory, fname)
            header, data_base = _read_safetensors_header(path)
            entry = header.get(name)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}: index lists {name!r} but the header omits it"
                )
            on_disk = entry.get("dtype")
            if on_disk not in _SAFETENSORS_DTYPES:
                raise ValueError(
                    f"{name}: unsupported PLE dtype {on_disk!r}; the "
                    f"safetensors PLE backend supports {sorted(_SAFETENSORS_DTYPES)}"
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
            if tensor_dim != int(expected_dim):
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
            begin = data_base + int(offsets[0])
            end = data_base + int(offsets[1])
            itemsize = torch.empty(0, dtype=_SAFETENSORS_DTYPES[on_disk]).element_size()
            if (
                end > os.path.getsize(path)
                or end - begin != rows * tensor_dim * itemsize
            ):
                raise ValueError(
                    f"{name}: data offsets [{begin}, {end}) do not cover "
                    f"{rows} x {tensor_dim} x {itemsize} bytes in {path}"
                )
            if shard_index in parsed:
                raise ValueError(f"{name}: duplicate PLE shard {shard_index}")
            if path not in file_indices:
                file_indices[path] = len(files)
                files.append(path)
            parsed[shard_index] = (rows, file_indices[path], begin)

        shard_ids = sorted(parsed)
        if shard_ids != list(range(len(shard_ids))):
            missing = sorted(set(range(max(shard_ids) + 1)) - set(shard_ids))
            raise ValueError(
                f"{directory}: PLE shard ids must be consecutive from 0; "
                f"missing shard(s) {missing}"
            )
        # The producer writes full shards except a possibly short final one;
        # the first shard's row count is the shard size. Coverage must reach
        # org_vocab_size (lookups never address the padding rows past it, so
        # a checkpoint that stops at org rows or at the padded row count both
        # serve; rows past coverage zero-fill in the scatter).
        shard_size = parsed[shard_ids[0]][0]
        if any(parsed[i][0] != shard_size for i in shard_ids[:-1]):
            raise ValueError(
                f"{directory}: only the final PLE shard may be short of "
                f"{shard_size} rows"
            )
        shards = []
        for shard_index in shard_ids:
            rows, file_index, offset = parsed[shard_index]
            shards.append(
                PleSafetensorsShard(
                    shard_index=shard_index,
                    row_start=shard_index * shard_size,
                    rows=rows,
                    file_index=file_index,
                    offset=offset,
                )
            )
        covered = shards[-1].row_start + shards[-1].rows
        if covered < int(expected_rows):
            raise ValueError(
                f"{directory}: PLE shards cover {covered} global rows, less "
                f"than the {expected_rows}-row table"
            )
        table = cls(
            files=tuple(files),
            shards=tuple(shards),
            dtype=_SAFETENSORS_DTYPES[dtype_str],
            dim=int(dim),
        )
        logger.info(
            "PLE table: safetensors backend, %s (%d shards covering %d rows "
            "x %d, %s, %d file(s))",
            directory,
            len(shards),
            table.num_rows,
            table.dim,
            dtype_str,
            len(files),
        )
        return table

    def locate(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Global row ids -> ``(shard position, valid, byte offset)``."""
        pos = np.searchsorted(self._starts, ids, side="right") - 1
        valid = pos >= 0
        pos = np.where(valid, pos, 0)
        local = ids - self._starts[pos]
        valid &= local < self._rows[pos]
        return pos, valid, self._offsets[pos] + local * self.row_bytes


def plan_io_spans(
    file_idx: np.ndarray,
    byte_offsets: np.ndarray,
    row_bytes: int,
    *,
    align_4k: bool,
    arena_bytes: int,
    max_spans: int = _MAX_SPANS_PER_ROUND,
    file_sizes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, list[Tuple[np.ndarray, list[Tuple[int, int, int, int]]]]]:
    """Group row addresses into merged IO spans, in arena-sized rounds.

    Rows are sorted by ``(file, offset)``; touching or overlapping spans
    merge, which is what makes a 320 B row that straddles a 4 KiB boundary
    one span instead of two IOs. ``align_4k`` floors/ceils span bounds to
    4 KiB pages, which the gds/p2pdma data path requires; compat mode keeps
    spans exact so a row costs 320 B of staging instead of up to a page.
    A merged run longer than the arena splits at row boundaries (adjacent
    split spans may re-read up to one shared page, which is harmless).

    Returns ``(arena_offset_per_row, rounds)``: each row's byte offset
    inside its round's arena window (input order), and one
    ``(row_indices, spans)`` per round, the spans as
    ``(file_index, file_offset, arena_offset, size)``. Row indices are into
    the input arrays; rounds are bounded by the arena size and
    ``max_spans``. ``file_sizes`` (bytes, indexed by file) caps aligned span
    ends at end of file, where a page-rounded read would come back short.
    """
    n = byte_offsets.size
    if n == 0:
        return np.empty(0, dtype=np.int64), []
    order = np.lexsort((byte_offsets, file_idx))
    sorted_off = byte_offsets[order]
    sorted_file = file_idx[order]
    starts = sorted_off if not align_4k else sorted_off & ~(_PAGE_BYTES - 1)
    ends = sorted_off + row_bytes
    if align_4k:
        ends = (ends + _PAGE_BYTES - 1) & ~(_PAGE_BYTES - 1)
        if file_sizes is not None:
            ends = np.minimum(ends, file_sizes[sorted_file])
    # Merge run: a span extends while the next row starts before or at the
    # accumulated end of its run. The accumulate must reset at file
    # boundaries -- a carried-over end would swallow every later row of the
    # next file into one giant span.
    run_end = ends.copy()
    new_file = np.concatenate(([True], sorted_file[1:] != sorted_file[:-1]))
    file_start_idx = np.flatnonzero(new_file)
    for a, b in zip(file_start_idx, np.concatenate((file_start_idx[1:], [n]))):
        run_end[a:b] = np.maximum.accumulate(ends[a:b])
    same_file = ~new_file[1:]
    new_group = np.concatenate(([True], (starts[1:] > run_end[:-1]) | ~same_file))
    group_start_idx = np.flatnonzero(new_group)
    group_sizes = np.diff(np.concatenate((group_start_idx, [n])))
    # Rows of one group are contiguous in the file; a group longer than the
    # arena splits into row batches, each batch its own span.
    per_row_cost = row_bytes + (2 * _PAGE_BYTES if align_4k else 0)
    rows_per_batch = max(1, arena_bytes // per_row_cost)
    pos_in_group = np.arange(n) - np.repeat(group_start_idx, group_sizes)
    # A unit is one span: a whole merge group, or one batch of a split one.
    # The composite key is densified below; it is not itself an index.
    group_id = np.cumsum(new_group) - 1
    if rows_per_batch > 1:
        batches_per_group = (n + rows_per_batch - 1) // rows_per_batch + 1
        composite = group_id * batches_per_group + pos_in_group // rows_per_batch
    else:
        composite = group_id
    first_of_unit = np.concatenate(([True], composite[1:] != composite[:-1]))
    unit_id = np.cumsum(first_of_unit) - 1
    unit_start_idx = np.flatnonzero(first_of_unit)
    unit_sizes = np.diff(np.concatenate((unit_start_idx, [n])))
    unit_start = starts[unit_start_idx]
    # Within a unit the rows are ascending, so the last member's end is the
    # unit end (run_end at that position equals it after the reset fix).
    unit_end = ends[unit_start_idx + unit_sizes - 1]
    unit_size = unit_end - unit_start

    # Greedy round packing over units, vectorized: repeated searchsorted on
    # the cumulative unit sizes replaces a per-unit Python loop, which cost
    # ~250 ms per 131k-row prefill gather.
    n_units = int(unit_id[-1]) + 1 if n else 0
    cum = np.concatenate(([0], np.cumsum(unit_size)))
    round_starts = [0]
    start = 0
    while start < n_units:
        end_by_spans = min(n_units, start + max_spans)
        end_by_bytes = int(np.searchsorted(cum, cum[start] + arena_bytes, "right")) - 1
        start = max(start + 1, min(end_by_spans, end_by_bytes))
        round_starts.append(start)
    round_starts = np.array(round_starts[:-1], dtype=np.int64)
    round_of_unit = np.searchsorted(round_starts, np.arange(n_units), "right") - 1
    units_per_round = np.diff(np.concatenate((round_starts, [n_units])))
    # cum[u] is the bytes before unit u (leading zero), so a unit's arena
    # window is its cumulative offset minus the round's first unit's.
    cum_round_start = np.repeat(cum[round_starts], units_per_round)
    arena_of_unit = cum[:n_units] - cum_round_start

    # Per-row arena offsets, back in input order.
    arena_offset_per_row = np.full(n, -1, dtype=np.int64)
    arena_offset_per_row[order] = (
        np.repeat(arena_of_unit, unit_sizes)
        + sorted_off
        - np.repeat(unit_start, unit_sizes)
    )

    # Rounds hold their units' rows (input order) and span arrays. The
    # per-unit arrays below are unit-id indexed; unit_start_idx is NOT (it
    # is sorted-row positions, one per unit).
    row_round = np.repeat(round_of_unit, unit_sizes)
    unit_file = sorted_file[unit_start_idx]
    rounds = []
    for r in range(len(round_starts)):
        rows = order[np.flatnonzero(row_round == r)]
        u0, u1 = (
            int(round_starts[r]),
            int(round_starts[r + 1] if r + 1 < len(round_starts) else n_units),
        )
        rounds.append(
            (
                rows,
                (
                    unit_file[u0:u1],
                    unit_start[u0:u1],
                    arena_of_unit[u0:u1],
                    unit_size[u0:u1],
                ),
            )
        )
    return arena_offset_per_row, rounds


@triton.jit
def _scatter_ple_rows_kernel(
    arena_ptr,
    inverse_ptr,
    arena_offset_ptr,
    out_ptr,
    embedding_dim,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per lookup position. arena_offset_ptr is indexed by the
    # unique-row id torch.unique produced for this gather. Byte offsets:
    # >= 0 the row's arena window this round; -1 a row never read (outside
    # this rank's shard -- the pinned backend zero-fills the same ids); -2 a
    # row loaded by an EARLIER round, whose arena window this round's IO has
    # since overwritten -- its output was written then and must stay.
    pos = tl.program_id(0)
    unique_idx = tl.load(inverse_ptr + pos)
    byte_off = tl.load(arena_offset_ptr + unique_idx)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    valid = byte_off >= 0
    if is_fp8:
        src = arena_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
        elem_off = byte_off
    else:
        src = arena_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        elem_off = byte_off // 2
    values = tl.load(src + elem_off + offsets, mask=mask & valid, other=0.0).to(
        tl.bfloat16
    )
    tl.store(
        out_ptr + pos * embedding_dim + offsets,
        tl.where(valid, values, 0.0),
        mask=mask & (byte_off != -2),
    )


class _GatherJob:
    """One gather handed from the caller's thread to the worker."""

    __slots__ = (
        "out",
        "inverse",
        "unique_ids",
        "n_lookup",
        "stream",
        "d2h_event",
        "done_event",
        "error",
    )

    def __init__(
        self,
        *,
        out: torch.Tensor,
        inverse: torch.Tensor,
        unique_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        d2h_event: torch.cuda.Event,
    ) -> None:
        self.out = out
        self.inverse = inverse
        self.unique_ids = unique_ids
        self.n_lookup = int(inverse.numel())
        self.stream = stream
        self.d2h_event = d2h_event
        self.done_event = threading.Event()
        self.error: Optional[BaseException] = None


class _IoVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


class PleSafetensorsGatherer:
    """Reads requested rows from the checkpoint files into a device arena.

    One worker thread owns all file IO (cuFile batch, or the preadv pool
    when libcufile is unavailable) and enqueues the scatter kernel on the
    gather's stream, serially per job and per arena-reusing round, so the
    arena is never overwritten while a previous scatter still reads it.
    The caller's thread never blocks in ``submit``; the job's completion
    event is set once the scatter is enqueued.
    """

    def __init__(
        self,
        table: PleSafetensorsTable,
        *,
        device: torch.device,
        vocab_start: int,
        vocab_end: int,
    ) -> None:
        self._table = table
        self._device = device
        self._vocab_start = int(vocab_start)
        self._vocab_end = int(vocab_end)
        arena_mb = int(envs.SGLANG_QWEN4_PLE_SAFETENSORS_ARENA_MB.get())
        if arena_mb <= 0:
            raise ValueError("SGLANG_QWEN4_PLE_SAFETENSORS_ARENA_MB must be positive")
        arena_bytes = arena_mb << 20
        self._arena = torch.empty(arena_bytes, dtype=torch.uint8, device=device)
        if cufile.cufile_available():
            self._mode = cufile.CuFileDriver.instance().mode()
            cufile.register_buffer(self._arena, arena_bytes)
            self._handles = [cufile.CuFileHandle(p) for p in table.files]
            self._file_sizes = np.array(
                [os.path.getsize(p) for p in table.files], dtype=np.int64
            )
            self._batch = cufile.CuFileBatch(max_ios=_MAX_SPANS_PER_ROUND)
            self._pread_pool = None
            self._host_arena = None
            self._fds = []
        else:
            # Same span plan, served by O_DIRECT preadv into a pinned host
            # arena and one async H2D per round into the device arena.
            self._mode = "pread"
            self._handles = []
            self._file_sizes = None
            self._batch = None
            self._pread_pool = ThreadPoolExecutor(max_workers=_PREAD_THREADS)
            self._host_arena = torch.empty(
                arena_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            self._fds = [
                os.open(p, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
                for p in table.files
            ]
        self._unique_ids_pinned = torch.empty(
            0, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._arena_offset_pinned = torch.empty(
            0, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self._arena_offset_gpu = torch.empty(0, dtype=torch.int64, device=device)
        self._queue: queue.SimpleQueue[Optional[_GatherJob]] = queue.SimpleQueue()
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop, name="ple-safetensors-gather", daemon=True
        )
        self._worker.start()

    @property
    def mode(self) -> str:
        return self._mode

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=_IO_TIMEOUT_S)
        if self._batch is not None:
            self._batch.close()
        for handle in self._handles:
            handle.close()
        if self._pread_pool is not None:
            self._pread_pool.shutdown(wait=False)
            for fd in self._fds:
                os.close(fd)
        else:
            # The arena's device address can be recycled by the caching
            # allocator to the next gatherer; cuFile would refuse to
            # re-register it as "already registered".
            try:
                cufile.deregister_buffer(self._arena)
            except Exception:
                pass

    # -- caller thread ------------------------------------------------------

    def submit(
        self,
        unique_ids: torch.Tensor,
        inverse: torch.Tensor,
        out: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> _GatherJob:
        """Enqueue one gather; ``unique_ids`` must be produced on ``stream``."""
        n_unique = int(unique_ids.numel())
        if self._unique_ids_pinned.numel() < n_unique:
            self._unique_ids_pinned = torch.empty(
                max(n_unique, 2 * self._unique_ids_pinned.numel()),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
        # A view of the current pinned buffer: a later, larger gather that
        # reallocates it leaves this job's storage alive.
        unique_view = self._unique_ids_pinned[:n_unique]
        unique_view.copy_(unique_ids, non_blocking=True)
        d2h_event = torch.cuda.Event()
        d2h_event.record(stream)
        job = _GatherJob(
            out=out,
            inverse=inverse,
            unique_ids=unique_view,
            stream=stream,
            d2h_event=d2h_event,
        )
        if breakable_capture_in_progress():
            # Global-mode capture rejects CUDA sync calls from any thread, so
            # capture-time gathers finish inside the eager break instead.
            self._run_job(job)
            job.done_event.set()
            return job
        self._queue.put(job)
        return job

    # -- worker thread ------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._run_job(job)
                job.done_event.set()
            except BaseException as exc:  # surfaced by the caller's wait()
                job.error = exc
                job.done_event.set()
                logger.exception("PLE safetensors gather failed")

    def _run_job(self, job: _GatherJob) -> None:
        job.d2h_event.synchronize()
        table = self._table
        n_unique = int(job.unique_ids.numel())
        ids = job.unique_ids.numpy()
        pos, in_shard, byte_off = table.locate(ids)
        in_shard &= (ids >= self._vocab_start) & (ids < self._vocab_end)
        row_idx = np.flatnonzero(in_shard)  # positions into the unique ids
        if row_idx.size:
            arena_off, rounds = plan_io_spans(
                table._file_of[pos][row_idx],
                byte_off[row_idx],
                table.row_bytes,
                align_4k=self._mode in ("gds", "p2pdma"),
                arena_bytes=self._arena.numel(),
                file_sizes=self._file_sizes,
            )
        else:
            arena_off, rounds = None, []
        if self._arena_offset_gpu.numel() < n_unique:
            self._arena_offset_gpu = torch.empty(
                n_unique, dtype=torch.int64, device=self._device
            )
            self._arena_offset_pinned = torch.empty(
                n_unique, dtype=torch.int64, device="cpu", pin_memory=True
            )
        offsets_np = self._arena_offset_pinned.numpy()
        offsets_np[:n_unique] = -1
        stream = job.stream
        with torch.cuda.stream(stream):
            for round_rows, spans in rounds:
                if self._batch is not None:
                    self._read_spans_cufile(spans)
                else:
                    self._read_spans_pread(spans)
                # Round rows index the subset passed to plan_io_spans; map
                # them back to unique ids for the scatter lookup table.
                selected = row_idx[round_rows]
                offsets_np[selected] = arena_off[round_rows]
                self._arena_offset_gpu[:n_unique].copy_(
                    self._arena_offset_pinned[:n_unique], non_blocking=True
                )
                self._scatter(job)
                # Loaded rows become skips: a later round's IO reuses their
                # arena window, and their output must survive it.
                offsets_np[selected] = -2
                # The next round's IO overwrites the arena this scatter reads.
                round_done = torch.cuda.Event()
                round_done.record(stream)
                round_done.synchronize()
            if not rounds and job.n_lookup:
                # Nothing to read: still scatter zeros through inverse.
                self._arena_offset_gpu.fill_(-1)
                self._scatter(job)

    def _read_spans_cufile(self, spans) -> None:
        # spans: (file_idx, file_offsets, arena_offsets, sizes) arrays.
        file_idx, file_offsets, arena_offsets, sizes = spans
        arena_ptr = self._arena.data_ptr()
        # Spans are file-sorted; each contiguous file group goes through the
        # vectorized batch path (one IoDesc per span costs ~20 us of Python
        # per IO, which dominates a 131k-span prefill gather).
        boundaries = np.flatnonzero(np.diff(file_idx)) + 1
        for group in np.split(np.arange(len(file_idx)), boundaries):
            self._batch.submit_spans(
                self._handles[int(file_idx[group[0]])],
                buf_ptr=arena_ptr,
                file_offsets=file_offsets[group],
                buf_offsets=arena_offsets[group],
                sizes=sizes[group],
            )
        rets = self._batch.wait(timeout_s=_IO_TIMEOUT_S)
        short = [r for r, s in zip(rets, sizes) if r != s]
        if short:
            raise RuntimeError(
                f"PLE safetensors gather: short cuFile read(s): {short[:4]}"
            )

    def _read_spans_pread(self, spans) -> None:
        file_idx, file_offsets, arena_offsets, sizes = spans
        host_ptr = self._host_arena.data_ptr()
        fds = self._fds
        libc = _LIBC
        jobs = [
            (int(f), int(fo), int(ao), int(sz))
            for f, fo, ao, sz in zip(file_idx, file_offsets, arena_offsets, sizes)
        ]

        def _pread(job):
            file_index, file_off, arena_off, size = job
            iovec = _IoVec(ctypes.c_void_p(host_ptr + arena_off), ctypes.c_size_t(size))
            done = 0
            while done < size:
                n = libc.preadv(
                    ctypes.c_int(fds[file_index]),
                    ctypes.byref(iovec),
                    ctypes.c_int(1),
                    ctypes.c_longlong(file_off + done),
                )
                if n <= 0:
                    errno = ctypes.get_errno()
                    raise OSError(errno, os.strerror(errno))
                done += n
            return size

        list(self._pread_pool.map(_pread, jobs))
        for file_index, file_off, arena_off, size in jobs:
            self._arena[arena_off : arena_off + size].copy_(
                self._host_arena[arena_off : arena_off + size], non_blocking=True
            )

    def _scatter(self, job: _GatherJob) -> None:
        _scatter_ple_rows_kernel[(job.n_lookup,)](
            self._arena,
            job.inverse,
            self._arena_offset_gpu,
            job.out,
            embedding_dim=self._table.dim,
            is_fp8=self._table.dtype == torch.float8_e4m3fn,
            BLOCK_D=triton.next_power_of_2(self._table.dim),
        )


def ple_prefetch_excluded_files(model_config) -> tuple:
    """Checkpoint files the safetensors PLE backend reads itself at gather
    time, so the weight-loading prefetch must skip them: 95.4 GiB for
    Qwen3.8-Flash-Next would be pulled through the page cache for nothing.
    Empty for every other configuration."""
    hf_config = model_config.hf_config
    architectures = hf_config.architectures or []
    if "Qwen4ExpForConditionalGeneration" not in architectures:
        return ()
    text_config = hf_config.text_config
    if (
        text_config.ple_offload_backend != "safetensors"
        or not text_config.ple_offload_embedding
        or not text_config.ple_safetensors_dir
    ):
        return ()
    directory = os.path.expanduser(text_config.ple_safetensors_dir)
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return ()
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError):
        return ()
    return tuple(
        sorted(
            {
                os.path.realpath(os.path.join(directory, fname))
                for name, fname in weight_map.items()
                if PLE_SHARD_TENSOR_RE.match(name)
            }
        )
    )


class Qwen4ExpSafetensorsEmbedding(nn.Module):
    """Drop-in PLE embedding backed by the checkpoint's own safetensors.

    Mirrors ``Qwen4ExpPinnedHostEmbedding``'s interface -- ``gather``,
    ``reduce``, ``allocate_output``, ``weight_scale`` and the shard-index
    attributes the model reads -- but the table lives on storage: the
    weight parameter stays on the meta device (never allocated, never
    loaded) and gathers copy just the requested rows to the device.
    """

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        *,
        directory: str,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "PLE embedding offload requires bfloat16 or fp8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The generic loader stages loaded weights back to GPU through the
        # quant method; this table never loads, so keep it out of that path.
        self.quant_method = None

        self._table = PleSafetensorsTable.build(
            directory,
            expected_dim=self.embedding_dim,
            expected_rows=int(self.org_vocab_size),
        )
        if self._table.dtype != embedding.weight.dtype:
            raise ValueError(
                "the checkpoint's PLE shards are "
                f"{self._table.dtype} but the embedding storage is "
                f"{embedding.weight.dtype}; set text_config.ple_embedding_dtype "
                "to the on-disk dtype"
            )
        # Structural parity with the pinned embedding: the parameter exists
        # (meta, never materialized) so nothing downstream re-bases on a
        # missing attribute; the safetensors-shard loader marks it loaded.
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(
                    (self._table.num_rows, self._table.dim),
                    dtype=self._table.dtype,
                    device="meta",
                ),
                requires_grad=False,
            ),
        )
        # The scale is tiny; keep it with the model instead of offloading it.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight
        self._gatherer: Optional[PleSafetensorsGatherer] = None
        self._active_job: Optional[_GatherJob] = None

    def _ensure_gatherer(self, device: torch.device) -> PleSafetensorsGatherer:
        if self._gatherer is None:
            self._gatherer = PleSafetensorsGatherer(
                self._table,
                device=device,
                vocab_start=self.shard_indices.org_vocab_start_index,
                vocab_end=self.shard_indices.org_vocab_end_index,
            )
        return self._gatherer

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The scatter kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if not flat_ids.numel():
            return output
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "--ple-offload-backend safetensors cannot run its file IO inside "
                "a CUDA graph capture; run decode with the breakable graph "
                "backend (--cuda-graph-config 'decode:backend=breakable') or "
                "with decode CUDA graphs disabled"
            )
        gatherer = self._ensure_gatherer(input_ids.device)
        # Sorted unique ids double as span-planning order and shrink the IO
        # set; inverse maps every lookup position to its unique row.
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        unique_ids.record_stream(torch.cuda.current_stream())
        inverse.record_stream(torch.cuda.current_stream())
        self._active_job = gatherer.submit(
            unique_ids,
            inverse,
            output.view(flat_ids.numel(), self.embedding_dim),
            stream=torch.cuda.current_stream(),
        )
        return output

    def join_prefetch(self) -> None:
        """Block until the running gather's scatter is enqueued on its stream.

        Called by the PLE layer's consumer before it waits on the gather
        stream (a no-op after the synchronous ``forward`` path).
        """
        job = self._active_job
        if job is not None:
            self._active_job = None
            job.done_event.wait()
            if job.error is not None:
                raise job.error

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.gather(input_ids)
        self.join_prefetch()
        return self.reduce(output)

    def close(self) -> None:
        if self._gatherer is not None:
            self._gatherer.close()
            self._gatherer = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
