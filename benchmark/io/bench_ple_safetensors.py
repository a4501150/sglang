"""Gather latency of the safetensors PLE offload backend.

Feeds the gatherer the id sets a real forward would produce -- a 6-token
decode batch (6 x 16 n-gram lookups) and an 8192-token prefill chunk
(8192 x 16) -- against the Qwen3.8-Flash-Next checkpoint (320M rows x
320 B bf16, 95.4 GiB) and reports the end-to-end gather latency
(submit -> scatter enqueued) plus the active cuFile mode. Fresh random ids
every iteration: reused ones come out of the SSD's internal cache, which
would flatter the number.

Run (against the real checkpoint when cached, a synthetic one otherwise):
    python benchmark/io/bench_ple_safetensors.py \
        [--iterations 10] [--arena-mb 256] [--batch-tokens 6 8192 ...]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_safetensors import (
    PleSafetensorsGatherer,
    PleSafetensorsTable,
)

_REAL_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--orcarouter--"
    "Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/"
    "c1209bda15a6bbc4c68b585e93d40c0d85f50306"
)
_ROW_BYTES = 320
_DIM = 160


def _synthetic_checkpoint(rows: int) -> str:
    root = os.path.expanduser(
        os.environ.get("SGLANG_CACHE_DIR", os.path.join("~", ".cache", "sglang"))
    )
    d = os.path.join(root, "sglang-test", "ple_safetensors", "bench")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "shard_0.safetensors")
    data_bytes = rows * _ROW_BYTES
    if not os.path.exists(path) or os.path.getsize(path) != 8 + data_bytes:
        import json
        import struct

        header = json.dumps(
            {
                "ple.ngram_embedding.shard_0.weight": {
                    "dtype": "BF16",
                    "shape": [rows, _DIM],
                    "data_offsets": [0, data_bytes],
                }
            }
        ).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(os.urandom(data_bytes))
    return d


def _synthetic_table(d: str) -> PleSafetensorsTable:
    # Bare safetensors layout: the table builder scans *.safetensors files.
    return PleSafetensorsTable.build(
        d,
        expected_dim=_DIM,
        expected_rows=(os.path.getsize(os.path.join(d, "shard_0.safetensors")) - 16)
        // _ROW_BYTES,
    )


def _gather_once(gatherer, ids, dim=_DIM):
    device_ids = torch.tensor(np.asarray(ids), dtype=torch.long, device="cuda")
    unique_ids, inverse = torch.unique(device_ids, return_inverse=True)
    out = torch.empty((device_ids.numel(), dim), dtype=torch.bfloat16, device="cuda")
    t0 = time.perf_counter()
    job = gatherer.submit(unique_ids, inverse, out, torch.cuda.current_stream())
    job.done_event.wait()
    if job.error is not None:
        raise job.error
    dt = time.perf_counter() - t0
    torch.cuda.synchronize()
    return dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--arena-mb", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, nargs="+", default=[6, 8192])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.zeros(8, device="cuda")
    envs.SGLANG_QWEN4_PLE_SAFETENSORS_ARENA_MB.set(args.arena_mb)

    if os.path.isdir(_REAL_CKPT):
        table = PleSafetensorsTable.build(
            _REAL_CKPT, expected_dim=_DIM, expected_rows=320001536
        )
        source = _REAL_CKPT
    else:
        d = _synthetic_checkpoint(1 << 20)
        table = _synthetic_table(d)
        source = d
    gatherer = PleSafetensorsGatherer(
        table, device=torch.device("cuda"), vocab_start=0, vocab_end=table.num_rows
    )
    print(f"cuFile mode: {gatherer.mode}   arena: {args.arena_mb} MiB")
    print(f"table: {source} ({table.num_rows} rows x {table.row_bytes} B)\n")

    rng = np.random.default_rng(0)
    try:
        for tokens in args.batch_tokens:
            n_ids = tokens * 16  # 16 n-gram lookups per token
            # Fresh ids every iteration: the SSD's cache must not serve them.
            id_sets = [
                rng.integers(0, table.num_rows, size=n_ids)
                for _ in range(args.iterations)
            ]
            warm = _gather_once(gatherer, id_sets[0])  # compile the kernel
            times = [_gather_once(gatherer, ids) for ids in id_sets[1:]]
            med = float(np.median(times))
            print(
                f"{tokens:>5}-token batch ({n_ids:>6} ids): median {med * 1e3:8.2f} ms"
                f"  min {min(times) * 1e3:6.2f}  max {max(times) * 1e3:6.2f}"
                f"  (warm-up {warm * 1e3:.2f} ms, {len(times)} fresh iterations)"
            )
    finally:
        gatherer.close()
    print(f"\n(gather mode: {gatherer.mode})")


if __name__ == "__main__":
    main()
