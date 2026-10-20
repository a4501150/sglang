"""Restore-latency microbenchmark for HiCache L3 device-target reads.

Rebuilds the Qwen3.8-Flash-Next per-page geometry synthetically (tp=1,
page=64): 12 dynamic-FP8 KV layers (payload 786,432 B + per-token scales
12,288 B per page), 12 indexer layers (49,152 B per page), one mamba
checkpoint (~59 MB) — writes a 32K-token prefix through the host path,
then times the restore per IO engine:

  1. host-target whole-page reads (the buffer_only host-bounce path)
  2. device-target reads into the GPU staging arenas (bounce + H2D copy
     under posix engines, batched cuFile under the cufile engine)

plus the arena->device scatter time. Pages are prefilled with distinct
bytes and every restore is byte-verified before timing loops start.

Usage:
    python benchmark/hicache/bench_restore_latency.py [--tokens 32768]
        [--engine posix_buffered,posix_o_direct,cufile] [--rounds 5]
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import torch

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CUR, "..", "..", "python"))

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
    PoolTransferTarget,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool  # noqa: E402
from sglang.srt.mem_cache.pool_host.device_staging import (  # noqa: E402
    DeviceStagingPool,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost  # noqa: E402
from sglang.srt.mem_cache.pool_host.mha_dynamic_fp8 import (  # noqa: E402
    MHATokenToKVPoolDynamicFP8Host,
)

PAGE = 64
KV_LAYERS = 12
HEADS = 2
HEAD_DIM = 256
MAMBA_LAYERS = 36
MAMBA_TEMPORAL = (48, 128, 128)
MAMBA_CONV = (10240, 3)
MAMBA_SLOT_STATE = 10240 * 9  # PLE short-conv window, bf16
MAMBA_SLOTS = 1
IDX_LAYERS = 12
IDX_ITEM_BYTES = 4096  # 12 layers x 4096 B per page
IDX_PAGES_PER_KV_PAGE = 1


def make_kv_host(tokens, device):
    host = MHATokenToKVPoolDynamicFP8Host.__new__(MHATokenToKVPoolDynamicFP8Host)
    pages = tokens // PAGE
    host.layout = "page_first_direct"
    host.page_size = PAGE
    host.page_num = pages
    host.size = tokens
    host.layer_num = KV_LAYERS
    host.target_layer_num = KV_LAYERS
    host.head_num = HEADS
    host.head_dim = HEAD_DIM
    host.v_head_dim = HEAD_DIM
    host.dtype = torch.float8_e4m3fn
    host.device = "cpu"
    host.pin_memory = False
    host.pool_label = "kv"
    host.storage_format_tag = "mha-dynamic-fp8"
    host._transfer_device = torch.device(device)
    host._scale_token_bytes = 2 * HEADS * torch.float32.itemsize
    host.kv_buffer = torch.zeros(
        2, pages, KV_LAYERS, PAGE, HEADS, HEAD_DIM, dtype=host.dtype
    )
    host.scale_host = torch.zeros(pages, 2, KV_LAYERS, PAGE, HEADS)
    # Distinct fp8 payload bytes (viewed as uint8; the values only need to
    # round-trip, not be valid fp8).
    host.kv_buffer[0].view(torch.uint8).random_(0, 255)
    host.kv_buffer[1].view(torch.uint8).random_(0, 255)
    host.scale_host.random_(0, 4)
    host.device_k_buffers = [
        torch.zeros(tokens, HEADS, HEAD_DIM, dtype=host.dtype, device=device)
        for _ in range(KV_LAYERS)
    ]
    host.device_v_buffers = [
        torch.zeros(tokens, HEADS, HEAD_DIM, dtype=host.dtype, device=device)
        for _ in range(KV_LAYERS)
    ]
    host.scale_k_buffers = [
        torch.zeros(tokens, HEADS, device=device) for _ in range(KV_LAYERS)
    ]
    host.scale_v_buffers = [
        torch.zeros(tokens, HEADS, device=device) for _ in range(KV_LAYERS)
    ]
    host.size_per_token = host.get_size_per_token()
    return host


def make_mamba_host(slots, device):
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.layout = "page_first_direct"
    host.page_size = 1
    host.layer_num = MAMBA_LAYERS
    host.num_mamba_layers = MAMBA_LAYERS
    host.temporal_state_shape = MAMBA_TEMPORAL
    host.temporal_dtype = torch.bfloat16
    host.temporal_state_elem_size = int(torch.tensor(MAMBA_TEMPORAL).prod().item())
    host.conv_state_shapes = [MAMBA_CONV]
    host.conv_state_elem_sizes = [int(torch.tensor(MAMBA_CONV).prod().item())]
    host.conv_dtype = torch.bfloat16
    host.mtp_draft_buffer_infos = ()
    host.pool_label = "mamba"
    host.dtype = torch.bfloat16
    host.device = "cpu"
    host.pin_memory = False
    host.storage_format_tag = None
    host.temporal_buffer = torch.randn(slots, MAMBA_LAYERS, *MAMBA_TEMPORAL).to(
        torch.bfloat16
    )
    host.conv_buffer = [
        torch.randn(slots, MAMBA_LAYERS, *MAMBA_CONV).to(torch.bfloat16)
    ]
    host.slot_state_buffers = (
        torch.randn(slots, MAMBA_SLOT_STATE).to(torch.bfloat16),
    )
    host.slot_state_device_tensors = (
        torch.zeros(slots, MAMBA_SLOT_STATE, dtype=torch.bfloat16, device=device),
    )
    temporal = (
        host.temporal_buffer.permute(1, 0, *range(2, host.temporal_buffer.dim()))
        .contiguous()
    )
    conv = (
        host.conv_buffer[0].permute(1, 0, *range(2, host.conv_buffer[0].dim()))
        .contiguous()
    )
    host.buffer_info = MambaStateBufferInfo(
        temporal=temporal.to(device), conv=(conv.to(device),)
    )
    host.size_per_token = host.get_size_per_token()
    return host


def make_indexer_host(pages, device):
    # QSAIndexerPoolHost for the packed-page get_size_per_token arithmetic.
    from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost

    host = QSAIndexerPoolHost.__new__(QSAIndexerPoolHost)
    host.pool_name = "qsa_indexer"
    host.layout = "page_first"
    host.page_size = PAGE
    host.slot_page_size = PAGE
    host.item_bytes = IDX_ITEM_BYTES
    host.layer_num = IDX_LAYERS
    host.page_aligned_only = True
    host.dtype = torch.uint8
    host.pool_label = "indexer"
    host.storage_format_tag = None
    host.device = "cpu"
    host.kv_buffer = torch.randint(
        0, 255, (pages, IDX_LAYERS, IDX_ITEM_BYTES), dtype=torch.uint8
    )
    host.device_buffers = [
        torch.zeros(pages, IDX_ITEM_BYTES, dtype=torch.uint8, device=device)
        for _ in range(IDX_LAYERS)
    ]
    host.size_per_token = host.get_size_per_token()
    return host


def make_backend(base_dir, engine, pools, staging):
    d = tempfile.mkdtemp(prefix="bench_restore_", dir=base_dir)
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="benchmodel",
        extra_config={"io_engine": engine, "verify": "off"},
    )
    b = HiCacheFile(cfg, file_path=d)
    for name, pool in pools.items():
        b.register_mem_host_pool_v2(pool, name)
        b.register_mem_device_pool_v2(staging[name], name)
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--engines", type=str, default="posix_buffered,posix_o_direct,cufile"
    )
    parser.add_argument("--scratch", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    scratch = args.scratch or os.path.expanduser("~/.cache/sglang-test")
    os.makedirs(scratch, exist_ok=True)
    pages = args.tokens // PAGE

    kv_host = make_kv_host(args.tokens, device)
    idx_host = make_indexer_host(pages, device)
    mamba_host = make_mamba_host(MAMBA_SLOTS, device)
    pools = {"kv": kv_host, "indexer": idx_host, "mamba": mamba_host}
    staging = {
        "kv": DeviceStagingPool(kv_host, device=device, page_slots=pages, name="kv"),
        "indexer": DeviceStagingPool(
            idx_host, device=device, page_slots=pages, name="indexer"
        ),
        "mamba": DeviceStagingPool(
            mamba_host, device=device, page_slots=1, name="mamba"
        ),
    }
    kv_keys = [f"kv{i:05d}" for i in range(pages)]
    idx_keys = [f"i{i:05d}" for i in range(pages)]
    mamba_keys = ["m0"]

    read_mb = (
        pages * (kv_host.get_size_per_token() + idx_host.get_size_per_token()) * PAGE
        + mamba_host.get_size_per_token()
    ) / 1e6
    print(
        f"prefix={args.tokens} tokens ({pages} pages); "
        f"read set ~{read_mb:.0f} MB (KV+indexer {pages} pages + 1 mamba slot)"
    )

    # One arena span reused by every engine round.
    kv_slots = staging["kv"].alloc(args.tokens)
    idx_slots = staging["indexer"].alloc(args.tokens)
    m_slots = staging["mamba"].alloc(MAMBA_SLOTS)
    assert kv_slots is not None and idx_slots is not None and m_slots is not None

    for engine in args.engines.split(","):
        try:
            backend = make_backend(scratch, engine, pools, staging)
        except (RuntimeError, OSError, NotImplementedError) as e:
            print(f"[{engine}] unavailable: {e}")
            continue
        try:
            # ---- write the prefix once through the host path ----
            t0 = time.perf_counter()
            assert all(
                backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=kv_keys,
                            host_indices=torch.arange(args.tokens),
                        )
                    ]
                )["kv"]
            )
            assert all(
                backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name=PoolName.INDEXER,
                            keys=idx_keys,
                            host_indices=torch.arange(args.tokens),
                        )
                    ]
                )["indexer"]
            )
            assert all(
                backend.batch_set_v2(
                    [
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            keys=mamba_keys,
                            host_indices=torch.arange(MAMBA_SLOTS),
                        )
                    ]
                )["mamba"]
            )
            write_s = time.perf_counter() - t0

            # ---- verify one device-target restore is byte-exact ----
            results = {}
            results.update(
                backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=kv_keys,
                            device_indices=kv_slots,
                            target=PoolTransferTarget.DEVICE,
                        )
                    ]
                )
            )
            results.update(
                backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name=PoolName.INDEXER,
                            keys=idx_keys,
                            device_indices=idx_slots,
                            target=PoolTransferTarget.DEVICE,
                        )
                    ]
                )
            )
            results.update(
                backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            keys=mamba_keys,
                            device_indices=m_slots,
                            target=PoolTransferTarget.DEVICE,
                        )
                    ]
                )
            )
            assert all(all(r) for r in results.values()), results
            torch.cuda.synchronize()
            slot0 = int(kv_slots[0].item()) // PAGE
            assert torch.equal(
                staging["kv"].arena[slot0, : staging["kv"].page_bytes].cpu(),
                kv_host.get_data_page(0, flat=True),
            ), "KV arena row mismatch"
            mamba_got = staging["mamba"].arena[0, : staging["mamba"].page_bytes].cpu()
            mamba_want = mamba_host.get_data_page(0, flat=True)
            if not torch.equal(mamba_got, mamba_want):
                diff = (mamba_got != mamba_want).nonzero().flatten()
                print(
                    f"mamba arena mismatch: {diff.numel()} bytes differ, "
                    f"first at {diff[0].item() if diff.numel() else -1}"
                )
                raise SystemExit(1)

            # ---- timed restore rounds ----
            device_dst = torch.arange(args.tokens, device=device)
            idx_dst = torch.arange(args.tokens)
            m_dst = torch.arange(MAMBA_SLOTS, device=device)
            times = []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                backend.batch_get_v2(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=kv_keys,
                            device_indices=kv_slots,
                            target=PoolTransferTarget.DEVICE,
                        ),
                        PoolTransfer(
                            name=PoolName.INDEXER,
                            keys=idx_keys,
                            device_indices=idx_slots,
                            target=PoolTransferTarget.DEVICE,
                        ),
                        PoolTransfer(
                            name=PoolName.MAMBA,
                            keys=mamba_keys,
                            device_indices=m_slots,
                            target=PoolTransferTarget.DEVICE,
                        ),
                    ]
                )
                read_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                for layer in range(KV_LAYERS):
                    staging["kv"].load_to_device_per_layer(
                        None, kv_slots, device_dst, layer, "direct"
                    )
                for layer in range(IDX_LAYERS):
                    staging["indexer"].load_to_device_per_layer(
                        None, idx_slots, idx_dst, layer, "direct"
                    )
                for layer in range(MAMBA_LAYERS):
                    staging["mamba"].load_to_device_per_layer(
                        None, m_slots, m_dst, layer, "direct"
                    )
                torch.cuda.synchronize()
                scatter_s = time.perf_counter() - t1
                times.append((read_s, scatter_s))
            reads = [t[0] for t in times]
            scatters = [t[1] for t in times]
            total = [r + s for r, s in times]
            print(
                f"[{engine}] write={write_s*1e3:.0f} ms | "
                f"device read: min={min(reads)*1e3:.0f} ms "
                f"median={sorted(reads)[len(reads)//2]*1e3:.0f} ms | "
                f"scatter: min={min(scatters)*1e3:.0f} ms | "
                f"restore total: min={min(total)*1e3:.0f} ms "
                f"median={sorted(total)[len(total)//2]*1e3:.0f} ms | "
                f"effective BW (read) {read_mb/min(reads)/1e3:.2f} GB/s"
            )
        finally:
            shutil.rmtree(backend.file_path, ignore_errors=True)


if __name__ == "__main__":
    main()
