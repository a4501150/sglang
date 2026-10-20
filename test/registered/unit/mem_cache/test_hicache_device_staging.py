"""
Scatter-parity tests for device-target L3 reads: a page written through the
host path must restore into the same device pool rows it came from,
component for component (KV payload, dynamic-FP8 scales, indexer rows,
mamba temporal/conv/slot states), across every IO engine.

Drives the full mechanism without a controller: fake host pools over
__new__ (same field contract as the CPU unit tests), a real HiCacheFile
backend, a DeviceStagingPool arena, and the staging pool's scatter (the
same kernels the L2 engine submits).

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_device_staging.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="base-b-test")

import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
    PoolTransferTarget,
)
from sglang.srt.mem_cache.pool_host.device_staging import DeviceStagingPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha_dynamic_fp8 import (
    MHATokenToKVPoolDynamicFP8Host,
)
from sglang.test.test_utils import CustomTestCase

TEST_SCRATCH = os.path.expanduser("~/.cache/sglang-test")

KV_PAGE = 4
KV_LAYERS = 2
KV_HEADS = 2
KV_DIM = 8
MAMBA_LAYERS = 2
MAMBA_TEMPORAL_SHAPE = (3, 4)
MAMBA_CONV_SHAPE = (5,)
IDX_ITEM_BYTES = 16
IDX_PAGES = 4


def _make_fp8_host(device):
    host = MHATokenToKVPoolDynamicFP8Host.__new__(MHATokenToKVPoolDynamicFP8Host)
    host.layout = "page_first_direct"
    host.page_size = KV_PAGE
    host.page_num = 4
    host.size = 4 * KV_PAGE
    host.layer_num = KV_LAYERS
    host.target_layer_num = KV_LAYERS
    host.head_num = KV_HEADS
    host.head_dim = KV_DIM
    host.v_head_dim = KV_DIM
    host.dtype = torch.float8_e4m3fn
    host.device = "cpu"
    host.pin_memory = False
    host.pool_label = "kv"
    host.storage_format_tag = "mha-dynamic-fp8"
    host._transfer_device = torch.device(device)
    host._scale_token_bytes = 2 * KV_HEADS * torch.float32.itemsize
    host.kv_buffer = torch.zeros(
        2, 4, KV_LAYERS, KV_PAGE, KV_HEADS, KV_DIM, dtype=host.dtype
    )
    host.scale_host = torch.zeros(4, 2, KV_LAYERS, KV_PAGE, KV_HEADS)
    # The "device pool" rows the scatter fills.
    tokens = 4 * KV_PAGE
    host.device_k_buffers = [
        torch.zeros(tokens, KV_HEADS, KV_DIM, dtype=host.dtype, device=device)
        for _ in range(KV_LAYERS)
    ]
    host.device_v_buffers = [
        torch.zeros(tokens, KV_HEADS, KV_DIM, dtype=host.dtype, device=device)
        for _ in range(KV_LAYERS)
    ]
    host.scale_k_buffers = [
        torch.zeros(tokens, KV_HEADS, device=device) for _ in range(KV_LAYERS)
    ]
    host.scale_v_buffers = [
        torch.zeros(tokens, KV_HEADS, device=device) for _ in range(KV_LAYERS)
    ]
    host.size_per_token = host.get_size_per_token()
    # Distinct payload/scale bytes per page.
    for page in range(4):
        host.kv_buffer[0, page] = torch.full(
            (KV_LAYERS, KV_PAGE, KV_HEADS, KV_DIM),
            (page + 1) * 3 + 11,
            dtype=host.dtype,
        )
        host.kv_buffer[1, page] = torch.full(
            (KV_LAYERS, KV_PAGE, KV_HEADS, KV_DIM),
            (page + 1) * 3 + 97,
            dtype=host.dtype,
        )
        host.scale_host[page, 0] = (page + 1) * 1.5
        host.scale_host[page, 1] = (page + 1) * -2.5
    return host, tokens


def _make_mamba_host(device):
    host = MambaPoolHost.__new__(MambaPoolHost)
    host.layout = "page_first_direct"
    host.page_size = 1
    host.layer_num = MAMBA_LAYERS
    host.num_mamba_layers = MAMBA_LAYERS
    host.temporal_state_shape = MAMBA_TEMPORAL_SHAPE
    host.temporal_dtype = torch.float32
    host.temporal_state_elem_size = int(
        torch.tensor(MAMBA_TEMPORAL_SHAPE).prod().item()
    )
    host.conv_state_shapes = [MAMBA_CONV_SHAPE]
    host.conv_state_elem_sizes = [int(torch.tensor(MAMBA_CONV_SHAPE).prod().item())]
    host.conv_dtype = torch.bfloat16
    host.mtp_draft_buffer_infos = ()
    host.pool_label = "mamba"
    host.dtype = torch.bfloat16
    host.device = "cpu"
    host.pin_memory = False
    host.storage_format_tag = None

    # Host page buffers (page-first: slot-major, layer within a slot).
    host.temporal_buffer = torch.stack(
        [
            torch.full((MAMBA_LAYERS,) + MAMBA_TEMPORAL_SHAPE, float(slot + 1))
            + 0.25 * torch.arange(MAMBA_LAYERS).view(-1, *([1] * len(MAMBA_TEMPORAL_SHAPE)))
            for slot in range(4)
        ]
    )
    host.conv_buffer = [
        torch.stack(
            [
                torch.full((MAMBA_LAYERS,) + MAMBA_CONV_SHAPE, float(10 * (slot + 1)))
                + torch.arange(MAMBA_LAYERS).view(-1, 1).float()
                for slot in range(4)
            ]
        ).to(torch.bfloat16)
    ]
    host.slot_state_buffers = (
        torch.arange(4 * 7, dtype=torch.float32).view(4, 7) + 100,
    )
    host.size_per_token = host.get_size_per_token()

    # Device pool tensors the scatter fills, seeded empty.
    temporal = host.temporal_buffer.permute(1, 0, *range(2, 2 + len(MAMBA_TEMPORAL_SHAPE))).contiguous()
    conv = host.conv_buffer[0].permute(1, 0, *range(2, 2 + len(MAMBA_CONV_SHAPE))).contiguous()
    host.slot_state_device_tensors = (
        torch.zeros_like(host.slot_state_buffers[0], device=device),
    )
    host.buffer_info = MambaStateBufferInfo(
        temporal=temporal.to(device), conv=(conv.to(device),)
    )
    return host


def _make_indexer_host(device):
    # QSAIndexerPoolHost so get_size_per_token carries the packed-page
    # arithmetic (per-page item rows / full-KV page size) the real sidecar
    # uses; fields via __new__ like the other fakes.
    from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost

    host = QSAIndexerPoolHost.__new__(QSAIndexerPoolHost)
    host.pool_name = "qsa_indexer"
    host.layout = "page_first"
    host.page_size = KV_PAGE
    host.slot_page_size = KV_PAGE
    host.item_bytes = IDX_ITEM_BYTES
    host.layer_num = KV_LAYERS
    host.page_aligned_only = True
    host.dtype = torch.uint8
    host.pool_label = "indexer"
    host.storage_format_tag = None
    host.device = "cpu"
    host.device_buffers = [
        torch.zeros(IDX_PAGES, IDX_ITEM_BYTES, dtype=torch.uint8, device=device)
        for _ in range(KV_LAYERS)
    ]
    return host


def _make_backend(base_dir, pools, staging, engine="posix_buffered"):
    d = tempfile.mkdtemp(prefix="hicache_dev_", dir=base_dir)
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
        model_name="devmodel",
        extra_config={"io_engine": engine} if engine != "posix_buffered" else None,
    )
    b = HiCacheFile(cfg, file_path=d)
    for name, pool in pools.items():
        b.register_mem_host_pool_v2(pool, name)
        b.register_mem_device_pool_v2(staging[name], name)
    return b


class DeviceStagingParityBase(CustomTestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_parity_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)
        self.device = torch.device("cuda")
        torch.cuda.init()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _round_trip(self, engine):
        raise NotImplementedError

    def _write_and_restore(self, host, pool_name, staging, engine, keys):
        """Write pages host-side, read them back into the arena, return the
        per-page results."""
        backend = _make_backend(
            self.base, {pool_name: host}, {pool_name: staging}, engine
        )
        # Register the anchor too so the raw fingerprint path resolves.
        backend.register_mem_pool_host(host)
        writes = [
            PoolTransfer(
                name=PoolName(pool_name),
                keys=[k for k in keys],
                host_indices=torch.arange(len(keys) * (host.page_size or 1)),
            )
        ]
        # Seed the "host pool" pages if the pool kind reads them from
        # get_data_page (fp8, paged); mamba writes come from get_data_page
        # over the device-side buffer_info, so seed is pool-specific.
        self._seed_pages(host)
        results = backend.batch_set_v2(writes)
        assert all(results[pool_name]), results
        slots = staging.alloc(len(keys) * (staging.page_size))
        reads = backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName(pool_name),
                    keys=list(keys),
                    device_indices=slots,
                    target=PoolTransferTarget.DEVICE,
                )
            ]
        )
        return backend, slots, reads

    def _seed_pages(self, host):
        pass

    def _engines(self):
        engines = ["posix_buffered"]
        try:
            from sglang.srt.mem_cache.storage.file.page_io import make_page_io

            make_page_io(self.base, "posix_o_direct")
            engines.append("posix_o_direct")
        except (RuntimeError, OSError):
            pass
        return engines


class TestKVDynamicFP8ScatterParity(DeviceStagingParityBase):
    """KV payload + per-token scales: file -> arena -> device rows."""

    def _seed_pages(self, host):
        # fp8 get_data_page reads kv_buffer + scale_host, both seeded in
        # the factory.
        pass

    def _round_trip(self, engine):
        host, tokens = _make_fp8_host(self.device)
        staging = DeviceStagingPool(
            host, device=self.device, page_slots=4, name="kv"
        )
        backend, slots, reads = self._write_and_restore(
            host, "kv", staging, engine, keys=["k0", "k1", "k2", "k3"]
        )
        self.assertEqual(reads, {"kv": [True, True, True, True]})
        torch.cuda.synchronize()

        # Arena rows hold the exact flat pages that were written.
        for i, key in enumerate(["k0", "k1", "k2", "k3"]):
            slot = int(slots[i * KV_PAGE].item()) // KV_PAGE
            expected = host.get_data_page(slot * KV_PAGE, flat=True).to("cuda")
            self.assertTrue(
                torch.equal(staging.arena[slot, : staging.page_bytes], expected),
                f"arena row {slot} must equal the stored flat page ({engine})",
            )

        # Scatter into device pool rows and compare with the seed.
        device_dst = torch.arange(4 * KV_PAGE, device=self.device)
        for layer in range(KV_LAYERS):
            staging.load_to_device_per_layer(
                None, slots, device_dst, layer, "direct"
            )
        torch.cuda.synchronize()
        src_k = host.kv_buffer[0]  # [page, layer, ps, head, dim]
        src_v = host.kv_buffer[1]
        for layer in range(KV_LAYERS):
            want_k = src_k[:, layer].reshape(-1, KV_HEADS, KV_DIM).to(self.device)
            want_v = src_v[:, layer].reshape(-1, KV_HEADS, KV_DIM).to(self.device)
            self.assertTrue(
                torch.equal(host.device_k_buffers[layer][device_dst], want_k)
            )
            self.assertTrue(
                torch.equal(host.device_v_buffers[layer][device_dst], want_v)
            )
            want_ks = host.scale_host[:, 0, layer].reshape(-1, KV_HEADS).to(
                self.device
            )
            want_vs = host.scale_host[:, 1, layer].reshape(-1, KV_HEADS).to(
                self.device
            )
            self.assertTrue(
                torch.equal(host.scale_k_buffers[layer][device_dst], want_ks)
            )
            self.assertTrue(
                torch.equal(host.scale_v_buffers[layer][device_dst], want_vs)
            )

    def test_scatter_parity_all_engines(self):
        for engine in self._engines():
            with self.subTest(engine=engine):
                self._round_trip(engine)


class TestMambaScatterParity(DeviceStagingParityBase):
    """Mamba temporal/conv/slot-state envelope: file -> arena -> slots."""

    def _seed_pages(self, host):
        # Mamba pages come from the device-side buffer_info through
        # get_data_page; seed distinct per-slot values there.
        pass

    def _round_trip(self, engine):
        host = _make_mamba_host(self.device)
        staging = DeviceStagingPool(
            host, device=self.device, page_slots=4, name="mamba"
        )
        backend, slots, reads = self._write_and_restore(
            host, "mamba", staging, engine, keys=["m0", "m1", "m2", "m3"]
        )
        self.assertEqual(
            reads, {"mamba": [True, True, True, True]}, f"engine={engine}"
        )
        torch.cuda.synchronize()

        device_dst = torch.arange(4, device=self.device)
        for layer in range(MAMBA_LAYERS):
            staging.load_to_device_per_layer(
                None, slots, device_dst, layer, "direct"
            )
        torch.cuda.synchronize()

        temporal = host.buffer_info.temporal  # [layer, slot, *shape]
        conv = host.buffer_info.conv[0]
        for layer in range(MAMBA_LAYERS):
            want = host.temporal_buffer[:4, layer]  # slot-major seed
            self.assertTrue(
                torch.equal(temporal[layer][:4].cpu(), want),
                f"temporal layer {layer} ({engine})",
            )
            want_conv = host.conv_buffer[0][:4, layer]
            self.assertTrue(
                torch.equal(conv[layer][:4].cpu(), want_conv),
                f"conv layer {layer} ({engine})",
            )
        want_state = host.slot_state_buffers[0][:4]
        self.assertTrue(
            torch.equal(
                host.slot_state_device_tensors[0][:4].cpu(), want_state
            ),
            f"slot state ({engine})",
        )

    def test_scatter_parity_all_engines(self):
        for engine in self._engines():
            with self.subTest(engine=engine):
                self._round_trip(engine)


class TestIndexerScatterParity(DeviceStagingParityBase):
    """Packed indexer rows: file -> arena -> device page rows."""

    def _seed_pages(self, host):
        for layer in range(KV_LAYERS):
            host.kv_buffer = torch.arange(
                IDX_PAGES * KV_LAYERS * IDX_ITEM_BYTES,
                dtype=torch.uint8,
            ).view(IDX_PAGES, KV_LAYERS, IDX_ITEM_BYTES) % 251

    def _round_trip(self, engine):
        host = _make_indexer_host(self.device)
        staging = DeviceStagingPool(
            host, device=self.device, page_slots=IDX_PAGES, name="indexer"
        )
        backend, slots, reads = self._write_and_restore(
            host, "indexer", staging, engine, keys=["i0", "i1"]
        )
        self.assertEqual(reads, {"indexer": [True, True]}, f"engine={engine}")
        torch.cuda.synchronize()

        device_dst = torch.arange(2 * KV_PAGE)
        for layer in range(KV_LAYERS):
            staging.load_to_device_per_layer(
                None, slots[: 2 * KV_PAGE], device_dst, layer, "direct"
            )
        torch.cuda.synchronize()
        rows = (device_dst[::KV_PAGE] // KV_PAGE).to(self.device)
        for layer in range(KV_LAYERS):
            want = host.kv_buffer[:2, layer].to(self.device)
            self.assertTrue(
                torch.equal(host.device_buffers[layer][rows], want),
                f"indexer layer {layer} rows ({engine})",
            )

    def test_scatter_parity_all_engines(self):
        for engine in self._engines():
            with self.subTest(engine=engine):
                self._round_trip(engine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
