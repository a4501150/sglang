import inspect
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


class TestDSAOffloadSignatures(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for method_name in ("get_cpu_copy", "load_cpu_copy"):
            with self.subTest(method_name=method_name):
                signature = inspect.signature(getattr(DSATokenToKVPool, method_name))
                self.assertIn("mamba_indices", signature.parameters)


class TestSparseIndexerHostPages(unittest.TestCase):
    def test_page_first_storage_round_trip_uses_device_page_contract(self):
        device_buffers = [
            torch.arange(96, dtype=torch.uint8).reshape(3, 32),
            torch.arange(96, 192, dtype=torch.uint8).reshape(3, 32),
        ]
        device_pool = SimpleNamespace(
            device="cpu",
            start_layer=0,
            layer_shard_enabled=False,
            get_hicache_indexer_page_buffers=lambda: device_buffers,
        )
        anchor_host = SimpleNamespace(
            page_size=4,
            size=12,
            page_num=3,
            mtp_draft_device_pools=(),
        )
        host = DSAIndexerPoolHost(
            device_pool=device_pool,
            anchor_host=anchor_host,
            layout="page_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        page = torch.arange(64, dtype=torch.uint8)

        host.set_from_flat_data_page(4, page)

        self.assertEqual(host.indexer_page_stride_size, 32)
        self.assertEqual(host.size_per_token, 16)
        self.assertTrue(torch.equal(host.get_data_page(4), page))


class TestDSAHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("DSA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    @staticmethod
    def _token_indices_for_pages(pages: torch.Tensor, page_size: int, device: str):
        parts = [
            torch.arange(
                int(page_id) * page_size,
                (int(page_id) + 1) * page_size,
                device=device,
                dtype=torch.int64,
            )
            for page_id in pages.tolist()
        ]
        return torch.cat(parts, dim=0)

    def _run_device_to_host_indexer_copy(self, io_backend: str):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        size = page_size * 4

        device_pool = DSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
        )
        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout="layer_first",
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            indexer_host = DSAIndexerPoolHost(
                device_pool=device_pool,
                anchor_host=mla_host,
                layout="layer_first",
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.uint8
            ).view_as(buf)
            buf.copy_((data + layer_id) % 256)
            kv_buf = device_pool.kv_buffer[layer_id]
            kv_data = torch.arange(
                kv_buf.numel(), device=kv_buf.device, dtype=kv_buf.dtype
            ).view_as(kv_buf)
            kv_buf.copy_(kv_data + layer_id)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor(
            [0, 1, 2],
            device="cuda" if io_backend == "kernel" else "cpu",
            dtype=torch.int64,
        )
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = self._token_indices_for_pages(
            host_pages,
            page_size,
            device="cuda" if io_backend == "kernel" else "cpu",
        )

        mla_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                got = indexer_host.index_k_with_scale_buffer[layer_id][host_page].cpu()
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))
                host_start = host_page * page_size
                device_start = device_page * page_size
                got_kv = mla_host.kv_buffer[layer_id][
                    host_start : host_start + page_size
                ].cpu()
                expected_kv = device_pool.kv_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_kv, expected_kv))

    def test_page_first_direct_page_contract_round_trip(self):
        page_size = 64
        page_bytes = 256
        layer_num = 2
        page_num = 4
        device_buffers = [
            torch.arange(page_num * page_bytes, device="cuda", dtype=torch.int64)
            .to(torch.uint8)
            .reshape(page_num, page_bytes)
            + layer_id
            for layer_id in range(layer_num)
        ]
        device_pool = SimpleNamespace(
            device="cuda",
            start_layer=0,
            layer_shard_enabled=False,
            get_hicache_indexer_page_buffers=lambda: device_buffers,
        )
        anchor_host = SimpleNamespace(
            page_size=page_size,
            size=page_num * page_size,
            page_num=page_num,
            mtp_draft_device_pools=(),
        )
        host = DSAIndexerPoolHost(
            device_pool=device_pool,
            anchor_host=anchor_host,
            layout="page_first",
            pin_memory=True,
            device="cpu",
            allocator_type="default",
        )
        device_pages = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor([0, 1], dtype=torch.int64)
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = self._token_indices_for_pages(
            host_pages, page_size, device="cpu"
        )
        expected = [buffer[device_pages].clone() for buffer in device_buffers]

        host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, "direct"
        )
        torch.cuda.synchronize()
        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    host.index_k_with_scale_buffer[:2, layer_id, 0],
                    expected[layer_id].cpu(),
                )
            )
            device_buffers[layer_id][device_pages] = 0
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                "direct",
            )
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(device_buffers[layer_id][device_pages], expected[layer_id])
            )

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in MLATokenToKVPoolHost.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        self._run_device_to_host_indexer_copy(io_backend="kernel")

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        self._run_device_to_host_indexer_copy(io_backend="direct")


if __name__ == "__main__":
    unittest.main()
