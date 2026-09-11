import ctypes
import gc
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.pool_host import common as host_common
from sglang.srt.mem_cache.pool_host import common as pool_host_common
from sglang.srt.mem_cache.pool_host import mha as mha_pool_host
from sglang.srt.mem_cache.pool_host import mla as mla_pool_host
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_register,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeBuffer:
    def __init__(self, base: int, size: int):
        self._base = base
        self._size = size

    def data_ptr(self) -> int:
        return self._base

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1


class _FakeCudart:
    def __init__(self, fail_on_registration: int | None = None):
        self.registrations = []
        self.unregistrations = []
        self.fail_on_registration = fail_on_registration

    def cudaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        self.registrations.append((ptr, size, flags))
        if len(self.registrations) == self.fail_on_registration:
            return 1
        return 0

    def cudaHostUnregister(self, ptr: int) -> int:
        self.unregistrations.append(ptr)
        return 0

    def cudaGetErrorString(self, rc: int) -> str:
        return "injected error"


class TestHiCacheHostRegister(unittest.TestCase):
    def _hip_non_userptr(self):
        return (
            mock.patch.object(host_common, "_is_hip", True),
            mock.patch.dict(os.environ, {"HSA_USERPTR_FOR_PAGED_MEM": "0"}),
        )

    def test_hip_non_userptr_pool_is_exact_size_and_never_unregistered(self):
        """The pool must not be rounded up to a power of two (torch pin_memory pins
        256 GiB for a 180 GB pool) and destroy() must not cudaHostUnregister it."""
        dims = (3, 5, 7)  # 105 bf16 elements: 210 bytes, not a power of two
        backing = (ctypes.c_uint8 * 210)()
        allocator = host_common.HostTensorAllocator()
        cudart = _FakeCudart()
        is_hip, env = self._hip_non_userptr()
        with (
            is_hip,
            env,
            mock.patch.object(
                host_common, "_hip_host_malloc", return_value=ctypes.addressof(backing)
            ) as malloc,
            mock.patch.object(host_common, "_hip_host_free"),
            mock.patch.object(allocator, "allocate") as allocate,
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            buffer = host_common.alloc_with_host_register(
                dims=dims,
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
                allocator=allocator,
            )
            malloc.assert_called_once_with(210)
            allocate.assert_not_called()
            self.assertEqual(buffer.shape, dims)
            self.assertEqual(buffer.data_ptr(), ctypes.addressof(backing))
            _cuda_host_unregister(buffer)
        self.assertEqual(cudart.registrations, [])
        self.assertEqual(cudart.unregistrations, [])

    def test_hip_host_pool_is_freed_once_after_its_last_view(self):
        """hipHostFree must wait for every view (the pools keep per-layer views)
        and run exactly once."""
        backing = (ctypes.c_uint8 * 64)()
        ptr = ctypes.addressof(backing)
        is_hip, env = self._hip_non_userptr()
        with (
            is_hip,
            env,
            mock.patch.object(host_common, "_hip_host_malloc", return_value=ptr),
            mock.patch.object(host_common, "_hip_host_free") as free,
        ):
            buffer = host_common.alloc_with_host_register(
                dims=(4, 16),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
                allocator=host_common.HostTensorAllocator(),
            )
            layer_view = buffer[1]
            del buffer
            gc.collect()
            free.assert_not_called()
            del layer_view
            gc.collect()
            free.assert_called_once_with(ptr)

    def test_non_userptr_path_preserves_other_allocators_and_platforms(self):
        class StorageAllocator(host_common.HostTensorAllocator):
            pass

        for hip, userptr, custom, pinned in (
            (False, "0", False, True),
            (True, "1", False, True),
            (True, None, False, True),
            (True, "0", True, True),
            (True, "0", False, False),
        ):
            with self.subTest(hip=hip, userptr=userptr, custom=custom, pinned=pinned):
                allocator = (
                    StorageAllocator if custom else host_common.HostTensorAllocator
                )()
                buffer = _FakeBuffer(0x10000000, 4096)
                with (
                    mock.patch.object(host_common, "_is_hip", hip),
                    mock.patch.dict(os.environ),
                    mock.patch.object(host_common, "_hip_host_malloc") as malloc,
                    mock.patch.object(
                        allocator, "allocate", return_value=buffer
                    ) as allocate,
                    mock.patch.object(host_common, "_cuda_host_register") as register,
                ):
                    os.environ.pop("HSA_USERPTR_FOR_PAGED_MEM", None)
                    if userptr is not None:
                        os.environ["HSA_USERPTR_FOR_PAGED_MEM"] = userptr
                    got = host_common.alloc_with_host_register(
                        (4096,), torch.uint8, "cpu", pinned, allocator, 512
                    )
                    self.assertIs(got, buffer)
                    malloc.assert_not_called()
                    allocate.assert_called_once_with(
                        (4096,), dtype=torch.uint8, device="cpu"
                    )
                    if pinned:
                        register.assert_called_once_with(buffer, 512)
                    else:
                        register.assert_not_called()

    def test_gtt_exhaustion_names_the_cap_and_does_not_fall_back_to_userptr(self):
        """Running out of GTT must fail startup with the remedy, not silently
        fall back to a USERPTR pool."""
        allocator = host_common.HostTensorAllocator()
        out_of_memory = SimpleNamespace(
            hipHostMalloc=lambda ptr, nbytes, flags: 2,
            hipGetErrorString=lambda rc: b"out of memory",
        )
        is_hip, env = self._hip_non_userptr()
        with (
            is_hip,
            env,
            mock.patch.object(host_common, "_hip_runtime", return_value=out_of_memory),
            mock.patch.object(allocator, "allocate") as allocate,
            self.assertRaisesRegex(RuntimeError, "mem_info_gtt_total.*--hicache-size"),
        ):
            host_common.alloc_with_host_register(
                dims=(8,),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
                allocator=allocator,
            )
        allocate.assert_not_called()

    def test_dsa_page_layouts_with_draft_use_page_registration_granularity(self):
        target_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
        draft_buffer = torch.empty(1, dtype=torch.uint8)

        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
                host.device_pool = SimpleNamespace(
                    device="cpu",
                    get_hicache_indexer_page_buffers=lambda: target_buffers,
                )
                host.mtp_draft_device_pools = [
                    SimpleNamespace(
                        get_hicache_indexer_page_buffers=lambda: [draft_buffer]
                    )
                ]
                host.layout = layout
                host._live_target_layers = [0, 1, 2]
                host.layer_num = 4
                host.indexer_page_num = 3
                host.indexer_page_stride_size = 512
                host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
                host.indexer_dtype = torch.uint8
                host.device = "cpu"
                host.pin_memory = True
                host.allocator = mock.sentinel.allocator
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}):
                    host.init_kv_buffer()

                self.assertEqual(len(host.packed_device_index_buffers), 4)
                self.assertIs(host.packed_device_index_buffers[-1], draft_buffer)
                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    host.indexer_layout_dim,
                )

    def test_page_first_direct_mla_uses_page_registration_granularity(self):
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.layout = "page_first_direct"
        pool.page_num = 4
        pool.layer_num = 3
        pool.page_size = 2
        pool.kv_cache_dim = 5
        pool.dtype = torch.float16
        pool.device_pool = SimpleNamespace(device="cuda")
        pool._transfer_device = torch.device("cuda")
        pool.device = "cpu"
        pool.pin_memory = True
        pool.allocator = object()
        alloc = mock.Mock(return_value=object())

        with mock.patch.dict(mla_pool_host.ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
            pool.init_kv_buffer()

        self.assertEqual(
            alloc.call_args.kwargs["registration_granularity_bytes"],
            pool.page_size * pool.layer_num * pool.kv_cache_dim * pool.dtype.itemsize,
        )

    def test_page_first_direct_mha_uses_page_registration_granularity(self):
        pool = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        pool.layout = "page_first_direct"
        pool.page_num = 4
        pool.layer_num = 3
        pool.page_size = 2
        pool.head_num = 2
        pool.head_dim = 4
        pool.dtype = torch.float16
        pool.device_pool = SimpleNamespace(device="cuda")
        pool._transfer_device = torch.device("cuda")
        pool.device = "cpu"
        pool.pin_memory = True
        pool.allocator = object()
        alloc = mock.Mock(return_value=object())

        with mock.patch.dict(mha_pool_host.ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
            pool.init_kv_buffer()

        self.assertEqual(
            alloc.call_args.kwargs["registration_granularity_bytes"],
            pool.page_size
            * pool.layer_num
            * pool.head_num
            * pool.head_dim
            * pool.dtype.itemsize,
        )

    def test_mamba_page_layouts_use_per_buffer_page_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = MambaPoolHost.__new__(MambaPoolHost)
                pool.layout = layout
                pool.size = 4
                pool.num_mamba_layers = 3
                pool.temporal_state_shape = (2, 5)
                pool.conv_state_shapes = [(7,), (2, 2)]
                pool.temporal_dtype = torch.float16
                pool.conv_dtype = torch.float32
                pool._buffer_device = torch.device("cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(
                    side_effect=lambda *args, **kwargs: torch.empty(
                        1, dtype=torch.uint8
                    )
                )

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    [
                        call.kwargs["registration_granularity_bytes"]
                        for call in alloc.call_args_list
                    ],
                    [
                        3 * 2 * 5 * torch.float16.itemsize,
                        3 * 7 * torch.float32.itemsize,
                        3 * 2 * 2 * torch.float32.itemsize,
                    ],
                )

    def test_mamba_ple_sibling_uses_aligned_backing_rows(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.layout = "page_first"
        pool.page_size = 1
        pool.size = 2
        pool.num_mamba_layers = 1
        pool.temporal_state_shape = (0,)
        pool.temporal_state_elem_size = 0
        pool.conv_state_shapes = []
        pool.conv_state_elem_sizes = []
        pool.temporal_dtype = torch.float16
        pool.conv_dtype = torch.float16
        pool._buffer_device = torch.device("cpu")
        pool.device = "cpu"
        pool.pin_memory = False
        pool.allocator = object()
        ngram_device = torch.empty((3, 2), dtype=torch.int64)
        pool.slot_state_device_tensors = [ngram_device]
        pool.slot_state_names = ("ple_ngram",)
        pool.slot_state_elem_sizes = [2]
        pool.slot_state_storage_bytes = [4096]

        def allocate(dims, **kwargs):
            return torch.empty(
                dims,
                dtype=kwargs["dtype"],
                device=kwargs["device"],
            )

        alloc = mock.Mock(side_effect=allocate)
        with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}):
            pool.kv_buffer = pool.init_kv_buffer()

        logical = pool.slot_state_buffers[0]
        storage = pool.slot_state_storage_buffers[0]
        self.assertEqual(logical.shape, (2, 2))
        self.assertEqual(logical.stride(0) * logical.element_size(), 4096)
        self.assertEqual(storage.stride(0) * storage.element_size(), 4096)
        self.assertEqual(alloc.call_args.kwargs["registration_granularity_bytes"], 4096)
        self.assertIs(pool.get_hybrid_pool_buffer()[-1], storage)
        self.assertEqual(torch.count_nonzero(storage).item(), 0)

        logical[0] = torch.tensor([101, 202])
        self.assertTrue(torch.equal(storage[0, :2], torch.tensor([101, 202])))
        self.assertEqual(torch.count_nonzero(storage[0, 2:]).item(), 0)

        pool.size_per_token = pool.get_size_per_token()
        pool.set_from_flat_data_page(1, pool.get_data_page(0))
        self.assertTrue(torch.equal(logical[1], logical[0]))
        self.assertEqual(torch.count_nonzero(storage[1, 2:]).item(), 0)

        ptrs, sizes = pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(ptrs, [storage.data_ptr(), storage.data_ptr() + 4096])
        self.assertEqual(sizes, [4096, 4096])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_mamba_direct_copy_honors_padded_logical_stride(self):
        storage = torch.zeros((4, 512), dtype=torch.int64, pin_memory=True)
        host = storage.as_strided(size=(4, 2), stride=(512, 1))
        device = torch.arange(8, device="cuda", dtype=torch.int64).reshape(4, 2)
        host_indices = torch.tensor([0, 3], dtype=torch.int64)
        device_indices = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
        expected = device[device_indices].cpu()

        MambaPoolHost._copy_tensor(
            device, host, device_indices, host_indices, io_backend="direct"
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(host[host_indices], expected))
        self.assertEqual(torch.count_nonzero(storage[0, 2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(storage[3, 2:]).item(), 0)

        device[device_indices] = 0
        MambaPoolHost._copy_tensor(
            host, device, host_indices, device_indices, io_backend="direct"
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(device[device_indices].cpu(), expected))

    def test_mamba_page_roundtrip_includes_ple_siblings(self):
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.layout = "page_first"
        pool.page_size = 1
        pool.size = 2
        pool.num_mamba_layers = 1
        pool.temporal_state_elem_size = 2
        pool.conv_state_elem_sizes = [3]
        pool.temporal_dtype = torch.float16
        pool.conv_dtype = torch.float16
        pool.temporal_buffer = torch.arange(4, dtype=torch.float16).reshape(2, 1, 1, 2)
        pool.conv_buffer = [torch.arange(6, dtype=torch.float16).reshape(2, 1, 1, 3)]
        short_conv_device = torch.empty((2, 2, 4), dtype=torch.float16)
        ngram_device = torch.empty((3, 3), dtype=torch.int64)
        pool.slot_state_device_tensors = [short_conv_device, ngram_device]
        pool.slot_state_names = ("ple_short_conv", "ple_ngram")
        pool.slot_state_buffers = [
            torch.arange(16, dtype=torch.float16).reshape(2, 2, 4),
            torch.empty((2, 3), dtype=torch.int64),
        ]
        pool.slot_state_buffers[0][0] = torch.arange(8, dtype=torch.float16).reshape(
            2, 4
        )
        pool.slot_state_buffers[1][0] = torch.tensor([101, 202, 303])
        pool.size_per_token = pool.get_size_per_token()

        expected = [tensor.clone() for tensor in pool._iter_page_tensors(0)]
        data_page = pool.get_data_page(0)
        self.assertEqual(data_page.numel(), pool.size_per_token)
        for tensor in pool._iter_page_tensors(1):
            tensor.zero_()

        pool.set_from_flat_data_page(1, data_page)

        for actual, wanted in zip(pool._iter_page_tensors(1), expected):
            self.assertTrue(torch.equal(actual, wanted))
        self.assertEqual(
            [name for name, _ in pool.get_debug_page_tensors(1)],
            ["mamba_temporal", "mamba_conv_0", "ple_short_conv", "ple_ngram"],
        )

    def test_deepseek_v4_page_layouts_use_page_registration_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(pool="paged", layout=layout):
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
                device_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(3)]
                with (
                    mock.patch.object(
                        memory_pool_host,
                        "host_memory_budget_bytes",
                        return_value=1024**3,
                    ),
                    mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
                ):
                    DeepSeekV4PagedHostPool(
                        pool_name="test",
                        device_buffers=device_buffers,
                        item_bytes=11,
                        num_host_pages=4,
                        slot_page_size=2,
                        layout=layout,
                    )

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    3 * 11,
                )

            with self.subTest(pool="state", layout=layout):
                alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
                state_pools = [
                    SimpleNamespace(
                        ring_size=2,
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty((4, 3), dtype=torch.uint8)
                        ),
                    )
                    for _ in range(2)
                ]
                with (
                    mock.patch.object(
                        memory_pool_host,
                        "host_memory_budget_bytes",
                        return_value=1024**3,
                    ),
                    mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
                ):
                    DeepSeekV4StateHostPool(
                        pool_name="test",
                        state_pools=state_pools,
                        num_host_pages=4,
                        swa_page_size=2,
                        layout=layout,
                    )

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    2 * 2 * 3,
                )

    def test_k_only_mha_page_layouts_use_page_registration_granularity(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = MHATokenToKOnlyPoolHost.__new__(MHATokenToKOnlyPoolHost)
                pool.layout = layout
                pool.size = 8
                pool.page_num = 4
                pool.page_size = 2
                pool.layer_num = 3
                pool.head_num = 2
                pool.head_dim = 5
                pool.dtype = torch.float16
                pool.layout_dim = (
                    pool.layer_num * pool.head_num * pool.head_dim * pool.dtype.itemsize
                )
                pool.device_pool = SimpleNamespace(device="cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(return_value=object())

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    alloc.call_args.kwargs["registration_granularity_bytes"],
                    pool.page_size * pool.layout_dim,
                )

    def test_asymmetric_mha_page_layouts_use_native_page_granularities(self):
        for layout in ("page_first", "page_first_direct"):
            with self.subTest(layout=layout):
                pool = AsymmetricMHATokenToKVPoolHost.__new__(
                    AsymmetricMHATokenToKVPoolHost
                )
                pool.layout = layout
                pool.size = 8
                pool.page_num = 4
                pool.page_size = 2
                pool.layer_num = 3
                pool.head_num = 2
                pool.head_dim = 5
                pool.v_head_dim = 7
                pool.dtype = torch.float16
                pool.device_pool = SimpleNamespace(device="cuda")
                pool._transfer_device = torch.device("cuda")
                pool.device = "cpu"
                pool.pin_memory = True
                pool.allocator = object()
                alloc = mock.Mock(side_effect=[object(), object()])

                with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cuda": alloc}):
                    pool.init_kv_buffer()

                self.assertEqual(
                    [
                        call.kwargs["registration_granularity_bytes"]
                        for call in alloc.call_args_list
                    ],
                    [
                        pool.page_size * pool._k_layout_dim(),
                        pool.page_size * pool._v_layout_dim(),
                    ],
                )

    def test_unregister_releases_every_registered_chunk_once(self):
        gib = 1024**3
        base = 0x10000000
        buffer = _FakeBuffer(base, 2 * gib + 17)
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(buffer, registration_granularity_bytes=gib)
            _cuda_host_unregister(buffer)
            _cuda_host_unregister(buffer)

        self.assertEqual(
            cudart.unregistrations,
            [base + 2 * gib, base + gib, base],
        )

    def test_registration_failure_rolls_back_prior_chunks(self):
        gib = 1024**3
        base = 0x10000000
        buffer = _FakeBuffer(base, 2 * gib + 17)
        cudart = _FakeCudart(fail_on_registration=2)

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
            self.assertRaisesRegex(RuntimeError, "offset=1073741824"),
        ):
            _cuda_host_register(buffer, registration_granularity_bytes=gib)

        self.assertEqual(
            cudart.registrations,
            [(base, gib, 0), (base + gib, gib, 0)],
        )
        self.assertEqual(cudart.unregistrations, [base])

    def test_missing_copy_granularity_preserves_single_registration(self):
        gib = 1024**3
        base = 0x10000000
        total = 2 * gib + 17
        buffer = _FakeBuffer(base, total)
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(buffer)

        self.assertEqual(cudart.registrations, [(base, total, 0)])

    def test_registration_boundaries_honor_page_copy_granularity(self):
        mib = 1024**2
        gib = 1024**3
        base = 0x10000000
        total = 2500 * mib
        page_copy_bytes = 300 * mib
        cudart = _FakeCudart()

        with (
            mock.patch.object(
                envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB,
                "get",
                return_value=1,
            ),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            _cuda_host_register(
                _FakeBuffer(base, total),
                registration_granularity_bytes=page_copy_bytes,
            )

        aligned_chunk = 900 * mib
        self.assertLessEqual(aligned_chunk, gib)
        self.assertEqual(
            cudart.registrations,
            [
                (base, aligned_chunk, 0),
                (base + aligned_chunk, aligned_chunk, 0),
                (base + 2 * aligned_chunk, 700 * mib, 0),
            ],
        )
        for ptr, _, _ in cudart.registrations:
            self.assertEqual((ptr - base) % page_copy_bytes, 0)


class _FakeAttrCudart:
    def __init__(self, result=0, reported=1, raises=None):
        self.result = result
        self.reported = reported
        self.raises = raises
        self.calls = []

    def cudaDeviceGetAttribute(self, ptr, attr, device):
        self.calls.append((attr, device))
        if self.raises is not None:
            raise self.raises
        ptr._obj.value = self.reported
        return self.result


class TestHostPointerCapabilityQuery(unittest.TestCase):
    """can_use_host_pointer_for_registered_mem must be portable: no bare
    libcudart dlopen, safe (False, never raising) on HIP / non-CUDA, and a
    plain attribute query through torch.cuda.cudart() on CUDA."""

    def setUp(self):
        pool_host_common.can_use_host_pointer_for_registered_mem.cache_clear()

    tearDown = setUp

    def test_no_cuda_returns_false_without_querying(self):
        cudart = mock.Mock()
        with (
            mock.patch.object(pool_host_common, "_is_hip", False),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch.object(torch.cuda, "cudart", cudart),
        ):
            self.assertFalse(
                pool_host_common.can_use_host_pointer_for_registered_mem(0)
            )
        cudart.assert_not_called()

    def test_hip_returns_false_without_querying(self):
        cudart = mock.Mock()
        with (
            mock.patch.object(pool_host_common, "_is_hip", True),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.cuda, "cudart", cudart),
        ):
            self.assertFalse(
                pool_host_common.can_use_host_pointer_for_registered_mem(0)
            )
        cudart.assert_not_called()

    def test_supported_query_returns_true(self):
        cudart = _FakeAttrCudart(result=0, reported=1)
        with (
            mock.patch.object(pool_host_common, "_is_hip", False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            self.assertTrue(pool_host_common.can_use_host_pointer_for_registered_mem(1))
        attr = pool_host_common._CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        self.assertEqual(cudart.calls, [(attr, 1)])

    def test_absent_capability_returns_false(self):
        cudart = _FakeAttrCudart(result=0, reported=0)
        with (
            mock.patch.object(pool_host_common, "_is_hip", False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            self.assertFalse(
                pool_host_common.can_use_host_pointer_for_registered_mem(0)
            )

    def test_api_error_returns_false_without_raising(self):
        cudart = _FakeAttrCudart(result=3)
        with (
            mock.patch.object(pool_host_common, "_is_hip", False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            self.assertFalse(
                pool_host_common.can_use_host_pointer_for_registered_mem(0)
            )

    def test_cudart_unavailable_returns_false_without_raising(self):
        cudart = _FakeAttrCudart(raises=OSError("libcudart not loaded"))
        with (
            mock.patch.object(pool_host_common, "_is_hip", False),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
        ):
            self.assertFalse(
                pool_host_common.can_use_host_pointer_for_registered_mem(0)
            )


if __name__ == "__main__":
    unittest.main()
