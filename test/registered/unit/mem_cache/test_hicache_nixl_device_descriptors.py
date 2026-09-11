"""Unit tests for the HiCacheNixl device (GDS) descriptor path.

The nixl package is stubbed so these run on hosts without NIXL or nvidia-fs
(for example WSL2). Only descriptor construction and validation are covered;
real GDS I/O is validated on native Linux.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferTarget,
)
from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlBackendSelection


def _load_hicache_nixl_with_stub():
    nixl_pkg = types.ModuleType("nixl")
    api = types.ModuleType("nixl._api")

    class nixl_agent:  # noqa: N801
        pass

    class nixl_agent_config:  # noqa: N801
        pass

    class nixlBind:  # noqa: N801
        NIXL_THREAD_SYNC_STRICT = 0
        NIXL_THREAD_SYNC_RW = 1
        nixlAgentConfig = object
        nixlAgent = object

    api.nixl_agent = nixl_agent
    api.nixl_agent_config = nixl_agent_config
    api.nixlBind = nixlBind
    nixl_pkg._api = api

    module_name = (
        "sglang.srt.mem_cache.storage.nixl._hicache_nixl_device_descriptor_test"
    )
    source_path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source_path}")
    module = importlib.util.module_from_spec(spec)
    previous = {name: sys.modules.get(name) for name in ("nixl", "nixl._api")}
    try:
        sys.modules["nixl"] = nixl_pkg
        sys.modules["nixl._api"] = api
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return module.HiCacheNixl


HiCacheNixl = _load_hicache_nixl_with_stub()

_HAS_CUDA = torch.cuda.is_available()


class _FakeAgent:
    def __init__(self):
        self.registrations = []

    def get_reg_descs(self, descs, mem_type):
        return (descs, mem_type)

    def register_memory(self, reg_descs):
        descs, mem_type = reg_descs
        self.registrations.append((descs, mem_type))
        return f"handle-{len(self.registrations)}"

    def deregister_memory(self, handle):
        pass


class _StubPool:
    """Minimal device pool with fixed two-component page segments."""

    def __init__(self, tensors, page_size=4):
        self._tensors = tensors
        self.page_size = page_size

    def get_hicache_transfer_tensors(self):
        return list(self._tensors)

    def set_segments(self, segments):
        self._segments = segments

    def get_hicache_page_segments(self, target_indices):
        return list(self._segments)


def _make_storage(backend_name="GDS", compat=False):
    storage = object.__new__(HiCacheNixl)
    storage.config_suffix = "_model_0_1"
    storage.registered_device_pools = {}
    storage._device_regs = []
    storage._device_reg_addrs = set()
    storage._host_regs = []
    storage._gds_compatibility_mode = compat
    storage._storage_mode = "auto"
    selector = NixlBackendSelection(backend_name, None)
    selector.backend_name = backend_name
    selector.mem_type = "OBJ" if backend_name == "OBJ" else "FILE"
    selector.device_backend_name = (
        backend_name if backend_name in ("GDS", "GDS_MT") else None
    )
    storage.backend_selector = selector
    storage._device_target_enabled = selector.device_backend_name is not None
    storage.agent = _FakeAgent()
    return storage


@unittest.skipUnless(_HAS_CUDA, "device registration needs CUDA memory")
class TestDeviceRegistration(unittest.TestCase):
    def test_registers_vram_regions(self):
        storage = _make_storage()
        tensors = [
            torch.zeros(16, dtype=torch.uint8, device="cuda"),
            torch.zeros(32, dtype=torch.uint8, device="cuda"),
        ]
        pool = _StubPool(tensors)
        storage.register_mem_device_pool_v2(pool, PoolName.KV)
        self.assertIs(storage.registered_device_pools[PoolName.KV], pool)
        self.assertEqual(len(storage.agent.registrations), 2)
        for (descs, mem_type), tensor in zip(storage.agent.registrations, tensors):
            self.assertEqual(mem_type, "VRAM")
            self.assertEqual(len(descs), 1)
            self.assertEqual(descs[0][0], tensor.data_ptr())
            self.assertEqual(descs[0][1], tensor.numel())
        # Re-registering the same pool object is a no-op.
        storage.register_mem_device_pool_v2(pool, PoolName.KV)
        self.assertEqual(len(storage.agent.registrations), 2)

    def test_rejects_cpu_tensors(self):
        storage = _make_storage()
        pool = _StubPool([torch.zeros(4, dtype=torch.uint8)])
        with self.assertRaises(ValueError):
            storage.register_mem_device_pool_v2(pool, PoolName.KV)

    def test_rejects_non_gds_backend(self):
        storage = _make_storage(backend_name="POSIX")
        pool = _StubPool([torch.zeros(4, dtype=torch.uint8, device="cuda")])
        with self.assertRaises(RuntimeError):
            storage.register_mem_device_pool_v2(pool, PoolName.KV)

    def test_rejects_compat_mode(self):
        storage = _make_storage(compat=True)
        pool = _StubPool([torch.zeros(4, dtype=torch.uint8, device="cuda")])
        with self.assertRaises(RuntimeError):
            storage.register_mem_device_pool_v2(pool, PoolName.KV)

    def test_supports_device_target_capability(self):
        self.assertTrue(_make_storage().supports_device_target)
        self.assertFalse(_make_storage(backend_name="POSIX").supports_device_target)
        self.assertFalse(_make_storage(compat=True).supports_device_target)

    def test_rejects_second_pool(self):
        storage = _make_storage()
        tensor = torch.zeros(4, dtype=torch.uint8, device="cuda")
        storage.register_mem_device_pool_v2(_StubPool([tensor]), PoolName.KV)
        with self.assertRaises(ValueError):
            storage.register_mem_device_pool_v2(_StubPool([tensor]), PoolName.KV)

    def test_re_registering_under_second_name(self):
        storage = _make_storage()
        tensor = torch.zeros(16, dtype=torch.uint8, device="cuda")
        pool = _StubPool([tensor])
        storage.register_mem_device_pool_v2(pool, PoolName.KV)
        self.assertEqual(len(storage.agent.registrations), 1)
        # Same buffer exposed under a second pool name (KV + INDEXER on one
        # QSA pool) maps the name without a duplicate NIXL registration.
        storage.register_mem_device_pool_v2(pool, PoolName.INDEXER)
        self.assertEqual(len(storage.agent.registrations), 1)
        self.assertIs(storage.registered_device_pools[PoolName.INDEXER], pool)


class TestPrepareDeviceTransfer(unittest.TestCase):
    def _registered(self, pool, pool_name=PoolName.KV):
        storage = _make_storage()
        # Registration checks is_cuda; descriptor building only reads
        # data_ptr, so these tests use CPU tensors.
        storage.registered_device_pools[pool_name] = pool
        return storage

    def test_kv_component_keys(self):
        t1 = torch.zeros(4, dtype=torch.uint8)
        t2 = torch.zeros(4, dtype=torch.uint8)
        pool = _StubPool([t1, t2])
        seg_a = ("k_0", t1.data_ptr(), 4)
        seg_b = ("v_0", t2.data_ptr(), 4)
        pool.set_segments([seg_a, seg_b, seg_a, seg_b])
        storage = self._registered(pool)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["pageA", "pageB"],
            target=PoolTransferTarget.DEVICE,
            target_pool=pool,
            target_indices=torch.arange(8),
        )
        keys, descs, multiplier = storage._prepare_device_transfer(transfer)
        self.assertEqual(multiplier, 2)
        self.assertEqual(
            keys,
            [
                "pageA_model_0_1_k_0",
                "pageA_model_0_1_v_0",
                "pageB_model_0_1_k_0",
                "pageB_model_0_1_v_0",
            ],
        )
        self.assertEqual(
            descs,
            [
                (t1.data_ptr(), 4),
                (t2.data_ptr(), 4),
                (t1.data_ptr(), 4),
                (t2.data_ptr(), 4),
            ],
        )

    def test_other_pools_get_pool_name_in_key(self):
        t1 = torch.zeros(4, dtype=torch.uint8)
        pool = _StubPool([t1], page_size=1)
        pool.set_segments([("conv_0_0", t1.data_ptr(), 4)])
        storage = self._registered(pool, PoolName.MAMBA)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["pageA"],
            target=PoolTransferTarget.DEVICE,
            target_pool=pool,
            target_indices=torch.tensor([3]),
        )
        keys, descs, multiplier = storage._prepare_device_transfer(transfer)
        self.assertEqual(keys, ["pageA_model_0_1_mamba_conv_0_0"])
        self.assertEqual(multiplier, 1)

    def test_index_count_mismatch(self):
        t1 = torch.zeros(4, dtype=torch.uint8)
        pool = _StubPool([t1])
        pool.set_segments([("k_0", t1.data_ptr(), 4)])
        storage = self._registered(pool)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["pageA"],
            target=PoolTransferTarget.DEVICE,
            target_pool=pool,
            # page_size is 4, so one key expects exactly 4 indices.
            target_indices=torch.arange(5),
        )
        storage = self._registered(pool)
        keys, descs, multiplier = storage._prepare_device_transfer(transfer)
        self.assertEqual((keys, descs, multiplier), ([], [], 0))

    def test_unregistered_pool(self):
        t1 = torch.zeros(4, dtype=torch.uint8)
        pool = _StubPool([t1])
        storage = _make_storage()
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["pageA"],
            target=PoolTransferTarget.DEVICE,
            target_pool=pool,
            target_indices=torch.arange(4),
        )
        keys, descs, multiplier = storage._prepare_device_transfer(transfer)
        self.assertEqual((keys, descs, multiplier), ([], [], 0))

    def test_component_order_change_raises(self):
        t1 = torch.zeros(4, dtype=torch.uint8)
        t2 = torch.zeros(4, dtype=torch.uint8)
        pool = _StubPool([t1, t2])
        pool.set_segments(
            [
                ("k_0", t1.data_ptr(), 4),
                ("v_0", t2.data_ptr(), 4),
                ("v_0", t2.data_ptr(), 4),
                ("k_0", t1.data_ptr(), 4),
            ]
        )
        storage = self._registered(pool)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["pageA", "pageB"],
            target=PoolTransferTarget.DEVICE,
            target_pool=pool,
            target_indices=torch.arange(8),
        )
        with self.assertRaises(ValueError):
            storage._prepare_device_transfer(transfer)


class TestStorageModeConfig(unittest.TestCase):
    def test_defaults_and_validation(self):
        from sglang.srt.mem_cache.storage.nixl.nixl_utils import NixlBackendConfig

        self.assertEqual(NixlBackendConfig({}).get_storage_mode(), "auto")
        self.assertEqual(
            NixlBackendConfig({"storage_mode": "GDS"}).get_storage_mode(), "gds"
        )
        with self.assertRaises(ValueError):
            NixlBackendConfig({"storage_mode": "nfs"}).get_storage_mode()


class _PluginAgent:
    def __init__(self, plugins):
        self._plugins = plugins
        self.created = []

    def get_plugin_list(self):
        return list(self._plugins)

    def create_backend(self, name, params):
        self.created.append(name)


class TestBackendResolution(unittest.TestCase):
    def _selector(self, plugin):
        selector = NixlBackendSelection(plugin, None)
        selector.backend_name = plugin
        selector.mem_type = "FILE"
        return selector

    def test_device_companion_created(self):
        selector = self._selector("POSIX")
        agent = _PluginAgent(["POSIX", "GDS"])
        self.assertTrue(selector.resolve_device_backend(agent))
        self.assertEqual(selector.device_backend_name, "GDS")
        self.assertEqual(agent.created, ["GDS"])
        self.assertEqual(selector.backend_name, "POSIX")

    def test_no_gds_plugin_disables_device(self):
        selector = self._selector("POSIX")
        agent = _PluginAgent(["POSIX"])
        self.assertFalse(selector.resolve_device_backend(agent))
        self.assertIsNone(selector.device_backend_name)

    def test_gds_mt_preferred(self):
        selector = self._selector("POSIX")
        agent = _PluginAgent(["POSIX", "GDS", "GDS_MT"])
        self.assertTrue(selector.resolve_device_backend(agent))
        self.assertEqual(selector.device_backend_name, "GDS_MT")

    def test_gds_primary_gets_posix_host_backend(self):
        selector = self._selector("GDS")
        selector.device_backend_name = "GDS"
        agent = _PluginAgent(["POSIX", "GDS"])
        self.assertTrue(selector.resolve_host_backend(agent))
        self.assertEqual(selector.backend_name, "POSIX")
        self.assertEqual(selector.device_backend_name, "GDS")
        self.assertEqual(agent.created, ["POSIX"])

    def test_gds_primary_without_host_companion(self):
        selector = self._selector("GDS")
        selector.device_backend_name = "GDS"
        agent = _PluginAgent(["GDS"])
        self.assertFalse(selector.resolve_host_backend(agent))


if __name__ == "__main__":
    unittest.main()
