"""Unit tests for the HiCacheController device-direct (GDS) routing.

The controller is built via __new__ with only the fields the routed paths
read, so no GPU, host pool, or storage backend installation is required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransferTarget,
)


class _StubBackend:
    def __init__(self, results=None):
        self.set_transfers = []
        self.get_transfers = []
        self._results = results

    def register_mem_device_pool_v2(self, device_pool, device_pool_name):
        if not hasattr(self, "registered_device_pools"):
            self.registered_device_pools = {}
        self.registered_device_pools[device_pool_name] = device_pool

    def batch_set_v2(self, transfers, extra_info=None):
        transfer = transfers[0]
        self.set_transfers.append(transfer)
        count = len(transfer.keys or [])
        if self._results is not None:
            return {transfer.name: self._results}
        return {transfer.name: [True] * count}

    def batch_get_v2(self, transfers, extra_info=None):
        transfer = transfers[0]
        self.get_transfers.append(transfer)
        count = len(transfer.keys or [])
        if self._results is not None:
            return {transfer.name: self._results}
        return {transfer.name: [True] * count}


class _StubDevicePool:
    def get_hicache_transfer_tensors(self):
        return [torch.zeros(8, dtype=torch.uint8)]


def _controller(page_size=4, device_direct=True, backend=None, results=None):
    controller = object.__new__(HiCacheController)
    controller.page_size = page_size
    controller.storage_device_direct = device_direct
    controller.storage_backend = backend or _StubBackend(results=results)
    controller.mem_pool_device = _StubDevicePool()
    controller.page_set_func = lambda hashes, host_indices, extra_info: True
    controller.page_get_func = lambda operation, hashes, host_indices, e: len(hashes)
    controller.prefetch_hits_sync_groups = []
    return controller


class TestStorageFormatSpec(unittest.TestCase):
    def test_runtime_fields_use_their_own_config_namespaces(self):
        controller = _controller()
        controller.mem_pool_host = SimpleNamespace()
        memory = SimpleNamespace(hicache_mem_layout="page_first")
        model = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            model_path="model-path",
            revision="model-revision",
        )
        mamba = SimpleNamespace(
            mamba_radix_cache_strategy="lru",
            mamba_ssm_dtype="bfloat16",
            mamba_track_interval=256,
            enable_int8_mamba_checkpoint=True,
            int8_mamba_ckpt_size=1024,
            enable_linear_replayssm=True,
            linear_replayssm_cache_len=64,
        )
        with (
            patch("sglang.srt.runtime_context.get_memory", return_value=memory),
            patch("sglang.srt.runtime_context.get_model", return_value=model),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(mamba=mamba),
            ),
        ):
            runtime = controller._storage_format_spec("fallback", None)["runtime"]

        self.assertEqual(runtime["hicache_mem_layout"], "page_first")
        self.assertEqual(runtime["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(runtime["mamba_track_interval"], 256)
        self.assertTrue(runtime["enable_int8_mamba_checkpoint"])
        self.assertEqual(runtime["model_path"], "model-path")
        self.assertEqual(runtime["model_revision"], "model-revision")


class TestDeviceBackup(unittest.TestCase):
    def test_device_indices_route_device_target(self):
        controller = _controller()
        operation = StorageOperation(
            host_indices=torch.arange(8),
            token_ids=list(range(8)),
            hash_value=["h0", "h1"],
            device_indices=torch.tensor([64, 65, 66, 67, 128, 129, 130, 131]),
        )
        controller._page_backup(operation)
        backend = controller.storage_backend
        self.assertEqual(len(backend.set_transfers), 1)
        transfer = backend.set_transfers[0]
        self.assertEqual(transfer.name, PoolName.KV)
        self.assertIs(transfer.target, PoolTransferTarget.DEVICE)
        self.assertIs(transfer.target_pool, controller.mem_pool_device)
        self.assertEqual(
            transfer.target_indices.tolist(), [64, 65, 66, 67, 128, 129, 130, 131]
        )
        self.assertEqual(operation.completed_tokens, 8)

    def test_host_only_operation_keeps_host_path(self):
        controller = _controller()
        calls = []
        controller.page_set_func = lambda hashes, host_indices, extra_info: (
            calls.append(len(hashes)) or True
        )
        operation = StorageOperation(
            host_indices=torch.arange(8),
            token_ids=list(range(8)),
            hash_value=["h0", "h1"],
        )
        controller._page_backup(operation)
        self.assertEqual(calls, [2])
        self.assertEqual(controller.storage_backend.set_transfers, [])

    def test_flag_off_keeps_host_path(self):
        controller = _controller(device_direct=False)
        calls = []
        controller.page_set_func = lambda hashes, host_indices, extra_info: (
            calls.append(len(hashes)) or True
        )
        operation = StorageOperation(
            host_indices=torch.arange(8),
            token_ids=list(range(8)),
            hash_value=["h0", "h1"],
            device_indices=torch.arange(8),
        )
        controller._page_backup(operation)
        self.assertEqual(calls, [2])

    def test_failed_device_batch_stops(self):
        backend = _StubBackend(results=[True, False])
        controller = _controller(backend=backend)
        operation = StorageOperation(
            host_indices=torch.arange(8),
            token_ids=list(range(8)),
            hash_value=["h0", "h1"],
            device_indices=torch.arange(8),
        )
        controller._page_backup(operation)
        self.assertEqual(operation.completed_tokens, 0)


class TestDevicePrefetchRead(unittest.TestCase):
    def test_device_batch_reads_device_target(self):
        controller = _controller()
        hits = controller._get_kv_device_batch(["h0", "h1"], torch.arange(8))
        self.assertEqual(hits, 2)
        recorded = controller.storage_backend.get_transfers[0]
        self.assertIs(recorded.target, PoolTransferTarget.DEVICE)

    def test_partial_read_clamps_hits(self):
        backend = _StubBackend(results=[True, False])
        controller = _controller(backend=backend)
        hits = controller._get_kv_device_batch(["h0", "h1"], torch.arange(8))
        self.assertEqual(hits, 1)


class TestDevicePoolRegistration(unittest.TestCase):
    def test_register_kv_pool(self):
        controller = _controller(backend=_StubBackend())
        controller._register_device_pools()
        self.assertIs(
            controller.storage_backend.registered_device_pools[PoolName.KV],
            controller.mem_pool_device,
        )
        self.assertTrue(controller.storage_device_direct)

    def test_unsupported_pool_disables_direct(self):
        controller = _controller(backend=_StubBackend())
        controller.mem_pool_device = object()
        controller._register_device_pools()
        self.assertFalse(controller.storage_device_direct)


if __name__ == "__main__":
    unittest.main()
