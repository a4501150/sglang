"""Unit tests for the device-direct (GDS) prefetch consumption path.

Covers tree-core device splices (insert_device), the cache-side completion
handler, and controller routing of device-target reads including KV-derived
sidecars. Runs without GPU, NIXL, or a storage backend."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from array import array
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferTarget,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV, FreeDeviceKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

_FULL = ComponentType.FULL
_COMPONENTS = (_FULL,)


def _key(tokens):
    return RadixKey(array("q", tokens))


class _CoreStub:
    """Hosts the insert_device walk: real nodes, minimal core services."""

    def __init__(self, page_size=2):
        self.page_size = page_size
        self.components = []
        self.components_by_type = {}
        self.created = []
        self.touched = []
        self._nodes = {}

    def node_by_id(self, node_id):
        return self._nodes[node_id]

    def add(self, node):
        self._nodes[node.id] = node
        return node

    def _touch_node(self, node):
        self.touched.append(node.id)

    def _split_node(self, key, child, split_len):
        raise AssertionError("split not expected")

    def _add_new_node(self, parent, key, value, priority=0):
        node = UnifiedTreeNode(_COMPONENTS, priority=priority)
        node.parent = parent
        node.key = key
        node.component_data[_FULL].value = value.clone()
        parent.children[key.child_key(self.page_size)] = node
        self.created.append(node)
        return node

    def _build_backup_kv_action(self, node):
        return BackupKV([node.id])


class TestInsertDevice(unittest.TestCase):
    def _anchor(self):
        anchor = UnifiedTreeNode(_COMPONENTS)
        anchor.parent = UnifiedTreeNode(_COMPONENTS)
        anchor.component_data[_FULL].host_value = torch.arange(4)
        return anchor

    def test_fresh_splice_creates_device_leaf_with_backup(self):
        core = _CoreStub()
        anchor = core.add(self._anchor())
        key = _key([1, 2, 3, 4])
        value = torch.tensor([10, 11, 12, 13])

        result = UnifiedTreeCore.insert_device(
            core, anchor.id, key, value, ["h1", "h2"]
        )

        self.assertEqual(result.prefix_len, 0)
        self.assertEqual(len(core.created), 1)
        node = core.created[0]
        self.assertEqual(node.component_data[_FULL].value.tolist(), [10, 11, 12, 13])
        self.assertEqual(node.hash_value, ["h1", "h2"])
        self.assertIn(key.child_key(core.page_size), anchor.children)
        self.assertEqual(result.inserted_host_node, node.id)
        self.assertEqual(result.last_device_node, node.id)
        self.assertEqual(result.cache_actions, [BackupKV([node.id])])

    def test_full_overlap_keeps_existing_data(self):
        core = _CoreStub()
        anchor = core.add(self._anchor())
        key = _key([1, 2, 3, 4])
        child = UnifiedTreeNode(_COMPONENTS)
        child.parent = anchor
        child.key = key
        child.component_data[_FULL].value = torch.tensor([90, 91, 92, 93])
        anchor.children[key.child_key(core.page_size)] = child
        core.add(child)

        result = UnifiedTreeCore.insert_device(
            core, anchor.id, key, torch.tensor([10, 11, 12, 13]), ["h1", "h2"]
        )

        self.assertEqual(result.prefix_len, 4)
        self.assertEqual(core.created, [])
        self.assertEqual(result.cache_actions, [])
        # The existing node's slots are untouched; the caller frees the
        # duplicate bounce separately.
        self.assertEqual(child.component_data[_FULL].value.tolist(), [90, 91, 92, 93])

    def test_partial_overlap_splices_the_suffix(self):
        core = _CoreStub()
        anchor = core.add(self._anchor())
        head_key = _key([1, 2])
        child = UnifiedTreeNode(_COMPONENTS)
        child.parent = anchor
        child.key = head_key
        child.component_data[_FULL].value = torch.tensor([90, 91])
        anchor.children[head_key.child_key(core.page_size)] = child
        core.add(child)

        result = UnifiedTreeCore.insert_device(
            core,
            anchor.id,
            _key([1, 2, 3, 4]),
            torch.tensor([10, 11, 12, 13]),
            ["h1", "h2"],
        )

        self.assertEqual(result.prefix_len, 2)
        self.assertEqual(len(core.created), 1)
        node = core.created[0]
        self.assertEqual(node.component_data[_FULL].value.tolist(), [12, 13])
        self.assertEqual(node.hash_value, ["h2"])
        self.assertEqual(node.parent, child)

    def test_longer_child_splits_before_descending(self):
        core = _CoreStub()
        anchor = core.add(self._anchor())
        key = _key([1, 2, 3, 4])
        child = UnifiedTreeNode(_COMPONENTS)
        child.parent = anchor
        child.key = _key([1, 2, 3, 4, 5, 6])
        child.component_data[_FULL].value = torch.tensor([90, 91, 92, 93, 94, 95])
        anchor.children[child.key.child_key(core.page_size)] = child
        core.add(child)
        split = mock.Mock(return_value=(child, None))
        core._split_node = split

        result = UnifiedTreeCore.insert_device(
            core, anchor.id, key, torch.tensor([10, 11, 12, 13]), ["h1", "h2"]
        )

        split.assert_called_once_with(child.key, child, 4)
        self.assertEqual(result.prefix_len, 4)
        self.assertEqual(core.created, [])


class _CompletionStub:
    """Hosts _handle_device_prefetch_result with mock tree and controller."""

    def __init__(self, insert_result, request):
        self.page_size = 2
        self.tree_core = mock.Mock()
        self.tree_core.insert_device.return_value = insert_result
        self.cache_controller = mock.Mock()
        self.cache_controller.prefetch_tokens_occupied = 8
        self.applied = []
        self.resolved = []
        self.dec_locks = []
        self.ongoing_prefetch = {request: object()}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.prefetch_loaded_storage_start_by_reqid = {}

    def _apply_cache_actions(self, actions):
        self.applied.extend(actions)

    def _apply_cache_action(self, action):
        self.applied.append(action)

    def _resolve_storage_prefetch_tokens(self, req_id, num_tokens, reason=None):
        self.resolved.append((req_id, num_tokens))

    def dec_host_lock_ref(self, node_id, params):
        self.dec_locks.append(node_id)


def _completed_operation(request, completed=8, device_span=10):
    operation = PrefetchOperation(request, list(range(device_span)))
    operation.completed_tokens = completed
    operation.device_indices = torch.arange(device_span)
    operation.hash_value = ["h0", "h1", "h2", "h3", "h4"]
    operation.storage_start = 64
    return operation


class TestDevicePrefetchCompletion(unittest.TestCase):
    def test_splice_frees_prefix_and_accounts_loaded_span(self):
        insert_result = InsertResult(
            prefix_len=4,
            total_len=8,
            cache_actions=[BackupKV([7])],
            inserted_host_node=7,
        )
        request = CacheRequestHandle("r", 0)
        stub = _CompletionStub(insert_result, request)
        operation = _completed_operation(request, completed=8)
        prefetch_key = _key(list(range(8)))

        UnifiedRadixCache._handle_device_prefetch_result(
            stub, request, operation, 5, prefetch_key, mock.Mock()
        )

        _, args = stub.tree_core.insert_device.call_args
        # Called positionally: (anchor, key, device prefix, hash prefix).
        self.assertEqual(args, {})
        anchor_id, key, device_prefix, hashes = stub.tree_core.insert_device.call_args[
            0
        ]
        self.assertEqual(anchor_id, 5)
        self.assertEqual(list(key), list(range(8)))
        self.assertEqual(device_prefix.tolist(), list(range(8)))
        self.assertEqual(hashes, ["h0", "h1", "h2", "h3"])
        # Backup action applied, then the duplicate prefix freed.
        self.assertIsInstance(stub.applied[0], BackupKV)
        self.assertIsInstance(stub.applied[1], FreeDeviceKV)
        self.assertEqual(stub.applied[1].indices[0].tolist(), [0, 1, 2, 3])
        self.assertNotIn(request, stub.ongoing_prefetch)
        self.assertEqual(stub.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(stub.prefetch_loaded_tokens_by_reqid[request], 4)
        self.assertEqual(stub.prefetch_loaded_storage_start_by_reqid[request], 68)
        self.assertEqual(stub.dec_locks, [5])

    def test_zero_prefix_frees_nothing(self):
        insert_result = InsertResult(prefix_len=0, total_len=8, cache_actions=[])
        request = CacheRequestHandle("r", 0)
        stub = _CompletionStub(insert_result, request)
        operation = _completed_operation(request, completed=8)

        UnifiedRadixCache._handle_device_prefetch_result(
            stub, request, operation, 5, _key(list(range(8))), mock.Mock()
        )

        self.assertEqual(len(stub.applied), 0)
        self.assertEqual(stub.prefetch_loaded_tokens_by_reqid[request], 8)
        # loaded=8 > 0 with a zero match: the span starts at the fetch start.
        self.assertEqual(stub.prefetch_loaded_storage_start_by_reqid[request], 64)


class _RoutingBackend:
    def __init__(self, kv_pool, indexer_pool, register_indexer=True):
        self.registered_device_pools = {PoolName.KV: kv_pool}
        if register_indexer:
            self.registered_device_pools[PoolName.INDEXER] = indexer_pool
        self.indexer_pool = indexer_pool
        self.get_transfers = []

    def batch_get_v2(self, transfers, extra_info=None):
        self.get_transfers.append(list(transfers))
        return {t.name: [True] * len(t.keys) for t in transfers}


def _io_controller(backend, device_direct=True):
    cc = object.__new__(HiCacheController)
    cc.page_size = 4
    cc.storage_device_direct = device_direct
    cc.storage_backend = backend
    cc.mem_pool_device = getattr(backend, "registered_device_pools", {}).get(
        PoolName.KV
    )
    cc.prefetch_sync_queue = Queue()
    return cc


def _prefetch_operation(pool_transfers=None):
    operation = PrefetchOperation("r", list(range(8)))
    operation.hash_value = ["h0", "h1"]
    operation.storage_hit_count = 8
    operation.device_indices = torch.arange(8)
    operation.pool_transfers = pool_transfers
    return operation


class TestDevicePrefetchRouting(unittest.TestCase):
    def test_device_read_routes_kv_and_sidecar_to_device(self):
        kv_pool = object()
        backend = _RoutingBackend(kv_pool, "indexer-pool")
        cc = _io_controller(backend)
        sidecar = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        operation = _prefetch_operation([sidecar])

        completed_pages = cc._page_transfer(operation)

        self.assertEqual(completed_pages, 2)
        kv_call = backend.get_transfers[0][0]
        self.assertEqual(kv_call.name, PoolName.KV)
        self.assertIs(kv_call.target, PoolTransferTarget.DEVICE)
        self.assertIs(kv_call.target_pool, kv_pool)
        self.assertEqual(kv_call.target_indices.tolist(), list(range(8)))
        sidecar_call = backend.get_transfers[1][0]
        self.assertEqual(sidecar_call.name, PoolName.INDEXER)
        self.assertIs(sidecar_call.target, PoolTransferTarget.DEVICE)
        self.assertEqual(sidecar_call.target_pool, "indexer-pool")
        self.assertEqual(sidecar_call.target_indices.tolist(), list(range(8)))
        self.assertEqual(sidecar_call.indices_from_pool, PoolName.KV)
        ack = cc.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.completed_tokens, 8)

    def test_supported_flag_rules(self):
        backend = _RoutingBackend("kv-pool", "indexer-pool")
        cc = _io_controller(backend)
        self.assertTrue(cc.device_prefetch_supported(_prefetch_operation(None)))
        sidecar = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        self.assertTrue(cc.device_prefetch_supported(_prefetch_operation([sidecar])))
        # Aux pool needing host staging: device path off for this operation.
        aux = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.arange(4))
        self.assertFalse(cc.device_prefetch_supported(_prefetch_operation([aux])))
        # KV-derived sidecar without a registered device pool: no device path.
        no_indexer = _RoutingBackend("kv-pool", None, register_indexer=False)
        cc2 = _io_controller(no_indexer)
        self.assertFalse(cc2.device_prefetch_supported(_prefetch_operation([sidecar])))
        # Device-direct disabled globally.
        cc3 = _io_controller(backend, device_direct=False)
        self.assertFalse(cc3.device_prefetch_supported(_prefetch_operation(None)))

        # Backend without the device-pool registry at all.
        class _NoRegistry:
            def batch_get_v2(self, transfers, extra_info=None):
                return {}

        cc4 = _io_controller(_NoRegistry())
        self.assertFalse(cc4.device_prefetch_supported(_prefetch_operation(None)))

    def test_host_path_still_works_when_not_device_direct(self):
        backend = _RoutingBackend("kv-pool", "indexer-pool", register_indexer=False)
        cc = _io_controller(backend, device_direct=False)
        calls = []
        cc.page_get_func = lambda operation, hashes, host_indices, extra_info: (
            calls.append(host_indices.tolist()) or len(hashes)
        )
        operation = _prefetch_operation(None)
        operation.device_indices = None
        operation.host_indices = torch.arange(100, 108)

        completed_pages = cc._page_transfer(operation)

        self.assertEqual(completed_pages, 2)
        self.assertEqual(calls, [[100, 101, 102, 103, 104, 105, 106, 107]])
        self.assertEqual(backend.get_transfers, [])


if __name__ == "__main__":
    unittest.main()
