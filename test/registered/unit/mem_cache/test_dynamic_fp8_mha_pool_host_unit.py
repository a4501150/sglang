"""Unit tests for dynamic FP8 K/V scale transfer and storage pages."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolDynamicFP8
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mha_dynamic_fp8 import (
    MHATokenToKVPoolDynamicFP8Host,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 4
LAYER_NUM = 2
HEAD_NUM = 2
HEAD_DIM = 8
PAGE_NUM = 3
TOKEN_NUM = PAGE_NUM * PAGE_SIZE
DEVICE_START_LAYER = 4


def _make_host(**overrides):
    host = MHATokenToKVPoolDynamicFP8Host.__new__(MHATokenToKVPoolDynamicFP8Host)
    host.layout = "page_first_direct"
    host.page_size = PAGE_SIZE
    host.page_num = PAGE_NUM
    host.size = TOKEN_NUM
    host.layer_num = LAYER_NUM
    host.target_layer_num = LAYER_NUM
    host.head_num = HEAD_NUM
    host.head_dim = HEAD_DIM
    host.dtype = torch.float8_e4m3fn
    host.device = "cpu"
    host.pin_memory = False
    host.pool_label = "kv"
    host.kv_buffer = torch.zeros(
        2,
        PAGE_NUM,
        LAYER_NUM,
        PAGE_SIZE,
        HEAD_NUM,
        HEAD_DIM,
        dtype=host.dtype,
    )
    host.scale_host = torch.ones(
        PAGE_NUM,
        2,
        LAYER_NUM,
        PAGE_SIZE,
        HEAD_NUM,
        dtype=torch.float32,
    )
    device_pool = SimpleNamespace(
        layer_shard_enabled=False,
        layer_num=LAYER_NUM,
        layer_shard_size=1,
        start_layer=DEVICE_START_LAYER,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        row_dim=HEAD_NUM * HEAD_DIM,
        k_buffer=[
            torch.zeros(TOKEN_NUM, HEAD_NUM, HEAD_DIM, dtype=host.dtype)
            for _ in range(LAYER_NUM)
        ],
        v_buffer=[
            torch.zeros(TOKEN_NUM, HEAD_NUM, HEAD_DIM, dtype=host.dtype)
            for _ in range(LAYER_NUM)
        ],
        k_scale_buffer=[torch.zeros(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)],
        v_scale_buffer=[torch.zeros(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)],
    )
    for name, value in overrides.items():
        setattr(device_pool, name, value)
    host.device_pool = device_pool
    return host


def _seed_device_scales(host):
    source = torch.arange(LAYER_NUM * TOKEN_NUM * HEAD_NUM, dtype=torch.float32)
    source = source.reshape(LAYER_NUM, TOKEN_NUM, HEAD_NUM)
    for layer in range(LAYER_NUM):
        host.device_pool.k_scale_buffer[layer].copy_(source[layer] + 1)
        host.device_pool.v_scale_buffer[layer].copy_(source[layer] + 101)
    return source


def _digest_lines(logs, component):
    return [line for line in logs.output if f"component={component}" in line]


class TestDynamicFP8MHATokenToKVPoolHost(CustomTestCase):
    def test_factory_selects_dynamic_fp8_host_pool(self):
        pool = MHATokenToKVPoolDynamicFP8.__new__(MHATokenToKVPoolDynamicFP8)
        pool.head_dim = pool.v_head_dim = HEAD_DIM
        self.assertIs(get_mha_host_pool_cls(pool), MHATokenToKVPoolDynamicFP8Host)

    def test_scale_rows_round_trip_through_l2(self):
        host = _make_host()
        source = _seed_device_scales(host)
        device_indices = torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)

        with (
            mock.patch.object(MHATokenToKVPoolHost, "backup_from_device_all_layer"),
            mock.patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer"),
        ):
            host.backup_from_device_all_layer(
                host.device_pool, host_indices, device_indices, io_backend="direct"
            )
            for buffers in (
                host.device_pool.k_scale_buffer,
                host.device_pool.v_scale_buffer,
            ):
                for buffer in buffers:
                    buffer.zero_()
            for layer in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    layer,
                    io_backend="direct",
                )

        for layer in range(LAYER_NUM):
            torch.testing.assert_close(
                host.device_pool.k_scale_buffer[layer][device_indices],
                source[layer][device_indices] + 1,
            )
            torch.testing.assert_close(
                host.device_pool.v_scale_buffer[layer][device_indices],
                source[layer][device_indices] + 101,
            )

    def test_backup_maps_device_layers_with_nonzero_start_layer(self):
        # The scale loop must index the device pool's own (stage-local) buffers
        # and map host rows through the layer-local host mapping, exactly like
        # the payload's ptr-table backup: the stage's start_layer must not
        # shift either side.
        host = _make_host(
            layer_shard_enabled=True,
            layer_shard_size=2,
            _owned_local_layer_range=lambda: (1, 2),
        )
        device_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)

        # Only device layer 1 (the shard this rank owns) feeds host row 0;
        # host row 1 and device layer 0 must stay untouched.
        host.device_pool.k_scale_buffer[1].fill_(7.0)
        host.device_pool.v_scale_buffer[1].fill_(9.0)
        host.device_pool.k_scale_buffer[0].fill_(123.0)
        host.device_pool.v_scale_buffer[0].fill_(123.0)

        with mock.patch.object(MHATokenToKVPoolHost, "backup_from_device_all_layer"):
            host.backup_from_device_all_layer(
                host.device_pool, host_indices, device_indices, io_backend="direct"
            )

        torch.testing.assert_close(
            host.k_scale_host[0:2, 0],
            torch.full((2, PAGE_SIZE, HEAD_NUM), 7.0),
        )
        torch.testing.assert_close(
            host.v_scale_host[0:2, 0],
            torch.full((2, PAGE_SIZE, HEAD_NUM), 9.0),
        )
        torch.testing.assert_close(
            host.k_scale_host[0:2, 1], torch.ones(2, PAGE_SIZE, HEAD_NUM)
        )
        self.assertFalse(torch.isnan(host.device_pool.k_scale_buffer[0]).any())
        self.assertTrue(
            torch.equal(
                host.device_pool.k_scale_buffer[0],
                torch.full_like(host.device_pool.k_scale_buffer[0], 123.0),
            )
        )

        # H2D honors the same mapping: layer 0 is not owned and must not
        # overwrite device layer 0 from host row 0.
        host.device_pool.k_scale_buffer[1].zero_()
        host.device_pool.v_scale_buffer[1].zero_()
        with mock.patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer"):
            host.load_to_device_per_layer(
                host.device_pool,
                host_indices,
                device_indices,
                0,
                io_backend="direct",
            )
            torch.testing.assert_close(
                host.device_pool.k_scale_buffer[0],
                torch.full_like(host.device_pool.k_scale_buffer[0], 123.0),
            )
            host.load_to_device_per_layer(
                host.device_pool,
                host_indices,
                device_indices,
                1,
                io_backend="direct",
            )
        torch.testing.assert_close(
            host.device_pool.k_scale_buffer[1][device_indices],
            torch.full((2 * PAGE_SIZE, HEAD_NUM), 7.0),
        )
        torch.testing.assert_close(
            host.device_pool.v_scale_buffer[1][device_indices],
            torch.full((2 * PAGE_SIZE, HEAD_NUM), 9.0),
        )

    def test_backup_rejects_misaligned_host_page_run(self):
        host = _make_host()
        _seed_device_scales(host)
        # Second host run starts mid-page, so its scale block has no page id.
        host_indices = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.int64)
        device_indices = torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)

        with mock.patch.object(
            MHATokenToKVPoolHost, "backup_from_device_all_layer"
        ) as backup_payload:
            with self.assertRaisesRegex(RuntimeError, "host.*not aligned"):
                host.backup_from_device_all_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    io_backend="direct",
                )
            backup_payload.assert_not_called()
            torch.testing.assert_close(
                host.scale_host,
                torch.ones_like(host.scale_host),
                atol=0,
                rtol=0,
            )

    def test_backup_rejects_misaligned_device_page_run(self):
        host = _make_host()
        _seed_device_scales(host)
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)
        # Second device run is shuffled, so the reshape into page blocks would
        # pair device rows with the wrong host pages.
        device_indices = torch.tensor([4, 5, 6, 7, 9, 8, 10, 11], dtype=torch.int64)

        with mock.patch.object(
            MHATokenToKVPoolHost, "backup_from_device_all_layer"
        ) as backup_payload:
            with self.assertRaisesRegex(RuntimeError, "device.*not aligned"):
                host.backup_from_device_all_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    io_backend="direct",
                )
            backup_payload.assert_not_called()

    def test_load_rejects_non_contiguous_host_page_run(self):
        host = _make_host()
        _seed_device_scales(host)
        host_indices = torch.tensor([0, 1, 2, 3, 8, 9, 11, 10], dtype=torch.int64)
        device_indices = torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)

        with mock.patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer"):
            with self.assertRaisesRegex(RuntimeError, "host.*not aligned"):
                host.load_to_device_per_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    0,
                    io_backend="direct",
                )

    def test_load_rejects_non_contiguous_device_page_run_before_payload_copy(self):
        host = _make_host()
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7, 9, 8, 10, 11], dtype=torch.int64)

        with mock.patch.object(
            MHATokenToKVPoolHost, "load_to_device_per_layer"
        ) as load_payload:
            with self.assertRaisesRegex(RuntimeError, "device.*not aligned"):
                host.load_to_device_per_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    0,
                    io_backend="direct",
                )
            load_payload.assert_not_called()

    def test_flat_page_rejects_misaligned_index(self):
        host = _make_host()
        page = host.get_data_page(PAGE_SIZE)
        with self.assertRaisesRegex(RuntimeError, "not page-aligned"):
            host.get_data_page(PAGE_SIZE + 1)
        with self.assertRaisesRegex(RuntimeError, "not page-aligned"):
            host.set_from_flat_data_page(PAGE_SIZE + 1, page)

    def test_page_meta_rejects_misaligned_host_page_run(self):
        host = _make_host()
        with self.assertRaisesRegex(RuntimeError, "host.*not aligned"):
            host.get_page_buffer_meta(
                torch.tensor([1, 2, 3, 4, 8, 9, 10, 11], dtype=torch.int64)
            )

    def test_payload_and_scales_round_trip_through_flat_l3_page(self):
        host = _make_host()
        host.kv_buffer[:, 1].copy_(
            torch.arange(host.kv_buffer[:, 1].numel(), dtype=torch.float32).reshape_as(
                host.kv_buffer[:, 1]
            )
        )
        host.scale_host[1].copy_(
            torch.arange(host.scale_host[1].numel(), dtype=torch.float32).reshape_as(
                host.scale_host[1]
            )
            / 100
        )
        expected_payload = host.kv_buffer[:, 1].clone()
        expected_scales = host.scale_host[1].clone()

        page = host.get_data_page(PAGE_SIZE)
        self.assertEqual(page.dtype, torch.uint8)
        self.assertEqual(page.numel(), host.get_size_per_token() * PAGE_SIZE)
        host.kv_buffer[:, 1].zero_()
        host.scale_host[1].zero_()
        host.set_from_flat_data_page(PAGE_SIZE, page)

        self.assertTrue(torch.equal(host.kv_buffer[:, 1], expected_payload))
        self.assertTrue(torch.equal(host.scale_host[1], expected_scales))

    def test_zero_copy_page_has_payload_and_scale_segments(self):
        host = _make_host()
        ptrs, sizes = host.get_page_buffer_meta(
            torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)
        )
        self.assertEqual(len(ptrs), 6)
        payload_bytes = (
            LAYER_NUM * PAGE_SIZE * HEAD_NUM * HEAD_DIM * host.dtype.itemsize
        )
        expected_sizes = [payload_bytes, payload_bytes, host._scale_page_bytes] * 2
        self.assertEqual(sizes, expected_sizes)

    def test_transfer_digests_include_matching_scale_components(self):
        host = _make_host()
        source = _seed_device_scales(host)
        device_indices = torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)

        with (
            envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.override(True),
            mock.patch.object(MHATokenToKVPoolHost, "backup_from_device_all_layer"),
        ):
            with self.assertLogs(
                "sglang.srt.mem_cache.pool_host.mha_dynamic_fp8", level="WARNING"
            ) as logs:
                host.backup_from_device_all_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    io_backend="direct",
                )
        for component in ("kv_scale_k", "kv_scale_v"):
            lines = _digest_lines(logs, component)
            self.assertEqual(len(lines), 2)  # one per backed-up page
            for line in lines:
                self.assertIn("exact=True", line)

        # After a restore the H2D digest must also see matching scale rows.
        with (
            envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.override(True),
            mock.patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer"),
        ):
            host.load_to_device_per_layer(
                host.device_pool,
                host_indices,
                device_indices,
                0,
                io_backend="direct",
            )
            with self.assertLogs(
                "sglang.srt.mem_cache.pool_host.mha_dynamic_fp8", level="WARNING"
            ) as logs:
                host.log_transfer_digests(
                    "host_to_device", host_indices, device_indices
                )
        for component in ("kv_scale_k", "kv_scale_v"):
            lines = _digest_lines(logs, component)
            self.assertEqual(len(lines), 2)
            for line in lines:
                self.assertIn("exact=True", line)
        del source

    def test_transfer_digest_flags_scale_component_mismatch(self):
        host = _make_host()
        _seed_device_scales(host)
        device_indices = torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64)

        with (
            envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.override(True),
            mock.patch.object(MHATokenToKVPoolHost, "backup_from_device_all_layer"),
        ):
            with self.assertLogs(
                "sglang.srt.mem_cache.pool_host.mha_dynamic_fp8", level="WARNING"
            ):
                host.backup_from_device_all_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    io_backend="direct",
                )

        # Corrupt the backed-up K scales of the second host page only.
        host.scale_host[1, 0] += 1.0
        with envs.SGLANG_HICACHE_FILE_BACKEND_LOG_PAGE_DIGESTS.override(True):
            with self.assertLogs(
                "sglang.srt.mem_cache.pool_host.mha_dynamic_fp8", level="WARNING"
            ) as logs:
                host.log_transfer_digests(
                    "host_to_device", host_indices, device_indices
                )

        k_lines = _digest_lines(logs, "kv_scale_k")
        self.assertEqual(len(k_lines), 2)
        self.assertIn("exact=True", k_lines[0])  # first page untouched
        self.assertIn("exact=False", k_lines[1])  # corrupted page
        for line in _digest_lines(logs, "kv_scale_v"):
            self.assertIn("exact=True", line)


class TestDynamicFP8ScaleBufferWait(CustomTestCase):
    def test_get_kv_scale_buffer_waits_on_layer_transfer_counter(self):
        pool = MHATokenToKVPoolDynamicFP8.__new__(MHATokenToKVPoolDynamicFP8)
        pool.start_layer = DEVICE_START_LAYER
        pool.k_scale_buffer = [
            torch.zeros(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)
        ]
        pool.v_scale_buffer = [
            torch.ones(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)
        ]
        counter = mock.Mock()
        pool.register_layer_transfer_counter(counter)

        k_scale, v_scale = pool.get_kv_scale_buffer(DEVICE_START_LAYER + 1)
        counter.wait_until.assert_called_once_with(1)
        self.assertIs(k_scale, pool.k_scale_buffer[1])
        self.assertIs(v_scale, pool.v_scale_buffer[1])

    def test_get_kv_scale_buffer_without_counter(self):
        pool = MHATokenToKVPoolDynamicFP8.__new__(MHATokenToKVPoolDynamicFP8)
        pool.start_layer = DEVICE_START_LAYER
        pool.k_scale_buffer = [
            torch.zeros(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)
        ]
        pool.v_scale_buffer = [
            torch.ones(TOKEN_NUM, HEAD_NUM) for _ in range(LAYER_NUM)
        ]
        pool.layer_transfer_counter = None

        k_scale, v_scale = pool.get_kv_scale_buffer(DEVICE_START_LAYER)
        self.assertIs(k_scale, pool.k_scale_buffer[0])
        self.assertIs(v_scale, pool.v_scale_buffer[0])


if __name__ == "__main__":
    unittest.main()
