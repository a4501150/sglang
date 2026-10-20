"""
Unit tests for the HiCacheFile page-file format: header/fingerprint/crc
integrity, namespacing, atomic-write leftovers (janitor), page-group atomic
eviction, and per-page failure isolation (the failure-ack contract).

The page-file IO lives in mem_cache/storage/file/ (page_io.py) and the
format helpers in mem_cache/hicache_format.py; these tests drive them end
to end through ``HiCacheFile`` with small synthetic host pools.

The O_DIRECT tests need a filesystem that accepts aligned direct IO, so
they run under $HOME/.cache/sglang-test (not tmpfs) and skip when the
probe is rejected.

Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_page_format.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.hicache_format import (
    HEADER_BYTES,
    decode_page_header,
    encode_page_header,
    format_fingerprint,
    pad_to_io_alignment,
    page_file_size,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.test.test_utils import CustomTestCase

TEST_SCRATCH = os.path.expanduser("~/.cache/sglang-test")


class FakePagePool(HostKVCache):
    """Minimal HostKVCache: flat byte pages, no segment metadata.

    The storage backend only needs the page API (page_size, data pages,
    format spec); pools that skip segment metadata exercise the flat-page
    (bounce) IO path, which is the whole-page contract mamba uses.
    """

    def __init__(self, page_bytes: int, page_size: int = 1, label: str = "kv"):
        self.page_size = page_size
        self.layout = "page_first"
        self.dtype = torch.uint8
        self.pool_label = label
        self.page_bytes = page_bytes
        self.size_per_token = page_bytes // page_size
        self.layer_num = 1
        self.storage_format_tag = None
        self.buffer = torch.zeros(64 * page_bytes, dtype=torch.uint8)

    def get_size_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        return self.buffer

    def load_to_device_per_layer(self, *args, **kwargs):
        raise NotImplementedError("FakePagePool has no device side")

    def backup_from_device_all_layer(self, *args, **kwargs):
        raise NotImplementedError("FakePagePool has no device side")

    def storage_format_spec(self):
        return {
            "pool_label": self.pool_label,
            "layout": self.layout,
            "page_size": self.page_size,
            "layer_num": self.layer_num,
            "dtype": "uint8",
            "size_per_token": self.size_per_token,
            "storage_format_tag": self.storage_format_tag,
        }

    def get_data_page(self, index, flat: bool = True):
        page = index // self.page_size
        data = self.buffer[page * self.page_bytes : (page + 1) * self.page_bytes]
        return data if flat else data.reshape(1, -1)

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            self.page_size * self.size_per_token, dtype=torch.uint8
        )

    def set_from_flat_data_page(self, index, data_page):
        page = index // self.page_size
        self.buffer[
            page * self.page_bytes : (page + 1) * self.page_bytes
        ] = data_page.view(torch.uint8).reshape(-1)


def _make_config(**overrides) -> HiCacheStorageConfig:
    defaults = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="fmtmodel",
        extra_config=None,
    )
    defaults.update(overrides)
    return HiCacheStorageConfig(**defaults)


def _make_backend(base_dir, pools=None, **config_overrides):
    """HiCacheFile over a fresh dir, with optional {pool_name: FakePagePool}."""
    d = tempfile.mkdtemp(prefix="hicache_fmt_", dir=base_dir)
    cfg = _make_config(extra_config=config_overrides or None)
    b = HiCacheFile(cfg, file_path=d)
    for name, pool in (pools or {}).items():
        b.register_mem_host_pool_v2(pool, name)
    return b


class TestPageHeaderCodec(CustomTestCase):
    """Header encode/decode round trip and the alignment invariant."""

    def test_round_trip(self):
        header = encode_page_header("0123456789abcdef", 798720, 0xBEEF)
        self.assertEqual(len(header), HEADER_BYTES)
        parsed = decode_page_header(header)
        self.assertEqual(parsed.fingerprint, "0123456789abcdef")
        self.assertEqual(parsed.payload_len, 798720)
        self.assertEqual(parsed.payload_padded_len, pad_to_io_alignment(798720))
        self.assertEqual(parsed.crc32, 0xBEEF)

    def test_rejects_foreign_header(self):
        with self.assertRaises(ValueError):
            decode_page_header(b"\x00" * HEADER_BYTES)
        bad = bytearray(encode_page_header("0" * 16, 8, 0))
        bad[0] = ord("X")
        with self.assertRaises(ValueError):
            decode_page_header(bytes(bad))

    def test_page_file_size_is_4k_aligned(self):
        for payload in (16, 4096, 798720, 59_019_280):
            size = page_file_size(payload)
            self.assertEqual(size % 4096, 0)
            self.assertGreaterEqual(size, HEADER_BYTES + payload)


class TestFingerprintNamespacing(CustomTestCase):
    """A geometry change is a miss, not a torn restore."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_fp_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_different_spec_different_fingerprint(self):
        a = format_fingerprint({"page_size": 64, "dtype": "bfloat16"})
        b = format_fingerprint({"page_size": 64, "dtype": "float32"})
        self.assertNotEqual(a, b)
        self.assertEqual(a, format_fingerprint({"dtype": "bfloat16", "page_size": 64}))

    def test_chain_seed_namespaces_extra_key_and_cache_salt(self):
        # Storage keys are chain hashes seeded with the request namespace;
        # distinct namespaces must never collide.
        seed_a = storage_namespace_seed("lora-1", None)
        seed_b = storage_namespace_seed("lora-2", None)
        seed_c = storage_namespace_seed("lora-1", "salt-1")
        self.assertIsNone(storage_namespace_seed(None, None))
        self.assertEqual(len({seed_a, seed_b, seed_c}), 3)

    def test_pool_geometry_change_misses_on_read(self):
        kv_a = FakePagePool(8192, label="kv")
        kv_a.storage_format_tag = "mha-dynamic-fp8"
        kv_b = FakePagePool(8192, label="kv")
        kv_b.storage_format_tag = "mha-dynamic-fp8"
        kv_b.size_per_token = 4096  # e.g. ssm dtype change: same pool, new bytes
        writer = _make_backend(self.base, {"kv": kv_a})
        reader = _make_backend(self.base, {"kv": kv_b})
        data = torch.full((8192,), 7, dtype=torch.uint8)
        kv_a.buffer[:8192] = data
        self.assertTrue(
            writer.batch_set_v2(
                [PoolTransfer(name="kv", keys=["p1"], host_indices=torch.tensor([0]))]
            )["kv"][0]
        )
        results = reader.batch_get_v2(
            [PoolTransfer(name="kv", keys=["p1"], host_indices=torch.tensor([0]))]
        )
        self.assertEqual(results, {"kv": [False]})
        # The unreadable file was dropped so later existence checks miss.
        self.assertFalse(
            os.path.exists(
                os.path.join(
                    reader.file_path,
                    f"{reader._get_component_key('p1', 'kv')}.bin",
                )
            )
        )


class TestAtomicWriteAndJanitor(CustomTestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_jan_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_no_tmp_left_after_write(self):
        b = _make_backend(self.base, verify="full")
        kv = FakePagePool(8192)
        b.register_mem_pool_host(kv)
        self.assertTrue(b.set("k1", torch.full((8192,), 3, dtype=torch.uint8)))
        self.assertEqual([f for f in os.listdir(b.file_path) if ".tmp" in f], [])

    def test_crash_leftovers_swept_and_foreign_files_kept(self):
        b = _make_backend(self.base, orphan_grace_seconds=0.0)
        kv = FakePagePool(8192)
        b.register_mem_pool_host(kv)
        # Simulated crash leftovers sharing this instance's identity.
        stem = b._get_component_key("dead", "kv")
        tmp = os.path.join(b.file_path, f"{stem}.bin.tmp.1.2.deadbeef")
        probe = os.path.join(b.file_path, ".odirect_probe.1.2")
        raw = os.path.join(b.file_path, f"{stem}.bin")
        for path in (tmp, probe):
            with open(path, "wb") as f:
                f.write(b"x" * 128)
        with open(raw, "wb") as f:
            f.write(b"not-a-page-file")  # pre-format file: undecodable header
        # A foreign identity (other model) must never be touched.
        foreign = os.path.join(b.file_path, "k_othermodel_0_1.bin")
        with open(foreign, "wb") as f:
            f.write(b"x" * 64)

        reclaimed = b.sweep_orphans()

        self.assertGreater(reclaimed, 0)
        self.assertFalse(os.path.exists(tmp))
        self.assertFalse(os.path.exists(probe))
        self.assertFalse(os.path.exists(raw))
        self.assertTrue(os.path.exists(foreign))

    def test_stale_fingerprint_page_swept(self):
        kv = FakePagePool(8192)
        b = _make_backend(self.base, orphan_grace_seconds=0.0)
        b.register_mem_pool_host(kv)
        # A page written under a different pool geometry, sharing this
        # instance's identity (same model/rank) and directory.
        kv_other = FakePagePool(4096)
        other = HiCacheFile(_make_config(), file_path=b.file_path)
        other.register_mem_pool_host(kv_other)
        other.set("stale", torch.zeros(4096, dtype=torch.uint8))
        # Age it past the grace window.
        stale_path = os.path.join(
            b.file_path, f"{b._get_suffixed_key('stale')}.bin"
        )
        os.utime(stale_path, (0, 0))

        b.sweep_orphans()

        self.assertFalse(os.path.exists(stale_path))

    def test_young_tmp_files_survive_grace_window(self):
        b = _make_backend(self.base, orphan_grace_seconds=600.0)
        tmp = os.path.join(b.file_path, ".odirect_probe.1.2")
        with open(tmp, "wb") as f:
            f.write(b"x" * 16)

        b.sweep_orphans()

        self.assertTrue(os.path.exists(tmp))


class TestPageGroupAtomicEviction(CustomTestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_grp_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_group_evicted_together(self):
        pools = {
            "kv": FakePagePool(8192),
            "indexer": FakePagePool(4096),
            "mamba": FakePagePool(16384),
        }
        group_bytes = sum(page_file_size(p.page_bytes) for p in pools.values())
        b = _make_backend(
            self.base,
            pools,
            max_size=str(2 * group_bytes),
            eviction_ratio=1.0,
        )
        b.register_mem_pool_host(pools["kv"])

        def write_group(key):
            transfers = [
                PoolTransfer(
                    name=name,
                    keys=[key],
                    host_indices=torch.tensor([i * 4]),
                )
                for i, name in enumerate(("kv", "indexer", "mamba"))
            ]
            results = b.batch_set_v2(transfers)
            assert all(rs[0] for rs in results.values()), results

        write_group("old")
        write_group("new")
        # The cap admits exactly two groups; the third forces the oldest
        # group out whole.
        write_group("third")

        for name in pools:
            self.assertFalse(
                os.path.exists(
                    os.path.join(b.file_path, f"{b._get_component_key('old', name)}.bin")
                ),
                f"old {name} page should be evicted with its group",
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(b.file_path, f"{b._get_component_key('new', name)}.bin")
                ),
                f"new {name} page should survive",
            )


class TestFailureAcksAndMidBatchClamp(CustomTestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_ack_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _backend_with_pool(self, page_bytes):
        kv = FakePagePool(page_bytes)
        b = _make_backend(self.base, {"kv": kv})
        b.register_mem_pool_host(kv)
        return b, kv

    def test_missing_middle_page_returns_per_page_results(self):
        b, kv = self._backend_with_pool(8192)
        b.set("p0", torch.full((8192,), 1, dtype=torch.uint8))
        b.set("p2", torch.full((8192,), 3, dtype=torch.uint8))  # p1 absent

        results = b.batch_get_v2(
            [
                PoolTransfer(
                    name="kv",
                    keys=["p0", "p1", "p2"],
                    host_indices=torch.tensor([0, 1, 2]),
                )
            ]
        )

        # One result per page regardless of the miss: the caller clamps at
        # the first failure and the ack count stays balanced.
        self.assertEqual(results, {"kv": [True, False, True]})

    def test_shorter_stored_page_is_a_miss_not_a_torn_restore(self):
        # A page stored by a build whose pool had different page bytes:
        # the header length check must reject it before any payload copy.
        writer, _ = self._backend_with_pool(4096)
        reader, kv = self._backend_with_pool(8192)
        writer.set("p0", torch.full((4096,), 9, dtype=torch.uint8))
        # Same on-disk key: writer/reader share the identity suffix, but the
        # readers' pools differ in page bytes. Point both at one directory.
        shutil.rmtree(reader.file_path)
        os.rename(writer.file_path, reader.file_path)

        results = reader.batch_get_v2(
            [
                PoolTransfer(
                    name="kv", keys=["p0"], host_indices=torch.tensor([0])
                )
            ]
        )

        self.assertEqual(results, {"kv": [False]})
        self.assertTrue(
            torch.equal(
                kv.buffer[:8192],
                torch.zeros(8192, dtype=torch.uint8),
            ),
            "no partial page state may be published on a rejected read",
        )


class TestEngines(CustomTestCase):
    """Engine parity: buffered and O_DIRECT must round-trip identically."""

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="hicache_eng_", dir=TEST_SCRATCH)
        os.makedirs(self.base, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _round_trip(self, engine, payload_bytes):
        kv = FakePagePool(payload_bytes)
        b = _make_backend(self.base, {"kv": kv}, io_engine=engine)
        b.register_mem_pool_host(kv)
        src = torch.arange(payload_bytes, dtype=torch.int32).to(torch.uint8)
        kv.buffer[:payload_bytes] = src
        ok = b.batch_set_v2(
            [PoolTransfer(name="kv", keys=["e"], host_indices=torch.tensor([0]))]
        )["kv"][0]
        kv.buffer.zero_()
        read = b.batch_get_v2(
            [PoolTransfer(name="kv", keys=["e"], host_indices=torch.tensor([0]))]
        )["kv"][0]
        return ok, read, torch.equal(kv.buffer[:payload_bytes], src)

    def test_buffered_round_trip_various_payloads(self):
        for n in (4096, 798720, 4097):  # aligned, big aligned, unaligned
            ok, read, equal = self._round_trip("posix_buffered", n)
            self.assertTrue(ok and read and equal, f"payload {n}")

    def test_o_direct_round_trip_various_payloads(self):
        try:
            ok, read, equal = self._round_trip("posix_o_direct", 4096)
        except RuntimeError as e:
            self.skipTest(f"O_DIRECT probe rejected on this filesystem: {e}")
        self.assertTrue(ok and read and equal)
        for n in (798720, 4097):  # big aligned (in-place) and unaligned (bounce)
            ok, read, equal = self._round_trip("posix_o_direct", n)
            self.assertTrue(ok and read and equal, f"payload {n}")

    def test_engine_selected_from_extra_config(self):
        b = _make_backend(self.base, io_engine="posix_o_direct")
        self.assertEqual(b.io_engine.name, "posix_o_direct")
        b2 = _make_backend(self.base)
        self.assertEqual(b2.io_engine.name, "posix_buffered")

    def test_unknown_engine_rejected(self):
        with self.assertRaises(ValueError):
            _make_backend(self.base, io_engine="nio")

    def test_cufile_engine_constructs_with_binding(self):
        from sglang.srt.utils import cufile as binding

        try:
            b = _make_backend(self.base, io_engine="cufile")
        except RuntimeError as e:
            if not binding.cufile_available():
                self.skipTest(f"no libcufile on this host: {e}")
            raise
        self.assertEqual(b.io_engine.name, "cufile")

    def test_corrupt_payload_crc_mismatch_is_miss(self):
        kv = FakePagePool(8192)
        b = _make_backend(self.base, {"kv": kv}, verify="full")
        b.register_mem_pool_host(kv)
        self.assertTrue(
            b.set("c1", torch.full((8192,), 5, dtype=torch.uint8))
        )
        # Flip one payload byte behind the backend's back.
        path = os.path.join(b.file_path, f"{b._get_suffixed_key('c1')}.bin")
        with open(path, "r+b") as f:
            f.seek(HEADER_BYTES + 32)
            byte = f.read(1)
            f.seek(HEADER_BYTES + 32)
            f.write(bytes([byte[0] ^ 0xFF]))

        out = b.get("c1", target_location=torch.zeros(8192, dtype=torch.uint8))

        self.assertIsNone(out)
        self.assertFalse(os.path.exists(path), "corrupt page must be dropped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
