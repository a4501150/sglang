"""
Unit tests for HiCacheFile LRU/eviction logic (max_size cap, free-space
watermark, MLA owner gating, pre-reservation under concurrency) and the
CP-aware file-key suffix.

The eviction logic lives in ``LRUFileEvictor`` (mem_cache/storage/file/); these
tests drive it end-to-end through ``HiCacheFile`` and inspect the wired-up
evictor via ``backend._evictor``.

These are pure CPU tests; they do not launch a server or need CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_lru_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import ctypes
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    DIRECT_IO_ALIGNMENT,
    HiCacheFile,
    HiCacheStorageConfig,
    MetadataCache,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    hicache_format_fingerprint,
)
from sglang.srt.mem_cache.storage.file.lru_file_evictor import _parse_size_to_bytes
from sglang.test.test_utils import CustomTestCase


def _t(n_bytes: int, fill: int = 0) -> torch.Tensor:
    """Build a uint8 CPU tensor of n_bytes filled with `fill`."""
    return torch.full((n_bytes,), fill, dtype=torch.uint8)


def _make_config(
    *,
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    attn_cp_rank=0,
    attn_cp_size=1,
    is_mla=False,
    model="testmodel",
    extra_config=None,
    format_spec=None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model,
        extra_config=extra_config,
        format_spec=format_spec,
    )


class _BackendBuilder:
    """Build a HiCacheFile with explicit config in a fresh temp dir."""

    def __init__(self, base_tmp: str):
        self.base_tmp = base_tmp

    def __call__(
        self,
        *,
        max_size=None,
        min_free=None,
        eviction_ratio=None,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla=False,
        model="testmodel",
        subdir=None,
        metadata_ttl=None,
        enable_metadata_cache=None,
        io_mode=None,
        format_spec=None,
    ) -> HiCacheFile:
        # Each backend gets its own subdir so MLA / non-MLA tests don't
        # contaminate each other's file_path.
        d = os.path.join(
            self.base_tmp, subdir or f"r{tp_rank}_t{tp_size}_{int(time.time_ns())}"
        )
        os.makedirs(d, exist_ok=True)
        cfg = _make_config(
            tp_rank=tp_rank,
            tp_size=tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            is_mla=is_mla,
            model=model,
            format_spec=format_spec,
            extra_config={
                "max_size": max_size,
                "eviction_ratio": eviction_ratio,
                "min_free_space": min_free,
                "metadata_ttl": metadata_ttl,
                "enable_metadata_cache": enable_metadata_cache,
                "io_mode": io_mode,
            },
        )
        return HiCacheFile(cfg, file_path=d)


class TestParseSize(CustomTestCase):
    def test_zero_and_none(self):
        self.assertEqual(_parse_size_to_bytes(None), 0)
        self.assertEqual(_parse_size_to_bytes("0"), 0)
        self.assertEqual(_parse_size_to_bytes(""), 0)
        self.assertEqual(_parse_size_to_bytes("none"), 0)

    def test_units(self):
        self.assertEqual(_parse_size_to_bytes("1024"), 1024)
        self.assertEqual(_parse_size_to_bytes("1k"), 1000)
        self.assertEqual(_parse_size_to_bytes("1Ki"), 1024)
        self.assertEqual(_parse_size_to_bytes("1Mi"), 1 << 20)
        self.assertEqual(_parse_size_to_bytes("2Gi"), 2 * (1 << 30))
        self.assertEqual(_parse_size_to_bytes("1.5G"), int(1.5 * 10**9))

    def test_invalid_returns_zero(self):
        self.assertEqual(_parse_size_to_bytes("abc"), 0)
        self.assertEqual(_parse_size_to_bytes("10XY"), 0)


class HiCacheFileLRUTestBase(CustomTestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hicache_lru_unit_")
        self.make_backend = _BackendBuilder(self.tmpdir)
        # Neutralise env vars so user shell can't leak settings into tests.
        self._env_overrides = [
            envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.override("0"),
            envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.override("0"),
        ]
        for cm in self._env_overrides:
            cm.__enter__()

    def tearDown(self):
        for cm in self._env_overrides:
            cm.__exit__(None, None, None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestEnvDefaults(CustomTestCase):
    """Verify the env var defaults match the documented opt-in behavior."""

    def test_min_free_space_default_is_zero(self):
        # Default must keep eviction off so existing users are unaffected.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE", None)
            self.assertEqual(
                envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE.get(),
                "0",
            )

    def test_max_size_default_is_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE", None)
            self.assertIsNone(envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE.get())


class TestEvictionDisabledByDefault(HiCacheFileLRUTestBase):
    def test_no_config_no_eviction(self):
        b = self.make_backend(max_size="0", min_free="0")
        self.assertFalse(b._evictor.enabled)
        # Set/get should still work as raw file storage.
        self.assertTrue(b.set("k1", _t(50)))
        self.assertTrue(b.exists("k1"))
        # No tracking happens.
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertEqual(b._evictor._total_bytes, 0)


class TestCapBasedEviction(HiCacheFileLRUTestBase):
    def test_basic_lru_evicts_oldest(self):
        b = self.make_backend(max_size="300", eviction_ratio=1.0)
        self.assertTrue(b.set("a", _t(100)))
        self.assertTrue(b.set("b", _t(100)))
        self.assertTrue(b.set("c", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 300)
        # Adding "d" forces eviction of "a" (oldest).
        self.assertTrue(b.set("d", _t(100)))
        self.assertLessEqual(b._evictor._total_bytes, 300)
        self.assertFalse(b.exists("a"))
        for k in ("b", "c", "d"):
            self.assertTrue(b.exists(k), f"{k} should still be present")

    def test_get_touches_recency(self):
        b = self.make_backend(max_size="300", eviction_ratio=1.0)
        b.set("a", _t(100))
        b.set("b", _t(100))
        b.set("c", _t(100))
        # Access "a" -> now "b" is the LRU.
        b.get("a", target_location=_t(100))
        # Inserting "d" should evict "b", not "a".
        b.set("d", _t(100))
        self.assertTrue(b.exists("a"), "a was just-accessed and must survive")
        self.assertFalse(b.exists("b"), "b should be the new LRU and got evicted")

    def test_value_larger_than_cap_rejected(self):
        b = self.make_backend(max_size="100")
        self.assertFalse(b.set("too_big", _t(200)))
        self.assertFalse(b.exists("too_big"))
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertEqual(len(b._evictor._lru), 0)

    def test_eviction_ratio_drops_to_watermark(self):
        # ratio=0.5 -> evict down to ~50% of the cap before adding.
        b = self.make_backend(max_size="400", eviction_ratio=0.5)
        for k in ("a", "b", "c", "d"):
            b.set(k, _t(100))
        self.assertEqual(b._evictor._total_bytes, 400)
        # target = 0.5*400 - 100 = 100, then +100 -> 200.
        b.set("e", _t(100))
        self.assertLessEqual(b._evictor._total_bytes, 200)

    def test_repeated_set_same_key_is_noop(self):
        b = self.make_backend(max_size="300")
        self.assertTrue(b.set("a", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 100)
        # Same key, different value -- fast path skips rewrite.
        self.assertTrue(b.set("a", _t(100)))
        self.assertEqual(b._evictor._total_bytes, 100)
        self.assertEqual(len(b._evictor._lru), 1)

    def test_clear_resets_state(self):
        b = self.make_backend(max_size="300")
        b.set("a", _t(100))
        b.set("b", _t(100))
        self.assertEqual(b._evictor._total_bytes, 200)
        self.assertTrue(b.clear())
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertFalse(b.exists("a"))


class TestScanExistingFiles(HiCacheFileLRUTestBase):
    def test_scan_seeds_lru_in_mtime_order(self):
        # Pre-create files, then check older mtimes land at the LRU front.
        d = tempfile.mkdtemp(prefix="hicache_seed_", dir=self.tmpdir)
        cfg = _make_config(
            model="seedmodel",
            extra_config={"max_size": "1000", "min_free_space": "0"},
        )
        # Files must end with the expected suffix for the rank/model.
        suffix = f"_seedmodel_0_1_fmt{hicache_format_fingerprint(None)}"
        # Create older "old.bin" first, then newer "new.bin".
        old_path = os.path.join(d, f"old{suffix}.bin")
        new_path = os.path.join(d, f"new{suffix}.bin")
        with open(old_path, "wb") as f:
            f.write(b"x" * 50)
        # Force older mtime on old_path.
        old_t = time.time() - 100
        os.utime(old_path, (old_t, old_t))
        with open(new_path, "wb") as f:
            f.write(b"y" * 70)
        b = HiCacheFile(cfg, file_path=d)
        self.assertEqual(b._evictor._total_bytes, 50 + 70)
        # First key in _lru should be the oldest (front = LRU).
        keys = list(b._evictor._lru.keys())
        self.assertEqual(keys[0], f"old{suffix}")
        self.assertEqual(keys[1], f"new{suffix}")


class TestRestartRescanMru(HiCacheFileLRUTestBase):
    def test_read_touch_survives_restart_rescan(self):
        # Simulate a restart: a page written long ago but read recently must
        # re-enter the rescan-seeded LRU as MRU (touch refreshes mtime), or
        # the first post-restart eviction would drop the hot page.
        cfg = _make_config(
            extra_config={
                "max_size": "250",
                "eviction_ratio": 1.0,
                "min_free_space": "0",
            }
        )
        d = os.path.join(self.tmpdir, "restart")
        os.makedirs(d)
        b = HiCacheFile(cfg, file_path=d)
        self.assertTrue(b.set("hot", _t(100)))
        self.assertTrue(b.set("cold", _t(100)))
        # Both pages were written long ago; since then only "hot" is read.
        stale = time.time() - 100
        for key in ("hot", "cold"):
            fp = b._get_component_path(key)
            os.utime(fp, (stale, stale))
        self.assertIsNotNone(b.get("hot", target_location=_t(100)))
        self.assertGreater(os.path.getmtime(b._get_component_path("hot")), stale)

        # A fresh backend rescans the directory and seeds the LRU by mtime:
        # the read page must sit at the MRU end, not in write-time order.
        b2 = HiCacheFile(cfg, file_path=d)
        keys = list(b2._evictor._lru.keys())
        self.assertEqual(keys[-1], b2._get_suffixed_key("hot"))
        # A new write evicts the front of the rescan order ("cold") and the
        # recently read prefix survives.
        self.assertTrue(b2.set("new", _t(100)))
        self.assertTrue(b2.exists("hot"))
        self.assertFalse(b2.exists("cold"))


class TestCPSuffix(HiCacheFileLRUTestBase):
    """Distinct CP ranks must not share a file key."""

    def test_cp_disabled_has_no_cp_suffix(self):
        b = self.make_backend(attn_cp_size=1, attn_cp_rank=0)
        self.assertNotIn("_cp", b.config_suffix)

    def test_distinct_cp_ranks_get_distinct_suffix(self):
        b0 = self.make_backend(attn_cp_rank=0, attn_cp_size=8, subdir="cp")
        b1 = self.make_backend(attn_cp_rank=1, attn_cp_size=8, subdir="cp")
        self.assertNotEqual(b0.config_suffix, b1.config_suffix)
        self.assertIn("_cp0_8_fmt", b0.config_suffix)
        self.assertIn("_cp1_8_fmt", b1.config_suffix)
        # Same logical key maps to different files per CP rank -> no write race.
        self.assertNotEqual(b0._get_suffixed_key("k"), b1._get_suffixed_key("k"))

    def test_cp_suffix_applies_to_mla(self):
        # MLA drops tp from the suffix; the CP tag keeps ranks isolated.
        b0 = self.make_backend(is_mla=True, attn_cp_rank=0, attn_cp_size=4, subdir="m")
        b1 = self.make_backend(is_mla=True, attn_cp_rank=3, attn_cp_size=4, subdir="m")
        self.assertIn("_cp0_4_fmt", b0.config_suffix)
        self.assertIn("_cp3_4_fmt", b1.config_suffix)
        self.assertNotEqual(b0.config_suffix, b1.config_suffix)


class TestFormatFingerprint(HiCacheFileLRUTestBase):
    def test_canonical_format_spec_is_stable(self):
        first = self.make_backend(
            subdir="format",
            format_spec={"layout": "page_first", "pools": ["kv", "mamba"]},
        )
        second = self.make_backend(
            subdir="format",
            format_spec={"pools": ["kv", "mamba"], "layout": "page_first"},
        )
        self.assertEqual(first.format_fingerprint, second.format_fingerprint)
        self.assertEqual(first.config_suffix, second.config_suffix)

    def test_checkpoint_and_kv_mode_are_isolated(self):
        orca_unit = hicache_format_fingerprint(
            {"runtime": {"model_path": "/models/orca", "kv_cache_dtype": "fp8_e4m3"}}
        )
        orca_dynamic = hicache_format_fingerprint(
            {
                "runtime": {
                    "model_path": "/models/orca",
                    "kv_cache_dtype": "fp8_e4m3_dynamic",
                }
            }
        )
        lambsea_dynamic = hicache_format_fingerprint(
            {
                "runtime": {
                    "model_path": "/models/lambsea",
                    "kv_cache_dtype": "fp8_e4m3_dynamic",
                }
            }
        )
        self.assertEqual(len({orca_unit, orca_dynamic, lambsea_dynamic}), 3)

    def test_changed_raw_format_cannot_see_old_pages(self):
        old = self.make_backend(
            subdir="format-miss", format_spec={"layout": "page_first"}
        )
        self.assertTrue(old.set("shared-key", _t(100, fill=7)))

        changed = self.make_backend(
            subdir="format-miss", format_spec={"layout": "layer_first"}
        )
        self.assertNotEqual(old.format_fingerprint, changed.format_fingerprint)
        self.assertFalse(changed.exists("shared-key"))


class TestMLAOwnerGating(HiCacheFileLRUTestBase):
    def test_mla_rank0_owns_eviction(self):
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=0, tp_size=2)
        self.assertTrue(b._evictor.is_storage_owner)
        self.assertTrue(b._evictor.enabled)

    def test_mla_rank1_skips_eviction(self):
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=1, tp_size=2)
        self.assertFalse(b._evictor.is_storage_owner)
        self.assertFalse(b._evictor.enabled)
        # Non-owner MLA ranks must not create new files when eviction is on.
        self.assertFalse(b.set("a", _t(50)))
        self.assertFalse(b.exists("a"))
        self.assertEqual(len(b._evictor._lru), 0)
        self.assertEqual(b._evictor._total_bytes, 0)

    def test_mla_rank1_can_touch_existing_file(self):
        # Non-owner ranks may still touch existing files, just not create new ones.
        b = self.make_backend(max_size="200", is_mla=True, tp_rank=1, tp_size=2)
        path = os.path.join(b.file_path, f"{b._get_suffixed_key('a')}.bin")
        with open(path, "wb") as f:
            f.write(b"x" * 50)
        self.assertTrue(b.set("a", _t(50)))
        self.assertTrue(b.exists("a"))
        self.assertEqual(len(b._evictor._lru), 0)

    def test_non_mla_each_rank_owns_its_files(self):
        # Non-MLA: even rank > 0 is its own owner because suffix isolates files.
        b = self.make_backend(max_size="200", is_mla=False, tp_rank=3, tp_size=4)
        self.assertTrue(b._evictor.is_storage_owner)
        self.assertTrue(b._evictor.enabled)


class TestTrackOrTouch(HiCacheFileLRUTestBase):
    def test_set_fast_path_adopts_external_file(self):
        # A file written by another rank should be adopted on the next set().
        b = self.make_backend(max_size="500")
        # Manually drop a suffixed file with the right name on disk.
        suffixed = b._get_suffixed_key("xkey")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"a" * 80)
        self.assertEqual(b._evictor._total_bytes, 0)
        self.assertNotIn(suffixed, b._evictor._lru)
        # set() should hit the fast path and adopt the file.
        self.assertTrue(b.set("xkey", _t(80)))
        self.assertIn(suffixed, b._evictor._lru)
        self.assertEqual(b._evictor._total_bytes, 80)

    def test_get_adopts_external_file(self):
        b = self.make_backend(max_size="500")
        suffixed = b._get_suffixed_key("ykey")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 64)
        # get() should return the data and also adopt the file.
        out = b.get("ykey", target_location=_t(64))
        self.assertIsNotNone(out)
        self.assertIn(suffixed, b._evictor._lru)
        self.assertEqual(b._evictor._total_bytes, 64)


class TestMinFreeSpaceWatermark(HiCacheFileLRUTestBase):
    def test_refuses_when_fs_would_drop_below_min_free(self):
        # Force statvfs to report a tiny free figure so the watermark trips.
        b = self.make_backend(max_size="0", min_free="100")
        # 150B free, writing 100B leaves 50B < 100B watermark -> refuse.
        b._evictor._fs_stats = lambda: (1024, 150)
        self.assertFalse(b.set("nope", _t(100)))
        self.assertFalse(b.exists("nope"))

    def test_evicts_to_satisfy_min_free(self):
        b = self.make_backend(max_size="0", min_free="100")
        # Pre-seed LRU with one 80B entry that is on disk.
        suffixed = b._get_suffixed_key("victim")
        path = os.path.join(b.file_path, f"{suffixed}.bin")
        with open(path, "wb") as f:
            f.write(b"v" * 80)
        b._evictor._lru[suffixed] = 80
        b._evictor._total_bytes = 80
        # 130 free; +60 write needs evicting the 80B victim to clear the watermark.
        free = [130]

        def fake_fs_stats():
            return (1024, free[0])

        original_remove = os.remove

        def tracked_remove(p):
            # Simulate tmpfs immediate free on unlink.
            if os.path.exists(p):
                free[0] += os.path.getsize(p)
            return original_remove(p)

        with (
            mock.patch.object(b._evictor, "_fs_stats", side_effect=fake_fs_stats),
            mock.patch("os.remove", side_effect=tracked_remove),
        ):
            self.assertTrue(b.set("newk", _t(60)))
        self.assertFalse(b.exists("victim"))
        self.assertTrue(b.exists("newk"))


class TestPreReservationConcurrency(HiCacheFileLRUTestBase):
    def test_pre_reservation_visible_during_write(self):
        """An in-flight reservation must not be evicted by a concurrent set()."""
        b = self.make_backend(max_size="100", eviction_ratio=1.0)
        pending = b._get_suffixed_key("A")
        with b._evictor._lock:
            b._evictor._lru[pending] = 60
            b._evictor._pending_writes.add(pending)
            b._evictor._total_bytes = 60

        self.assertFalse(b.set("B", _t(60)))
        self.assertIn(pending, b._evictor._lru)
        self.assertIn(pending, b._evictor._pending_writes)
        self.assertEqual(b._evictor._total_bytes, sum(b._evictor._lru.values()))
        self.assertLessEqual(b._evictor._total_bytes, 100)


class TestMetadataCache(CustomTestCase):
    def test_metadata_cache_basic(self):
        cache = MetadataCache(ttl_seconds=1.0)
        cache.add("k1")
        self.assertTrue(cache.contains("k1"))
        self.assertFalse(cache.contains("k2"))
        cache.remove("k1")
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_ttl(self):
        cache = MetadataCache(ttl_seconds=0.1)
        cache.add("k1")
        self.assertTrue(cache.contains("k1"))
        time.sleep(0.2)
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_hard_ttl(self):
        cache = MetadataCache(ttl_seconds=0.3)
        cache.add("k1")
        time.sleep(0.15)
        # Try updating k1
        cache.add("k1")
        # Expiry is still 0.3s from original timestamp, i.e. 0.15s from now.
        time.sleep(0.2)
        self.assertFalse(cache.contains("k1"))

    def test_metadata_cache_infinite_ttl(self):
        cache = MetadataCache(ttl_seconds=-1.0)
        cache.add("k1")
        time.sleep(0.3)
        self.assertTrue(cache.contains("k1"))


class TestHiCacheFileMetadataIntegration(HiCacheFileLRUTestBase):
    def test_disabled_by_default(self):
        b = self.make_backend()
        self.assertIsNone(b.metadata_cache)
        self.assertFalse(b.enable_metadata_cache)

    def test_startup_scanning_populates_cache(self):
        d = tempfile.mkdtemp(prefix="hicache_metadata_seed_", dir=self.tmpdir)
        cfg = _make_config(
            model="seedmodel",
            extra_config={"metadata_ttl": 5.0, "enable_metadata_cache": True},
        )
        suffix = f"_seedmodel_0_1_fmt{hicache_format_fingerprint(None)}"

        # Pre-create a suffixed bin file on disk
        with open(os.path.join(d, f"k1{suffix}.bin"), "wb") as f:
            f.write(b"data")

        b = HiCacheFile(cfg, file_path=d)
        # It should be found in metadata cache on startup
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

    def test_write_and_read_populates_cache(self):
        b = self.make_backend(metadata_ttl=5.0, enable_metadata_cache=True)
        suffix = b.config_suffix

        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        b.set("k1", _t(50))
        # After set, it must be in the metadata cache
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

        # Evict manually from metadata cache and call get
        b.metadata_cache.clear()
        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        b.get("k1", target_location=_t(50))
        # Get should populate it back
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))

    def test_eviction_removes_from_metadata_cache(self):
        # max_size=200, so setting three 100B tensors will evict the oldest
        b = self.make_backend(
            max_size="200",
            eviction_ratio=1.0,
            metadata_ttl=-1.0,
            enable_metadata_cache=True,
        )
        suffix = b.config_suffix

        b.set("k1", _t(100))
        b.set("k2", _t(100))
        self.assertTrue(b.metadata_cache.contains(f"k1{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k2{suffix}"))

        # Forces eviction of k1
        b.set("k3", _t(100))
        self.assertFalse(b.metadata_cache.contains(f"k1{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k2{suffix}"))
        self.assertTrue(b.metadata_cache.contains(f"k3{suffix}"))

    def test_batch_exists_bypass_scandir(self):
        b = self.make_backend(metadata_ttl=5.0, enable_metadata_cache=True)
        suffix = b.config_suffix

        b.set("k1", _t(50))
        b.set("k2", _t(50))

        # Now patch os.scandir and os.path.exists
        with (
            mock.patch("os.scandir") as mock_scandir,
            mock.patch("os.path.exists") as mock_exists,
        ):
            mock_exists.return_value = True

            # batch_exists_v2 for k1 and k2 should hit the metadata cache and NOT call os.scandir or os.path.exists
            res = b.batch_exists_v2(["k1", "k2"])
            self.assertEqual(res.kv_hit_pages, 2)
            mock_scandir.assert_not_called()
            mock_exists.assert_not_called()

            # Querying "k3" (miss) should fall back to os.path.exists once but still NOT call os.scandir
            res = b.batch_exists_v2(["k3"])
            self.assertEqual(
                res.kv_hit_pages, 1
            )  # since mock_exists returns True, k3 exists physically
            mock_scandir.assert_not_called()
            mock_exists.assert_called_once()


# ---------------------------------------------------------------------------
# io_mode configuration and direct (O_DIRECT) segment I/O
# ---------------------------------------------------------------------------


def _direct_io_supported(dir_path: str) -> bool:
    """True when this filesystem accepts a page-aligned O_DIRECT read."""
    if not hasattr(os, "O_DIRECT"):
        return False
    probe = os.path.join(dir_path, ".odirect_probe")
    try:
        with open(probe, "wb") as f:
            f.truncate(DIRECT_IO_ALIGNMENT)
        raw = (ctypes.c_char * (2 * DIRECT_IO_ALIGNMENT))()
        off = (-ctypes.addressof(raw)) % DIRECT_IO_ALIGNMENT
        buf = memoryview(raw).cast("B")[off : off + DIRECT_IO_ALIGNMENT]
        fd = os.open(probe, os.O_RDONLY | os.O_DIRECT)
        try:
            return os.preadv(fd, [buf], 0) == DIRECT_IO_ALIGNMENT
        finally:
            os.close(fd)
    except OSError:
        return False
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass


class _StubPool:
    """CPU stand-in for a page_first host pool.

    One ctypes buffer laid out as num_pages x seg_bytes with page_size=1, so
    slot j maps to exactly one segment. Also implements the copy-based v2 API
    on the same bytes for the buffered-path tests.
    """

    def __init__(
        self,
        num_pages: int,
        seg_bytes: int,
        layout: str = "page_first",
        force_unaligned: bool = False,
    ):
        self.page_size = 1
        self.layout = layout
        self.seg_bytes = seg_bytes
        raw = (ctypes.c_char * (num_pages * seg_bytes + DIRECT_IO_ALIGNMENT))()
        off = (-ctypes.addressof(raw)) % DIRECT_IO_ALIGNMENT
        if force_unaligned:
            off += 1
        self._raw = raw
        self._view = memoryview(raw).cast("B")
        self._off = off
        self.base = ctypes.addressof(raw) + off

    def get_page_buffer_meta(self, indices):
        idxs = [int(i) for i in indices]
        return (
            [self.base + j * self.seg_bytes for j in idxs],
            [self.seg_bytes] * len(idxs),
        )

    def is_stride_page_aligned(self, page_size_bytes: int = DIRECT_IO_ALIGNMENT):
        return (
            self.layout in ("page_first", "page_first_direct")
            and self.base % page_size_bytes == 0
            and (self.page_size * self.seg_bytes) % page_size_bytes == 0
        )

    def fill_slot(self, slot: int, fill: int):
        self._view[
            self._off + slot * self.seg_bytes : self._off + (slot + 1) * self.seg_bytes
        ] = bytes([fill]) * self.seg_bytes

    def write_slot(self, slot: int, data: bytes):
        start = self._off + slot * self.seg_bytes
        self._view[start : start + len(data)] = data

    def read_slot(self, slot: int) -> bytes:
        start = self._off + slot * self.seg_bytes
        return bytes(self._view[start : start + self.seg_bytes])

    # Copy-based (buffered v2) path API
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(self.read_slot(int(index))), dtype=torch.uint8
        )

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(self.seg_bytes, dtype=torch.uint8)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.write_slot(int(index), bytes(data_page.numpy().tobytes()))


class TestIoModeConfig(HiCacheFileLRUTestBase):
    def test_default_is_buffered(self):
        b = self.make_backend()
        self.assertEqual(b.io_mode, "buffered")
        self.assertFalse(b.supports_zero_copy_page_io)
        self.assertEqual(b._o_direct, 0)

    def test_env_default_is_buffered(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_IO_MODE", None)
            self.assertEqual(envs.SGLANG_HICACHE_FILE_BACKEND_IO_MODE.get(), "buffered")

    def test_extra_config_wins_over_env(self):
        with envs.SGLANG_HICACHE_FILE_BACKEND_IO_MODE.override("direct"):
            b = self.make_backend(io_mode="buffered")
            self.assertEqual(b.io_mode, "buffered")

    def test_invalid_extra_config_raises(self):
        with self.assertRaises(ValueError):
            self.make_backend(io_mode="streaming")
        with self.assertRaises(ValueError):
            self.make_backend(io_mode=42)

    def test_invalid_env_raises(self):
        with envs.SGLANG_HICACHE_FILE_BACKEND_IO_MODE.override("weird"):
            with self.assertRaises(ValueError):
                self.make_backend()


class DirectIoTestBase(HiCacheFileLRUTestBase):
    """Shared setup for the strict direct-mode tests.

    Only the actual O_DIRECT filesystem interactions are skipped when the
    platform/FS lacks O_DIRECT; production behavior stays strict.
    """

    def setUp(self):
        super().setUp()
        if not _direct_io_supported(self.tmpdir):
            self.skipTest("filesystem rejects an aligned O_DIRECT probe")

    def _direct_backend(self, **kw):
        return self.make_backend(io_mode="direct", **kw)

    def _file(self, b, stem):
        return os.path.join(b.file_path, f"{stem}{b.config_suffix}.bin")


class TestDirectStrictValidation(DirectIoTestBase):
    def test_direct_reports_zero_copy_capability(self):
        b = self._direct_backend()
        self.assertEqual(b.io_mode, "direct")
        self.assertTrue(b.supports_zero_copy_page_io)

    def test_anchor_wrong_layout_rejected(self):
        b = self._direct_backend()
        with self.assertRaises(ValueError):
            b.register_mem_pool_host(
                _StubPool(2, DIRECT_IO_ALIGNMENT, layout="layer_first")
            )

    def test_anchor_missing_page_buffer_meta_rejected(self):
        class NoMeta(_StubPool):
            get_page_buffer_meta = None

        b = self._direct_backend()
        with self.assertRaises(ValueError):
            b.register_mem_pool_host(NoMeta(2, DIRECT_IO_ALIGNMENT))

    def test_anchor_unaligned_stride_rejected(self):
        b = self._direct_backend()
        pool = _StubPool(2, DIRECT_IO_ALIGNMENT, force_unaligned=True)
        self.assertFalse(pool.is_stride_page_aligned(DIRECT_IO_ALIGNMENT))
        with self.assertRaises(ValueError):
            b.register_mem_pool_host(pool)

    def test_sidecar_pools_validated_including_mamba(self):
        b = self._direct_backend()
        b.register_mem_pool_host(_StubPool(1, DIRECT_IO_ALIGNMENT))
        with self.assertRaises(ValueError):
            b.register_mem_host_pool_v2(
                _StubPool(1, DIRECT_IO_ALIGNMENT, layout="layer_first"), PoolName.MAMBA
            )
        with self.assertRaises(ValueError):
            b.register_mem_host_pool_v2(
                _StubPool(1, DIRECT_IO_ALIGNMENT, force_unaligned=True), PoolName.MAMBA
            )
        # A conforming sidecar registers fine.
        b.register_mem_host_pool_v2(_StubPool(1, DIRECT_IO_ALIGNMENT), PoolName.MAMBA)
        self.assertIs(b.registered_pools[PoolName.MAMBA].layout, "page_first")


class TestDirectSegmentRoundtrip(DirectIoTestBase):
    def test_v1_roundtrip_returns_per_page_booleans(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        src = _StubPool(2, seg)
        src.fill_slot(0, 0xAB)
        src.fill_slot(1, 0xCD)
        b.register_mem_pool_host(src)

        self.assertEqual(
            b.batch_set_v1(["p0", "p1"], torch.tensor([0, 1])), [True, True]
        )
        for k in ("p0", "p1"):
            path = self._file(b, k)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.getsize(path), seg)

        dst = _StubPool(2, seg)
        b.register_mem_pool_host(dst)
        res = b.batch_get_v1(["p0", "p1", "absent"], torch.tensor([0, 1, 0]))
        self.assertEqual(res, [True, True, False])
        self.assertEqual(dst.read_slot(0), bytes([0xAB]) * seg)
        self.assertEqual(dst.read_slot(1), bytes([0xCD]) * seg)

    def test_v1_write_uses_o_direct_fd(self):
        seg = DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        pool = _StubPool(1, seg)
        b.register_mem_pool_host(pool)
        seen = []
        real_open = os.open

        def spy(path, flags, *args, **kwargs):
            seen.append(flags)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch("os.open", side_effect=spy):
            self.assertTrue(b.batch_set_v1(["od"], torch.tensor([0]))[0])
        self.assertTrue(any(f & os.O_DIRECT for f in seen))

    def test_v1_read_requires_exact_file_length(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        pool = _StubPool(1, seg)
        b.register_mem_pool_host(pool)
        # A truncated file must be rejected, not partially consumed.
        with open(self._file(b, "half"), "wb") as f:
            f.write(b"z" * (seg - DIRECT_IO_ALIGNMENT))
        dst = _StubPool(1, seg)
        b.register_mem_pool_host(dst)
        self.assertEqual(b.batch_get_v1(["half"], torch.tensor([0])), [False])
        self.assertEqual(dst.read_slot(0), b"\x00" * seg)

    def test_aligned_short_read_advances_through_iovecs(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        pool = _StubPool(1, seg)
        pool.fill_slot(0, 0x5A)
        b.register_mem_pool_host(pool)
        self.assertTrue(b.batch_set_v1(["sh"], torch.tensor([0]))[0])

        dst = _StubPool(1, seg)
        b.register_mem_pool_host(dst)
        real_preadv = os.preadv
        calls = []

        def halving(fd, views, off):
            calls.append(sum(len(v) for v in views))
            # Emulate an aligned short read: request at most one page.
            capped = [views[0][:DIRECT_IO_ALIGNMENT]]
            return real_preadv(fd, capped, off)

        with mock.patch("os.preadv", side_effect=halving):
            self.assertEqual(b.batch_get_v1(["sh"], torch.tensor([0])), [True])
        self.assertGreater(len(calls), 1)
        self.assertEqual(dst.read_slot(0), bytes([0x5A]) * seg)

    def test_nonaligned_short_read_rejected(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        pool = _StubPool(1, seg)
        b.register_mem_pool_host(pool)
        self.assertTrue(b.batch_set_v1(["ns"], torch.tensor([0]))[0])
        dst = _StubPool(1, seg)
        b.register_mem_pool_host(dst)
        with mock.patch("os.preadv", return_value=DIRECT_IO_ALIGNMENT - 1):
            self.assertEqual(b.batch_get_v1(["ns"], torch.tensor([0])), [False])

    def test_eof_before_completion_rejected(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        pool = _StubPool(1, seg)
        b.register_mem_pool_host(pool)
        self.assertTrue(b.batch_set_v1(["ef"], torch.tensor([0]))[0])
        dst = _StubPool(1, seg)
        b.register_mem_pool_host(dst)
        with mock.patch("os.preadv", return_value=None):
            self.assertEqual(b.batch_get_v1(["ef"], torch.tensor([0])), [False])

    def test_direct_writes_go_through_lru_evictor(self):
        seg = 2 * DIRECT_IO_ALIGNMENT
        b = self._direct_backend(max_size="100000")
        pool = _StubPool(1, seg)
        b.register_mem_pool_host(pool)
        self.assertTrue(b.batch_set_v1(["lru"], torch.tensor([0]))[0])
        self.assertIn(b._get_suffixed_key("lru"), b._evictor._lru)
        self.assertNotIn(b._get_suffixed_key("lru"), b._evictor._pending_writes)
        self.assertEqual(b._evictor._total_bytes, seg)

    def test_v2_kv_and_mamba_component_separation(self):
        kv_seg, mb_seg = 2 * DIRECT_IO_ALIGNMENT, DIRECT_IO_ALIGNMENT
        b = self._direct_backend()
        b.register_mem_pool_host(_StubPool(1, kv_seg))
        kv_src = _StubPool(1, kv_seg)
        mb_src = _StubPool(1, mb_seg)
        kv_src.fill_slot(0, 0x11)
        mb_src.fill_slot(0, 0x22)
        b.register_mem_host_pool_v2(kv_src, PoolName.KV)
        b.register_mem_host_pool_v2(mb_src, PoolName.MAMBA)
        transfers = [
            PoolTransfer(name=PoolName.KV, host_indices=torch.tensor([0]), keys=["c0"]),
            PoolTransfer(
                name=PoolName.MAMBA, host_indices=torch.tensor([0]), keys=["c0"]
            ),
        ]
        res = b.batch_set_v2(transfers)
        self.assertEqual(res, {PoolName.KV: [True], PoolName.MAMBA: [True]})

        kv_path = self._file(b, "c0")
        mb_path = self._file(b, "c0.mamba")
        self.assertTrue(os.path.exists(kv_path))
        self.assertTrue(os.path.exists(mb_path))
        with open(kv_path, "rb") as f:
            self.assertEqual(f.read(), bytes([0x11]) * kv_seg)
        with open(mb_path, "rb") as f:
            self.assertEqual(f.read(), bytes([0x22]) * mb_seg)

        kv_dst = _StubPool(1, kv_seg)
        mb_dst = _StubPool(1, mb_seg)
        b.register_mem_host_pool_v2(kv_dst, PoolName.KV)
        b.register_mem_host_pool_v2(mb_dst, PoolName.MAMBA)
        res = b.batch_get_v2(transfers)
        self.assertEqual(res, {PoolName.KV: [True], PoolName.MAMBA: [True]})
        self.assertEqual(kv_dst.read_slot(0), bytes([0x11]) * kv_seg)
        self.assertEqual(mb_dst.read_slot(0), bytes([0x22]) * mb_seg)


class TestBufferedCompatibility(HiCacheFileLRUTestBase):
    """Buffered mode must keep working without any O_DIRECT requirement."""

    @staticmethod
    def _touch_component(backend, key, pool_name=PoolName.KV):
        component_key = backend._get_component_key(key, pool_name)
        open(os.path.join(backend.file_path, f"{component_key}.bin"), "wb").close()

    def test_trailing_mamba_requirement_rejects_kv_only_chain(self):
        backend = self.make_backend()
        keys = ["k0", "k1", "k2"]
        for key in keys:
            self._touch_component(backend, key)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["checkpoint"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        with self.assertLogs(
            "sglang.srt.mem_cache.hicache_storage", level="WARNING"
        ) as logs:
            result = backend.batch_exists_v2(keys, [transfer])

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages, {PoolName.KV: 3})
        self.assertIn(
            "3 KV pages but no trailing mamba sidecar", "\n".join(logs.output)
        )

    def test_trailing_mamba_requirement_accepts_latest_sidecar(self):
        backend = self.make_backend()
        keys = ["k0", "k1", "k2"]
        for key in keys:
            self._touch_component(backend, key)
        self._touch_component(backend, keys[-1], PoolName.MAMBA)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            keys=["checkpoint"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        result = backend.batch_exists_v2(keys, [transfer])

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(
            result.extra_pool_hit_pages,
            {PoolName.KV: 3, PoolName.MAMBA: 3},
        )

    def test_v1_buffered_roundtrip_without_alignment(self):
        # 1000-byte unaligned segments: fine for buffered, rejected in direct.
        b = self.make_backend()
        pool = _StubPool(2, 1000, force_unaligned=True)
        pool.fill_slot(0, 0x77)
        b.register_mem_pool_host(pool)
        self.assertEqual(b.batch_set_v1(["b0"], torch.tensor([0])), [True])
        dst = _StubPool(2, 1000, force_unaligned=True)
        b.register_mem_pool_host(dst)
        self.assertEqual(b.batch_get_v1(["b0"], torch.tensor([0])), [True])
        self.assertEqual(dst.read_slot(0), bytes([0x77]) * 1000)

    def test_v1_buffered_rejects_nonexact_file_sizes(self):
        b = self.make_backend()
        src = _StubPool(1, 1000)
        src.fill_slot(0, 0x44)
        b.register_mem_pool_host(src)
        self.assertEqual(b.batch_set_v1(["bad"], torch.tensor([0])), [True])
        path = self._file_name(b, "bad")

        for payload in (bytes(999), bytes(1001)):
            with self.subTest(size=len(payload)):
                with open(path, "wb") as f:
                    f.write(payload)
                dst = _StubPool(1, 1000)
                b.register_mem_pool_host(dst)
                self.assertEqual(b.batch_get_v1(["bad"], torch.tensor([0])), [False])

    def test_v2_buffered_keeps_copy_page_path(self):
        b = self.make_backend()
        src = _StubPool(1, 1000)
        src.fill_slot(0, 0x33)
        b.register_mem_host_pool_v2(src, PoolName.KV)
        transfers = [
            PoolTransfer(name=PoolName.KV, host_indices=torch.tensor([0]), keys=["z0"])
        ]
        res = b.batch_set_v2(transfers)
        self.assertEqual(res, {PoolName.KV: [True]})
        with open(self._file_name(b, "z0"), "rb") as f:
            self.assertEqual(f.read(), bytes([0x33]) * 1000)

        dst = _StubPool(1, 1000)
        b.register_mem_host_pool_v2(dst, PoolName.KV)
        res = b.batch_get_v2(transfers)
        self.assertEqual(res, {PoolName.KV: [True]})
        self.assertEqual(dst.read_slot(0), bytes([0x33]) * 1000)

    @staticmethod
    def _file_name(b, stem):
        return os.path.join(b.file_path, f"{stem}{b.config_suffix}.bin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
