"""CPU-only unit tests for the fail-closed Mamba checkpoint-depth validation.

The donated checkpoint depth must be a host-known int, bounded by the insert
span, and radix-page-aligned before any donate alloc/copy/commit runs. These tests
drive ``MambaComponent`` bare (``object.__new__``) so a rejected depth proves
itself by never touching the request's pools.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    MambaComponent,
)
from sglang.test.test_utils import CustomTestCase


def _component(*, extra_buffer: bool, page_size: int, pool):
    comp = object.__new__(MambaComponent)
    comp.cache = SimpleNamespace(
        enable_mamba_extra_buffer=extra_buffer,
        req_to_token_pool=pool,
    )
    comp._checkpoint_tree_page = page_size
    return comp


def _req(**kv_fields):
    return SimpleNamespace(rid="test-rid", kv=SimpleNamespace(**kv_fields))


class TestRequireDonatableCheckpointDepth(CustomTestCase):
    def _comp(self, page_size=64):
        return _component(extra_buffer=True, page_size=page_size, pool=object())

    def test_valid_depth_passes_through(self):
        comp = self._comp(page_size=64)
        self.assertEqual(
            comp._require_donatable_checkpoint_depth(128, _req(), 200), 128
        )
        self.assertEqual(comp._require_donatable_checkpoint_depth(0, _req(), 200), 0)

    def test_device_tensor_depth_rejected(self):
        comp = self._comp()
        with self.assertRaisesRegex(AssertionError, "device tensor"):
            comp._require_donatable_checkpoint_depth(torch.tensor(64), _req(), 200)

    def test_non_int_depth_rejected(self):
        comp = self._comp()
        for depth in (64.0, "64", None):
            with (
                self.subTest(depth=depth),
                self.assertRaisesRegex(AssertionError, "not a host int"),
            ):
                comp._require_donatable_checkpoint_depth(depth, _req(), 200)

    def test_out_of_bounds_depth_rejected(self):
        comp = self._comp()
        for depth in (-1, 201):
            with (
                self.subTest(depth=depth),
                self.assertRaisesRegex(AssertionError, r"outside \[0, 200\]"),
            ):
                comp._require_donatable_checkpoint_depth(depth, _req(), 200)

    def test_page_misaligned_depth_rejected(self):
        comp = self._comp(page_size=64)
        with self.assertRaisesRegex(AssertionError, "not aligned to"):
            comp._require_donatable_checkpoint_depth(96, _req(), 200)


class TestMambaTopologyOnlyInsert(CustomTestCase):
    def test_existing_state_is_preserved(self):
        comp = object.__new__(MambaComponent)
        existing = torch.tensor([7])
        node = SimpleNamespace(
            component_data={ComponentType.MAMBA: SimpleNamespace(value=existing)}
        )
        params = SimpleNamespace(mamba_value=None)
        result = SimpleNamespace(mamba_exist=False)

        comp.commit_insert_component_data(
            node=node,
            is_new_leaf=False,
            params=params,
            result=result,
            cache_actions=[],
        )

        self.assertIs(node.component_data[ComponentType.MAMBA].value, existing)
        self.assertTrue(result.mamba_exist)

    def test_new_leaf_without_state_is_rejected(self):
        comp = object.__new__(MambaComponent)
        node = SimpleNamespace(
            component_data={ComponentType.MAMBA: SimpleNamespace(value=None)}
        )

        with self.assertRaises(AssertionError):
            comp.commit_insert_component_data(
                node=node,
                is_new_leaf=True,
                params=SimpleNamespace(mamba_value=None),
                result=SimpleNamespace(mamba_exist=False),
                cache_actions=[],
            )


class TestPrepareForCachingReqValidation(CustomTestCase):
    def test_extra_buffer_rejects_before_touching_pools(self):
        # pool=object(): any pool access would raise AttributeError, proving
        # validation precedes donation.
        comp = _component(extra_buffer=True, page_size=64, pool=object())
        req = _req(mamba_last_track_seqlen=96)
        insert_params = SimpleNamespace(mamba_value=None)
        result = comp.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=200,
            is_finished=True,
        )
        self.assertEqual(result, 0)
        self.assertIsNone(insert_params.mamba_value)

    def test_extra_buffer_unfinished_rejects_before_donation(self):
        comp = _component(extra_buffer=True, page_size=64, pool=object())
        req = _req(mamba_last_track_seqlen=128)  # over token_ids_len
        insert_params = SimpleNamespace(mamba_value=None)
        result = comp.prepare_for_caching_req(
            req=req,
            insert_params=insert_params,
            token_ids_len=100,
            is_finished=False,
        )
        self.assertEqual(result, 0)
        self.assertIsNone(insert_params.mamba_value)

    def test_extra_buffer_unfinished_without_tracked_depth_skips(self):
        # A depth the host never learned (None) is "nothing to cache", not a
        # violation: return 0 without validating or donating.
        comp = _component(extra_buffer=True, page_size=64, pool=object())
        req = _req(mamba_last_track_seqlen=None)
        insert_params = SimpleNamespace(mamba_value=None)
        result = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=200, is_finished=False
        )
        self.assertEqual(result, 0)
        self.assertIsNone(insert_params.mamba_value)

    def test_extra_buffer_finished_without_new_checkpoint_reuses_protected_prefix(self):
        comp = _component(extra_buffer=True, page_size=64, pool=object())
        req = SimpleNamespace(
            rid="test-rid",
            cached_tokens=128,
            kv=SimpleNamespace(mamba_last_track_seqlen=None, cache_protected_len=128),
        )
        insert_params = SimpleNamespace(mamba_value=None)

        cache_len = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=150, is_finished=True
        )

        self.assertEqual(cache_len, 128)
        self.assertIsNone(insert_params.mamba_value)

    def test_topology_reinsert_is_capped_at_predating_cache_hit(self):
        comp = _component(extra_buffer=True, page_size=64, pool=object())
        req = SimpleNamespace(
            rid="test-rid",
            cached_tokens=64,
            kv=SimpleNamespace(mamba_last_track_seqlen=None, cache_protected_len=128),
        )
        insert_params = SimpleNamespace(mamba_value=None)

        cache_len = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=150, is_finished=True
        )

        self.assertEqual(cache_len, 64)
        self.assertIsNone(insert_params.mamba_value)

    def test_extra_buffer_finished_donates_valid_depth(self):
        pool = SimpleNamespace(
            mamba_ckpt_pool=None,
            get_mamba_ping_pong_keep_idx=lambda req: 1,
        )
        comp = _component(extra_buffer=True, page_size=64, pool=pool)
        req = _req(
            mamba_last_track_seqlen=128,
            mamba_ping_pong_track_buffer=torch.tensor([7, 9]),
        )
        insert_params = SimpleNamespace(mamba_value=None)
        cache_len = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=200, is_finished=True
        )
        self.assertEqual(cache_len, 128)
        self.assertEqual(insert_params.mamba_value.tolist(), [9])

    def test_no_buffer_trimmed_span_skips_stale_live_state(self):
        pool = SimpleNamespace(mamba_ckpt_pool=None)
        comp = _component(extra_buffer=False, page_size=1, pool=pool)
        req = _req(kv_committed_len=20)
        insert_params = SimpleNamespace(mamba_value=None)

        result = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=12, is_finished=True
        )

        self.assertEqual(result, 0)
        self.assertIsNone(insert_params.mamba_value)

    def test_no_buffer_finished_donates_at_token_len(self):
        pool = SimpleNamespace(
            mamba_ckpt_pool=None,
            mamba_pool=SimpleNamespace(replayssm_write_pos=None),
        )
        comp = _component(extra_buffer=False, page_size=1, pool=pool)
        req = _req(mamba_pool_idx=torch.tensor(4), kv_committed_len=17)
        insert_params = SimpleNamespace(mamba_value=None)
        cache_len = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=17, is_finished=True
        )
        self.assertEqual(cache_len, 17)
        self.assertEqual(insert_params.mamba_value.tolist(), [4])

    def test_no_buffer_replayssm_negative_depth_rejected(self):
        # Ring depth 5 past a 3-token request: the flush-boundary cap drives
        # the depth negative and the donate must be refused.
        pool = SimpleNamespace(
            mamba_ckpt_pool=None,
            mamba_pool=SimpleNamespace(replayssm_write_pos=torch.tensor([5])),
        )
        comp = _component(extra_buffer=False, page_size=1, pool=pool)
        req = _req(mamba_pool_idx=torch.tensor(0), kv_committed_len=3)
        insert_params = SimpleNamespace(mamba_value=None)
        result = comp.prepare_for_caching_req(
            req=req, insert_params=insert_params, token_ids_len=3, is_finished=True
        )
        self.assertEqual(result, 0)
        self.assertEqual(pool.mamba_pool.replayssm_write_pos.tolist(), [5])
        self.assertIsNone(insert_params.mamba_value)


if __name__ == "__main__":
    unittest.main()
