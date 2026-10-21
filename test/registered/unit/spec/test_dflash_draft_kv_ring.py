"""Unit tests for the DFLASH ring draft-KV layout: page-granular slot
mapping, window clamping against the per-request valid start, ring geometry
covering the widest window the worker builds, and the eager ring row
rebuild. CPU-only; the Triton rebuild is exercised on GPU by the serving
bench, this file pins the arithmetic it must agree with."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-b", runner_config="cpu")

from sglang.srt.speculative.dflash_draft_kv_layout import (
    PrivateRingLayout,
    SharedTargetLayout,
    resolve_draft_kv_layout,
)

WINDOW, BLOCK, PAGE = 2048, 4, 64


def make_ring():
    return PrivateRingLayout(window=WINDOW, block=BLOCK, page=PAGE)


class TestRingGeometry(unittest.TestCase):
    def test_ring_covers_widest_window(self):
        # The compact window keeps up to page-1 extra tokens on the left and
        # appends one verify block; the ring must hold that whole span with
        # the previous span's tail still live.
        ring = make_ring()
        widest = WINDOW + PAGE - 1 + BLOCK
        self.assertGreaterEqual(ring.ring_tokens, widest)
        self.assertGreaterEqual(ring.ring_tokens, WINDOW + PAGE + BLOCK)
        # A window of ring_pages pages never repeats a ring page.
        span_pages = (widest + PAGE - 1) // PAGE
        self.assertLessEqual(span_pages, ring.ring_pages)

    def test_pool_tokens_scales_with_requests_not_target(self):
        ring = make_ring()
        self.assertEqual(ring.pool_tokens(6), 6 * ring.ring_tokens)
        self.assertIsNone(SharedTargetLayout().pool_tokens(6))

    def test_page1_ring_degenerates_to_flat_modulus(self):
        ring = PrivateRingLayout(window=2048, block=4, page=1)
        self.assertEqual(ring.ring_tokens, 2048 + 4)
        slots = ring.slots(torch.tensor([2]), torch.arange(2059))
        self.assertTrue(
            torch.equal(slots, 2 * ring.ring_tokens + torch.arange(2059) % 2052)
        )

    def test_resolve_layout_defaults(self):
        self.assertIsInstance(
            resolve_draft_kv_layout(
                layout_name=None, window_size=None, block=4, page=64
            ),
            SharedTargetLayout,
        )
        self.assertIsInstance(
            resolve_draft_kv_layout(
                layout_name=None, window_size=2048, block=4, page=64
            ),
            PrivateRingLayout,
        )
        with self.assertRaises(ValueError):
            resolve_draft_kv_layout(
                layout_name="ring", window_size=None, block=4, page=64
            )


class TestRingSlotMapping(unittest.TestCase):
    def test_pages_stay_contiguous_and_aligned(self):
        ring = make_ring()
        req = torch.tensor([0, 3])
        pos = torch.arange(0, 3 * WINDOW)
        slots = ring.slots(req[:, None], pos[None, :])
        for row in slots.tolist():
            for start in range(0, len(row) - PAGE, PAGE):
                page = row[start : start + PAGE]
                self.assertEqual(page[0] % PAGE, 0)
                self.assertEqual(page, list(range(page[0], page[0] + PAGE)))

    def test_window_pages_are_distinct(self):
        ring = make_ring()
        slots = ring.slots(
            torch.tensor([0]), torch.arange(ring.ring_tokens)[None, :]
        )[0]
        pages = set((slots // PAGE).tolist())
        self.assertEqual(len(pages), ring.ring_pages)
        # Two positions collide only when they are exactly ring_tokens apart.
        again = ring.slots(torch.tensor([0]), torch.arange(WINDOW) + ring.ring_tokens)
        self.assertTrue(
            torch.equal(again, ring.slots(torch.tensor([0]), torch.arange(WINDOW)))
        )

    def test_request_base_is_off_the_request_slot(self):
        ring = make_ring()
        slots = ring.slots(torch.tensor([5]), torch.arange(8))
        self.assertTrue(
            torch.equal(slots, 5 * ring.ring_tokens + torch.arange(8))
        )


class TestWindowClamping(unittest.TestCase):
    def test_fresh_request_sees_the_full_window(self):
        ring = make_ring()
        clamped = ring.clamp_window_start(
            aligned_start=torch.tensor([0]), valid_start=torch.tensor([0])
        )
        self.assertEqual(clamped.tolist(), [0])

    def test_prefix_hit_clamps_to_materialized_region(self):
        # A 45K radix hit: the window wants [seq-window, seq) but the request
        # only materialized [45000, seq).
        ring = make_ring()
        seq = 45200
        aligned = seq - WINDOW
        clamped = ring.clamp_window_start(
            aligned_start=torch.tensor([aligned]), valid_start=torch.tensor([45000])
        )
        self.assertEqual(clamped.tolist(), [45056])  # 45000 rounds up a page

    def test_mid_page_valid_start_rounds_up(self):
        ring = make_ring()
        clamped = ring.clamp_window_start(
            aligned_start=torch.tensor([1024]), valid_start=torch.tensor([1090])
        )
        self.assertEqual(clamped.tolist(), [1152])

    def test_valid_start_after_extend(self):
        ring = make_ring()
        # Short extends keep everything from their start.
        starts = torch.tensor([0, 1000, 1090])
        ends = torch.tensor([2048, 1100, 4096])
        got = ring.valid_start_after_extend(extend_start=starts, extend_end=ends)
        self.assertEqual(got.tolist(), [0, 1000, 4096 - ring.ring_tokens])

    def test_longer_than_ring_extend_keeps_only_the_tail(self):
        ring = make_ring()
        got = ring.valid_start_after_extend(
            extend_start=torch.tensor([0]), extend_end=torch.tensor([10000])
        )
        self.assertEqual(got.tolist(), [10000 - ring.ring_tokens])


class _FakeWorker:
    """The eager ring row rebuild, driven the way DFlashWorkerV2 calls it."""

    def __init__(self, ring, num_reqs, width):
        self.ring = ring
        self.req_to_token = torch.zeros(
            (num_reqs, width), dtype=torch.int64
        )

    def rebuild(self, req_pool_indices, suffix_start, draft_prefix_lens, block):
        width = self.ring.ring_tokens
        cols = torch.arange(width)
        total = draft_prefix_lens.to(torch.int64) + block
        mask = cols.unsqueeze(0) < total.unsqueeze(1)
        positions = suffix_start.unsqueeze(1) + cols
        slots = self.ring.slots(req_pool_indices, positions)
        rows = req_pool_indices.to(torch.int64).unsqueeze(1).expand(-1, width)
        flat = self.req_to_token.reshape(-1)
        flat_dst = rows * self.req_to_token.shape[1] + cols.unsqueeze(0)
        flat[flat_dst[mask]] = slots[mask]


class TestRingRowRebuild(unittest.TestCase):
    def test_row_holds_ring_slots_of_positions(self):
        ring = make_ring()
        worker = _FakeWorker(ring, num_reqs=4, width=ring.ring_tokens)
        suffix_start = torch.tensor([0, 45056, 1920, 64])
        prefix_lens = torch.tensor([512, 128, 2048, 1])
        worker.rebuild(
            torch.tensor([0, 1, 2, 3]), suffix_start, prefix_lens, BLOCK
        )
        for req_idx, (row, start, length) in enumerate(
            zip(worker.req_to_token.tolist(), suffix_start.tolist(), prefix_lens.tolist())
        ):
            for col in range(length + BLOCK):
                pos = start + col
                want = req_idx * ring.ring_tokens + (
                    (pos // PAGE) % ring.ring_pages
                ) * PAGE + pos % PAGE
                self.assertEqual(row[col], want, f"pos={pos}")

    def test_slot_reuse_across_requests(self):
        # A req slot's ring row is rebuilt wholesale every step, so a new
        # owner of the request slot never sees the old owner's slots.
        ring = make_ring()
        worker = _FakeWorker(ring, num_reqs=1, width=ring.ring_tokens)
        worker.rebuild(
            torch.tensor([0]),
            torch.tensor([8192]),
            torch.tensor([64]),
            BLOCK,
        )
        first = worker.req_to_token[0, : 64 + BLOCK].tolist()
        worker.rebuild(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([64]),
            BLOCK,
        )
        second = worker.req_to_token[0, : 64 + BLOCK].tolist()
        self.assertNotEqual(first[:64], second[:64])


if __name__ == "__main__":
    unittest.main()
