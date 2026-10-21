"""How the DFLASH draft's KV pool indexes its slots.

``SharedTargetLayout`` (default): the draft pool spans the target's token
index space, one draft slot per target token. A radix prefix hit reuses the
draft KV the tree preserved, but the draft pool rivals the target's: a
5-layer draft over an 850K-token target pool costs its own 850K slots.

``PrivateRingLayout`` (--speculative-draft-kv-layout ring, requires
--speculative-draft-window-size): each request slot owns a fixed ring of
pages. A token at global position ``p`` of request ``r`` lives at

    r * ring_tokens + ((p // page) % ring_pages) * page + p % page

The mapping is page-granular, so every page of the draft's visible window
stays contiguous and page-aligned wherever the window wraps; pages of the
window are always distinct because the window spans at most ``ring_pages``
pages. The pool costs ``num_req_slots * ring_tokens`` slots instead of one
per target token. Positions the request never materialized (a radix prefix
hit, or ring-recycled history) are excluded from the window via a
per-request ``draft_valid_start`` clamp, so the draft never reads a slot it
did not write for the current request; acceptance dips after a prefix hit
until the window slides past the materialized region.
"""

from __future__ import annotations

from typing import Optional

import torch


class DraftKVLayout:
    """The draft pool's slot index space and per-request lifecycle rules.

    ``pool_tokens`` is the number of slots the draft pool needs (None: size
    it to the target's token count). The layout is the extension point for
    future draft-KV placements: a layout owns its slot mapping and its
    window-validity rules; the worker and the pool sizing consult it and
    nothing else.
    """

    name: str = "abstract"

    def pool_tokens(self, num_req_slots: int) -> Optional[int]:
        raise NotImplementedError


class SharedTargetLayout(DraftKVLayout):
    name = "shared"

    def pool_tokens(self, num_req_slots: int) -> Optional[int]:
        return None


class PrivateRingLayout(DraftKVLayout):
    name = "ring"

    def __init__(self, *, window: int, block: int, page: int):
        self.window = int(window)
        self.block = int(block)
        self.page = int(page)

    @property
    def ring_pages(self) -> int:
        # The visible window spans at most window + page - 1 tokens (the
        # page-aligned start keeps up to page-1 extra), plus one verify
        # block; the ring must also survive having that span written while
        # the previous span's last page is still inside it, hence the
        # +block recycle margin. Page granularity rounds it up.
        if self.page > 1:
            span = self.window + self.page + self.block
            return (span + self.page - 1) // self.page
        return self.window + self.block

    @property
    def ring_tokens(self) -> int:
        return self.ring_pages * self.page

    def pool_tokens(self, num_req_slots: int) -> Optional[int]:
        return int(num_req_slots) * self.ring_tokens

    def slots(self, req_pool_indices: torch.Tensor, positions: torch.Tensor):
        """Ring slots of ``positions`` (any shape broadcastable against the
        per-request ``req_pool_indices``). Pure elementwise int64 ops, so it
        is safe inside a captured region and adds no host syncs."""
        req = req_pool_indices.to(torch.int64)
        while req.dim() < positions.dim():
            req = req.unsqueeze(-1)
        pos = positions.to(torch.int64)
        page = self.page
        ring_pages = self.ring_pages
        if page > 1:
            slot = ((pos // page) % ring_pages) * page + (pos % page)
        else:
            slot = pos % ring_pages
        return req * self.ring_tokens + slot

    def clamp_window_start(
        self, *, aligned_start: torch.Tensor, valid_start: torch.Tensor
    ) -> torch.Tensor:
        """The visible window's start: the page-aligned window start, but
        never below the first page the request materialized. ``valid_start``
        may sit mid-page; rounding it up drops at most one partial page."""
        if self.page > 1:
            valid_page = (valid_start + self.page - 1) // self.page * self.page
        else:
            valid_page = valid_start
        return torch.maximum(aligned_start, valid_page)

    def valid_start_after_extend(
        self, *, extend_start: torch.Tensor, extend_end: torch.Tensor
    ) -> torch.Tensor:
        """First readable position after one extend materialization.

        The extend writes [extend_start, extend_end); when it is longer than
        the ring only its last ring_tokens survive, and the ring's recycling
        invalidates everything below extend_end - ring_tokens. Both effects
        (and only these) make positions below this value unreadable.
        """
        return torch.maximum(
            extend_start.to(torch.int64),
            extend_end.to(torch.int64) - self.ring_tokens,
        )


def resolve_draft_kv_layout(
    *,
    layout_name: Optional[str],
    window_size: Optional[int],
    block: int,
    page: int,
) -> DraftKVLayout:
    if layout_name == "ring" or (layout_name is None and window_size is not None):
        if window_size is None:
            raise ValueError(
                "--speculative-draft-kv-layout ring requires "
                "--speculative-draft-window-size."
            )
        return PrivateRingLayout(
            window=int(window_size), block=int(block), page=int(page)
        )
    return SharedTargetLayout()


def active_dflash_ring_layout(page_size: int) -> Optional[PrivateRingLayout]:
    """The ring layout when DFLASH resolved to it, else None.

    Reads the published spec context, so callers must run after arg
    resolution (pool sizing and the worker both do). DSPARK never returns a
    ring: its worker is a different draft pipeline this layout does not
    cover.
    """
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None or not spec.speculative_algorithm.is_dflash():
        return None
    if spec.speculative_draft_kv_layout != "ring":
        return None
    window = spec.speculative_draft_window_size
    if window is None:
        raise ValueError(
            "--speculative-draft-kv-layout ring requires "
            "--speculative-draft-window-size."
        )
    block = spec.speculative_num_draft_tokens
    if block is None:
        raise ValueError(
            "DFLASH resolved no --speculative-num-draft-tokens; the ring "
            "layout needs the block size."
        )
    return PrivateRingLayout(window=int(window), block=int(block), page=int(page_size))
