"""GPU staging arena for device-target L3 reads.

A ``DeviceStagingPool`` mirrors one host pool's whole flat pages in a GPU
arena: one 4 KiB-aligned row per page, byte-identical to the host pool's
``get_data_page`` layout (payload plus the FP32 scale block for
dynamic-FP8 KV; the temporal/conv/slot-state envelope for mamba; packed
item rows for paged sidecars like the QSA indexer). L3 page files land
here directly (positional read into a pinned bounce, async copy onto
the arena; the cuFile engine will read straight into the rows), and the
existing pf->lf scatter kernels then move the staged pages into
layer-first device pool rows — the same kernels the host pools use,
because ``transfer_kv_per_layer_direct_pf_lf`` is a plain ``copy_``
under a pointer walk and accepts device-source pointers.

Arena rows are padded to ``IO_ALIGNMENT`` so every row pointer and size
stays 4 KiB-aligned for the cuFile engine.

The class implements the host-pool transfer surface the L2 engine drives
(``prepare_transfer_indices`` / ``load_to_device_per_layer``) plus a
host-pool-like page allocator (token-indexed, page-granular free list),
so the buffer pipeline can treat an arena slot span exactly like a host
bounce span.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.hicache_format import IO_ALIGNMENT, pad_to_io_alignment
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha_dynamic_fp8 import (
    MHATokenToKVPoolDynamicFP8Host,
)

logger = logging.getLogger(__name__)


def _elem_strides(shape: Sequence[int]) -> list[int]:
    """Row-major element strides for *shape* (no leading page axis)."""
    acc = []
    run = 1
    for d in reversed(shape):
        acc.append(run)
        run *= d
    return acc[::-1]


class DeviceStagingPool:
    """GPU arena of whole flat cache pages for one host pool's geometry."""

    @staticmethod
    def _check_geometry(host_pool) -> None:
        """Raise ``NotImplementedError`` when *host_pool* has no arena
        geometry; lets callers probe support without allocating."""
        if not isinstance(
            host_pool,
            (MHATokenToKVPoolDynamicFP8Host, MambaPoolHost, DeepSeekV4PagedHostPool),
        ):
            raise NotImplementedError(
                f"no device staging geometry for {type(host_pool).__name__}"
            )

    def __init__(
        self,
        host_pool,
        *,
        device,
        page_slots: int,
        name: str,
    ):
        self.host_pool = host_pool
        self.name = name
        self.page_size = host_pool.page_size or 1
        self.page_bytes = host_pool.get_size_per_token() * self.page_size
        self.row_bytes = pad_to_io_alignment(self.page_bytes)
        self.device = torch.device(device)
        self.page_slots = page_slots
        # Host-pool-like token capacity: one page per arena row.
        self.size = page_slots * self.page_size

        raw = torch.empty(
            page_slots * self.row_bytes + IO_ALIGNMENT,
            dtype=torch.uint8,
            device=self.device,
        )
        offset = (-raw.data_ptr()) % IO_ALIGNMENT
        # One big allocation keeps every row 4 KiB-aligned without relying
        # on the caching allocator's alignment.
        self.arena = raw.narrow(
            0, offset, page_slots * self.row_bytes
        ).view(page_slots, self.row_bytes)

        self.stream = torch.cuda.Stream(device=self.device)
        self._copy_events: list = []
        self._free_slots = list(range(page_slots))
        self.lock = threading.Lock()

        self._init_geometry()

    # ------------------------------------------------------------------
    # Geometry (per host pool kind)
    # ------------------------------------------------------------------

    def _view(self, shape, strides, byte_offset: int, dtype) -> torch.Tensor:
        """A strided GPU view into the arena rows at *byte_offset*.

        The row axis strides by ``row_bytes`` (rows are padded); the
        trailing axes pack densely, which is exactly the page-first
        layout the scatter kernels step with ``stride(0)``.
        """
        elem_size = torch.empty((), dtype=dtype).element_size()
        if byte_offset % elem_size or self.row_bytes % elem_size:
            raise ValueError(
                f"arena carve at {byte_offset} is not {dtype}-aligned"
            )
        base = self.arena.view(dtype)
        return torch.as_strided(
            base,
            (self.page_slots, *shape),
            (self.row_bytes // elem_size, *strides),
            storage_offset=base.storage_offset() + byte_offset // elem_size,
        )

    def _init_geometry(self) -> None:
        host = self.host_pool
        if isinstance(host, MHATokenToKVPoolDynamicFP8Host):
            self.kind = "kv_dynamic_fp8"
            self.layer_num = host.target_layer_num
            ps, head, dim = host.page_size, host.head_num, host.head_dim
            per_layer_payload = ps * head * dim * host.dtype.itemsize
            self.k_page_bytes = self.layer_num * per_layer_payload
            self.payload_page_bytes = 2 * self.k_page_bytes
            scale_page = host._scale_page_bytes
            if self.payload_page_bytes + scale_page != self.page_bytes:
                raise ValueError(
                    "device staging carve does not match the dynamic-fp8 "
                    f"flat page layout: {self.payload_page_bytes}+{scale_page} "
                    f"!= {self.page_bytes}"
                )
            payload_shape = (self.layer_num, ps, head, dim)
            payload_strides = _elem_strides(payload_shape)
            self.k_view = self._view(
                payload_shape,
                payload_strides,
                0,
                host.dtype,
            )
            self.v_view = self._view(
                payload_shape,
                payload_strides,
                self.k_page_bytes,
                host.dtype,
            )
            scale_shape = (2, self.layer_num, ps, head)
            self.scale_view = self._view(
                scale_shape,
                _elem_strides(scale_shape),
                self.payload_page_bytes,
                torch.float32,
            )
        elif isinstance(host, MambaPoolHost):
            self.kind = "mamba"
            self.layer_num = host.num_mamba_layers
            offset = 0
            if host.temporal_state_elem_size > 0:
                shape = (self.layer_num, *host.temporal_state_shape)
                self.temporal_view = self._view(
                    shape, _elem_strides(shape), 0, host.temporal_dtype
                )
                offset += (
                    host.temporal_state_elem_size
                    * host.temporal_dtype.itemsize
                    * self.layer_num
                )
            else:
                self.temporal_view = None
            self.conv_views = []
            for conv_shape in host.conv_state_shapes:
                shape = (self.layer_num, *conv_shape)
                self.conv_views.append(
                    self._view(
                        shape, _elem_strides(shape), offset, host.conv_dtype
                    )
                )
                offset += (
                    math.prod(conv_shape) * host.conv_dtype.itemsize
                ) * self.layer_num
            self.slot_state_views = []
            for state in host.slot_state_device_tensors:
                shape = tuple(state.shape[1:])
                self.slot_state_views.append(
                    self._view(
                        shape, _elem_strides(shape), offset, state.dtype
                    )
                )
                offset += state[0].numel() * state.element_size()
        elif isinstance(host, DeepSeekV4PagedHostPool):
            self.kind = "paged"
            self.layer_num = host.layer_num
            self.item_bytes = host.item_bytes
            self.kv_view = self._view(
                (self.layer_num, self.item_bytes),
                (self.item_bytes, 1),
                0,
                torch.uint8,
            )
        else:
            raise NotImplementedError(
                f"Device-target staging has no geometry for host pool "
                f"{type(host).__name__}"
            )

    # ------------------------------------------------------------------
    # Page-slot allocator (host-pool contract, token-indexed)
    # ------------------------------------------------------------------

    def available_size(self) -> int:
        with self.lock:
            return len(self._free_slots) * self.page_size

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        pages = need_size // self.page_size
        with self.lock:
            if pages > len(self._free_slots):
                return None
            # Ascending slots, matching the host pools' free-list order so
            # page order in a span follows allocation order.
            slots = self._free_slots[:pages]
            self._free_slots = self._free_slots[pages:]
        tokens = []
        for slot in slots:
            tokens.extend(range(slot * self.page_size, (slot + 1) * self.page_size))
        return torch.tensor(tokens, dtype=torch.int64)

    def free(self, indices: torch.Tensor) -> int:
        indices = indices.detach().cpu().reshape(-1)
        if indices.numel() == 0:
            return 0
        assert indices.numel() % self.page_size == 0
        slots = (indices[:: self.page_size] // self.page_size).tolist()
        with self.lock:
            self._free_slots.extend(slots)
        return len(indices)

    def page_slots_of(self, host_indices: torch.Tensor) -> list[int]:
        """Arena rows addressed by (possibly scattered) token indices."""
        pages = (
            host_indices.detach().cpu().reshape(-1)[:: self.page_size]
            // self.page_size
        )
        return pages.tolist()

    # ------------------------------------------------------------------
    # Storage side: read a page into the arena
    # ------------------------------------------------------------------

    def copy_page_from_host(self, slot: int, host_page: torch.Tensor) -> None:
        """Copy one whole flat page (pinned host) into an arena row.

        The copy is async on the staging stream but the caller's pinned
        bounce is reused for the next page, so this waits for the copy to
        leave the host before returning. The cuFile engine removes the
        bounce entirely by reading straight into the row.
        """
        with torch.cuda.stream(self.stream):
            self.arena[slot, : self.page_bytes].copy_(host_page, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        event.synchronize()
        self._copy_events.append(event)

    def take_pending_copy_events(self) -> list:
        """Events gating any scatter that consumes unread arena rows."""
        events, self._copy_events = self._copy_events, []
        return events

    def get_debug_page_tensors(self, slot: int):
        """Component tensors of one arena row, named like the host pool's
        page digests so write and device-read markers pair up."""
        if self.kind == "kv_dynamic_fp8":
            yield "kv_k", self.k_view[slot]
            yield "kv_v", self.v_view[slot]
            yield "kv_scale_k", self.scale_view[slot, 0]
            yield "kv_scale_v", self.scale_view[slot, 1]
        elif self.kind == "mamba":
            if self.temporal_view is not None:
                yield "mamba_temporal", self.temporal_view[slot]
            for conv_idx, view in enumerate(self.conv_views):
                yield f"mamba_conv_{conv_idx}", view[slot]
            for state_idx, view in enumerate(self.slot_state_views):
                yield f"mamba_slot_state_{state_idx}", view[slot]
        else:
            yield self.host_pool.pool_label, self.arena[slot, : self.page_bytes]

    # ------------------------------------------------------------------
    # Host-pool transfer surface (L2 engine contract)
    # ------------------------------------------------------------------

    def storage_format_spec(self) -> dict:
        # Identical to the host pool's, so pages written through the host
        # path fingerprint-match pages read into this arena.
        return self.host_pool.storage_format_spec()

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError("Device staging has no draft layers.")
        if self.kind == "kv_dynamic_fp8":
            self._scatter_kv_dynamic_fp8(host_indices, device_indices, layer_id)
        elif self.kind == "mamba":
            self._scatter_mamba(host_indices, device_indices, layer_id)
        elif self.kind == "paged":
            self._scatter_paged(host_indices, device_indices, layer_id)
        else:
            raise NotImplementedError(self.kind)

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        self.load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )

    def _scatter_kv_dynamic_fp8(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor, layer_id: int
    ) -> None:
        if not 0 <= layer_id < self.layer_num:
            return
        host = self.host_pool
        device_k = host.device_k_buffers[layer_id]
        pages = torch.tensor(
            self.page_slots_of(host_indices),
            device=device_k.device,
            dtype=torch.long,
        )
        scale_dst = device_indices.to(device_k.device, dtype=torch.long)
        # Plain D2D index copies: the installed direct pf->lf kernel's batch
        # memcpy path rejects device-source pointers (cudaErrorInvalidValue
        # from cudaMemcpyBatchAsync), so M2a scatters with gather/scatter
        # indexing; the kernel path returns with the cuFile engine.
        host.device_k_buffers[layer_id][scale_dst] = self.k_view[
            pages, layer_id
        ].reshape(-1, host.head_num, host.head_dim)
        host.device_v_buffers[layer_id][scale_dst] = self.v_view[
            pages, layer_id
        ].reshape(-1, host.head_num, host.head_dim)
        host.scale_k_buffers[layer_id][scale_dst] = self.scale_view[
            pages, 0, layer_id
        ].reshape(-1, host.head_num)
        host.scale_v_buffers[layer_id][scale_dst] = self.scale_view[
            pages, 1, layer_id
        ].reshape(-1, host.head_num)

    def _scatter_mamba(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor, layer_id: int
    ) -> None:
        if not 0 <= layer_id < self.layer_num:
            return
        host = self.host_pool
        temporal_layers = (
            host.temporal_device_layers
            if host.mtp_draft_buffer_infos
            else host.buffer_info.temporal
        )
        conv_layers = (
            host.conv_device_layers
            if host.mtp_draft_buffer_infos
            else host.buffer_info.conv
        )
        # Plain D2D index copies (see _scatter_kv_dynamic_fp8: the installed
        # direct kernel's batch memcpy rejects device-source pointers).
        slots = host_indices.detach().cpu().reshape(-1)
        dst_slots = device_indices.detach().cpu().reshape(-1)
        if self.temporal_view is not None:
            src = slots.to(self.temporal_view.device, dtype=torch.long)
            dst = dst_slots.to(temporal_layers[layer_id].device, dtype=torch.long)
            temporal_layers[layer_id][dst] = self.temporal_view[src, layer_id]
        for conv_idx, layers in enumerate(conv_layers):
            src = slots.to(self.conv_views[conv_idx].device, dtype=torch.long)
            dst = dst_slots.to(layers[layer_id].device, dtype=torch.long)
            layers[layer_id][dst] = self.conv_views[conv_idx][src, layer_id]
        if layer_id == 0 and host.slot_state_device_tensors:
            # Side states ride on the first layer's call, before its
            # completion event — same contract as the host pool's load.
            for state_idx, state in enumerate(host.slot_state_device_tensors):
                view = self.slot_state_views[state_idx]
                state[dst_slots.to(state.device, dtype=torch.long)] = view[
                    slots.to(view.device, dtype=torch.long)
                ]

    def _scatter_paged(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor, layer_id: int
    ) -> None:
        if not 0 <= layer_id < self.layer_num:
            return
        host = self.host_pool
        # Plain D2D index copy: same rows the host pool's page_first kernel
        # path moves, without requiring a host-source pointer.
        src_rows = torch.tensor(
            self.page_slots_of(host_indices), dtype=torch.long
        ).to(host.device_buffers[layer_id].device)
        dst = host.device_buffers[layer_id]
        dst_rows = (
            device_indices.detach().cpu().reshape(-1)[:: host.slot_page_size]
            // host.slot_page_size
        ).to(dst.device, dtype=torch.long)
        dst[dst_rows] = self.kv_view[src_rows, layer_id]
