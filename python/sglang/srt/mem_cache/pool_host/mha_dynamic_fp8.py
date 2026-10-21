"""Host (L2) pool for the per-token dynamically scaled FP8 MHA KV cache.

The e4m3 payload rides the regular MHA host pool; this subclass adds the
per-token, per-head FP32 K/V descale buffers, which move as validated
whole-page blocks so a scale row never lands beside a payload row from
another page.

Per-page byte layout (the contract for L3 storage and the cuFile SSD->GPU
backend):

* Payload segment — ``2 * layer_num * page_size * head_num * head_dim`` e4m3
  bytes in the base pool's page_first/page_first_direct page order: K pages
  first (``[layer, page_slot, head, dim]`` per page), then V pages.
* Scale segment — ``2 * layer_num * page_size * head_num`` FP32 bytes laid
  out as ``[component (K=0, V=1), layer, page_slot, head]``: one contiguous
  block per page, addressed from the page's base pointer.
* A flat L3 page (``get_data_page``/``set_from_flat_data_page``) is the
  payload segment followed by the scale segment, as uint8.
* Zero-copy metadata (``get_page_buffer_meta``) emits three segments per
  page in order: K payload, V payload, scale block. A transfer backend that
  cannot take the three-way split can copy the flat page instead.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.mem_cache.device_pool_info import KVScaleLayout, MHABufferInfo
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)

logger = logging.getLogger(__name__)


class MHATokenToKVPoolDynamicFP8Host(MHATokenToKVPoolHost):
    # Payload+scale pages cannot be read back by a static fp8/bf16 pool;
    # the tag makes those L3 keys miss instead of loading corrupt pages.
    storage_format_tag = "mha-dynamic-fp8"

    def __init__(
        self,
        buffer_info: MHABufferInfo,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        draft_buffer_infos: Sequence[MHABufferInfo] = (),
        write_back_staging: tuple[torch.Tensor, torch.Tensor] | None = None,
        pool_label: str = "kv",
        head_group_num: int = 1,
    ):
        if layout not in ("page_first", "page_first_direct"):
            raise NotImplementedError(
                "Dynamic FP8 KV host pool supports only page_first and "
                f"page_first_direct layouts, got {layout!r}."
            )
        if draft_buffer_infos:
            raise NotImplementedError(
                "Dynamic FP8 KV host pool does not pack MTP draft KV layers."
            )
        if (
            buffer_info.scales is None
            or buffer_info.scales.layout != KVScaleLayout.DYNAMIC_FP8_TOKEN
        ):
            raise ValueError(
                "Dynamic FP8 KV host pool requires per-token FP32 "
                "(DYNAMIC_FP8_TOKEN layout) K/V scale buffers."
            )
        if buffer_info.head_dim != buffer_info.v_head_dim:
            raise NotImplementedError(
                "Dynamic FP8 HiCache does not support asymmetric K/V head "
                "dimensions yet."
            )
        self.scale_k_buffers = tuple(buffer_info.scales.k)
        self.scale_v_buffers = tuple(buffer_info.scales.v)
        self.scale_host: torch.Tensor | None = None
        # HostKVCache sizes the pool through get_size_per_token during the
        # base constructor, so the per-token scale cost is derived first.
        self._scale_token_bytes = 2 * buffer_info.head_num * torch.float32.itemsize
        super().__init__(
            buffer_info,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            device_capacity_tokens=device_capacity_tokens,
            start_layer=start_layer,
            end_layer=end_layer,
            draft_buffer_infos=draft_buffer_infos,
            write_back_staging=write_back_staging,
            pool_label=pool_label,
            head_group_num=head_group_num,
        )
        if self.page_size != buffer_info.page_size:
            raise ValueError(
                "Dynamic FP8 KV host pool moves scales per page, so the host "
                f"page size ({self.page_size}) must equal the device page size "
                f"({buffer_info.page_size})."
            )
        self._init_scale_buffer()

    def get_size_per_token(self):
        payload = super().get_size_per_token()
        return payload + self._scale_token_bytes * self.layer_num

    def _init_scale_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        self.scale_host = alloc_func(
            (self.page_num, 2, self.layer_num, self.page_size, self.head_num),
            dtype=torch.float32,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=self._scale_page_bytes,
        )

    @property
    def k_scale_host(self) -> torch.Tensor:
        return self.scale_host[:, 0]

    @property
    def v_scale_host(self) -> torch.Tensor:
        return self.scale_host[:, 1]

    @property
    def _scale_page_bytes(self) -> int:
        return (
            2 * self.layer_num * self.page_size * self.head_num * torch.float32.itemsize
        )

    def _validate_page_runs(self, name: str, indices: torch.Tensor) -> list[int]:
        flat = indices.detach().cpu().reshape(-1)
        if flat.numel() % self.page_size:
            raise RuntimeError(
                f"HiCache KV {name} index count {flat.numel()} is not a multiple "
                f"of page_size={self.page_size}"
            )
        runs = flat.reshape(-1, self.page_size)
        bad = runs[:, 0] % self.page_size != 0
        if self.page_size > 1:
            bad = bad | (runs.diff(dim=1) != 1).any(dim=1)
        if bool(bad.any()):
            page = bad.nonzero()[0].item() * self.page_size
            run = runs[page // self.page_size]
            raise RuntimeError(
                f"HiCache KV {name} page run is not aligned and contiguous: "
                f"offset={page} first={run[0].item()} last={run[-1].item()} "
                f"page_size={self.page_size}"
            )
        return flat.tolist()

    def _page_ids(self, indices: torch.Tensor, name: str) -> torch.Tensor:
        # Scales move as whole page blocks keyed by the run start, so a partial
        # or shuffled run would silently land scale blocks on the wrong pages.
        self._validate_page_runs(name, indices)
        pages = indices[:: self.page_size] // self.page_size
        return pages.long().cpu()

    def _aligned_page_id(self, index: int) -> int:
        if index % self.page_size:
            raise RuntimeError(
                f"HiCache KV scale page index {index} is not page-aligned: "
                f"page_size={self.page_size}"
            )
        return index // self.page_size

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_pages = self._page_ids(host_indices, "host")
        self._validate_page_runs("device", device_indices)
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        device_indices = device_indices.to(
            device=self._transfer_device, dtype=torch.long
        )
        page_count = host_pages.numel()
        # One blocking D2H per direction for the whole batch instead of one
        # per layer: the write-back pipeline no longer stalls 2 x num_layers
        # times per backup batch.
        k_rows = torch.stack(
            [
                buffer.index_select(0, device_indices).reshape(
                    page_count, self.page_size, self.head_num
                )
                for buffer in self.scale_k_buffers
            ]
        ).cpu()
        v_rows = torch.stack(
            [
                buffer.index_select(0, device_indices).reshape(
                    page_count, self.page_size, self.head_num
                )
                for buffer in self.scale_v_buffers
            ]
        ).cpu()
        for layer in range(self.target_layer_num):
            self.k_scale_host[host_pages, layer] = k_rows[layer]
            self.v_scale_host[host_pages, layer] = v_rows[layer]

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
            raise NotImplementedError("Dynamic FP8 KV host pool has no draft layers.")
        host_pages = self._page_ids(host_indices, "host")
        self._validate_page_runs("device", device_indices)
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if not 0 <= layer_id < self.target_layer_num:
            return
        device = self.scale_k_buffers[layer_id].device
        device_indices = device_indices.to(device=device, dtype=torch.long)
        self.scale_k_buffers[layer_id][device_indices] = (
            self.k_scale_host[host_pages, layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )
        self.scale_v_buffers[layer_id][device_indices] = (
            self.v_scale_host[host_pages, layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page = self._aligned_page_id(index)
        payload = super().get_data_page(index, flat=True).view(torch.uint8)
        scales = self.scale_host[page : page + 1].reshape(-1).view(torch.uint8)
        data = torch.cat((payload, scales))
        return data if flat else data.reshape(1, -1)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.get_size_per_token() * self.page_size,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page = self._aligned_page_id(index)
        payload_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        payload = data_page[:payload_bytes].view(self.dtype)
        super().set_from_flat_data_page(index, payload)
        self.scale_host[page : page + 1].copy_(
            data_page[payload_bytes:]
            .view(torch.float32)
            .reshape(1, 2, self.layer_num, self.page_size, self.head_num)
        )

    def get_page_buffer_meta(self, indices):
        # The scale segment address comes from the run start, so every run
        # must be a complete aligned contiguous page.
        host_values = self._validate_page_runs("host", indices)
        payload_ptrs, payload_sizes = super().get_page_buffer_meta(indices)
        scale_base = self.scale_host.data_ptr()
        ptrs = []
        sizes = []
        for page_offset, token_offset in enumerate(
            range(0, len(host_values), self.page_size)
        ):
            ptrs.extend(payload_ptrs[2 * page_offset : 2 * page_offset + 2])
            sizes.extend(payload_sizes[2 * page_offset : 2 * page_offset + 2])
            page = host_values[token_offset] // self.page_size
            ptrs.append(scale_base + page * self._scale_page_bytes)
            sizes.append(self._scale_page_bytes)
        return ptrs, sizes

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise NotImplementedError(
            "Dynamic FP8 KV host pages do not support split-head transfers."
        )

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return (
            super().is_stride_page_aligned(page_size_bytes)
            and self.scale_host.data_ptr() % page_size_bytes == 0
            and self._scale_page_bytes % page_size_bytes == 0
        )

    def destroy(self):
        if self.scale_host is not None and self.pin_memory and (_is_cuda or _is_hip):
            _cuda_host_unregister(self.scale_host)
        self.scale_host = None
        super().destroy()
