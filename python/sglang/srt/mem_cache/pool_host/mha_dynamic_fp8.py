"""Host pool for per-token, per-head dynamically scaled FP8 MHA KV."""

from __future__ import annotations

from typing import Sequence

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolDynamicFP8
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)


class MHATokenToKVPoolDynamicFP8Host(MHATokenToKVPoolHost):
    device_pool: MHATokenToKVPoolDynamicFP8

    def __init__(
        self,
        device_pool: MHATokenToKVPoolDynamicFP8,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
    ):
        if layout not in ("page_first", "page_first_direct"):
            raise NotImplementedError(
                "Dynamic FP8 KV host pool supports only page_first and "
                f"page_first_direct layouts, got {layout!r}."
            )
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Dynamic FP8 KV host pool does not pack MTP draft KV layers."
            )
        self.scale_host: torch.Tensor | None = None
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )
        if self.page_size != device_pool.page_size:
            raise ValueError(
                "Dynamic FP8 KV host pool moves scales per page, so the host "
                f"page size ({self.page_size}) must equal the device page size "
                f"({device_pool.page_size})."
            )
        self._init_scale_buffer()

    def get_size_per_token(self):
        payload = super().get_size_per_token()
        scales = 2 * self.head_num * torch.float32.itemsize * self.layer_num
        return payload + scales

    def _init_scale_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
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

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        pages = indices[:: self.page_size] // self.page_size
        return pages.long().cpu()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        host_pages = self._page_ids(host_indices)
        device_indices = device_indices.to(
            self.device_pool.k_scale_buffer[0].device, dtype=torch.long
        )
        page_count = host_pages.numel()
        for layer_id in range(self.layer_num):
            self.k_scale_host[host_pages, layer_id] = (
                self.device_pool.k_scale_buffer[layer_id]
                .index_select(0, device_indices)
                .reshape(page_count, self.page_size, self.head_num)
                .cpu()
            )
            self.v_scale_host[host_pages, layer_id] = (
                self.device_pool.v_scale_buffer[layer_id]
                .index_select(0, device_indices)
                .reshape(page_count, self.page_size, self.head_num)
                .cpu()
            )

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
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if is_draft:
            raise NotImplementedError("Dynamic FP8 KV host pool has no draft layers.")
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer_id = self._host_layer_index(layer_id)
        host_pages = self._page_ids(host_indices)
        device = self.device_pool.k_scale_buffer[layer_id].device
        device_indices = device_indices.to(device=device, dtype=torch.long)
        self.device_pool.k_scale_buffer[layer_id][device_indices] = (
            self.k_scale_host[host_pages, host_layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )
        self.device_pool.v_scale_buffer[layer_id][device_indices] = (
            self.v_scale_host[host_pages, host_layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        payload = super().get_data_page(index, flat=True).view(torch.uint8)
        page = index // self.page_size
        scales = self.scale_host[page : page + 1].view(torch.uint8)
        data = torch.cat((payload, scales.flatten()))
        return data if flat else data.reshape(1, -1)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.get_size_per_token() * self.page_size,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
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
        page = index // self.page_size
        self.scale_host[page : page + 1].copy_(
            data_page[payload_bytes:]
            .view(torch.float32)
            .reshape(1, 2, self.layer_num, self.page_size, self.head_num)
        )

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        payload_ptrs, payload_sizes = super().get_page_buffer_meta(indices)
        scale_base = self.scale_host.data_ptr()
        ptrs = []
        sizes = []
        indices = indices.tolist()
        for page_offset, token_offset in enumerate(
            range(0, len(indices), self.page_size)
        ):
            ptrs.extend(payload_ptrs[2 * page_offset : 2 * page_offset + 2])
            sizes.extend(payload_sizes[2 * page_offset : 2 * page_offset + 2])
            page = indices[token_offset] // self.page_size
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
