"""Host pool for per-token, per-head dynamically scaled FP8 MHA KV."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolDynamicFP8
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    log_hicache_transfer_digests,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)

logger = logging.getLogger(__name__)


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
            device_pool.k_scale_buffer[0].device, dtype=torch.long
        )
        page_count = host_pages.numel()
        # Device scale buffers are indexed in the device pool's own layer space
        # (like the payload's k_data_ptrs/v_data_ptrs tables, which already
        # fold in start_layer), while host rows use the CP-shard-local mapping
        # applied by load_to_device_per_layer; a stage with a non-zero
        # start_layer must not shift this mapping.
        # One blocking D2H per direction for the whole batch instead of one
        # per layer: the write-back pipeline no longer stalls 2 x num_layers
        # times per backup batch.
        owned_device_layer_ids = self._owned_device_layer_ids(device_pool)
        if owned_device_layer_ids:
            k_rows = torch.stack(
                [
                    device_pool.k_scale_buffer[layer_id]
                    .index_select(0, device_indices)
                    .reshape(page_count, self.page_size, self.head_num)
                    for layer_id in owned_device_layer_ids
                ]
            ).cpu()
            v_rows = torch.stack(
                [
                    device_pool.v_scale_buffer[layer_id]
                    .index_select(0, device_indices)
                    .reshape(page_count, self.page_size, self.head_num)
                    for layer_id in owned_device_layer_ids
                ]
            ).cpu()
            for i, device_layer_id in enumerate(owned_device_layer_ids):
                host_layer_id = self._host_layer_index(device_layer_id, device_pool)
                self.k_scale_host[host_pages, host_layer_id] = k_rows[i]
                self.v_scale_host[host_pages, host_layer_id] = v_rows[i]
        # Emitted here (after the scale rows landed) rather than from the
        # payload digest hook inside super(), which runs before this copy.
        self._log_scale_transfer_digests(
            "device_to_host", device_pool, host_indices, device_indices
        )

    def log_transfer_digests(
        self,
        direction: str,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> None:
        super().log_transfer_digests(direction, host_indices, device_indices)
        if direction == "host_to_device":
            # The transfer engine calls this after every layer's load has been
            # submitted, so both payload and scale rows are already in place.
            self._log_scale_transfer_digests(
                direction, self.device_pool, host_indices, device_indices
            )

    def _log_scale_transfer_digests(
        self,
        direction: str,
        device_pool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> None:
        def _pairs():
            host_values = self._validate_page_runs("host", host_indices)
            device_values = self._validate_page_runs("device", device_indices)
            owned_device_layer_ids = self._owned_device_layer_ids(device_pool)
            for offset in range(0, len(host_values), self.page_size):
                host_page = host_values[offset] // self.page_size
                device_index = device_values[offset]
                yield (
                    host_page,
                    device_index,
                    {
                        "kv_scale_k": self.scale_host[
                            host_page, 0, : len(owned_device_layer_ids)
                        ],
                        "kv_scale_v": self.scale_host[
                            host_page, 1, : len(owned_device_layer_ids)
                        ],
                    },
                    {
                        "kv_scale_k": torch.stack(
                            [
                                device_pool.k_scale_buffer[layer_id][
                                    device_index : device_index + self.page_size
                                ]
                                for layer_id in owned_device_layer_ids
                            ]
                        ),
                        "kv_scale_v": torch.stack(
                            [
                                device_pool.v_scale_buffer[layer_id][
                                    device_index : device_index + self.page_size
                                ]
                                for layer_id in owned_device_layer_ids
                            ]
                        ),
                    },
                )

        log_hicache_transfer_digests(direction, "index", _pairs())

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
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer_id = self._host_layer_index(layer_id, device_pool)
        device = device_pool.k_scale_buffer[layer_id].device
        device_indices = device_indices.to(device=device, dtype=torch.long)
        device_pool.k_scale_buffer[layer_id][device_indices] = (
            self.k_scale_host[host_pages, host_layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )
        device_pool.v_scale_buffer[layer_id][device_indices] = (
            self.v_scale_host[host_pages, host_layer_id]
            .reshape(-1, self.head_num)
            .to(device, non_blocking=True)
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page = self._aligned_page_id(index)
        payload = super().get_data_page(index, flat=True).view(torch.uint8)
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
