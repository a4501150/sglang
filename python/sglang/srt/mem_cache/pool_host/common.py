from __future__ import annotations

import ctypes
import json
import logging
import os
from collections import defaultdict
from functools import cache, lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

_CUDA_HOST_REGISTERED_RANGES_ATTR = "_sglang_cuda_host_registered_ranges"
_CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91


@cache
def can_use_host_pointer_for_registered_mem(device_id: int) -> bool:
    """Whether CUDA hands back the same device pointer for registered host
    memory (needed by the direct page-first kernel path). Returns False —
    safe async-DMA fallback — on HIP or any runtime where the query is
    unavailable, and never raises."""
    if _is_hip or not torch.cuda.is_available():
        return False
    try:
        value = ctypes.c_int()
        result = int(
            torch.cuda.cudart().cudaDeviceGetAttribute(
                ctypes.byref(value),
                _CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
                device_id,
            )
        )
    except Exception as exc:
        logger.warning(
            "Could not query CUDA registered-host pointer capability (%s); "
            "HiCache page-first transfers will use asynchronous DMA copies.",
            exc,
        )
        return False
    supported = result == 0 and value.value != 0
    if not supported:
        logger.warning(
            "CUDA reports distinct registered-host pointer aliases; HiCache "
            "page-first transfers will use asynchronous DMA copies."
        )
    return supported


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
) -> None:
    if can_use_host_pointer_for_registered_mem(torch.cuda.current_device()):
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_direct_lf_pf as transfer

        transfer(src_ptrs, dst_ptrs, src_indices, dst_indices, page_size)
        return

    src_indices_cpu = src_indices.cpu()
    dst_indices_cpu = dst_indices.cpu()
    num_pages = src_indices_cpu.numel() // page_size
    is_mla = len(dst_ptrs) == 1
    num_layers = len(src_ptrs) if is_mla else len(src_ptrs) // 2
    for page_offset in range(num_pages):
        index_offset = page_offset * page_size
        src_index = int(src_indices_cpu[index_offset])
        dst_page = int(dst_indices_cpu[index_offset]) // page_size
        for layer_id in range(num_layers):
            dst_ptrs[0][dst_page, layer_id].copy_(
                src_ptrs[layer_id][src_index : src_index + page_size],
                non_blocking=True,
            )
            if not is_mla:
                dst_ptrs[1][dst_page, layer_id].copy_(
                    src_ptrs[layer_id + num_layers][src_index : src_index + page_size],
                    non_blocking=True,
                )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: list[torch.Tensor],
    dst_ptrs: list[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    if can_use_host_pointer_for_registered_mem(torch.cuda.current_device()):
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_direct_pf_lf as transfer

        transfer(src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size)
        return

    src_indices_cpu = src_indices.cpu()
    dst_indices_cpu = dst_indices.cpu()
    num_pages = src_indices_cpu.numel() // page_size
    is_mla = len(src_ptrs) == 1
    num_layers = len(dst_ptrs) if is_mla else len(dst_ptrs) // 2
    for page_offset in range(num_pages):
        index_offset = page_offset * page_size
        src_page = int(src_indices_cpu[index_offset]) // page_size
        dst_index = int(dst_indices_cpu[index_offset])
        for layer_offset in range(num_layers):
            dst_ptrs[layer_offset][dst_index : dst_index + page_size].copy_(
                src_ptrs[0][src_page, layer_id + layer_offset],
                non_blocking=True,
            )
            if not is_mla:
                dst_ptrs[layer_offset + num_layers][
                    dst_index : dst_index + page_size
                ].copy_(
                    src_ptrs[1][src_page, layer_id + layer_offset],
                    non_blocking=True,
                )


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert device == "cpu", (
            f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        )
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    elif allocator_type == "tensorcast":
        try:
            from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
                get_tensorcast_host_allocator_from_runtime,
            )

            return get_tensorcast_host_allocator_from_runtime()
        except ImportError:
            logger.warning(
                "TensorCast's tensor allocator requires tensorcast >= 0.1.1. Please install TensorCast by 'pip install tensorcast' or build from source by following https://tensorcast.ai/development/build-from-source/. Fallback to use default allocator"
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            rc = int(cudart.cudaHostRegister(ptr, size, 0))
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{cudart.cudaGetErrorString(rc)}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        rc = int(cudart.cudaHostUnregister(ptr))
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(rc),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    registered_ranges = getattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    if registered_ranges is None:
        # Compatibility for buffers registered before range metadata was added.
        registered_ranges = [
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        ]
    if not registered_ranges:
        return

    remaining_ranges = _cuda_host_unregister_ranges(
        cudart, registered_ranges, operation="host-pool destroy"
    )
    setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer, registration_granularity_bytes)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
    registration_granularity_bytes: int | None = None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


@lru_cache(maxsize=1)
def _resolve_device_accessible_ptr_fn():
    try:
        from sgl_kernel.kvcacheio import get_device_accessible_ptr
    except ImportError:
        get_device_accessible_ptr = None
    else:
        if not hasattr(torch.ops.sgl_kernel, "get_device_accessible_ptr"):
            get_device_accessible_ptr = None

    if get_device_accessible_ptr is None:
        # CUDA's UVA makes host and device addresses equal; on HIP they differ.
        if _is_hip:
            raise ImportError(
                "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
                "installed sglang-kernel. It is required on ROCm, where registered "
                "host memory carries a distinct device address. Rebuild sglang-kernel "
                "from python/sglang/kernels/aot (setup_rocm.py)."
            )
        logger.warning(
            "sgl_kernel.kvcacheio.get_device_accessible_ptr is missing from the "
            "installed sglang-kernel; using raw host addresses for kernel pointer "
            "tables. Build sglang-kernel from python/sglang/kernels/aot to enable it."
        )
    return get_device_accessible_ptr


def make_kernel_ptr_table(
    tensors: list[torch.Tensor],
    target_device: torch.device | str,
    *,
    host_memory_registered: bool,
) -> torch.Tensor:
    device = torch.device(target_device)
    get_device_accessible_ptr = (
        _resolve_device_accessible_ptr_fn()
        if host_memory_registered and device.type == "cuda"
        else None
    )
    if get_device_accessible_ptr is not None:
        if device.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = device.index
        pointers = [
            get_device_accessible_ptr(tensor, device_index) for tensor in tensors
        ]
    else:
        pointers = [tensor.data_ptr() for tensor in tensors]
    return torch.tensor(
        pointers,
        dtype=torch.uint64,
        device=device,
    )


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
        "xpu": alloc_with_pin_memory,
    },
)
