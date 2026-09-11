#pragma once

#include "hicache.cuh"
#include "relayout.cuh"
#include <dlfcn.h>
#include <limits>
#include <vector>

namespace sglang {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
// cudaMemcpyBatchAsync changed ABI between CUDA 12.x (9 args, includes
// fail_idx) and CUDA 13.0 (8 args, fail_idx removed). The loaded CUDA
// runtime determines the ABI of the dlsym-resolved symbol.
using CudaMemcpyBatchPtr12 = void*;
using CudaMemcpyBatchAsyncFn12 = cudaError_t (*)(
    CudaMemcpyBatchPtr12*, CudaMemcpyBatchPtr12*, size_t*, size_t,
    cudaMemcpyAttributes*, size_t*, size_t, size_t*, cudaStream_t);
using CudaMemcpyBatchPtr13 = const void*;
using CudaMemcpyBatchAsyncFn13 = cudaError_t (*)(
    CudaMemcpyBatchPtr13*, CudaMemcpyBatchPtr13*, const size_t*, size_t,
    cudaMemcpyAttributes*, size_t*, size_t, cudaStream_t);

struct CudaMemcpyBatchState {
  void* symbol = nullptr;
  bool runtime_is_13 = false;
};

inline auto get_cuda_memcpy_batch_state() -> const CudaMemcpyBatchState& {
  static CudaMemcpyBatchState state = [] {
    CudaMemcpyBatchState s;
    s.symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyBatchAsync");
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
      s.runtime_is_13 = runtime_version >= 13000;
    }
    return s;
  }();
  return state;
}

inline auto call_cuda_memcpy_batch_async(
    const CudaMemcpyBatchState& state,
    const void** dsts,
    const void** srcs,
    const size_t* sizes,
    size_t count,
    cudaMemcpyAttributes* attrs,
    size_t* attrs_idxs,
    size_t num_attrs,
    cudaStream_t stream) -> cudaError_t {
  if (state.runtime_is_13) {
    auto fn = reinterpret_cast<CudaMemcpyBatchAsyncFn13>(state.symbol);
    return fn(
        dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, stream);
  } else {
    auto fn = reinterpret_cast<CudaMemcpyBatchAsyncFn12>(state.symbol);
    size_t fail_idx = std::numeric_limits<size_t>::max();
    return fn(
        const_cast<CudaMemcpyBatchPtr12*>(dsts),
        const_cast<CudaMemcpyBatchPtr12*>(srcs),
        const_cast<size_t*>(sizes), count, attrs, attrs_idxs, num_attrs, &fail_idx, stream);
  }
}
#endif

inline void copy_page_first_pages_fallback(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    cudaStream_t stream) {
  using namespace host;

  RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  for (const auto tensor_id : irange(src_ptrs.size())) {
    RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    for (const auto page_offset : irange(num_pages)) {
      const char* src_ptr = static_cast<const char*>(src_ptrs[tensor_id].data_ptr()) +
                            static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      RuntimeDeviceCheck(cudaMemcpyAsync(dst_ptr, src_ptr, src_page_bytes, cudaMemcpyDeviceToHost, stream));
    }
  }
}

inline bool try_copy_page_first_pages_batch(
    const std::vector<tvm::ffi::TensorView>& src_ptrs,
    std::vector<tvm::ffi::TensorView> dst_ptrs,
    const int64_t* dst_indices_ptr,
    int64_t num_pages,
    int64_t page_size,
    int device_id,
    cudaStream_t stream) {
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || (CUDA_VERSION < 12080)
  return false;
#else

  // A genuine CUDA API failure must surface through the runtime check path
  // rather than being silently converted into a fallback. A normal query
  // that reports the capability as absent (value 0) still returns false so
  // callers keep the per-page copy fallback.
  int can_use_host_pointer = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &can_use_host_pointer, cudaDevAttrCanUseHostPointerForRegisteredMem, device_id));
  if (can_use_host_pointer == 0) {
    return false;
  }

  host::RuntimeCheck(src_ptrs.size() == dst_ptrs.size(), "Source and destination tensors must have the same count");
  constexpr size_t kLargeCopyThresholdBytes = 128 * 1024;
  thread_local std::vector<const void*> batch_srcs;
  thread_local std::vector<const void*> batch_dsts;
  thread_local std::vector<size_t> batch_sizes;

  const auto& batch_state = get_cuda_memcpy_batch_state();
  if (batch_state.symbol == nullptr) {
    return false;
  }

  const size_t num_copies = static_cast<size_t>(src_ptrs.size()) * static_cast<size_t>(num_pages);
  batch_srcs.clear();
  batch_dsts.clear();
  batch_sizes.clear();
  batch_srcs.reserve(num_copies);
  batch_dsts.reserve(num_copies);
  batch_sizes.reserve(num_copies);

  size_t first_page_bytes = 0;
  for (const auto tensor_id : host::irange(src_ptrs.size())) {
    host::RuntimeCheck(
        src_ptrs[tensor_id].dtype() == dst_ptrs[tensor_id].dtype(),
        "Source and destination tensors must have the same dtype");
    const int64_t elem_size = host::dtype_bytes(src_ptrs[tensor_id].dtype());
    const int64_t src_stride0 = src_ptrs[tensor_id].stride(0);
    const int64_t dst_stride0 = dst_ptrs[tensor_id].stride(0);
    const size_t src_page_bytes = static_cast<size_t>(page_size * src_stride0 * elem_size);
    const size_t dst_page_bytes = static_cast<size_t>(page_size * dst_stride0 * elem_size);
    host::RuntimeCheck(src_page_bytes == dst_page_bytes, "Source and destination page spans must match");
    if (tensor_id == 0) {
      first_page_bytes = src_page_bytes;
    }
    for (const auto page_offset : host::irange(num_pages)) {
      char* src_ptr = static_cast<char*>(src_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(page_offset * page_size * src_stride0 * elem_size);
      char* dst_ptr = static_cast<char*>(dst_ptrs[tensor_id].data_ptr()) +
                      static_cast<size_t>(dst_indices_ptr[page_offset * page_size] * dst_stride0 * elem_size);
      batch_srcs.push_back(src_ptr);
      batch_dsts.push_back(dst_ptr);
      batch_sizes.push_back(src_page_bytes);
    }
  }
  if (first_page_bytes < kLargeCopyThresholdBytes) {
    return false;
  }

  std::vector<size_t> attrs_idxs(1, 0);
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeDevice;
  attrs.srcLocHint.id = device_id;
  attrs.dstLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.id = 0;
  attrs.flags = 0;

  cudaError_t err = call_cuda_memcpy_batch_async(
      batch_state,
      batch_dsts.data(),
      batch_srcs.data(),
      batch_sizes.data(),
      num_copies,
      &attrs,
      attrs_idxs.data(),
      1,
      stream);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
#endif
}

template <int64_t kElementSize, uint32_t kUnroll, uint32_t kBlockQuota, uint32_t kBlockSize>
struct HiCacheStagedWriteBackKernel {
 private:
  template <bool kIsMLA>
  static void run_staged_impl(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    using namespace host;

    auto T = SymbolicSize{"num_tokens"};
    auto N = SymbolicSize{"num_layers"};
    auto D = SymbolicSize{"element_dim"};
    auto P = SymbolicSize{"num_pages"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto dst_indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({T, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLGPU>(device_)
        .verify(staging_k);
    if constexpr (!kIsMLA) {
      TensorMatcher({T, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLGPU>(device_)
          .verify(staging_v);
    }
    TensorMatcher({-1, N, D})  //
        .with_dtype(cache_dtype)
        .with_device<kDLCPU, kDLGPUHost>()
        .verify(k_cache_dst);
    if constexpr (!kIsMLA) {
      TensorMatcher({-1, N, D})  //
          .with_dtype(cache_dtype)
          .with_device<kDLCPU, kDLGPUHost>()
          .verify(v_cache_dst);
    }
    TensorMatcher({N})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(k_ptr_src);
    if constexpr (!kIsMLA) {
      TensorMatcher({N})  //
          .with_dtype<uint64_t>()
          .with_device<kDLGPU>(device_)
          .verify(v_ptr_src);
    }
    TensorMatcher({P})  //
        .with_dtype<int32_t, int64_t>(indices_dtype)
        .with_device<kDLGPU>(device_)
        .verify(page_indices_src);
    TensorMatcher({T})  //
        .with_dtype<int64_t>(dst_indices_dtype)
        .with_device<kDLCPU, kDLGPUHost>()
        .verify(dst_indices_cpu);

    RuntimeCheck(page_size > 0, "HiCache staged relayout: page_size must be positive");
    RuntimeCheck(T.unwrap() == P.unwrap() * page_size, "HiCache staged relayout: staging token count mismatch");
    RuntimeCheck(
        kElementSize == D.unwrap() * dtype_bytes(cache_dtype.unwrap()),
        "HiCache staged relayout: element size mismatch");
    RuntimeCheck(kElementSize % 16 == 0, "HiCache staged relayout: element size must be 16-byte aligned");

    const auto params = HicacheRelayoutParams{
        .k_cache_dst = staging_k.data_ptr(),
        .v_cache_dst = kIsMLA ? nullptr : staging_v.data_ptr(),
        .indices_src = page_indices_src.data_ptr(),
        .k_ptr_src = k_ptr_src.data_ptr(),
        .v_ptr_src = kIsMLA ? nullptr : v_ptr_src.data_ptr(),
        .num_pages = static_cast<uint32_t>(P.unwrap()),
        .num_layers = static_cast<uint32_t>(N.unwrap()),
        .page_size = static_cast<uint32_t>(page_size),
    };
    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    launch_hicache_relayout_kernel<kElementSize, kIsMLA>(params, P.unwrap(), N.unwrap(), page_size, use_int32, device);

    auto stream = LaunchKernel::resolve_device(device);
    const int64_t* dst_indices_ptr = static_cast<const int64_t*>(dst_indices_cpu.data_ptr());
    if constexpr (kIsMLA) {
      if (!try_copy_page_first_pages_batch(
              {staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, device.device_id, stream)) {
        copy_page_first_pages_fallback({staging_k}, {k_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    } else {
      if (!try_copy_page_first_pages_batch(
              {staging_k, staging_v},
              {k_cache_dst, v_cache_dst},
              dst_indices_ptr,
              P.unwrap(),
              page_size,
              device.device_id,
              stream)) {
        copy_page_first_pages_fallback(
            {staging_k, staging_v}, {k_cache_dst, v_cache_dst}, dst_indices_ptr, P.unwrap(), page_size, stream);
      }
    }
  }

  static void run_page_first_to_layer_dma_impl(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices_cpu,
      const tvm::ffi::TensorView dst_indices_cpu,
      const int64_t layer_id,
      const int64_t page_size) {
    using namespace host;

    auto T = SymbolicSize{"num_host_tokens"};
    auto U = SymbolicSize{"num_device_tokens"};
    auto N = SymbolicSize{"num_layers"};
    auto D = SymbolicSize{"element_dim"};
    auto I = SymbolicSize{"num_indices"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({T, N, D})  //
        .with_dtype(dtype)
        .with_device<kDLCPU, kDLGPUHost>()
        .verify(src);
    TensorMatcher({U, D})  //
        .with_dtype(dtype)
        .with_device<kDLGPU>(device)
        .verify(dst);
    TensorMatcher({I})  //
        .with_dtype<int64_t>()
        .with_device<kDLCPU, kDLGPUHost>()
        .verify(src_indices_cpu);
    TensorMatcher({I})  //
        .with_dtype<int64_t>()
        .with_device<kDLCPU, kDLGPUHost>()
        .verify(dst_indices_cpu);

    RuntimeCheck(page_size > 0, "HiCache page-first DMA: page_size must be positive");
    RuntimeCheck(I.unwrap() % page_size == 0, "HiCache page-first DMA: index count must be page aligned");
    RuntimeCheck(layer_id >= 0 && layer_id < N.unwrap(), "HiCache page-first DMA: layer index out of bounds");
    RuntimeCheck(src.stride(2) == 1 && dst.stride(1) == 1, "HiCache page-first DMA: element dimensions must be contiguous");

    const auto elem_size = dtype_bytes(dtype.unwrap());
    const auto row_bytes = static_cast<size_t>(D.unwrap() * elem_size);
    const auto src_pitch = static_cast<size_t>(src.stride(0) * elem_size);
    const auto dst_pitch = static_cast<size_t>(dst.stride(0) * elem_size);
    RuntimeCheck(row_bytes <= src_pitch && row_bytes <= dst_pitch, "HiCache page-first DMA: row width exceeds tensor pitch");

    const auto* src_indices = static_cast<const int64_t*>(src_indices_cpu.data_ptr());
    const auto* dst_indices = static_cast<const int64_t*>(dst_indices_cpu.data_ptr());
    const auto num_pages = I.unwrap() / page_size;
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    const auto* src_base = static_cast<const char*>(src.data_ptr()) +
                           static_cast<size_t>(layer_id * src.stride(1) * elem_size);
    auto* dst_base = static_cast<char*>(dst.data_ptr());

    for (const auto page_offset : irange(num_pages)) {
      const auto index_offset = page_offset * page_size;
      const auto src_index = src_indices[index_offset];
      const auto dst_index = dst_indices[index_offset];
      RuntimeCheck(
          src_index >= 0 && src_index + page_size <= T.unwrap(),
          "HiCache page-first DMA: source page out of bounds");
      RuntimeCheck(
          dst_index >= 0 && dst_index + page_size <= U.unwrap(),
          "HiCache page-first DMA: destination page out of bounds");

      RuntimeDeviceCheck(cudaMemcpy2DAsync(
          dst_base + static_cast<size_t>(dst_index) * dst_pitch,
          dst_pitch,
          src_base + static_cast<size_t>(src_index) * src_pitch,
          src_pitch,
          row_bytes,
          static_cast<size_t>(page_size),
          cudaMemcpyHostToDevice,
          stream));
    }
  }

 public:
  static void run_all_lf_pf_staged(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging_k,
      const tvm::ffi::TensorView staging_v,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const int64_t page_size) {
    run_staged_impl<false>(
        k_cache_dst,
        v_cache_dst,
        dst_indices_cpu,
        staging_k,
        staging_v,
        page_indices_src,
        k_ptr_src,
        v_ptr_src,
        page_size);
  }

  static void run_page_first_to_layer_dma(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices_cpu,
      const tvm::ffi::TensorView dst_indices_cpu,
      const int64_t layer_id,
      const int64_t page_size) {
    run_page_first_to_layer_dma_impl(
        src, dst, src_indices_cpu, dst_indices_cpu, layer_id, page_size);
  }

  static void run_all_mla_lf_pf_staged(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView dst_indices_cpu,
      const tvm::ffi::TensorView staging,
      const tvm::ffi::TensorView page_indices_src,
      const tvm::ffi::TensorView ptr_src,
      const int64_t page_size) {
    run_staged_impl<true>(
        cache_dst, cache_dst, dst_indices_cpu, staging, staging, page_indices_src, ptr_src, ptr_src, page_size);
  }
};

}  // namespace sglang
