# SPDX-License-Identifier: Apache-2.0
"""SM120 (RTX PRO 6000 Blackwell / GeForce Blackwell) low-M BF16 GEMM.

cuBLAS under CUDA-graph capture serves the qwen3.5/qwen4 decode projections
(small m, k in {1536, 2560}) with 1-warp split-K WMMA kernels that reach only
~20-70% of DRAM bandwidth on sm120 (measured 61.9 us for m=4, n=4120, k=2560
= 320 GB/s on an RTX PRO 6000). The SM100 CuTeDSL/split-K paths in
``unquant.py`` need tcgen05 and cannot run here. This module provides a plain
Triton split-K GEMV with a generic heuristic tactic for sm120 low-M shapes
(m <= ``_MAX_M``), dispatched from ``bf16_gemm_dispatch``.

Split-K partials are accumulated with fp32 atomics: non-deterministic
summation order, so this path is disabled under
``--enable-deterministic-inference`` (mirrors the SM100 split-K policy).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Eligible shapes. Covers decode (m = bs <= 4), bs=1 target-verify (m = 4-5)
# and bs<=4 C4 verify (m <= 16), plus small eager draft-extend batches.
_MAX_M = 32
_MIN_WEIGHT_BYTES = 512 * 1024  # below this, launch overhead dominates; keep cuBLAS
# Above this, cuBLAS GEMV/split-K already streams at ~94% of DRAM bandwidth
# (measured on the 1.27 GB lm_head); keep it.
_MAX_WEIGHT_BYTES = 128 * 1024 * 1024


@triton.jit
def _sm120_lowm_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    n_mask = rn < N
    m_mask = rm < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    k_end = tl.minimum(k_start + k_per, K)
    for k0 in tl.range(k_start, k_end, BLOCK_K, num_stages=NUM_STAGES):
        rk = k0 + tl.arange(0, BLOCK_K)
        k_mask = rk < k_end
        xb = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(xb, tl.trans(wb), out_dtype=tl.float32)
    if OUT_BF16:
        tl.store(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
            sem="relaxed",
        )


def _generic_tactic(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """(1, 16, 128) measured best for the wide shapes (81% of DRAM bandwidth
    at m=4, n=4120, k=2560); split K only when 16-wide N tiles cannot fill
    the 188 sm120 SMs."""
    n_ctas = triton.cdiv(n, 16)
    split_k = 1
    while split_k < 16 and n_ctas * split_k < 128 and (k // (split_k * 2)) >= 256:
        split_k *= 2
    return (split_k, 16, 128, 4, 4)


def use_sm120_lowm_bf16_gemm(m: int, n: int, k: int) -> bool:
    return m <= _MAX_M and _MIN_WEIGHT_BYTES <= n * k * 2 <= _MAX_WEIGHT_BYTES


def sm120_lowm_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """out = x @ weight.T for bf16 x[m, k], weight[n, k]; bias-free, m <= _MAX_M."""
    x_2d = x.view(-1, x.shape[-1])
    m, k = x_2d.shape
    n = weight.shape[0]
    tactic = _generic_tactic(m, n, k)
    split_k, block_n, block_k, num_warps, num_stages = tactic
    block_m = max(16, triton.next_power_of_2(m))
    grid = (triton.cdiv(n, block_n), split_k)
    if split_k == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _sm120_lowm_gemm_kernel[grid](
            x_2d,
            weight,
            out,
            m,
            n,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_M=block_m,
            SPLIT_K=1,
            NUM_STAGES=num_stages,
            OUT_BF16=True,
            num_warps=num_warps,
        )
    else:
        acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        _sm120_lowm_gemm_kernel[grid](
            x_2d,
            weight,
            acc,
            m,
            n,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            weight.stride(1),
            acc.stride(0),
            acc.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_M=block_m,
            SPLIT_K=split_k,
            NUM_STAGES=num_stages,
            OUT_BF16=False,
            num_warps=num_warps,
        )
        out = acc.to(torch.bfloat16)
    return out.view(*x.shape[:-1], n)
