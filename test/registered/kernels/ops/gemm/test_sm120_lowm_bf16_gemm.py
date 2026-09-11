"""
Tests the SM120 low-M BF16 Triton split-K GEMM path: kernel numerics on the
dispatch domain (SM120 GPU only), the shape predicate, and the BF16 GEMM
dispatch routing (CPU-only, with stubbed kernel hooks).
"""

import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_sm120
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-large")
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@unittest.skipIf(not is_sm120(), "SM120 low-M BF16 GEMM requires an SM120 GPU")
class TestSm120LowmBf16GemmNumerical(unittest.TestCase):
    def _run_case(self, m, n, k, seed=0):
        from sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm import (
            sm120_lowm_bf16_gemm,
            use_sm120_lowm_bf16_gemm,
        )

        self.assertTrue(use_sm120_lowm_bf16_gemm(m, n, k), (m, n, k))
        torch.manual_seed(seed)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = sm120_lowm_bf16_gemm(x, w)
        ref = x.float() @ w.float().t()
        cub = (x @ w.t()).float()
        err = (out.float() - ref).abs().max().item()
        err_cub = (cub - ref).abs().max().item()
        # fp32 accumulation; with split-K > 1 the atomic summation order is
        # nondeterministic, so compare against the cuBLAS error budget.
        self.assertLessEqual(err, max(err_cub * 3.0, 5e-2), (m, n, k, err, err_cub))
        self.assertFalse(torch.isnan(out).any().item(), (m, n, k))

    def test_dispatch_domain_shapes(self):
        # Representative decode projections: wide-N shapes take the
        # split_k == 1 direct-store path, narrow-N shapes take split-K atomics.
        for m, n, k in [
            (1, 2560, 1536),
            (4, 4120, 2560),
            (8, 24576, 1536),
            (16, 1536, 2560),
            (32, 3072, 3072),
        ]:
            self._run_case(m, n, k)

    def test_3d_input(self):
        from sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm import sm120_lowm_bf16_gemm

        x = torch.randn(2, 4, 1536, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(2560, 1536, dtype=torch.bfloat16, device="cuda") * 0.05
        out = sm120_lowm_bf16_gemm(x, w)
        self.assertEqual(out.shape, (2, 4, 2560))
        ref = x.float() @ w.float().t()
        err = (out.float() - ref).abs().max().item()
        self.assertLessEqual(err, 5e-2)


class TestSm120LowmBf16GemmPredicate(unittest.TestCase):
    def test_predicate(self):
        from sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm import (
            use_sm120_lowm_bf16_gemm,
        )

        # In-domain decode shapes (512 KB <= weight bytes <= 128 MB, m <= 32).
        self.assertTrue(use_sm120_lowm_bf16_gemm(4, 4120, 2560))
        self.assertTrue(use_sm120_lowm_bf16_gemm(32, 4120, 2560))
        # Batched above the low-M cutoff.
        self.assertFalse(use_sm120_lowm_bf16_gemm(33, 4120, 2560))
        # Small weights (launch overhead dominates) and the lm_head-scale
        # weights cuBLAS already streams near peak bandwidth must fall back.
        self.assertFalse(use_sm120_lowm_bf16_gemm(1, 512, 256))
        self.assertFalse(use_sm120_lowm_bf16_gemm(1, 65536, 65536))


class TestBf16GemmDispatchSm120(unittest.TestCase):
    """CPU routing coverage for the sm120 branch of _bf16_gemm_dispatch_impl."""

    def setUp(self):
        from sglang.srt.layers.quantization import unquant

        self.unquant = unquant
        self.x = torch.zeros(4, 64, dtype=torch.bfloat16)
        self.w = torch.zeros(32, 64, dtype=torch.bfloat16)

    def _dispatch(self, *, sm120, in_domain=True, bias=None, addend=None):
        uq = self.unquant
        calls = []

        def fake_kernel(x, weight):
            calls.append(tuple(x.shape))
            return torch.zeros(*x.shape[:-1], weight.shape[0], dtype=x.dtype)

        with (
            mock.patch.object(uq, "_enable_sm120_lowm_bf16_gemm", sm120),
            mock.patch.object(uq, "_sm120_lowm_bf16_gemm", fake_kernel),
            mock.patch.object(
                uq, "_use_sm120_lowm_bf16_gemm", lambda m, n, k: in_domain
            ),
            mock.patch.object(uq, "_enable_bf16_splitk_gemm", False),
            mock.patch.object(uq, "_use_hopper_bf16_gemv", None),
            mock.patch.object(uq, "_use_cutedsl_bf16_gemm", None),
        ):
            out = uq._bf16_gemm_dispatch_impl(self.x, self.w, bias, addend=addend)
        return out, calls

    def test_routes_low_m_bias_free_gemm(self):
        out, calls = self._dispatch(sm120=True)
        self.assertEqual(calls, [(4, 64)])
        self.assertEqual(out.shape, (4, 32))

    def test_bias_falls_back(self):
        bias = torch.zeros(32, dtype=torch.bfloat16)
        out, calls = self._dispatch(sm120=True, bias=bias)
        self.assertEqual(calls, [])
        torch.testing.assert_close(
            out, torch.nn.functional.linear(self.x, self.w, bias)
        )

    def test_addend_falls_back(self):
        addend = torch.ones(4, 32, dtype=torch.bfloat16)
        out, calls = self._dispatch(sm120=True, addend=addend)
        self.assertEqual(calls, [])
        self.assertIs(out, addend)

    def test_flag_off_falls_back(self):
        out, calls = self._dispatch(sm120=False)
        self.assertEqual(calls, [])
        torch.testing.assert_close(out, torch.nn.functional.linear(self.x, self.w))

    def test_out_of_domain_falls_back(self):
        out, calls = self._dispatch(sm120=True, in_domain=False)
        self.assertEqual(calls, [])
        torch.testing.assert_close(out, torch.nn.functional.linear(self.x, self.w))

    def test_sm120_precedes_cutedsl(self):
        uq = self.unquant
        sm120_calls = []
        cutedsl_calls = []

        def fake_sm120(x, weight):
            sm120_calls.append(1)
            return torch.zeros(x.shape[0], weight.shape[0], dtype=x.dtype)

        def fake_cutedsl(x, weight, bias):
            cutedsl_calls.append(1)
            return torch.zeros(x.shape[0], weight.shape[0], dtype=x.dtype)

        with (
            mock.patch.object(uq, "_enable_sm120_lowm_bf16_gemm", True),
            mock.patch.object(uq, "_sm120_lowm_bf16_gemm", fake_sm120),
            mock.patch.object(uq, "_use_sm120_lowm_bf16_gemm", lambda m, n, k: True),
            mock.patch.object(uq, "_enable_bf16_splitk_gemm", False),
            mock.patch.object(uq, "_use_hopper_bf16_gemv", None),
            mock.patch.object(uq, "_cutedsl_bf16_gemm", fake_cutedsl),
            mock.patch.object(uq, "_use_cutedsl_bf16_gemm", lambda m, n, k: True),
        ):
            uq._bf16_gemm_dispatch_impl(self.x, self.w, None)
        self.assertEqual(sm120_calls, [1])
        self.assertEqual(cutedsl_calls, [])


if __name__ == "__main__":
    unittest.main()
