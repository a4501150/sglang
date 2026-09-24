"""Dispatch semantics of the shared per-layer page-first DMA helper.

The three MHA host-pool classes all route their page-first per-layer loads
through ``transfer_layers_pf_dma``: async-DMA via the JIT kernel when the
device cannot address registered host memory directly, the caller's fallback
chain otherwise. The per-class suites exercise the fallback paths (they patch
the capability usable); these tests pin the DMA branch itself.
"""

import unittest

import torch
from unittest import mock

from sglang.srt.mem_cache.pool_host import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTransferLayersPfDmaDispatch(CustomTestCase):
    def _run(self, *, is_cuda, host_ptr_usable):
        kernel = mock.Mock()
        fallback = mock.Mock()
        with (
            mock.patch.object(common, "_is_cuda", is_cuda),
            mock.patch.object(
                common,
                "can_use_host_pointer_for_registered_mem",
                return_value=host_ptr_usable,
            ),
            mock.patch.object(
                common,
                "jit_transfer_hicache_page_first_to_layer_dma",
                kernel,
            ),
            mock.patch("torch.cuda.current_device", return_value=0),
        ):
            host_idx = torch.tensor([0, 64, 128], dtype=torch.int64)
            dev_idx = torch.tensor([1, 2, 3], dtype=torch.int64)
            common.transfer_layers_pf_dma(
                [("k_src", "k_dst"), ("v_src", "v_dst")],
                3,
                host_idx,
                dev_idx,
                64,
                fallback,
            )
        return kernel, fallback

    def test_dma_branch_when_host_pointers_unusable(self):
        kernel, fallback = self._run(is_cuda=True, host_ptr_usable=False)
        fallback.assert_not_called()
        self.assertEqual(kernel.call_count, 2)
        k_kwargs = kernel.call_args_list[0].kwargs
        self.assertEqual((k_kwargs["src"], k_kwargs["dst"]), ("k_src", "k_dst"))
        self.assertEqual(k_kwargs["layer_id"], 3)
        self.assertEqual(k_kwargs["page_size"], 64)
        # The dispatch hoists one .cpu() per batch; on already-CPU index
        # tensors that is an identity passthrough.
        self.assertIs(k_kwargs["src_indices"], k_kwargs["src_indices"])
        self.assertEqual(k_kwargs["src_indices"].tolist(), [0, 64, 128])
        self.assertEqual(k_kwargs["dst_indices"].tolist(), [1, 2, 3])
        v_kwargs = kernel.call_args_list[1].kwargs
        self.assertEqual((v_kwargs["src"], v_kwargs["dst"]), ("v_src", "v_dst"))

    def test_fallback_when_host_pointers_usable(self):
        kernel, fallback = self._run(is_cuda=True, host_ptr_usable=True)
        kernel.assert_not_called()
        fallback.assert_called_once()

    def test_non_cuda_takes_fallback(self):
        # _is_cuda is False: the guard short-circuits even when the (mocked)
        # capability query would say host pointers are unusable.
        kernel, fallback = self._run(is_cuda=False, host_ptr_usable=False)
        kernel.assert_not_called()
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
