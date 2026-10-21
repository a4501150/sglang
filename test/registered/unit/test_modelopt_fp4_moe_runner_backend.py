"""Unit tests for the modelopt_fp4 moe_runner_backend resolution in
srt/arg_groups/overrides.py.

`auto` must resolve once, in the post-process pass, before any MoE layer is
built: ModelOptNvFp4FusedMoEMethod snapshots the backend at construction
(`auto` means "not trtllm" there, so the CUTLASS weight layout is prepared),
while its create_moe_runner independently resolves `auto` to
FLASHINFER_TRTLLM. Leaving `auto` alive past this pass split-brains the two
and the trtllm forward then reads tensors (g1_scale_c) that were never built.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _moe_runner_backend_quant_constraints,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase


def _view(
    backend: str = "auto",
    quantization: str | None = None,
    quant_algo: str | None = None,
    quant_method: str | None = None,
    moe_a2a_backend: str = "none",
) -> ResolvedView:
    """A view whose checkpoint carries the given HF quantization_config.

    The ModelConfig memo is fixture-supplied, so model_config_of hands it
    back without touching the filesystem.
    """
    record = SimpleNamespace(
        moe_runner_backend=backend,
        quantization=quantization,
        moe_a2a_backend=moe_a2a_backend,
        model_path="dummy",
    )
    quantization_config = {}
    if quant_method is not None:
        quantization_config["quant_method"] = quant_method
    if quant_algo is not None:
        quantization_config["quant_algo"] = quant_algo
    record._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(quantization_config=quantization_config)
    )
    return ResolvedView(record)


def _resolve(view: ResolvedView) -> str:
    declared = _moe_runner_backend_quant_constraints(view)
    return declared.get("moe_runner_backend", view.moe_runner_backend)


class TestModelOptFp4MoeRunnerBackend(CustomTestCase):
    # The checkpoint did not name a quantization the operator passed on the
    # CLI; the pass must read it off the HF config and resolve `auto`.

    def test_auto_resolves_cutlass_on_sm120(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_cutlass",
            )

    def test_canonical_quant_method_resolves_cutlass_on_sm120(self):
        # The parsed hf_config may already carry the canonical method name
        # instead of the raw "modelopt" the checkpoint's config.json ships.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(_view(quant_method="modelopt_fp4")),
                "flashinfer_cutlass",
            )

    def test_auto_resolves_cutlass_on_sm110(self):
        with override_platform(is_sm120=False, is_sm110=True, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_cutlass",
            )

    def test_auto_resolves_trtllm_on_sm100(self):
        # trtllm-gen MoE kernels run on SM100; the mismatch bug hit every
        # Blackwell part, just with a different wrong answer.
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_trtllm",
            )

    def test_explicit_modelopt_fp4_still_resolves_cutlass_on_sm120(self):
        # Pre-existing behavior for --quantization modelopt_fp4 is preserved.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(_view(quantization="modelopt_fp4")),
                "flashinfer_cutlass",
            )

    def test_generic_modelopt_arg_uses_the_checkpoint_algo(self):
        # --quantization modelopt means "auto-detect from quant_algo".
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(
                        quantization="modelopt",
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                    )
                ),
                "flashinfer_cutlass",
            )

    # Configurations the rule must not touch.

    def test_modelopt_fp8_checkpoint_is_untouched(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            view = _view(quant_method="modelopt", quant_algo="FP8")
            self.assertEqual(_moe_runner_backend_quant_constraints(view), {})

    def test_compressed_tensors_checkpoint_is_untouched(self):
        # The compressed-tensors scheme resolves internally and stays
        # self-consistent; resolving for it here would change its behavior.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            view = _view(quant_method="compressed-tensors")
            self.assertEqual(_moe_runner_backend_quant_constraints(view), {})

    def test_non_blackwell_is_untouched(self):
        # SM80/90 keep `auto`: both the method __init__ and create_moe_runner
        # resolve it to the marlin fallback there, consistently.
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=False):
            self.assertEqual(
                _resolve(_view(quant_method="modelopt", quant_algo="NVFP4")),
                "auto",
            )

    def test_megamoe_a2a_is_untouched(self):
        # handle_a2a_moe resolves `auto` to flashinfer_megamoe itself;
        # pre-empting it here would trip its assert.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="flashinfer_megamoe",
                    )
                ),
                "auto",
            )

    def test_deepep_a2a_is_untouched(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="deepep",
                    )
                ),
                "auto",
            )

    def test_explicit_backend_is_untouched(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve(
                    _view(
                        backend="flashinfer_cutlass",
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                    )
                ),
                "flashinfer_cutlass",
            )


if __name__ == "__main__":
    unittest.main()
