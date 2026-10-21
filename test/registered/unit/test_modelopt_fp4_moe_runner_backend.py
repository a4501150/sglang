"""Unit tests for the modelopt_fp4 moe_runner_backend resolution in
srt/arg_groups/overrides.py.

`auto` must resolve once, in the post-process passes, before any MoE layer is
built: ModelOptNvFp4FusedMoEMethod snapshots the backend at construction
(`auto` means "not trtllm" there, so the CUTLASS weight layout is prepared),
while its create_moe_runner used to resolve `auto` to FLASHINFER_TRTLLM on its
own. Leaving `auto` alive past the passes split-brains the two and the trtllm
forward then reads tensors (g1_scale_c) that were never built.

The quant pass owns a2a=none; `_moe_runner_backend_a2a_default` owns the
flashinfer and deepep A2A gaps (nothing resolved those, so `--moe-a2a-backend
flashinfer` with auto failed handle_a2a_moe's supported-runner assert and
deepep+NVFP4 fell to the layer's split-brain resolution).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _moe_runner_backend_a2a_default,
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
    nvfp4_moe_meta=None,
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
        hf_config=SimpleNamespace(quantization_config=quantization_config),
        nvfp4_moe_meta=nvfp4_moe_meta,
    )
    return ResolvedView(record)


def _resolve_quant(view: ResolvedView) -> str:
    declared = _moe_runner_backend_quant_constraints(view)
    return declared.get("moe_runner_backend", view.moe_runner_backend)


def _resolve_a2a(view: ResolvedView) -> str:
    declared = _moe_runner_backend_a2a_default(view)
    return declared.get("moe_runner_backend", view.moe_runner_backend)


class TestModelOptFp4MoeRunnerBackend(CustomTestCase):
    # The checkpoint did not name a quantization the operator passed on the
    # CLI; the pass must read it off the HF config and resolve `auto`.

    def test_auto_resolves_cutlass_on_sm120(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_cutlass",
            )

    def test_canonical_quant_method_resolves_cutlass_on_sm120(self):
        # The parsed hf_config may already carry the canonical method name
        # instead of the raw "modelopt" the checkpoint's config.json ships.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(_view(quant_method="modelopt_fp4")),
                "flashinfer_cutlass",
            )

    def test_auto_resolves_cutlass_on_sm110(self):
        with override_platform(is_sm120=False, is_sm110=True, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_cutlass",
            )

    def test_auto_resolves_trtllm_on_sm100(self):
        # trtllm-gen MoE kernels run on SM100; the mismatch bug hit every
        # Blackwell part, just with a different wrong answer.
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                "flashinfer_trtllm",
            )

    def test_auto_resolves_marlin_on_sm80_sm90(self):
        # The W4A16 marlin fallback the method __init__ used to pick for
        # (8,0) <= capability < (10,0); it is now the pass's decision, so
        # weight prep (marlin repack) and the runner cannot disagree.
        with override_platform(
            is_sm120=False,
            is_sm110=False,
            is_blackwell=False,
            is_cuda=True,
            is_hip=False,
            device_capability=(9, 0),
        ):
            self.assertEqual(
                _resolve_quant(_view(quant_method="modelopt", quant_algo="NVFP4")),
                "marlin",
            )

    def test_explicit_modelopt_fp4_still_resolves_cutlass_on_sm120(self):
        # Pre-existing behavior for --quantization modelopt_fp4 is preserved.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(_view(quantization="modelopt_fp4")),
                "flashinfer_cutlass",
            )

    def test_generic_modelopt_arg_uses_the_checkpoint_algo(self):
        # --quantization modelopt means "auto-detect from quant_algo".
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(
                    _view(
                        quantization="modelopt",
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                    )
                ),
                "flashinfer_cutlass",
            )

    # Configurations the quant pass must not touch.

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

    def test_non_cuda_is_untouched(self):
        # No CUDA capability to read: the pass must not guess a fallback
        # where the old __init__ raised a platform error instead.
        with override_platform(
            is_sm120=False,
            is_sm110=False,
            is_blackwell=False,
            is_cuda=False,
            is_hip=False,
        ):
            self.assertEqual(
                _resolve_quant(_view(quant_method="modelopt", quant_algo="NVFP4")),
                "auto",
            )

    def test_megamoe_a2a_is_untouched(self):
        # handle_a2a_moe resolves `auto` to flashinfer_megamoe itself;
        # pre-empting it here would trip its assert.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="flashinfer_megamoe",
                    )
                ),
                "auto",
            )

    def test_deepep_a2a_resolves_cutedsl(self):
        # Regression: deepep + NVFP4 + auto used to fall through to the
        # layer, whose __init__ and create_moe_runner resolved `auto`
        # differently (CUTLASS weight layout vs the trtllm runner).
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="deepep",
                    )
                ),
                "flashinfer_cutedsl",
            )

    def test_deepep_a2a_hybrid_nvfp4_meta_resolves_cutedsl(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(moe_a2a_backend="deepep", nvfp4_moe_meta=object())
                ),
                "flashinfer_cutedsl",
            )

    def test_deepep_a2a_non_fp4_keeps_auto(self):
        # fp8 resolves in its own create_moe_runner from per-layer facts and
        # its weight prep never reads the backend, so auto is safe there.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(quantization="fp8", moe_a2a_backend="deepep")
                ),
                "auto",
            )

    def test_explicit_backend_is_untouched(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_quant(
                    _view(
                        backend="flashinfer_cutlass",
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                    )
                ),
                "flashinfer_cutlass",
            )


class TestFlashinferA2aRunnerDefault(CustomTestCase):
    # Regression: --moe-a2a-backend flashinfer with the runner left on auto
    # reached handle_a2a_moe's supported-runner assert and failed startup
    # with a confusing message; nothing resolved `auto` for non-DeepSeek
    # architectures.

    def test_nvfp4_resolves_cutlass_on_sm120(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="flashinfer",
                    )
                ),
                "flashinfer_cutlass",
            )

    def test_nvfp4_resolves_trtllm_on_sm100(self):
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="flashinfer",
                    )
                ),
                "flashinfer_trtllm",
            )

    def test_bf16_resolves_cutlass_on_sm120(self):
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(_view(moe_a2a_backend="flashinfer")),
                "flashinfer_cutlass",
            )

    def test_fp8_resolves_trtllm_on_sm100(self):
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _resolve_a2a(
                    _view(quantization="fp8", moe_a2a_backend="flashinfer")
                ),
                "flashinfer_trtllm",
            )

    def test_fp8_rejects_sm120(self):
        # trtllm-gen MoE kernels are SM100-only and flashinfer_cutlass takes
        # modelopt_fp4/modelopt_fp8/bf16 only: plain fp8 experts have no
        # FlashInfer-A2A runner on SM110/SM120. This used to die at the
        # handle_a2a_moe assert; now it fails at resolution with a reason.
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            with self.assertRaisesRegex(ValueError, "SM110/SM120"):
                _resolve_a2a(_view(quantization="fp8", moe_a2a_backend="flashinfer"))

    def test_unsupported_quantization_rejected(self):
        with override_platform(is_sm120=False, is_sm110=False, is_blackwell=True):
            with self.assertRaisesRegex(ValueError, "flashinfer"):
                _resolve_a2a(
                    _view(quantization="gptq", moe_a2a_backend="flashinfer")
                )

    def test_non_blackwell_rejected(self):
        with override_platform(
            is_sm120=False, is_sm110=False, is_blackwell=False, is_cuda=True
        ):
            with self.assertRaisesRegex(ValueError, "Blackwell"):
                _resolve_a2a(
                    _view(
                        quant_method="modelopt",
                        quant_algo="NVFP4",
                        moe_a2a_backend="flashinfer",
                    )
                )

    def test_none_a2a_is_untouched(self):
        # a2a=none belongs to the quant pass; the a2a pass must not restate
        # its rules (a second resolution point is what this fix removes).
        with override_platform(is_sm120=True, is_sm110=False, is_blackwell=True):
            self.assertEqual(
                _moe_runner_backend_a2a_default(
                    _view(quant_method="modelopt", quant_algo="NVFP4")
                ),
                {},
            )


class TestModelOptNvFp4MethodBackendFrozen(CustomTestCase):
    # The layer side of the invariant: the method refuses `auto` (the passes
    # own resolution) and refuses a backend flip between construction and
    # create_moe_runner (weight prep picked its layout from the first value).

    def _method(self, backend):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization import modelopt_quant as mq

        with patch.object(mq, "get_moe_runner_backend", return_value=backend):
            return mq.ModelOptNvFp4FusedMoEMethod(SimpleNamespace())

    def test_init_rejects_auto(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        with self.assertRaisesRegex(ValueError, "auto"):
            self._method(MoeRunnerBackend.AUTO)

    def test_create_moe_runner_rejects_auto(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        method = self._method(MoeRunnerBackend.MARLIN)
        from sglang.srt.layers.quantization import modelopt_quant as mq

        with patch.object(
            mq, "get_moe_runner_backend", return_value=MoeRunnerBackend.AUTO
        ):
            with self.assertRaisesRegex(ValueError, "auto"):
                method.create_moe_runner(
                    SimpleNamespace(), SimpleNamespace(is_gated=False)
                )

    def test_create_moe_runner_rejects_flip(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        method = self._method(MoeRunnerBackend.MARLIN)
        from sglang.srt.layers.quantization import modelopt_quant as mq

        with patch.object(
            mq,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with self.assertRaisesRegex(ValueError, "flipped"):
                method.create_moe_runner(
                    SimpleNamespace(), SimpleNamespace(is_gated=False)
                )

    def test_agreeing_backend_passes_the_guards(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        method = self._method(MoeRunnerBackend.MARLIN)
        from sglang.srt.layers.quantization import modelopt_quant as mq

        with patch.object(
            mq, "get_moe_runner_backend", return_value=MoeRunnerBackend.MARLIN
        ):
            with patch.object(mq, "MoeRunner") as runner_cls:
                method.create_moe_runner(
                    SimpleNamespace(), SimpleNamespace(is_gated=False)
                )
        runner_cls.assert_called_once()
        self.assertEqual(
            method._moe_runner_backend, MoeRunnerBackend.MARLIN
        )


if __name__ == "__main__":
    unittest.main()
