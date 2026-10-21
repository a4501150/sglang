"""DFLASH aux-hidden capture on Qwen4-Exp.

DFlash2 drafts trained on completed zero-based layer outputs averaged over
the four hyper-connection streams (concatenated stream blocks, not
interleaved). The pre-patch model captured nothing: the generic
layer-input hook is a no-op on this architecture (the layers never read
``captured_last_layer_outputs``), and even a wired-up input capture would
feed 10240-wide raw stream states where the draft's fc expects 2560-wide
completed outputs. These cases pin the capture contract: the completed
output contraction, the layer-id validation, and the (hidden, packed aux)
tuple that must pass through the VL wrapper unchanged.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    Qwen4ExpVLModel,
    _dflash_completed_layer_output,
    _validate_dflash_completed_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The geometry of the real deployment: Qwen3.8-Flash-Next target (48 layers,
# hidden 2560, hc_count 4) with the incoai DFlash2 draft capturing
# completed outputs of layers [5, 15, 25, 35, 45].
PRODUCTION_LAYER_IDS = [5, 15, 25, 35, 45]
NUM_HIDDEN_LAYERS = 48
HC_COUNT = 4
HIDDEN_SIZE = 2560


class TestDflashLayerIdValidation(CustomTestCase):
    def test_production_ids_resolve_without_shift(self):
        # The Qwen3-VL base setter shifts ids by +1 (input capture); this
        # model must keep them, or captures land on the wrong layers.
        resolved = _validate_dflash_completed_layer_ids(
            PRODUCTION_LAYER_IDS, num_hidden_layers=NUM_HIDDEN_LAYERS
        )
        self.assertEqual(resolved, tuple(PRODUCTION_LAYER_IDS))

    def test_out_of_range_ids_raise(self):
        for bad in ([-1], [NUM_HIDDEN_LAYERS], [0, NUM_HIDDEN_LAYERS - 1, 48]):
            with self.assertRaises(ValueError):
                _validate_dflash_completed_layer_ids(
                    bad, num_hidden_layers=NUM_HIDDEN_LAYERS
                )

    def test_unsorted_or_duplicate_ids_raise(self):
        for bad in ([15, 5], [5, 15, 15], [45, 35, 25, 15, 5]):
            with self.assertRaises(ValueError):
                _validate_dflash_completed_layer_ids(
                    bad, num_hidden_layers=NUM_HIDDEN_LAYERS
                )

    def test_empty_ids_raise(self):
        with self.assertRaises(ValueError):
            _validate_dflash_completed_layer_ids(
                [], num_hidden_layers=NUM_HIDDEN_LAYERS
            )


class TestDflashStreamAveragedCapture(CustomTestCase):
    def test_capture_averages_concatenated_streams(self):
        torch.manual_seed(0)
        tokens = 6
        streams = torch.randn(tokens, HC_COUNT, 8)
        # The stack's widened state concatenates one hidden-wide block per
        # stream (see _widen_streams); build it that way, not by interleave.
        widened = torch.cat([streams[:, i, :] for i in range(HC_COUNT)], dim=-1)

        captured = _dflash_completed_layer_output(widened, HC_COUNT, 8)

        self.assertEqual(tuple(captured.shape), (tokens, 8))
        torch.testing.assert_close(captured, streams.mean(dim=1))

    def test_capture_contracts_production_width(self):
        widened = torch.randn(3, HC_COUNT * HIDDEN_SIZE)
        captured = _dflash_completed_layer_output(widened, HC_COUNT, HIDDEN_SIZE)
        # 5 x 2560 captures pack to the draft fc's 12800-wide input.
        self.assertEqual(tuple(captured.shape), (3, HIDDEN_SIZE))

    def test_capture_rejects_non_widened_state(self):
        with self.assertRaises(AssertionError):
            _dflash_completed_layer_output(
                torch.randn(3, HIDDEN_SIZE), HC_COUNT, HIDDEN_SIZE
            )


class TestDflashTuplePassThrough(CustomTestCase):
    def _vl_model(self, layer_ids):
        model = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
        nn.Module.__init__(model)
        model._dflash_completed_layer_ids = tuple(layer_ids)
        model.last_hc_hidden_states = object()
        return model

    def _run_vl_forward(self, model):
        hidden, aux = torch.randn(2, 8), torch.randn(2, 40)
        forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))
        with mock.patch.object(
            Qwen4ExpModel, "forward", return_value=(hidden, aux)
        ) as forward:
            out = model.forward(
                input_ids=torch.zeros(2, dtype=torch.long),
                positions=torch.arange(2),
                forward_batch=forward_batch,
            )
        forward.assert_called_once()
        return out, hidden, aux

    def test_dflash_tuple_passes_through_unchanged(self):
        # The VL wrapper must not swallow the capture tuple into
        # last_hc_hidden_states (the MTP path); the logits processor
        # unpacks it via capture_aux_hidden_states.
        model = self._vl_model(PRODUCTION_LAYER_IDS)
        out, hidden, aux = self._run_vl_forward(model)
        self.assertIsInstance(out, tuple)
        self.assertIs(out[0], hidden)
        self.assertIs(out[1], aux)
        self.assertIsNone(model.last_hc_hidden_states)

    def test_without_dflash_the_hc_tuple_is_unpacked(self):
        # MTP capture keeps its contract: the wrapper stores the second
        # element as the hc streams and returns the hidden states alone.
        model = self._vl_model([])
        out, hidden, aux = self._run_vl_forward(model)
        self.assertIs(out, hidden)
        self.assertIs(model.last_hc_hidden_states, aux)


class TestDflashCondGenWiring(CustomTestCase):
    def _cond_gen(self):
        vl = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
        nn.Module.__init__(vl)
        vl._dflash_completed_layer_ids = ()
        vl.hidden_size = HIDDEN_SIZE
        vl.config = SimpleNamespace(num_hidden_layers=NUM_HIDDEN_LAYERS)
        model = Qwen4ExpForConditionalGeneration.__new__(
            Qwen4ExpForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.capture_aux_hidden_states = False
        model.pp_group = SimpleNamespace(is_last_rank=True, world_size=1)
        model.model = vl
        return model

    def test_set_dflash_layers_to_capture_configures_model(self):
        model = self._cond_gen()
        # No capture configured: no aux columns preallocated.
        self.assertEqual(model.get_aux_hidden_states_width(), 0)
        model.set_dflash_layers_to_capture(PRODUCTION_LAYER_IDS)
        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(
            model.model._dflash_completed_layer_ids, tuple(PRODUCTION_LAYER_IDS)
        )
        # The decode graph runner sizes its aux buffer from this: 5 layers x
        # 2560 must equal the draft fc's 12800-wide input.
        self.assertEqual(model.get_aux_hidden_states_width(), 5 * HIDDEN_SIZE)

    def test_set_dflash_layers_to_capture_rejects_invalid_ids(self):
        model = self._cond_gen()
        with self.assertRaises(ValueError):
            model.set_dflash_layers_to_capture([5, 15, 48])
        self.assertFalse(model.capture_aux_hidden_states)
        self.assertEqual(model.model._dflash_completed_layer_ids, ())


if __name__ == "__main__":
    unittest.main()
