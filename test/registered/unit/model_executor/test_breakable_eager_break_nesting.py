"""Eager-break nesting and capture-exit cleanup in the breakable CUDA graph.

A nested ``eager_on_graph`` call inside an enclosing break body must run
eagerly without ending a second segment (the enclosing break already closed
it), and a capture whose break body raised must let the original exception
propagate instead of tripping the missing-segment assert in ``__exit__``.
Both were found by server-boot failures on Qwen4-Exp's nested PLE breaks; the
first is proven here without a GPU by driving the wrapper against a stub
capture.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _current_capture_var,
    _current_stream_var,
    _forked_streams_var,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _stub_capture():
    return SimpleNamespace(
        _end_current_segment=mock.Mock(),
        _begin_new_segment=mock.Mock(),
        _barrier_fn=None,
        cuda_graph=SimpleNamespace(_break_fns=[]),
    )


class TestEagerBreakNesting(CustomTestCase):
    def test_nested_break_ends_only_one_segment(self):
        capture = _stub_capture()

        @eager_on_graph(True)
        def inner(x):
            return x * 2

        @eager_on_graph(True)
        def outer(x):
            # The nested decorator must see the outer break's token and run
            # this eagerly instead of ending a second segment.
            return inner(x) + 1

        token = _current_capture_var.set(capture)
        try:
            self.assertEqual(outer(3), 7)
        finally:
            _current_capture_var.reset(token)

        capture._end_current_segment.assert_called_once()
        capture._begin_new_segment.assert_called_once()
        self.assertEqual(len(capture.cuda_graph._break_fns), 1)

    def test_eager_calls_outside_capture_run_undecorated(self):
        @eager_on_graph(True)
        def inner(x):
            return x + 1

        # No capture is active: the wrapper must not touch any capture state.
        self.assertEqual(inner(2), 3)

    def test_capture_exit_preserves_error_after_failed_break(self):
        # A break body that raised already ended its segment and never began
        # the next one; __exit__ must not end a missing segment (whose assert
        # would mask the in-flight exception) and must still unwind cleanly.
        # __exit__ resets the three contextvars itself; the fake supplies the
        # tokens it must reset.
        capture = SimpleNamespace(
            _current_graph=None,
            _end_current_segment=mock.Mock(),
            _capture_token=_current_capture_var.set(None),
            _stream_token=_current_stream_var.set(None),
            _forked_token=_forked_streams_var.set(set()),
            _stream_ctx=None,
        )
        self.assertFalse(BreakableCUDAGraphCapture.__exit__(capture, None, None, None))
        capture._end_current_segment.assert_not_called()

        # The happy path (an open segment) still ends the segment on exit.
        capture2 = SimpleNamespace(
            _current_graph=object(),
            _end_current_segment=mock.Mock(),
            _capture_token=_current_capture_var.set(None),
            _stream_token=_current_stream_var.set(None),
            _forked_token=_forked_streams_var.set(set()),
            _stream_ctx=None,
        )
        self.assertFalse(BreakableCUDAGraphCapture.__exit__(capture2, None, None, None))
        capture2._end_current_segment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
