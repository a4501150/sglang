import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_utils import _verify_commit_step_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


# The decode checkpoint grid is lcm(mamba_cache_chunk_size, tree page,
# interval); keeping all three equal leaves it at the interval under test.
TRACK_INTERVAL = 4


def _make_batch() -> tuple[Req, ScheduleBatch]:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.kv.kv_committed_len = 2

    batch = ScheduleBatch(reqs=[req])
    batch.tree_cache = SimpleNamespace(page_size=TRACK_INTERVAL)
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    return req, batch


def _make_processor() -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=SimpleNamespace(page_size=TRACK_INTERVAL),
        hisparse_coordinator=None,
        beam_coordinator=MagicMock(),
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _make_result():
    return GenerationBatchResult(
        logits_output=SimpleNamespace(
            hidden_states=None, customized_info=None, sampling_mask_output=None
        ),
        next_token_ids=[4],
        speculative_num_draft_tokens=0,
    )


class TestMambaBoundaryMaskReuse(unittest.TestCase):
    def test_overlap_scheduler_handles_zero_and_one_batch_lookahead(self):
        cases = (
            (False, False, 0, 1),
            (True, False, 1, 0),
            (True, True, 1, 1),
        )
        for (
            schedule_next_decode,
            disable_second_overlap,
            expected_lookahead,
            expected_recovery_waits,
        ) in cases:
            with self.subTest(
                schedule_next_decode=schedule_next_decode,
                disable_second_overlap=disable_second_overlap,
            ):
                req, batch = _make_batch()
                processor = _make_processor()
                result = _make_result()

                scheduler = Scheduler.__new__(Scheduler)
                scheduler.gracefully_exit = False
                scheduler.ingest_requests = MagicMock(
                    side_effect=[[], [], StopIteration]
                )
                scheduler.process_input_requests = MagicMock()
                scheduler._engine_paused = False
                scheduler.running_batch = batch
                scheduler.is_disable_overlap_for_batch = MagicMock(
                    side_effect=lambda *_args, **_kwargs: (
                        disable_second_overlap and plan_count == 2
                    )
                )
                scheduler.run_batch = MagicMock(return_value=result)
                scheduler._apply_war_barrier = MagicMock()
                scheduler.model_worker = MagicMock()
                scheduler.enable_unified_memory = False
                scheduler.is_generation = False
                scheduler.last_batch = None

                plan_count = 0

                def get_next_batch_to_run(*, running_batch, last_batch):
                    nonlocal plan_count
                    del running_batch, last_batch
                    plan_count += 1
                    if plan_count == 1:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    if plan_count == 2 and schedule_next_decode:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    return SimpleNamespace(
                        running_batch=batch,
                        batch_to_run=None,
                    )

                scheduler.get_next_batch_to_run = get_next_batch_to_run
                observed_lookahead = []

                def process_batch_result(result_batch, batch_result):
                    observed_lookahead.append(
                        req.decode_batch_idx
                        - result_batch.mamba_decode_batch_idx_cpu[0]
                    )
                    processor.process_batch_result_decode(result_batch, batch_result)

                scheduler.process_batch_result = process_batch_result

                with (
                    # The mamba predicates and the track interval read the
                    # published bags, so publish the configuration under test
                    # (non-lazy extra buffer, interval 4); observability and
                    # disagg reads are served by the same publish at their
                    # defaults.
                    get_context().override_server_args(
                        mamba_radix_cache_strategy="extra_buffer",
                        mamba_track_interval=TRACK_INTERVAL,
                        _mamba_cache_chunk_size=TRACK_INTERVAL,
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_decode",
                        return_value=torch.tensor([3], dtype=torch.int64),
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs"
                    ),
                    patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
                    patch.object(
                        SchedulerBatchResultProcessor,
                        "_mamba_prefix_cache_update",
                    ) as cache_update,
                ):
                    with self.assertRaises(StopIteration):
                        scheduler.event_loop_overlap()

                self.assertEqual(observed_lookahead, [expected_lookahead])
                self.assertEqual(
                    scheduler.model_worker.wait_for_pending_state_recovery.call_count,
                    expected_recovery_waits,
                )
                if expected_lookahead == 0:
                    cache_update.assert_not_called()
                else:
                    self.assertTrue(cache_update.call_args.kwargs["known_boundary"])

    def test_normal_scheduler_waits_before_processing_result(self):
        _, batch = _make_batch()
        result = _make_result()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler.ingest_requests = MagicMock(side_effect=[[], StopIteration])
        scheduler._engine_paused = False
        scheduler.running_batch = batch
        scheduler.get_next_batch_to_run = MagicMock(
            return_value=SimpleNamespace(running_batch=batch, batch_to_run=batch)
        )
        scheduler.run_batch = MagicMock(return_value=result)
        scheduler.last_batch = None
        events = []
        scheduler.model_worker = MagicMock()
        scheduler.model_worker.wait_for_pending_state_recovery.side_effect = lambda: (
            events.append("wait")
        )
        scheduler.process_batch_result = MagicMock(
            side_effect=lambda *_args: events.append("process")
        )

        with self.assertRaises(StopIteration):
            scheduler.event_loop_normal()

        self.assertEqual(events, ["wait", "process"])

    def test_deferred_mamba_cow_joins_recovery_before_pool_reads(self):
        class FakeHybridReqToTokenPool:
            def __init__(self):
                self.mamba_pool = MagicMock()
                self.mamba_ckpt_pool = None

            def translate_mamba_indices(self, indices):
                return indices

            def wait_for_hicache_load_complete(self):
                events.append("wait_hicache")

            def copy_mamba_state(self, src, dst):
                events.append("copy")

        events = []
        runner = ModelRunner.__new__(ModelRunner)
        runner.req_to_token_pool = FakeHybridReqToTokenPool()
        runner.is_draft_worker = False
        runner.attn_backend = MagicMock()
        runner.attn_backend.join_pending_state_recovery.side_effect = lambda: (
            events.append("join")
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
            mamba_clear_indices=None,
            mamba_cow_src_indices=torch.tensor([1]),
            mamba_cow_dst_indices=torch.tensor([2]),
        )

        with patch(
            "sglang.srt.model_executor.model_runner.HybridReqToTokenPool",
            FakeHybridReqToTokenPool,
        ):
            runner._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

        self.assertEqual(events, ["join", "wait_hicache", "copy"])
        self.assertIsNone(forward_batch.mamba_cow_src_indices)
        self.assertIsNone(forward_batch.mamba_cow_dst_indices)


class TestSpecVerifyCommitTrackStep(unittest.TestCase):
    """Spec-verify commit index math: an interval crossing in the MIDDLE of
    the accepted path must track the state where the sequence reaches the
    boundary token, not the state one accepted step later."""

    def _commit_step_indices(self, seq_lens_pre, accept_lens, interval):
        bs = len(accept_lens)
        draft_token_num = max(accept_lens)
        # Linear accepted chain: node i of req b is slot b * draft_token_num + i.
        accept_index = torch.arange(bs * draft_token_num).reshape(bs, draft_token_num)
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(page_size=interval),
            mamba_track_indices=object(),
            seq_lens=torch.tensor(seq_lens_pre, dtype=torch.int64),
        )
        last, track = _verify_commit_step_indices(
            batch=batch,
            accept_index=accept_index,
            accept_lens=torch.tensor(accept_lens, dtype=torch.int64),
            draft_token_num=draft_token_num,
        )
        return last.tolist(), track.tolist()

    def test_mid_path_crossing_tracks_boundary_step(self):
        interval = 64
        with get_context().override_server_args(
            mamba_track_interval=interval,
            _mamba_cache_chunk_size=interval,
        ):
            # req0 crosses token 64 after two of five accepted tokens: the
            # checkpoint is accepted step 1 (state at seq_len 64), not step 2.
            # req1 never crosses (already past the grid line). req2 crosses
            # exactly at its last accepted token.
            last, track = self._commit_step_indices(
                seq_lens_pre=[62, 64, 59],
                accept_lens=[5, 3, 5],
                interval=interval,
            )
        self.assertEqual(last, [4, 2, 4])
        self.assertEqual(track, [1, -1, 4])

    def test_boundary_before_first_accepted_token_clamps_to_zero(self):
        interval = 64
        with get_context().override_server_args(
            mamba_track_interval=interval,
            _mamba_cache_chunk_size=interval,
        ):
            # pre=63: the boundary token is the first accepted token, so the
            # tracked step is 0.
            _last, track = self._commit_step_indices(
                seq_lens_pre=[63],
                accept_lens=[2],
                interval=interval,
            )
        self.assertEqual(track, [0])


if __name__ == "__main__":
    unittest.main()
