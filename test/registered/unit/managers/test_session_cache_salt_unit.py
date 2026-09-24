"""Tests that session identity does not implicitly isolate prefix caches."""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase


def _make_req(**kwargs):
    return Req(
        "test-rid",
        "",
        array("q", [1, 2, 3]),
        SamplingParams(),
        **kwargs,
    )


def _make_tokenized_req(*, rid="turn-1", session_id="legacy-session", cache_salt=None):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [4, 5, 6]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=session_id),
        cache_salt=cache_salt,
    )


class TestReqCacheSaltPolicy(CustomTestCase):
    def test_no_salt_stays_shared(self):
        self.assertIsNone(_make_req().cache_salt)
        self.assertIsNone(_make_req(cache_salt="").cache_salt)

    def test_session_ids_stay_in_shared_namespace(self):
        first = _make_req(session_id="session-a")
        second = _make_req(session_id="session-b")

        self.assertIsNone(first.cache_salt)
        self.assertIsNone(second.cache_salt)

    def test_legacy_session_objects_stay_in_shared_namespace(self):
        first = _make_req(
            session=Session(capacity_of_str_len=1024, session_id="session-a")
        )
        second = _make_req(
            session=Session(capacity_of_str_len=1024, session_id="session-b")
        )

        self.assertIsNone(first.cache_salt)
        self.assertIsNone(second.cache_salt)

    def test_explicit_salt_is_preserved(self):
        self.assertEqual(
            _make_req(cache_salt="tenant-a", session_id="session-a").cache_salt,
            "tenant-a",
        )


class TestLegacySessionCreateReqCacheSaltPolicy(CustomTestCase):
    def test_different_sessions_stay_in_shared_namespace(self):
        first_session = Session(capacity_of_str_len=1024, session_id="session-a")
        second_session = Session(capacity_of_str_len=1024, session_id="session-b")

        first = first_session.create_req(
            _make_tokenized_req(session_id="session-a"),
            tokenizer=None,
            vocab_size=1000,
        )
        second = second_session.create_req(
            _make_tokenized_req(session_id="session-b"),
            tokenizer=None,
            vocab_size=1000,
        )

        self.assertIsNone(first.cache_salt)
        self.assertIsNone(second.cache_salt)

    def test_explicit_salt_is_preserved(self):
        session = Session(capacity_of_str_len=1024, session_id="session-a")
        req = session.create_req(
            _make_tokenized_req(cache_salt="tenant-a"),
            tokenizer=None,
            vocab_size=1000,
        )

        self.assertEqual(req.cache_salt, "tenant-a")


if __name__ == "__main__":
    unittest.main()
