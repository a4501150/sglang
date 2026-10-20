"""The safetensors PLE offload backend (``--ple-offload-backend safetensors``).

CPU part: a synthetic checkpoint (real safetensors files, decoy tensors,
shuffled index entries) must parse into shard descriptors that locate rows
exactly, reject every structurally broken variant, and plan IO spans that
merge straddling rows and split by arena size with per-row arena offsets
that always point at the row's own bytes.

GPU part: the gatherer reads requested rows back through the full
pipeline (unique -> D2H -> spans -> cuFile batch or preadv -> Triton
scatter) and matches a plain ``numpy`` reference read of the same file:
deduplicated ids, ids outside the vocabulary shard, fp8 rows converted to
bf16, and multi-round gathers past a deliberately tiny arena. The
real-checkpoint test (Qwen3.8-Flash-Next, 95.4 GiB table) runs only where
that model is cached.
"""

from __future__ import annotations

import json
import os
import types
import unittest

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp_ple_safetensors import (
    PleSafetensorsTable,
    plan_io_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

_REAL_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--orcarouter--"
    "Qwen3.8-Flash-Next-Uncensored-NVFP4/snapshots/"
    "c1209bda15a6bbc4c68b585e93d40c0d85f50306"
)

_PREFIX = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding"


def _shard_name(i: int) -> str:
    return f"{_PREFIX}.shard_{i}.weight"


def _write_checkpoint(
    dir_path: str,
    *,
    rows_per_shard: int,
    shards: int,
    dim: int = 8,
    dtype: str = "BF16",
    extra_last_rows: int = 0,
) -> np.ndarray:
    """Write a synthetic PLE checkpoint; returns the row bytes as uint8.

    Row ``i``'s content is its global row id broadcast into the row's
    dtype, so a wrong address never compares equal.
    """
    os.makedirs(dir_path, exist_ok=True)
    total = rows_per_shard * shards + extra_last_rows
    torch_dtype = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}[dtype]
    ids = np.arange(total, dtype=np.int64)
    if dtype == "BF16":
        row_data = np.repeat(ids[:, None], dim, axis=1).astype("<f2")
    else:
        row_data = np.repeat(np.remainder(ids, 200)[:, None], dim, axis=1).astype(
            np.uint8
        )
    raw = row_data.tobytes()
    row_bytes = dim * (2 if dtype == "BF16" else 1)

    files: dict[str, dict] = {}
    for s in range(shards):
        fname = f"model-{s:05d}-of-{shards:05d}.safetensors"
        tensor = (
            torch.frombuffer(
                bytearray(
                    raw[s * rows_per_shard * row_bytes :][: rows_per_shard * row_bytes]
                ),
                dtype=torch.uint8,
            )
            .view(torch_dtype)
            .reshape(rows_per_shard, dim)
        )
        payload = {
            _shard_name(s): tensor,
            # A decoy sharing the file: the loader skips PLE tensors, so the
            # parser must find shards among other tensors.
            f"model.language_model.layers.{s}.self_attn.q_proj.weight": torch.zeros(
                4, 4, dtype=torch.bfloat16
            ),
        }
        save_file(payload, os.path.join(dir_path, fname))
        files[_shard_name(s)] = fname
    if extra_last_rows:
        s = shards - 1
        fname = f"model-{s:05d}-of-{shards:05d}.safetensors"
        start = shards * rows_per_shard - rows_per_shard
        n = rows_per_shard + extra_last_rows
        tensor = (
            torch.frombuffer(
                bytearray(raw[start * row_bytes :][: n * row_bytes]),
                dtype=torch.uint8,
            )
            .view(torch_dtype)
            .reshape(n, dim)
        )
        save_file(
            {
                _shard_name(s): tensor,
                f"model.language_model.layers.{s}.self_attn.q_proj.weight": torch.zeros(
                    4, 4, dtype=torch.bfloat16
                ),
            },
            os.path.join(dir_path, fname),
        )
        files[_shard_name(s)] = fname
    # Shuffled weight map with decoys; the parser must not rely on order.
    weight_map = dict(reversed(sorted(files.items())))
    weight_map["model.language_model.embed_tokens.weight"] = "model-00000.safetensors"
    with open(os.path.join(dir_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return raw


def _cache_dir(name: str) -> str:
    root = os.path.expanduser(
        os.environ.get("SGLANG_CACHE_DIR", os.path.join("~", ".cache", "sglang"))
    )
    path = os.path.join(root, "sglang-test", "ple_safetensors", name)
    os.makedirs(path, exist_ok=True)
    return path


class TestPleSafetensorsTable(unittest.TestCase):
    ROWS = 300
    SHARDS = 4
    DIM = 8

    def setUp(self):
        self.dir = _cache_dir("synthetic")
        self.rows = np.arange(self.ROWS * self.SHARDS, dtype=np.int64)
        self.row_data = _write_checkpoint(
            self.dir, rows_per_shard=self.ROWS, shards=self.SHARDS, dim=self.DIM
        )
        self.table = PleSafetensorsTable.build(
            self.dir, expected_dim=self.DIM, expected_rows=self.ROWS * self.SHARDS
        )

    def test_locate_matches_reference_reads(self):
        ids = np.concatenate(
            [
                np.array([0, 1, self.ROWS - 1, self.ROWS, self.ROWS * self.SHARDS - 1]),
                np.sort(
                    np.random.default_rng(0).integers(
                        0, self.ROWS * self.SHARDS, size=500
                    )
                ),
            ]
        ).astype(np.int64)
        pos, valid, offsets = self.table.locate(ids)
        self.assertTrue(valid.all())
        row_bytes = self.table.row_bytes
        for row_id, shard_pos, off in zip(ids, pos, offsets):
            expect = self.row_data[
                int(row_id) * row_bytes : (int(row_id) + 1) * row_bytes
            ]
            path = self.table.files[int(self.table._file_of[shard_pos])]
            with open(path, "rb") as f:
                f.seek(int(off))
                self.assertEqual(f.read(row_bytes), bytes(expect))

    def test_locate_rejects_rows_past_the_table(self):
        _, valid, _ = self.table.locate(
            np.array([self.ROWS * self.SHARDS + 7], dtype=np.int64)
        )
        self.assertFalse(valid[0])

    def test_invalid_checkpoints_rejected(self):
        cases = {
            "dim": dict(
                expected_dim=self.DIM + 1, expected_rows=self.ROWS * self.SHARDS
            ),
            "rows": dict(
                expected_dim=self.DIM, expected_rows=self.ROWS * self.SHARDS + 1
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    PleSafetensorsTable.build(self.dir, **kwargs)
        # A dropped middle shard breaks consecutiveness.
        broken = _cache_dir("broken")
        _write_checkpoint(
            broken, rows_per_shard=self.ROWS, shards=self.SHARDS, dim=self.DIM
        )
        with open(os.path.join(broken, "model.safetensors.index.json")) as f:
            index = json.load(f)
        index["weight_map"].pop(_shard_name(2))
        with open(os.path.join(broken, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)
        with self.assertRaises(ValueError):
            PleSafetensorsTable.build(
                broken, expected_dim=self.DIM, expected_rows=self.ROWS * self.SHARDS
            )

    def test_short_final_shard_accepted(self):
        extra = 17
        padded = _cache_dir("short_last")
        _write_checkpoint(
            padded,
            rows_per_shard=self.ROWS,
            shards=self.SHARDS,
            dim=self.DIM,
            extra_last_rows=extra,
        )
        table = PleSafetensorsTable.build(
            padded, expected_dim=self.DIM, expected_rows=self.ROWS * self.SHARDS
        )
        self.assertEqual(table.num_rows, self.ROWS * self.SHARDS + extra)


class TestPlanIoSpans(unittest.TestCase):
    ROW_BYTES = 320  # the real table's row size; 12.8 rows per 4 KiB page

    def test_straddling_rows_merge_into_one_span(self):
        # Row 0 at page-1, row 1 past the boundary: aligned mode must read
        # one span, not four boundary IOs.
        offsets = np.array([4000, 4320], dtype=np.int64)
        arena_off, rounds = plan_io_spans(
            np.zeros(2, dtype=np.int64),
            offsets,
            self.ROW_BYTES,
            align_4k=True,
            arena_bytes=1 << 20,
        )
        self.assertEqual(len(rounds), 1)
        _, off_arr, _, size_arr = rounds[0][1]
        self.assertEqual(len(size_arr), 1)
        self.assertEqual(off_arr[0], 0)
        self.assertEqual(size_arr[0], 2 * 4096)
        for off, arena in zip(offsets, arena_off):
            self.assertEqual(arena, off)

    def test_touching_rows_merge_in_compat_mode(self):
        offsets = np.array([1000, 1000 + self.ROW_BYTES], dtype=np.int64)
        _, rounds = plan_io_spans(
            np.zeros(2, dtype=np.int64),
            offsets,
            self.ROW_BYTES,
            align_4k=False,
            arena_bytes=1 << 20,
        )
        self.assertEqual(len(rounds[0][1][3]), 1)
        self.assertEqual(rounds[0][1][3][0], 2 * self.ROW_BYTES)

    def test_rows_in_different_files_do_not_merge(self):
        offsets = np.array([1000, 1000 + self.ROW_BYTES], dtype=np.int64)
        _, rounds = plan_io_spans(
            np.array([0, 1], dtype=np.int64),
            offsets,
            self.ROW_BYTES,
            align_4k=False,
            arena_bytes=1 << 20,
        )
        self.assertEqual(len(rounds[0][1][3]), 2)

    def test_rounds_respect_the_arena(self):
        n = 64
        offsets = np.arange(n, dtype=np.int64) * self.ROW_BYTES
        arena = 8 * self.ROW_BYTES
        arena_off, rounds = plan_io_spans(
            np.zeros(n, dtype=np.int64),
            offsets,
            self.ROW_BYTES,
            align_4k=False,
            arena_bytes=arena,
        )
        total = 0
        seen = np.zeros(n, dtype=bool)
        for rows, (_, _, arena_arr, size_arr) in rounds:
            for span_arena, size in zip(arena_arr, size_arr):
                self.assertLessEqual(span_arena + size, arena)
                total += size
            seen[rows] = True
        self.assertEqual(total, n * self.ROW_BYTES)
        self.assertTrue(seen.all())

    def test_unsorted_offsets_are_sorted_internally(self):
        # Shard files interleave arbitrarily, so a round's rows are not
        # offset-ordered across files; the planner lexsorts, never assumes.
        offsets = np.array([900, 1000], dtype=np.int64)
        arena_off, rounds = plan_io_spans(
            np.array([1, 0], dtype=np.int64),
            offsets,
            self.ROW_BYTES,
            align_4k=False,
            arena_bytes=1 << 20,
        )
        self.assertEqual(len(rounds[0][1][3]), 2)
        # File 0 sorts first: its row (input row 1) takes arena offset 0.
        self.assertEqual(arena_off[1], 0)
        self.assertEqual(arena_off[0], self.ROW_BYTES)


class TestPleSafetensorsGathererGpu(unittest.TestCase):
    """The full gather pipeline against synthetic and real checkpoints.

    Every comparison is byte-exact: the scatter is a pure copy (bf16) or an
    exact widening (fp8 -> bf16), so the reference is the on-disk bytes.
    """

    ROWS = 4096
    SHARDS = 3
    DIM = 160

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        from sglang.srt.models.qwen4_exp_ple_safetensors import (
            PleSafetensorsGatherer,
        )

        cls.Gatherer = PleSafetensorsGatherer
        cls._warm = torch.zeros(8, device="cuda")
        # 1 MiB forces multi-round gathers for everything below.
        envs.SGLANG_QWEN4_PLE_SAFETENSORS_ARENA_MB.set(1)
        cls.dir = _cache_dir("gpu_synthetic")
        cls.row_data = _write_checkpoint(
            cls.dir, rows_per_shard=cls.ROWS, shards=cls.SHARDS, dim=cls.DIM
        )
        cls.table = PleSafetensorsTable.build(
            cls.dir, expected_dim=cls.DIM, expected_rows=cls.ROWS * cls.SHARDS
        )

    @classmethod
    def tearDownClass(cls):
        envs.SGLANG_QWEN4_PLE_SAFETENSORS_ARENA_MB.clear()

    def _gather(self, gatherer, ids, dim=None):
        dim = self.DIM if dim is None else dim
        device_ids = torch.tensor(np.asarray(ids), dtype=torch.long, device="cuda")
        unique_ids, inverse = torch.unique(device_ids, return_inverse=True)
        out = torch.empty(
            (device_ids.numel(), dim), dtype=torch.bfloat16, device="cuda"
        )
        job = gatherer.submit(unique_ids, inverse, out, torch.cuda.current_stream())
        job.done_event.wait()
        if job.error is not None:
            raise job.error
        torch.cuda.synchronize()
        return out, device_ids

    def _reference_rows_bf16(self, row_data, ids, dim, vocab_start=0, vocab_end=None):
        rows = np.frombuffer(row_data, dtype=np.uint8).reshape(-1, dim * 2)
        vocab_end = rows.shape[0] if vocab_end is None else vocab_end
        out = np.zeros((len(ids), dim * 2), dtype=np.uint8)
        for i, row_id in enumerate(ids):
            if vocab_start <= row_id < vocab_end:
                out[i] = rows[row_id]
        return (
            torch.frombuffer(bytearray(out.tobytes()), dtype=torch.uint8)
            .view(torch.bfloat16)
            .reshape(-1, dim)
        )

    def test_roundtrip_matches_reference(self):
        gatherer = self.Gatherer(
            self.table,
            device=torch.device("cuda"),
            vocab_start=0,
            vocab_end=self.ROWS * self.SHARDS,
        )
        self.addCleanup(gatherer.close)
        rng = np.random.default_rng(1)
        ids = rng.integers(0, self.ROWS * self.SHARDS, size=2000)
        # Duplicate ids exercise the unique/inverse dedup path.
        ids[:25] = ids[25:50]
        out, _ = self._gather(gatherer, ids)
        self.assertTrue(
            torch.equal(
                out.cpu(), self._reference_rows_bf16(self.row_data, ids, self.DIM)
            )
        )

    def test_out_of_shard_ids_zero_fill(self):
        gatherer = self.Gatherer(
            self.table,
            device=torch.device("cuda"),
            vocab_start=1000,
            vocab_end=2000,
        )
        self.addCleanup(gatherer.close)
        rng = np.random.default_rng(2)
        ids = rng.integers(0, self.ROWS * self.SHARDS, size=500)
        out, _ = self._gather(gatherer, ids)
        self.assertTrue(
            torch.equal(
                out.cpu(),
                self._reference_rows_bf16(
                    self.row_data, ids, self.DIM, vocab_start=1000, vocab_end=2000
                ),
            )
        )

    def test_multi_round_gather(self):
        # 12k rows against the 1 MiB arena: dozens of rounds, still exact.
        gatherer = self.Gatherer(
            self.table,
            device=torch.device("cuda"),
            vocab_start=0,
            vocab_end=self.ROWS * self.SHARDS,
        )
        self.addCleanup(gatherer.close)
        rng = np.random.default_rng(3)
        ids = rng.integers(0, self.ROWS * self.SHARDS, size=12000)
        out, _ = self._gather(gatherer, ids)
        self.assertTrue(
            torch.equal(
                out.cpu(), self._reference_rows_bf16(self.row_data, ids, self.DIM)
            )
        )

    def test_fp8_rows_convert_to_bf16(self):
        dir_fp8 = _cache_dir("gpu_synthetic_fp8")
        row_data = _write_checkpoint(
            dir_fp8, rows_per_shard=512, shards=2, dim=64, dtype="F8_E4M3"
        )
        table = PleSafetensorsTable.build(
            dir_fp8, expected_dim=64, expected_rows=512 * 2
        )
        gatherer = self.Gatherer(
            table, device=torch.device("cuda"), vocab_start=0, vocab_end=1024
        )
        self.addCleanup(gatherer.close)
        ids = np.arange(0, 1024, 7)
        out, device_ids = self._gather(gatherer, ids, dim=64)
        rows = np.frombuffer(row_data, dtype=np.uint8).reshape(-1, 64)
        unique, inverse = np.unique(ids, return_inverse=True)
        unique_rows = (
            torch.frombuffer(bytearray(rows[unique].tobytes()), dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
            .reshape(-1, 64)
        )
        expect = unique_rows[torch.from_numpy(inverse)]
        self.assertTrue(torch.equal(out.cpu(), expect))

    def test_real_checkpoint_random_rows(self):
        if not os.path.isdir(_REAL_CKPT):
            raise unittest.SkipTest(f"Qwen3.8-Flash-Next not cached at {_REAL_CKPT}")
        table = PleSafetensorsTable.build(
            _REAL_CKPT, expected_dim=160, expected_rows=320001536
        )
        gatherer = self.Gatherer(
            table, device=torch.device("cuda"), vocab_start=0, vocab_end=320001536
        )
        self.addCleanup(gatherer.close)
        rng = np.random.default_rng(4)
        ids = rng.integers(0, 320001536, size=4096)
        out, _ = self._gather(gatherer, ids)
        unique, inverse = np.unique(ids, return_inverse=True)
        rows = np.frombuffer(
            bytearray(_read_rows(table, unique)), dtype=np.uint8
        ).reshape(-1, 320)
        expect = (
            torch.frombuffer(bytearray(rows[inverse].tobytes()), dtype=torch.uint8)
            .view(torch.bfloat16)
            .reshape(-1, 160)
        )
        self.assertTrue(torch.equal(out.cpu(), expect))


def _read_rows(table, unique_ids):
    """Reference read of checkpoint rows via plain pread, in id order."""
    row_bytes = table.row_bytes
    _, _, offsets = table.locate(unique_ids)
    blob = bytearray()
    with open(table.files[0], "rb") as f:
        for off in offsets:
            f.seek(int(off))
            blob += f.read(row_bytes)
    return blob


class _FakeVocabEmbedding(nn.Module):
    """Just enough of VocabParallelEmbedding for the wrapper: TP rank 0 of 1
    owning the whole table, meta weight, the attributes the model reads."""

    def __init__(self, num_rows: int, dim: int, dtype: torch.dtype):
        super().__init__()
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
        )

        self.quant_method = UnquantizedEmbeddingMethod()
        self.quant_config = None
        self.enable_tp = True
        self.use_attn_tp_group = False
        self.tp_size = 1
        self.num_embeddings = num_rows
        self.org_vocab_size = num_rows
        self.padding_size = 0
        self.num_added_embeddings = 0
        self.use_presharded_weights = False
        self.org_vocab_size_padded = num_rows
        self.num_embeddings_padded = num_rows
        self.shard_indices = types.SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=num_rows
        )
        self.embedding_dim = dim
        self.num_embeddings_per_partition = num_rows
        self.num_org_embeddings_per_partition = num_rows
        self.num_added_embeddings_per_partition = 0
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty((num_rows, dim), dtype=dtype, device="meta"),
                requires_grad=False,
            ),
        )
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16))


class TestQwen4ExpSafetensorsEmbeddingGpu(unittest.TestCase):
    """The wrapper the model installs, against the real checkpoint.

    The full model needs the distributed runtime; the wrapper plus a
    duck-typed VocabParallelEmbedding covers the wiring the model touches:
    construction (descriptor build, dtype contract, meta weight), an
    async gather + join_prefetch round trip, and the synchronous forward.
    """

    def _embedding(self, directory):
        from sglang.srt.models.qwen4_exp_ple_safetensors import (
            Qwen4ExpSafetensorsEmbedding,
        )

        emb = _FakeVocabEmbedding(320001536, 160, torch.bfloat16)
        wrapper = Qwen4ExpSafetensorsEmbedding(emb, directory=directory)
        self.addCleanup(wrapper.close)
        return wrapper

    def test_wrapper_roundtrip_real_checkpoint(self):
        if not os.path.isdir(_REAL_CKPT):
            raise unittest.SkipTest(f"Qwen3.8-Flash-Next not cached at {_REAL_CKPT}")
        wrapper = self._embedding(_REAL_CKPT)
        self.assertTrue(wrapper.weight.is_meta)
        self.assertEqual(wrapper._table.num_rows, 320001536)
        rng = np.random.default_rng(9)
        ids = torch.tensor(
            rng.integers(0, 320001536, size=96), dtype=torch.long, device="cuda"
        )
        out = wrapper.gather(ids)  # async: worker owns the IO
        wrapper.join_prefetch()
        torch.cuda.synchronize()
        unique, inverse = np.unique(ids.cpu().numpy(), return_inverse=True)
        expect = (
            torch.frombuffer(
                bytearray(_read_rows(wrapper._table, unique)), dtype=torch.uint8
            )
            .view(torch.bfloat16)
            .reshape(-1, 160)
        )[torch.from_numpy(inverse)]
        self.assertTrue(torch.equal(out.cpu(), expect))
        # The synchronous forward path gathers, joins and (TP=1) reduces.
        out2 = wrapper(ids)
        self.assertTrue(torch.equal(out2.cpu(), expect))
        # Out-of-table ids zero-fill, matching the pinned backend.
        bad = torch.tensor([320001536 + 5], dtype=torch.long, device="cuda")
        self.assertTrue(
            torch.equal(wrapper(bad).cpu(), torch.zeros(1, 160, dtype=torch.bfloat16))
        )

    def test_wrapper_rejects_dtype_mismatch(self):
        if not os.path.isdir(_REAL_CKPT):
            raise unittest.SkipTest(f"Qwen3.8-Flash-Next not cached at {_REAL_CKPT}")
        from sglang.srt.models.qwen4_exp_ple_safetensors import (
            Qwen4ExpSafetensorsEmbedding,
        )

        emb = _FakeVocabEmbedding(320001536, 160, torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            wrapper = Qwen4ExpSafetensorsEmbedding(emb, directory=_REAL_CKPT)
            wrapper.close()


if __name__ == "__main__":
    unittest.main()
