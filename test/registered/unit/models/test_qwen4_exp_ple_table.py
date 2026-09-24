"""File-backed host storage for the offloaded Qwen4-Exp PLE table.

CPU part: the allocator builds a sparse file of exactly the table's size, hands
back a tensor with the requested shape/dtype whose writes land in the file and
survive a re-open, reuses the file across calls, replaces one of the wrong size,
and the prefetcher computes the right page set and honours its size floor. The
resident-set trimmer measures only its own mapping, drops its pages once over
budget without losing what was written through them, and is off when the budget
is zero or the mapping is pinned.

GPU part (skipped unless the device reads pageable host memory through the host
page tables, i.e. unified-memory parts such as GB10): the production Triton
gather kernel reading from the file-backed table matches a torch gather.

Direct safetensors part (SGLANG_QWEN4_PLE_SAFETENSORS /
Qwen4ExpSafetensorsEmbedding, which copies rows to the device and needs no
pageable GPU access): checkpoint index/header parsing into validated numeric
shard descriptors (multi-file, nonuniform, lexical-order shards, BF16 and
F8_E4M3), exact pread and grouped-mmap gathers by global row id with bounded
mapped residency, meta-device pre-allocation, mutual exclusion with built-in
offload, and skipping checkpoint PLE shards only for that backend.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.qwen4_exp_ple_table import (
    PleFilePrefetcher,
    PleFileRssTrimmer,
    _mapping_rss_bytes,
    allocate_ple_host_table,
    default_ple_table_dir,
    device_uses_host_page_tables,
    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
    ple_table_file_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestPleFileTableAllocator(CustomTestCase):
    def test_file_is_sparse_and_sized_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((1000, 160), torch.float8_e4m3fn, "file", d)
            path = os.path.join(
                d, ple_table_file_name((1000, 160), torch.float8_e4m3fn)
            )
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.getsize(path), 1000 * 160)
            self.assertEqual(tuple(table.shape), (1000, 160))
            self.assertEqual(table.dtype, torch.float8_e4m3fn)
            # Sparse: nothing written yet, so (almost) no blocks allocated.
            self.assertLess(os.stat(path).st_blocks * 512, 64 * 1024)

    def test_writes_persist_and_file_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            shape, dtype = (64, 32), torch.bfloat16
            table = allocate_ple_host_table(shape, dtype, "file", d)
            row = torch.arange(32, dtype=torch.float32).to(dtype)
            table[7].copy_(row)  # what the weight loader does, row by row
            del table
            again = allocate_ple_host_table(shape, dtype, "file", d)
            self.assertTrue(torch.equal(again[7].float(), row.float()))
            self.assertEqual(len(os.listdir(d)), 1)

    def test_wrong_sized_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ple_table_file_name((8, 8), torch.bfloat16))
            with open(path, "wb") as f:
                f.write(b"\x01" * 10)
            table = allocate_ple_host_table((8, 8), torch.bfloat16, "file", d)
            self.assertEqual(os.path.getsize(path), 8 * 8 * 2)
            self.assertEqual(tuple(table.shape), (8, 8))

    def test_tag_separates_tensor_parallel_shards(self):
        with tempfile.TemporaryDirectory() as d:
            a = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows0-8"
            )
            b = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows8-16"
            )
            a.fill_(1.0)
            self.assertEqual(len(os.listdir(d)), 2)
            self.assertTrue(torch.all(b.float() == 0.0))
            self.assertIn(
                "rows8-16", ple_table_file_name((8, 8), torch.bfloat16, "rows8-16")
            )

    def test_default_dir_is_per_checkpoint(self):
        with mock.patch.dict(os.environ, {"SGLANG_QWEN4_PLE_FILE_DIR": "/cache/ple"}):
            a = default_ple_table_dir("RadixArk/Qwen3.8-Flash-Next-NVFP4")
            b = default_ple_table_dir("/root/.cache/huggingface/flashnext-fp8/")
            self.assertEqual(a, "/cache/ple/RadixArk_Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(b, "/cache/ple/root_.cache_huggingface_flashnext-fp8")
            self.assertNotEqual(a, b)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            allocate_ple_host_table((4, 4), torch.bfloat16, "nvme", None)

    def test_pinned_backend_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertTrue(table.is_pinned())
        self.assertIsNone(make_ple_file_prefetcher(table))


class TestPleFilePrefetcher(CustomTestCase):
    def test_page_set_covers_row_start_and_end(self):
        # 160-byte rows: row 25 spans bytes 4000-4159, i.e. pages 0 and 1.
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([25, 0]), 160)
        self.assertEqual(pages, [0, 1])
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([1000, 1000]), 160)
        self.assertEqual(pages, [39])  # dedup, single page

    def test_enqueue_uses_local_tp_offsets_and_ignores_other_shards(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            with open(path, "wb") as f:
                f.truncate(8192)
            pf = PleFilePrefetcher(path, row_bytes=160, min_rows=1)
            try:
                with mock.patch("os.posix_fadvise") as fadvise:
                    self.assertFalse(
                        pf.enqueue(
                            torch.tensor([999, 1032]), vocab_start=1000, vocab_end=1032
                        )
                    )
                    self.assertTrue(
                        pf.enqueue(
                            torch.tensor([999, 1000, 1025, 1032]),
                            vocab_start=1000,
                            vocab_end=1032,
                        )
                    )
                    pf._pool.shutdown(wait=True)
                    self.assertEqual(
                        sorted(c.args[1] for c in fadvise.call_args_list), [0, 4096]
                    )
            finally:
                pf.close()

    def test_enqueue_respects_min_rows_and_advises_pages(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            with open(path, "wb") as f:
                f.truncate(1 << 20)
            pf = PleFilePrefetcher(path, row_bytes=160, min_rows=4)
            try:
                self.assertFalse(pf.enqueue(torch.tensor([1, 2, 3])))
                with mock.patch("os.posix_fadvise") as fadvise:
                    self.assertTrue(pf.enqueue(torch.tensor([0, 1, 2, 30])))
                    pf._pool.shutdown(wait=True)
                    offsets = sorted(c.args[1] for c in fadvise.call_args_list)
                    # rows 0-2 live in page 0; row 30 (bytes 4800-4959) in page 1
                    self.assertEqual(offsets, [0, 4096])
            finally:
                pf.close()


class TestSmapsParsing(CustomTestCase):
    """The parser runs anywhere; only the live mapping needs Linux."""

    SMAPS = """00400000-00401000 r--p 00000000 08:01 1  /usr/bin/x
Size:                  4 kB
Rss:                   4 kB
7f0000000000-7f0004000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:               32768 kB
7f0004000000-7f0008000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:                1024 kB
7f0100000000-7f0100001000 rw-p 00000000 00:00 0
Size:                  4 kB
Rss:                   4 kB
"""

    def _rss(self, addr, nbytes):
        with tempfile.NamedTemporaryFile("w", suffix=".smaps", delete=False) as f:
            f.write(self.SMAPS)
            path = f.name
        try:
            return _mapping_rss_bytes(addr, nbytes, smaps_path=path)
        finally:
            os.unlink(path)

    def test_sums_every_vma_of_the_table_and_nothing_else(self):
        # The table spans both of its VMAs; the unrelated ones must not count.
        self.assertEqual(self._rss(0x7F0000000000, 0x8000000), (32768 + 1024) * 1024)

    def test_counts_a_partially_overlapping_vma(self):
        # A range ending inside the first VMA still needs that VMA's pages.
        self.assertEqual(self._rss(0x7F0000000000, 0x1000), 32768 * 1024)

    def test_ignores_unrelated_mappings(self):
        self.assertEqual(self._rss(0x7F0200000000, 0x1000), 0)

    def test_missing_smaps_is_reported_as_unknown(self):
        self.assertIsNone(
            _mapping_rss_bytes(0x1000, 0x1000, smaps_path="/nonexistent/smaps")
        )


class TestPleFileRssTrimmerConfig(CustomTestCase):
    def test_budget_zero_disables_the_trimmer(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ, {"SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "0"}
            ):
                self.assertIsNone(make_ple_file_rss_trimmer(table))

    def test_pinned_table_has_no_trimmer(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertIsNone(make_ple_file_rss_trimmer(table))


@unittest.skipUnless(
    os.path.exists("/proc/self/smaps"),
    "the resident set of a mapping is only readable on Linux",
)
class TestPleFileRssTrimmer(CustomTestCase):
    # 64 MiB: large enough that the Rss of the mapping stands out, small
    # enough to write in a CPU test.
    SHAPE = (32768, 1024)
    NBYTES = 32768 * 1024 * 2

    def _trimmer(self, table, budget_bytes):
        return PleFileRssTrimmer(
            addr=table.data_ptr(),
            nbytes=self.NBYTES,
            budget_bytes=budget_bytes,
            interval_s=3600.0,
            chunk_bytes=16 << 20,  # several chunks, as in production
        )

    def test_measures_its_own_mapping_only(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            trimmer = self._trimmer(table, 0)
            empty = trimmer.mapping_rss_bytes()
            table.fill_(1.0)  # touches every page
            touched = trimmer.mapping_rss_bytes()
            self.assertIsNotNone(touched)
            self.assertGreater(touched, empty)
            self.assertGreater(touched, self.NBYTES // 2)
            # Never the whole process: only the VMAs backing this table.
            self.assertLessEqual(touched, self.NBYTES + (16 << 20))

    def test_over_budget_drops_pages_and_keeps_the_data(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            table[7][3] = 2.0
            trimmer = self._trimmer(table, budget_bytes=1 << 20)
            before = trimmer.mapping_rss_bytes()
            freed = trimmer.trim_once()
            after = trimmer.mapping_rss_bytes()
            self.assertGreater(freed, 0)
            self.assertLess(after, before // 2)
            # MADV_DONTNEED on a shared file mapping drops the page-table
            # entries, not the page cache: the writes are still there.
            self.assertEqual(table[7][3].item(), 2.0)
            self.assertEqual(table[9][3].item(), 1.0)

    def test_under_budget_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            trimmer = self._trimmer(table, budget_bytes=self.NBYTES * 4)
            before = trimmer.mapping_rss_bytes()
            self.assertEqual(trimmer.trim_once(), 0)
            self.assertEqual(trimmer.mapping_rss_bytes(), before)

    def test_factory_starts_and_stops_a_thread(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "1",
                    "SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S": "3600",
                },
            ):
                trimmer = make_ple_file_rss_trimmer(table)
            self.assertIsNotNone(trimmer)
            try:
                self.assertTrue(trimmer._thread.is_alive())
            finally:
                trimmer.close()
                trimmer._thread.join(timeout=5)
            self.assertFalse(trimmer._thread.is_alive())


@unittest.skipUnless(
    torch.cuda.is_available() and device_uses_host_page_tables(0) is True,
    "needs a device that reads pageable host memory through the host page tables",
)
class TestPleFileTableGatherOnDevice(CustomTestCase):
    def test_triton_gather_reads_file_backed_table(self):
        import triton

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        rows, dim = 4096, 160
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((rows, dim), torch.bfloat16, "file", d)
            table.copy_(torch.randn(rows, dim).to(torch.bfloat16))
            ids = torch.randint(0, rows, (2048,), device="cuda")
            out = torch.empty(2048, dim, dtype=torch.bfloat16, device="cuda")
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                table.data_ptr(),
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=rows,
                is_fp8=False,
                BLOCK_D=triton.next_power_of_2(dim),
            )
            torch.cuda.synchronize()
            expected = table[ids.cpu()].to("cuda")
            self.assertTrue(torch.equal(out, expected))


def _shard_tensor(rows, dim, row_base, dtype=torch.bfloat16):
    """Shard whose global row r is filled with the bf16-exact value r."""
    r = torch.arange(row_base, row_base + rows, dtype=torch.int32).unsqueeze(1)
    return (r + torch.zeros(rows, dim, dtype=torch.int32)).to(dtype)


def _write_safetensors(path, tensors):
    from safetensors.torch import save_file

    save_file({name: t.contiguous() for name, t in tensors.items()}, path)


def _write_index(directory, weight_map):
    import json

    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


_PLE_PREFIX = "model.layers.0.ple.ple_embedding.ngram_embedding"


def _ple_name(shard):
    return f"{_PLE_PREFIX}.shard_{shard}.weight"


def _write_direct_checkpoint(
    directory,
    layout,
    *,
    dtype=torch.bfloat16,
    row_bases,
    dim=4,
    with_index=True,
    extra_tensors=None,
):
    """Write a tiny PLE checkpoint. layout: file -> [shard index]; row_bases:
    shard -> (global first row, row count). Returns the written weight map."""
    weight_map = {}
    for file_name, shards in layout.items():
        tensors = {
            _ple_name(s): _shard_tensor(row_bases[s][1], dim, row_bases[s][0], dtype)
            for s in shards
        }
        if extra_tensors:
            tensors.update(
                {
                    name: tensor
                    for name, (target_file, tensor) in extra_tensors.items()
                    if target_file == file_name
                }
            )
        _write_safetensors(os.path.join(directory, file_name), tensors)
        for s in shards:
            weight_map[_ple_name(s)] = file_name
    for name, (file_name, _tensor) in (extra_tensors or {}).items():
        weight_map[name] = file_name
    if with_index:
        _write_index(directory, weight_map)
    return weight_map


# Small multi-file fixture: 23 table rows in shard slots of 2 rows, shard ids
# consecutive from 0 with a one-row final tensor (exactly the short tail that
# ceil(23 / 2) shard geometry implies). File 1 starts at shard 1 with shards
# 10-11, so file 1 does not hold row 0 and the files interleave. Mirrors the
# real checkpoint, where the physical order inside a file follows the lexical
# order of tensor names (shard 10 right after shard 1, never numeric order).
_DIRECT_LAYOUT = {
    "model_00001-of-00002.safetensors": [1, 10, 11],
    "model_00002-of-00002.safetensors": [0, 2, 3, 4, 5, 6, 7, 8, 9],
}
_DIRECT_ROW_BASES = {i: (2 * i, 2) for i in range(11)}
_DIRECT_ROW_BASES[11] = (22, 1)  # short final tensor implied by ceil geometry
_DIRECT_ROWS = 23


class TestPleSafetensorsTable(CustomTestCase):
    """build_ple_safetensors_table + PleSafetensorsTable.read_rows (CPU)."""

    def _build(self, directory, **kwargs):
        from sglang.srt.models.qwen4_exp_ple_table import (
            build_ple_safetensors_table,
        )

        params = dict(shard_size=2)
        params.update(kwargs)
        return build_ple_safetensors_table(directory, **params)

    def test_descriptors_come_from_headers_not_position(self):
        import struct

        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            table = self._build(d)
            self.assertEqual(table.dtype, torch.bfloat16)
            self.assertEqual(table.dim, 4)
            self.assertEqual(table.row_bytes, 8)
            self.assertEqual(table.num_rows, _DIRECT_ROWS)
            by_index = {s.shard_index: s for s in table.shards}
            self.assertEqual([s.row_start for s in table.shards], list(range(0, 24, 2)))
            self.assertEqual(by_index[5].rows, 2)
            self.assertEqual(by_index[11].rows, 1)  # short tail preserved
            # Offsets and file assignments match each file's own header entry,
            # regardless of the (lexical) physical order within the file.
            for shard in table.shards:
                path = table.files[shard.file_index]
                with open(path, "rb") as f:
                    (header_len,) = struct.unpack("<Q", f.read(8))
                    header = json.loads(f.read(header_len))
                entry = header[_ple_name(shard.shard_index)]
                begin, end = (8 + header_len + o for o in entry["data_offsets"])
                self.assertEqual(shard.offset, begin)
                self.assertEqual(shard.rows * table.row_bytes, end - begin)
            table.close()

    def test_index_free_scan_matches_indexed_build(self):
        with (
            tempfile.TemporaryDirectory() as d_indexed,
            tempfile.TemporaryDirectory() as d_scan,
        ):
            _write_direct_checkpoint(
                d_indexed, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES
            )
            _write_direct_checkpoint(
                d_scan, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES, with_index=False
            )
            a, b = self._build(d_indexed), self._build(d_scan)
            ka = [
                (s.row_start, s.rows, os.path.basename(a.files[s.file_index]), s.offset)
                for s in a.shards
            ]
            kb = [
                (s.row_start, s.rows, os.path.basename(b.files[s.file_index]), s.offset)
                for s in b.shards
            ]
            self.assertEqual(ka, kb)
            a.close()
            b.close()

    def test_read_rows_is_exact_across_files_gaps_and_both_paths(self):
        expected = {}  # global row -> value (row id), rows past the end read zero
        for shard, (base, rows) in _DIRECT_ROW_BASES.items():
            for r in range(rows):
                expected[base + r] = float(base + r)
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            table = self._build(d)
            for ids in (
                [0, 1, 10, 11, 23, 20, 22],  # pread path: file + shard borders
                list(range(0, 23)) * 4,  # 92 rows: grouped mmap path
            ):
                data = table.read_rows(torch.tensor(ids, dtype=torch.int64))
                self.assertEqual(tuple(data.shape), (len(ids), table.row_bytes))
                vals = data.view(torch.bfloat16).float()
                for i, row in enumerate(ids):
                    want = expected.get(row, 0.0)
                    self.assertTrue(
                        torch.equal(vals[i], torch.full((4,), want)),
                        f"row {row} of a {len(ids)}-row gather",
                    )
            table.close()

    def test_bulk_gather_starts_resident_set_trimmer(self):
        rows, dim, shard_size = 600, 4, 50
        layout = {"model_00001-of-00001.safetensors": list(range(rows // shard_size))}
        bases = {i: (i * shard_size, shard_size) for i in range(rows // shard_size)}
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, layout, row_bases=bases, dim=dim)
            table = self._build(d, shard_size=shard_size)
            with (
                mock.patch(
                    "sglang.srt.models.qwen4_exp_ple_table._mapping_rss_bytes",
                    return_value=0,
                ),
                mock.patch.object(PleFileRssTrimmer, "start") as start,
            ):
                data = table.read_rows(torch.arange(rows, dtype=torch.int64))
            self.assertEqual(tuple(data.shape), (rows, dim * 2))
            self.assertEqual(len(table._trimmers), 1)
            start.assert_called_once_with()
            table.close()

    def test_f8_e4m3_is_supported_byte_exact(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(
                d,
                _DIRECT_LAYOUT,
                row_bases=_DIRECT_ROW_BASES,
                dtype=torch.float8_e4m3fn,
            )
            table = self._build(d)
            self.assertEqual(table.dtype, torch.float8_e4m3fn)
            data = table.read_rows(torch.tensor([0, 5, 22]))
            expected = _shard_tensor(23, 4, 0, torch.float8_e4m3fn)
            self.assertTrue(
                torch.equal(
                    data.view(torch.float8_e4m3fn).to(torch.bfloat16),
                    expected[[0, 5, 22]].to(torch.bfloat16),
                )
            )
            table.close()

    def _expect_reject(self, directory, regex):
        with self.assertRaisesRegex(ValueError, regex):
            self._build(directory)

    def test_unsupported_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(
                d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES, dtype=torch.float32
            )
            self._expect_reject(d, "unsupported PLE dtype")

    def test_dim_disagreement_between_shards_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            # Same file, same index, but shard 3 was written with a wider dim.
            tensors = {
                _ple_name(s): _shard_tensor(r, 4, b)
                for s, (b, r) in _DIRECT_ROW_BASES.items()
                if s in (0, 2, 4, 5)
            }
            tensors[_ple_name(3)] = _shard_tensor(2, 8, 6)
            _write_safetensors(
                os.path.join(d, "model_00002-of-00002.safetensors"), tensors
            )
            self._expect_reject(d, "dim 8 disagrees")

    def test_expected_dim_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            with self.assertRaisesRegex(ValueError, "does not match embedding_dim"):
                self._build(d, expected_dim=6)

    def test_truncated_shard_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            with open(os.path.join(d, "model_00002-of-00002.safetensors"), "r+b") as f:
                f.truncate(os.path.getsize(f.name) - 1)
            self._expect_reject(d, "do not cover")

    def test_duplicate_shard_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            # No index: both files carry shard 3, so the scan finds a conflict.
            _write_direct_checkpoint(
                d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES, with_index=False
            )
            _write_safetensors(
                os.path.join(d, "model_00001-of-00002.safetensors"),
                {
                    _ple_name(1): _shard_tensor(2, 4, 2),
                    _ple_name(10): _shard_tensor(2, 4, 20),
                    _ple_name(11): _shard_tensor(1, 4, 22),
                    _ple_name(3): _shard_tensor(2, 4, 6),
                },
            )
            self._expect_reject(d, "duplicate PLE shard 3")

    def test_shard_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            bases = dict(_DIRECT_ROW_BASES)
            bases[4] = (8, 3)  # rows 8-10 spill into shard 5's slot
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=bases)
            self._expect_reject(d, "overflow")

    def test_missing_shard_rejected_not_zero_filled(self):
        # Shard 7 is entirely absent: rows 14-15 would read zero instead of
        # failing, so the build must reject the checkpoint.
        with tempfile.TemporaryDirectory() as d:
            layout = {
                "model_00001-of-00002.safetensors": [0, 1, 2, 8, 9],
                "model_00002-of-00002.safetensors": [3, 4, 5, 6],
            }
            _write_direct_checkpoint(
                d, layout, row_bases={i: (2 * i, 2) for i in range(10) if i != 7}
            )
            self._expect_reject(
                d, r"consecutive starting at 0; missing shard\(s\) \[7\]"
            )
            # The same checkpoint is rejected with expected_rows too: the
            # contiguity check runs before any coverage arithmetic, so a
            # missing shard can never be interpreted as zero-fillable rows.
            with self.assertRaisesRegex(ValueError, "consecutive starting at 0"):
                self._build(d, expected_rows=20)

    def test_expected_rows_accepts_ceil_short_final_tensor(self):
        # 23 rows in 12 shards of 2: the 1-row final tensor is exactly the
        # short tail ceil(23 / 2) geometry implies, so it must pass.
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            table = self._build(d, expected_rows=_DIRECT_ROWS)
            self.assertEqual(table.num_rows, _DIRECT_ROWS)
            table.close()

    def test_expected_rows_shortfall_rejected(self):
        # org_vocab_size=30 with shards covering 23 rows: 4 shards of 2 rows
        # short, which must fail instead of serving rows 23-29 as zeros.
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            with self.assertRaisesRegex(ValueError, "cannot cover org_vocab_size=30"):
                self._build(d, expected_rows=30)

    def test_expected_rows_missing_middle_rows_rejected(self):
        # Shard 5 holds 1 of 2 rows: one row gap, and one fewer row than
        # org_vocab_size. Neither may slip through as a zero-fill.
        with tempfile.TemporaryDirectory() as d:
            bases = dict(_DIRECT_ROW_BASES)
            bases[5] = (10, 1)
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=bases)
            with self.assertRaisesRegex(ValueError, "no missing-row gaps"):
                self._build(d, expected_rows=_DIRECT_ROWS)

    def test_expected_rows_long_final_tensor_rejected(self):
        # The final shard fills its 2-row slot (24 rows total) but the
        # embedding only has org_vocab_size=23: an over-long tail means the
        # shard geometry does not match the embedding, not a legal short tail.
        with tempfile.TemporaryDirectory() as d:
            bases = {i: (2 * i, 2) for i in range(12)}
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=bases)
            with self.assertRaisesRegex(ValueError, "expected org_vocab_size=23"):
                self._build(d, expected_rows=_DIRECT_ROWS)

    def test_index_without_ple_tensors_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(
                d,
                {"model_00001-of-00001.safetensors": [0]},
                row_bases={0: (0, 2)},
            )
            _write_index(
                d, {"model.embed_tokens.weight": "model_00001-of-00001.safetensors"}
            )
            self._expect_reject(d, r"no \*\.ngram_embedding")

    def test_missing_directory_rejected(self):
        self._expect_reject("/nonexistent-ple-checkpoint", "not a directory")


class TestQwen4PleSafetensorsBackend(CustomTestCase):
    """Direct checkpoint table (SGLANG_QWEN4_PLE_SAFETENSORS,
    Qwen4ExpSafetensorsEmbedding).

    Reads rows from the checkpoint's own safetensors shards and copies them to
    the device, so unlike the file offload backend it needs no pageable GPU
    access. The two backends are mutually exclusive."""

    @staticmethod
    def _fake_embedding(rows, dim, vocab_start=0, vocab_end=None, scale=1.75):
        from sglang.srt.layers.vocab_parallel_embedding import (
            VocabParallelEmbeddingShardIndices,
        )

        if vocab_end is None:
            vocab_end = rows
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=vocab_start,
            padded_org_vocab_end_index=vocab_end,
            padded_added_vocab_start_index=rows,
            padded_added_vocab_end_index=rows,
            org_vocab_start_index=vocab_start,
            org_vocab_end_index=vocab_end,
            added_vocab_start_index=rows,
            added_vocab_end_index=rows,
        )
        local_rows = vocab_end - vocab_start
        return SimpleNamespace(
            quant_config=None,
            enable_tp=True,
            use_attn_tp_group=False,
            tp_size=1,
            num_embeddings=rows,
            org_vocab_size=rows,
            padding_size=64,
            num_added_embeddings=0,
            use_presharded_weights=False,
            org_vocab_size_padded=rows,
            num_embeddings_padded=rows,
            shard_indices=shard,
            embedding_dim=dim,
            num_embeddings_per_partition=local_rows,
            num_org_embeddings_per_partition=local_rows,
            num_added_embeddings_per_partition=0,
            weight=torch.nn.Parameter(
                torch.zeros((local_rows, dim), dtype=torch.bfloat16),
                requires_grad=False,
            ),
            weight_scale=torch.full((1,), scale, dtype=torch.bfloat16),
        )

    def _load(self, directory, embedding, shard_size=2):
        from sglang.srt.models.qwen4_exp import Qwen4ExpSafetensorsEmbedding

        return Qwen4ExpSafetensorsEmbedding(embedding, directory, shard_size=shard_size)

    @staticmethod
    def _patch_tp():
        # A no-TP CPU process: the runtime context has no published parallel
        # config here, so answer the module's own getter instead.
        from sglang.srt.layers import vocab_parallel_embedding as vpe

        fake = SimpleNamespace(tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1)
        return mock.patch.object(vpe, "get_parallel", lambda: fake)

    def test_gather_is_exact_and_zeroes_out_of_shard_rows(self):
        # A TP window [4, 11): checkpoint rows are global ids, so the gather
        # must read row == token id without re-basing, and zero the rest.
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            emb = self._load(
                d, self._fake_embedding(_DIRECT_ROWS, 4, vocab_start=4, vocab_end=11)
            )
            out = emb.gather(torch.tensor([4, 10, 3, 12, 1000]))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(tuple(out.shape), (5, 4))
            self.assertTrue(torch.equal(out[0], torch.full((4,), 4.0)))
            self.assertTrue(torch.equal(out[1], torch.full((4,), 10.0)))
            for i in (2, 3, 4):
                self.assertTrue(
                    torch.equal(out[i], torch.zeros(4, dtype=torch.bfloat16))
                )
            # The scale buffer survives the replacement: the direct backend
            # keeps the module's buffer, the checkpoint cannot overwrite it.
            self.assertAlmostEqual(float(emb.weight_scale), 1.75, delta=1e-2)

    def test_bulk_gather_grouped_mmap_out_buffer_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            emb = self._load(d, self._fake_embedding(_DIRECT_ROWS, 4))
            # Above the pread threshold: the grouped mmap path must gather
            # exactly, across both files.
            ids = torch.tensor([r for r in range(_DIRECT_ROWS)] * 8 + [20, 22, 23] * 8)
            # Row 23 is past the last checkpoint row and reads zero.
            present = set()
            for base, count in _DIRECT_ROW_BASES.values():
                present.update(range(base, base + count))
            expected = torch.tensor(
                [float(r) if r in present else 0.0 for r in ids.tolist()]
            )[:, None].repeat(1, 4)
            self.assertTrue(torch.equal(emb.gather(ids), expected.to(torch.bfloat16)))
            # Caller-provided output buffer (the decode prefetch path).
            buf = emb.allocate_output((5, 4), torch.device("cpu"))
            small = torch.tensor([0, 5, 22, 23, 1000])
            want = (
                torch.tensor([[0.0], [5.0], [22.0], [0.0], [0.0]])
                .repeat(1, 4)
                .to(torch.bfloat16)
            )
            self.assertTrue(torch.equal(emb.gather(small, out=buf), want))
            with self.assertRaisesRegex(ValueError, "invalid PLE prefetch output"):
                emb.gather(small, out=torch.empty((6, 4), dtype=torch.bfloat16))
            with self.assertRaisesRegex(ValueError, "must be bfloat16"):
                emb.gather(small, out=torch.empty((5, 4), dtype=torch.float32))

    def test_gather_validates_dimension_against_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            with self.assertRaisesRegex(ValueError, "does not match embedding_dim"):
                self._load(d, self._fake_embedding(_DIRECT_ROWS, 6))

    def _tiny_config(self, **extra):
        from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig

        return Qwen4ExpTextConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=1,
            eos_token_id=0,
            ngram_size=2,
            heads_per_ngram=2,
            ngram_vocab_size_base=101,
            make_ngram_vocab_size_divisible_by=1,
            ple_embed_dim=8,
            hc_count=2,
            **extra,
        )

    def test_table_is_not_allocated_before_replacement(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.qwen4_exp import Qwen4ExpNGramEmbedding

        with self._patch_tp():
            with envs.SGLANG_QWEN4_PLE_SAFETENSORS.override("/nonexistent-yet"):
                ngram = Qwen4ExpNGramEmbedding(self._tiny_config(), 8)
            self.assertEqual(ngram.ngram_embedding.weight.device.type, "meta")
            # Control: without the env, the table is a real allocation as upstream.
            ngram = Qwen4ExpNGramEmbedding(self._tiny_config(), 8)
            self.assertEqual(ngram.ngram_embedding.weight.device.type, "cpu")

    def test_direct_rejects_builtin_offload_backend(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.qwen4_exp import Qwen4ExpPLELayer

        config = self._tiny_config(
            ple_offload_embedding=True,
            ple_offload_backend="file",
        )
        with (
            self._patch_tp(),
            envs.SGLANG_QWEN4_PLE_SAFETENSORS.override("/unused"),
            self.assertRaisesRegex(ValueError, "cannot be combined"),
        ):
            Qwen4ExpPLELayer(config, prefix="model.layers.0.ple", layer_id=0)

    def test_prefetch_excludes_only_the_direct_table_files(self):
        from sglang.srt.model_loader.loader import DefaultModelLoader
        from sglang.srt.models.qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
            Qwen4ExpPinnedHostEmbedding,
            Qwen4ExpSafetensorsEmbedding,
        )

        prop = Qwen4ExpForConditionalGeneration.weight_loader_prefetch_exclude_files
        skip_prop = Qwen4ExpForConditionalGeneration.weight_loader_skip_tensor

        class _StubModel:
            # Real attribute access, so getattr resolves the property the
            # same way it does on the initialized Qwen4Exp model.
            weight_loader_prefetch_exclude_files = prop
            weight_loader_skip_tensor = skip_prop

            def __init__(self, modules):
                self._modules = modules

            def modules(self):
                return iter(self._modules)

        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            emb_a = self._load(d, self._fake_embedding(_DIRECT_ROWS, 4))
            emb_b = self._load(d, self._fake_embedding(_DIRECT_ROWS, 4))
            model = _StubModel([emb_a, emb_b])
            files = getattr(model, "weight_loader_prefetch_exclude_files")
            # Distinct files only (both embeddings read the same shards).
            self.assertEqual(
                files, sorted(set(emb_a.weight_loader_prefetch_exclude_files))
            )
            self.assertEqual(files, emb_b.weight_loader_prefetch_exclude_files)
            self.assertTrue(all(f.startswith(d) for f in files))
            # The built-in offload backends have no direct embedding, so
            # nothing is excluded from prefetching for them.
            plain = SimpleNamespace(
                modules=lambda: iter(
                    [Qwen4ExpPinnedHostEmbedding.__new__(Qwen4ExpPinnedHostEmbedding)]
                )
            )
            self.assertIsNone(prop.fget(plain))
            # The hook is generic on Source: an initialized direct model is
            # what Source.init_new reads, no loader-side model branching.
            source = DefaultModelLoader.Source.init_new(
                SimpleNamespace(model_path=d, revision=None), model
            )
            self.assertEqual(source.weight_loader_prefetch_exclude_files, files)
            self.assertTrue(
                source.weight_loader_skip_tensor(
                    "model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
                )
            )
            self.assertFalse(
                source.weight_loader_skip_tensor("model.embed_tokens.weight")
            )
        self.assertIsInstance(emb_a, Qwen4ExpSafetensorsEmbedding)

    def test_ple_shards_skipped_only_for_the_direct_backend(self):
        from sglang.srt.environ import envs
        from sglang.srt.models.qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
            Qwen4ExpNGramEmbedding,
        )

        head_dim = 4
        mod_prefix = "model.layers.0.ple.ple_embedding"
        weight_name = f"{mod_prefix}.ngram_embedding.shard_0.weight"
        loaded_weight = (
            torch.arange(3 * head_dim).reshape(3, head_dim).to(torch.bfloat16)
        )

        def make_self(ngram):
            return SimpleNamespace(
                config=SimpleNamespace(tie_word_embeddings=False),
                pp_group=SimpleNamespace(is_last_rank=False),
                start_layer=0,
                end_layer=1,
                named_parameters=lambda **kwargs: iter([]),
                named_buffers=lambda: iter([]),
                named_modules=lambda: iter([(mod_prefix, ngram)]),
                modules=lambda: iter([]),
                _load_qwen4_exp_ple_buffer=lambda *args: False,
            )

        with tempfile.TemporaryDirectory() as d:
            _write_direct_checkpoint(d, _DIRECT_LAYOUT, row_bases=_DIRECT_ROW_BASES)
            with self._patch_tp():
                with envs.SGLANG_QWEN4_PLE_SAFETENSORS.override(d):
                    ngram = Qwen4ExpNGramEmbedding(self._tiny_config(), 8)
                # The meta embedding sizes its table for the real n-gram
                # vocab; point it at the fixture's rows so the direct build's
                # coverage check matches this tiny checkpoint.
                ngram.ngram_embedding.org_vocab_size = _DIRECT_ROWS
                ngram.ngram_embedding = self._load(d, ngram.ngram_embedding)
                # The direct backend reads the checkpoint itself: the shard is
                # consumed without touching the meta-device weight and
                # reported unloaded.
                loaded = Qwen4ExpForConditionalGeneration.load_weights(
                    make_self(ngram), [(weight_name, loaded_weight)]
                )
                self.assertEqual(loaded, set())

                # Control: a plain (non-direct) embedding still loads shards.
                ngram = Qwen4ExpNGramEmbedding(self._tiny_config(), 8)
                loaded = Qwen4ExpForConditionalGeneration.load_weights(
                    make_self(ngram), [(weight_name, loaded_weight)]
                )
                self.assertIn(
                    f"{mod_prefix}.ngram_embedding.weight",
                    loaded,
                )
                self.assertTrue(
                    torch.equal(
                        ngram.ngram_embedding.weight.data[:3],
                        loaded_weight,
                    )
                )


if __name__ == "__main__":
    unittest.main()
