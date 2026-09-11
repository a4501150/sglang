"""Unit tests for HiCache GDS device page-segment descriptors.

Runs on CPU: descriptor construction only reads data_ptr/shape/stride.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import ctypes
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.srt.mem_cache.qsa_kv_pool import QwenDSATokenToKVPool, QSATokenToKVPool


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())


def _read(address: int, nbytes: int) -> bytes:
    return ctypes.string_at(address, nbytes)


def _mha_pool(layers, page_size, kv_heads, head_dim, use_scale=False):
    pool = object.__new__(MHATokenToKVPool)
    pages = 5
    total_slots = pages * page_size
    pool.size = (pages - 1) * page_size
    pool.page_size = page_size
    pool.k_buffer = [
        torch.arange(total_slots * kv_heads * head_dim, dtype=torch.int32).reshape(
            total_slots, kv_heads, head_dim
        )
        for _ in range(layers)
    ]
    pool.v_buffer = [
        -torch.arange(total_slots * kv_heads * head_dim, dtype=torch.int32).reshape(
            total_slots, kv_heads, head_dim
        )
        for _ in range(layers)
    ]
    if use_scale:
        pool.k_scale_buffer = [
            torch.full((pages, 4), 3, dtype=torch.float8_e8m0fnu) for _ in range(layers)
        ]
        pool.v_scale_buffer = [
            torch.full((pages, 4), 5, dtype=torch.float8_e8m0fnu) for _ in range(layers)
        ]
    else:
        pool.k_scale_buffer = None
        pool.v_scale_buffer = None
    return pool


def _component_names(segments):
    return [name for name, _, _ in segments]


class TestMHAPageSegments(unittest.TestCase):
    def test_nhd_segments_match_bytes(self):
        page_size, layers = 4, 2
        pool = _mha_pool(layers, page_size, kv_heads=2, head_dim=8)
        # Pages starting at slots 0 and 8.
        indices = torch.tensor(list(range(page_size)) + list(range(8, 8 + page_size)))
        segments = pool.get_hicache_page_segments(indices)
        per_page = 2 * layers
        self.assertEqual(len(segments), 2 * per_page)
        for offset in range(0, len(segments), per_page):
            self.assertEqual(
                _component_names(segments[offset : offset + per_page]),
                _component_names(segments[:per_page]),
            )
        for page_no, first in enumerate((0, 8)):
            for layer in range(layers):
                k_name, k_addr, k_size = segments[page_no * per_page + layer]
                v_name, v_addr, v_size = segments[page_no * per_page + layers + layer]
                self.assertEqual(k_name, f"k_{layer}")
                self.assertEqual(v_name, f"v_{layer}")
                token_bytes = 2 * 8 * 4
                self.assertEqual(k_size, page_size * token_bytes)
                self.assertEqual(
                    _read(k_addr, k_size),
                    _tensor_bytes(pool.k_buffer[layer][first : first + page_size]),
                )
                self.assertEqual(
                    _read(v_addr, v_size),
                    _tensor_bytes(pool.v_buffer[layer][first : first + page_size]),
                )

    def test_hnd_scale_segments_match_bytes(self):
        page_size, layers = 4, 2
        pool = _mha_pool(layers, page_size, kv_heads=2, head_dim=8, use_scale=True)
        # HND k/v: [num_pages, head, page, dim]; scale rows are per page.
        pool.k_buffer = [t.reshape(5, 2, page_size, 8) for t in pool.k_buffer]
        pool.v_buffer = [t.reshape(5, 2, page_size, 8) for t in pool.v_buffer]
        indices = torch.tensor(list(range(4, 4 + page_size)))
        segments = pool.get_hicache_page_segments(indices)
        per_page = 4 * layers
        self.assertEqual(len(segments), per_page)
        for layer in range(layers):
            name, addr, size = segments[layer]
            self.assertEqual(name, f"k_{layer}")
            self.assertEqual(size, 2 * page_size * 8 * 4)
            self.assertEqual(_read(addr, size), _tensor_bytes(pool.k_buffer[layer][1]))
        scale_entry = segments[2 * layers]
        self.assertEqual(scale_entry[0], "k_scale_0")
        self.assertEqual(
            scale_entry[2],
            pool.k_scale_buffer[0][1].numel() * pool.k_scale_buffer[0].element_size(),
        )
        self.assertEqual(
            _read(scale_entry[1], scale_entry[2]),
            _tensor_bytes(pool.k_scale_buffer[0][1]),
        )

    def test_rejects_partial_page(self):
        pool = _mha_pool(1, 4, kv_heads=2, head_dim=8)
        with self.assertRaises(ValueError):
            pool.get_hicache_page_segments(torch.tensor([0, 1, 2]))

    def test_rejects_unaligned_page(self):
        pool = _mha_pool(1, 4, kv_heads=2, head_dim=8)
        with self.assertRaises(ValueError):
            pool.get_hicache_page_segments(torch.tensor([2, 3, 4, 5]))


class _FakeSibling:
    def __init__(self):
        self.ngram = torch.arange(6 * 32, dtype=torch.int64).reshape(6, 32)
        self.short_conv = torch.arange(2 * 6 * 16, dtype=torch.float32).reshape(
            2, 6, 16
        )

    def get_storage_tensors(self):
        return [
            ("ple_ngram", self.ngram, 0),
            ("ple_short_conv", self.short_conv, 1),
        ]


class TestMambaPageSegments(unittest.TestCase):
    def _pool(self):
        pool = object.__new__(MambaPool)
        layers = 2
        slots = 6
        pool.mamba_layer_ids = [0, 3]
        pool.conv_slice_axis = 1
        conv = torch.arange(layers * slots * 8 * 4, dtype=torch.float32).reshape(
            layers, slots, 8, 4
        )
        temporal = torch.arange(layers * slots * 16, dtype=torch.float32).reshape(
            layers, slots, 16
        )
        pool.mamba_cache = MambaPool.State(conv=[conv], temporal=temporal)
        pool._slot_siblings = [_FakeSibling()]
        return pool

    def test_segments_match_bytes(self):
        pool = self._pool()
        sibling = pool._slot_siblings[0]
        segments = pool.get_hicache_page_segments(torch.tensor([2, 5]))
        expected_names = [
            "conv_0_0",
            "conv_0_3",
            "temporal_0_0",
            "temporal_0_3",
            "ple_ngram",
            "ple_short_conv_0",
            "ple_short_conv_1",
        ]
        self.assertEqual(_component_names(segments), expected_names * 2)
        per_slot = len(expected_names)
        for slot_no, slot in enumerate((2, 5)):
            base = slot_no * per_slot
            for layer_offset in (0, 1):
                _, addr, size = segments[base + layer_offset]
                expected = pool.mamba_cache.conv[0][layer_offset, slot]
                self.assertEqual(size, expected.numel() * 4)
                self.assertEqual(_read(addr, size), _tensor_bytes(expected))
                _, addr, size = segments[base + 2 + layer_offset]
                expected = pool.mamba_cache.temporal[layer_offset, slot]
                self.assertEqual(size, expected.numel() * 4)
                self.assertEqual(_read(addr, size), _tensor_bytes(expected))
            _, addr, size = segments[base + 4]
            expected = sibling.ngram[slot]
            self.assertEqual(size, expected.numel() * 8)
            self.assertEqual(_read(addr, size), _tensor_bytes(expected))
            for layer_offset in (0, 1):
                _, addr, size = segments[base + 5 + layer_offset]
                expected = sibling.short_conv[layer_offset, slot]
                self.assertEqual(size, expected.numel() * 4)
                self.assertEqual(_read(addr, size), _tensor_bytes(expected))

    def test_rejects_capacity_overrun(self):
        pool = self._pool()
        with self.assertRaises(ValueError):
            pool.get_hicache_page_segments(torch.tensor([6]))


def _qsa_pool(cls, page_size, ratio, heads, dim, layers):
    pool = object.__new__(cls)
    pool.page_size = page_size
    pool.qsa_index_kv_heads = heads
    pool.qsa_index_head_dim = dim
    pool.index_state_dtype = torch.bfloat16
    if cls is QSATokenToKVPool:
        pool.qsa_compress_ratio = ratio
        pool.qsa_compressed_page_size = page_size // ratio
        pool.qsa_compressed_capacity = (5 * page_size) // ratio
        rows = pool.qsa_compressed_capacity
        buffers = [
            torch.arange(rows * heads * dim // 2, dtype=torch.int32)
            .view(torch.bfloat16)
            .reshape(rows, heads, dim)
            for _ in range(layers)
        ]
        pool.qsa_compressed_k_buffer_pool = buffers
        page_bytes = pool.qsa_compressed_page_size * heads * dim * 2
    else:
        pool.qsa_compress_ratio = 1
        rows = 5 * page_size
        buffers = [
            torch.arange(rows * heads * dim // 2, dtype=torch.int32)
            .view(torch.bfloat16)
            .reshape(rows, heads, dim)
            for _ in range(layers)
        ]
        pool.dsa_index_k_buffer_pool = buffers
        page_bytes = page_size * heads * dim * 2
    pool.get_hicache_indexer_page_buffers = lambda: [
        buffer.view(torch.uint8).reshape(-1, page_bytes) for buffer in buffers
    ]
    return pool


class TestQSAIndexerPageSegments(unittest.TestCase):
    def test_compressed_segments_match_bytes(self):
        page_size, ratio, layers = 8, 2, 2
        pool = _qsa_pool(
            QSATokenToKVPool, page_size, ratio, heads=1, dim=4, layers=layers
        )
        segments = pool.get_hicache_page_segments(torch.tensor(list(range(page_size))))
        self.assertEqual(len(segments), layers)
        for layer in range(layers):
            name, addr, size = segments[layer]
            self.assertEqual(name, f"index_k_{layer}")
            row_bytes = (
                pool.qsa_compressed_page_size
                * pool.qsa_index_kv_heads
                * pool.qsa_index_head_dim
                * 2
            )
            self.assertEqual(size, row_bytes)
            source = pool.qsa_compressed_k_buffer_pool[layer]
            self.assertEqual(
                _read(addr, size),
                _tensor_bytes(source.view(torch.uint8).flatten()[:row_bytes]),
            )

    def test_tokenwise_segments_match_bytes(self):
        page_size, layers = 64, 2
        pool = _qsa_pool(
            QwenDSATokenToKVPool, page_size, 1, heads=1, dim=8, layers=layers
        )
        segments = pool.get_hicache_page_segments(
            torch.tensor(list(range(2 * page_size)))
        )
        self.assertEqual(len(segments), 2 * layers)
        page_bytes = page_size * 1 * 8 * 2
        for page_no in (0, 1):
            for layer in range(layers):
                name, addr, size = segments[page_no * layers + layer]
                self.assertEqual(name, f"index_k_{layer}")
                self.assertEqual(size, page_bytes)
                source = pool.dsa_index_k_buffer_pool[layer]
                page_view = source.view(torch.uint8).reshape(-1, page_bytes)
                self.assertEqual(_read(addr, size), _tensor_bytes(page_view[page_no]))

    def test_rejects_partial_page(self):
        pool = _qsa_pool(QwenDSATokenToKVPool, 64, 1, heads=1, dim=8, layers=1)
        with self.assertRaises(ValueError):
            pool.get_hicache_page_segments(torch.tensor([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
