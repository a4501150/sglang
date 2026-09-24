import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolDynamicFP8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _quantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return (tensor / scale).to(torch.float8_e4m3fn)


def test_qsa_sparse_attention_reference_applies_fp8_kv_descales():
    torch.manual_seed(23)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), k_scale)
    v = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), v_scale)
    slots = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)

    actual = qsa_sparse_attention(q, k, v, slots, k_scale=k_scale, v_scale=v_scale)
    expected = qsa_sparse_attention(
        q,
        k.float() * k_scale,
        v.float() * v_scale,
        slots,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_sparse_attention_reference_applies_dynamic_fp8_kv_descales():
    torch.manual_seed(24)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    source_k = torch.randn(7, 2, 16, dtype=torch.bfloat16) * 3
    source_v = torch.randn(7, 2, 16, dtype=torch.bfloat16) * 5
    k, k_scale = MHATokenToKVPoolDynamicFP8._quantize(source_k)
    v, v_scale = MHATokenToKVPoolDynamicFP8._quantize(source_v)
    slots = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)

    actual = qsa_sparse_attention(
        q,
        k,
        v,
        slots,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k.float() * k_scale.unsqueeze(-1),
        v.float() * v_scale.unsqueeze(-1),
        slots,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_dynamic_fp8_quantization_uses_per_token_head_descales():
    values = torch.zeros(3, 2, 8, dtype=torch.bfloat16)
    values[0, 0, 0] = 1
    values[0, 1, 0] = 8
    values[1, 0, 0] = 16
    values[1, 1, 0] = -32

    quantized, scales = MHATokenToKVPoolDynamicFP8._quantize(values)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (3, 2)
    torch.testing.assert_close(
        scales,
        torch.tensor(
            [[1 / 448, 8 / 448], [16 / 448, 32 / 448], [1, 1]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        quantized.float() * scales.unsqueeze(-1), values.float(), rtol=0, atol=1e-3
    )


def test_dynamic_fp8_pool_stores_and_moves_payload_with_scales():
    if not torch.cuda.is_available():
        return

    pool = MHATokenToKVPoolDynamicFP8(
        size=8,
        page_size=4,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=16,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_kv_cache_copy=True,
    )
    layer = SimpleNamespace(layer_id=0)
    loc = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    k = torch.randn(2, 2, 16, dtype=torch.bfloat16, device="cuda") * 3
    v = torch.randn(2, 2, 16, dtype=torch.bfloat16, device="cuda") * 5
    expected_k, expected_v = k.clone(), v.clone()

    pool.set_kv_buffer(layer, loc, k, v)
    stored_k, stored_v = pool.get_kv_buffer(0)
    k_scale, v_scale = pool.get_kv_scale_buffer(0)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    torch.testing.assert_close(
        stored_k[loc].float() * k_scale[loc].unsqueeze(-1),
        expected_k.float(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        stored_v[loc].float() * v_scale[loc].unsqueeze(-1),
        expected_v.float(),
        rtol=5e-2,
        atol=5e-2,
    )

    target = torch.tensor([5, 6], dtype=torch.int64, device="cuda")
    pool.move_kv_cache(target, loc)
    torch.testing.assert_close(stored_k[target].float(), stored_k[loc].float())
    torch.testing.assert_close(stored_v[target].float(), stored_v[loc].float())
    torch.testing.assert_close(k_scale[target], k_scale[loc])
    torch.testing.assert_close(v_scale[target], v_scale[loc])
    with pytest.raises(NotImplementedError, match="prefix-valid commit"):
        pool.set_kv_buffer_prefix_valid()


def test_qsa_fp8_prefill_matches_dequantized_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(27)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device=device), v_scale
    )
    indices = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2]],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton(
        q,
        k,
        v,
        max_seqlen_k=3,
        indices=indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_fp8_chunk_prefill_matches_dequantized_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(29)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device), v_scale
    )
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_dynamic_fp8_chunk_prefill_matches_dequantized_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(291)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    source_k = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device) * 3
    source_v = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device) * 5
    k, k_scale = MHATokenToKVPoolDynamicFP8._quantize(source_k)
    v, v_scale = MHATokenToKVPoolDynamicFP8._quantize(source_v)
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        softmax_scale,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
    )
    expected = qsa_sparse_attention(
        q,
        k,
        v,
        indices,
        softmax_scale,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_qsa_bf16_chunk_prefill_regression():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(30)
    device = "cuda"
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device)
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, kv_lens, softmax_scale
    )
    expected = qsa_sparse_attention(q, k, v, indices, softmax_scale)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_qsa_fp8_compact_gather_dequantizes_for_flash_attention():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(31)
    device = "cuda"
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device=device), k_scale
    )
    v = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device=device), v_scale
    )
    req_to_token = torch.tensor(
        [[3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12]],
        dtype=torch.int32,
        device=device,
    )
    req_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indices = torch.tensor(
        [[0, 3, 5, -1], [1, 4, -1, -1]], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([6, 5], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    out_k = torch.empty(8, 1, 16, dtype=torch.bfloat16, device=device)
    out_v = torch.empty_like(out_k)

    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        batch=2,
        topk=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    selected_slots = torch.tensor([3, 9, 13, 4, 10], device=device)
    torch.testing.assert_close(
        out_k[:5].float(),
        k[selected_slots].float() * k_scale,
        rtol=0,
        atol=2e-2,
    )
    torch.testing.assert_close(
        out_v[:5].float(),
        v[selected_slots].float() * v_scale,
        rtol=0,
        atol=2e-2,
    )

    dynamic_k_scale = torch.linspace(0.125, 0.5, 16, device=device).unsqueeze(1)
    dynamic_v_scale = torch.linspace(0.25, 0.75, 16, device=device).unsqueeze(1)
    out_k_dynamic = torch.empty_like(out_k)
    out_v_dynamic = torch.empty_like(out_v)
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k_dynamic,
        out_v_dynamic,
        batch=2,
        topk=4,
        k_scale_buffer=dynamic_k_scale,
        v_scale_buffer=dynamic_v_scale,
    )
    torch.testing.assert_close(
        out_k_dynamic[:5].float(),
        k[selected_slots].float() * dynamic_k_scale[selected_slots].unsqueeze(-1),
        rtol=0,
        atol=2e-2,
    )
    torch.testing.assert_close(
        out_v_dynamic[:5].float(),
        v[selected_slots].float() * dynamic_v_scale[selected_slots].unsqueeze(-1),
        rtol=0,
        atol=2e-2,
    )

    # TRTLLM consumes FP8 scratch directly and applies the descales in BMM1/BMM2.
    out_k_fp8 = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn, device=device)
    out_v_fp8 = torch.empty_like(out_k_fp8)
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k_fp8,
        out_v_fp8,
        batch=2,
        topk=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(
        out_k_fp8[:5].float(), k[selected_slots].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_v_fp8[:5].float(), v[selected_slots].float(), rtol=0, atol=0
    )


def test_qsa_fp8_cache_write_preserves_prefill_kv():
    class MutatingPool:
        dtype = torch.float8_e4m3fn

        def set_kv_buffer(
            self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None
        ):
            self.args = (cache_k, cache_v, k_scale, v_scale)
            if k_scale is not None:
                cache_k.div_(k_scale)
                cache_v.div_(v_scale)

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = MutatingPool()
    layer = SimpleNamespace(layer_id=0, k_scale_float=0.25, v_scale_float=0.5)
    k = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    v = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    expected_k, expected_v = k.clone(), v.clone()

    backend._store_kv(layer, torch.arange(3, dtype=torch.int32), k, v)

    written_k, written_v, written_k_scale, written_v_scale = (
        backend.token_to_kv_pool.args
    )
    assert written_k is not k and written_v is not v
    assert written_k_scale == 0.25 and written_v_scale == 0.5
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)


def test_qsa_trtllm_decode_receives_fp8_kv_descales(monkeypatch):
    seen = {}
    compact_seen = {}

    def fake_valid_counts(seq_lens, indices, counts, batch, topk):
        counts.fill_(topk)

    def fake_compact(*args, **kwargs):
        compact_seen.update(kwargs)

    def fake_decode(**kwargs):
        seen.update(kwargs)
        return torch.zeros_like(kwargs["query"])

    monkeypatch.setattr(
        qsa_backend_module, "qwen_sparse_valid_counts_triton", fake_valid_counts
    )
    monkeypatch.setattr(
        qsa_backend_module,
        "qwen_sparse_kv_extraction_compact_triton",
        fake_compact,
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8, dtype=torch.int32).reshape(1, 8)
    )
    backend._fa2_scratch = {}
    backend._trtllm_sparse_tables = {}
    backend._trtllm_workspace = torch.empty(1, dtype=torch.uint8)
    backend._cuda_graph_max_tokens = 0
    layer = SimpleNamespace(
        layer_id=0, scaling=0.125, k_scale_float=0.25, v_scale_float=0.5
    )
    q = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    k = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    topk_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([0]))
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        row_req_pool_indices=None,
        is_cuda_graph=False,
    )

    output = backend._forward_trtllm_sparse(
        q, k, v, layer, forward_batch, metadata, topk_indices, fake_decode
    )

    assert output.shape == (1, 64)
    assert compact_seen["k_scale"] == 0.25
    assert compact_seen["v_scale"] == 0.5
    assert seen["kv_cache"][0].dtype == torch.bfloat16
    assert seen["bmm1_scale"] == 0.125
    assert seen["bmm2_scale"] == 1.0

    dynamic_k_scale = torch.ones(8, 1, dtype=torch.float32)
    dynamic_v_scale = torch.full((8, 1), 2.0, dtype=torch.float32)
    backend.token_to_kv_pool = SimpleNamespace(
        dynamic_fp8_kv_cache=True,
        get_kv_scale_buffer=lambda layer_id: (dynamic_k_scale, dynamic_v_scale),
    )
    compact_seen.clear()
    backend._forward_trtllm_sparse(
        q, k, v, layer, forward_batch, metadata, topk_indices, fake_decode
    )
    assert compact_seen["k_scale"] == 1.0
    assert compact_seen["v_scale"] == 1.0
    assert compact_seen["k_scale_buffer"] is dynamic_k_scale
    assert compact_seen["v_scale_buffer"] is dynamic_v_scale


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
