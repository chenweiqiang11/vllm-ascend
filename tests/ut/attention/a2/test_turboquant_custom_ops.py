# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import math

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
importlib.import_module("vllm_ascend.vllm_ascend_C")
torch_npu.npu.config.allow_internal_format = True

TQ_HEAD_DIM = 512
TQ_ROPE_HEAD_DIM = 64
TQ_BLOCK_SIZE = 128
TQ_PACKED_BYTES = TQ_HEAD_DIM // 2
TQ_SLOT_BYTES = TQ_PACKED_BYTES + TQ_ROPE_HEAD_DIM * 2 + 2
TQ_CENTROIDS = np.array(
    [
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
    ],
    dtype=np.float32,
)


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str = "meta",
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _sfa_inputs(device: str = "meta") -> tuple[torch.Tensor, ...]:
    return (
        _tensor((2, 4, 576), torch.bfloat16, device),
        _tensor((2, 16, 1, 386), torch.int8, device),
        _tensor((2, 16, 1, 386), torch.int8, device),
        _tensor((2, 1, 8), torch.int32, device),
    )


def _sfa_kwargs(device: str = "meta") -> dict[str, object]:
    return {
        "block_table": _tensor((1, 2), torch.int32, device),
        "actual_seq_lengths_query": _tensor((1,), torch.int32, device),
        "actual_seq_lengths_kv": _tensor((1,), torch.int32, device),
        "scale_value": 0.125,
        "key_quant_mode": 3,
        "value_quant_mode": 3,
        "sparse_block_size": 1,
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": 128,
        "rope_head_dim": 64,
        "return_softmax_lse": True,
    }


def test_turboquant_sfa_schema_hides_cann_window_sentinels() -> None:
    schema = str(torch.ops._C_ascend.turboquant_sparse_flash_attention.default._schema)

    assert "pre_tokens" not in schema
    assert "next_tokens" not in schema


@pytest.mark.parametrize(
    "required_input",
    [
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
    ],
)
def test_turboquant_sfa_schema_requires_paged_inputs(required_input: str) -> None:
    kwargs = _sfa_kwargs()
    kwargs.pop(required_input)

    with pytest.raises(RuntimeError, match=required_input):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*_sfa_inputs(), **kwargs)


def test_turboquant_sfa_meta_shapes() -> None:
    output, softmax_max, softmax_sum = torch.ops._C_ascend.turboquant_sparse_flash_attention(
        *_sfa_inputs(),
        **_sfa_kwargs(),
    )

    assert output.shape == (2, 4, 512)
    assert output.dtype == torch.bfloat16
    assert softmax_max.shape == softmax_sum.shape == (1, 2, 4)
    assert softmax_max.dtype == softmax_sum.dtype == torch.float32


@pytest.mark.parametrize("scale_name", ["key_dequant_scale", "value_dequant_scale"])
def test_turboquant_sfa_meta_rejects_invalid_dequant_scale_dtype(scale_name: str) -> None:
    kwargs = _sfa_kwargs()
    kwargs[scale_name] = _tensor((1,), torch.bfloat16)

    with pytest.raises(RuntimeError, match=f"{scale_name} must have float32 dtype"):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*_sfa_inputs(), **kwargs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("sparse_head_mismatch", "token/KV-head dimensions"),
        ("empty_kv_blocks", "block count and block size must be non-zero"),
        ("empty_kv_block_size", "block count and block size must be non-zero"),
        ("unaligned_kv_block_size", "KV block size must be aligned"),
        ("empty_block_batch", "block_table must match the sequence batch"),
        ("empty_block_columns", "block_table must match the sequence batch"),
        ("empty_query_lengths", "actual_seq_lengths_query must be non-empty"),
        ("length_batch_mismatch", "sequence-length tensors must have the same batch size"),
        ("block_batch_mismatch", "block_table must match the sequence batch"),
    ],
)
def test_turboquant_sfa_meta_rejects_invalid_paged_contract(
    case: str,
    message: str,
) -> None:
    inputs = list(_sfa_inputs())
    kwargs = _sfa_kwargs()

    if case == "sparse_head_mismatch":
        inputs[3] = _tensor((2, 4, 8), torch.int32)
    elif case == "empty_kv_blocks":
        inputs[1] = inputs[2] = _tensor((0, 16, 1, 386), torch.int8)
    elif case == "empty_kv_block_size":
        inputs[1] = inputs[2] = _tensor((2, 0, 1, 386), torch.int8)
    elif case == "unaligned_kv_block_size":
        inputs[1] = inputs[2] = _tensor((2, 8, 1, 386), torch.int8)
    elif case == "empty_block_batch":
        kwargs["block_table"] = _tensor((0, 2), torch.int32)
    elif case == "empty_block_columns":
        kwargs["block_table"] = _tensor((1, 0), torch.int32)
    elif case == "empty_query_lengths":
        kwargs["actual_seq_lengths_query"] = _tensor((0,), torch.int32)
    elif case == "length_batch_mismatch":
        kwargs["actual_seq_lengths_query"] = _tensor((2,), torch.int32)
        kwargs["block_table"] = _tensor((2, 2), torch.int32)
    elif case == "block_batch_mismatch":
        kwargs["block_table"] = _tensor((2, 2), torch.int32)
    else:
        raise AssertionError(f"Unhandled test case: {case}")

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*inputs, **kwargs)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty_kv_blocks", "block count and block size must be non-zero"),
        ("sparse_head_mismatch", "token/KV-head dimensions"),
    ],
)
def test_turboquant_sfa_runtime_adapter_rejects_before_aclnn(
    case: str,
    message: str,
) -> None:
    inputs = list(_sfa_inputs("npu"))
    kwargs = _sfa_kwargs("npu")
    if case == "empty_kv_blocks":
        inputs[1] = inputs[2] = _tensor((0, 16, 1, 386), torch.int8, "npu")
    else:
        inputs[3] = _tensor((2, 4, 8), torch.int32, "npu")

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(*inputs, **kwargs)


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "message"),
    [
        ((2, 576), (2, 16, 1, 386), "query dimension must be 3"),
        ((2, 4, 576), (2, 16, 386), "key dimension must be 4"),
        ((2, 4, 576), (2, 16, 0, 386), "exactly one KV head"),
    ],
)
def test_turboquant_sfa_meta_rejects_invalid_ranks_and_heads(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    message: str,
) -> None:
    query = _tensor(query_shape, torch.bfloat16)
    key = _tensor(key_shape, torch.int8)
    value = _tensor(key_shape, torch.int8)
    sparse_indices = _tensor((2, 1, 8), torch.int32)

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turboquant_sparse_flash_attention(
            query,
            key,
            value,
            sparse_indices,
            **_sfa_kwargs(),
        )


def test_turboquant_compress_meta_shape() -> None:
    slot = torch.ops._C_ascend.turbo_quant_compress_latent(
        _tensor((3, 512), torch.float32),
        _tensor((16,), torch.float32),
    )

    assert slot.shape == (3, 320)
    assert slot.dtype == torch.uint8


@pytest.mark.parametrize(
    ("latent_shape", "latent_dtype", "centroid_shape", "centroid_dtype", "message"),
    [
        ((512,), torch.float32, (16,), torch.float32, r"\[N, 512\]"),
        ((3, 511), torch.float32, (16,), torch.float32, r"\[N, 512\]"),
        ((3, 512), torch.bfloat16, (16,), torch.float32, "float32 inputs"),
        ((3, 512), torch.float32, (15,), torch.float32, "exactly 16 centroids"),
        ((3, 512), torch.float32, (16,), torch.bfloat16, "float32 inputs"),
    ],
)
def test_turboquant_compress_meta_rejects_invalid_contract(
    latent_shape: tuple[int, ...],
    latent_dtype: torch.dtype,
    centroid_shape: tuple[int, ...],
    centroid_dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.turbo_quant_compress_latent(
            _tensor(latent_shape, latent_dtype),
            _tensor(centroid_shape, centroid_dtype),
        )


def _reference_compress(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared_norm = np.sum(latent * latent, axis=1, dtype=np.float32)
    norm = np.sqrt(squared_norm + np.float32(1e-16)).astype(np.float32)
    inverse_norm = (np.float32(1.0) / norm).astype(np.float32)
    unit = (latent * inverse_norm[:, None]).astype(np.float32)
    boundaries = ((TQ_CENTROIDS[:-1] + TQ_CENTROIDS[1:]) * np.float32(0.5)).astype(np.float32)
    nibbles = (unit[:, :, None] >= boundaries[None, None, :]).sum(axis=2).astype(np.uint8)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    return packed, norm


@torch.inference_mode()
def test_turboquant_compress_npu_matches_reference() -> None:
    rng = np.random.default_rng(2026)
    latent = (rng.standard_normal((5, TQ_HEAD_DIM)) / math.sqrt(TQ_HEAD_DIM)).astype(np.float32)
    expected_packed, expected_norm = _reference_compress(latent)

    actual = torch.ops._C_ascend.turbo_quant_compress_latent(
        torch.from_numpy(latent).npu(),
        torch.from_numpy(TQ_CENTROIDS).npu(),
    ).cpu()

    assert actual.shape == (5, 320)
    np.testing.assert_array_equal(actual[:, :TQ_PACKED_BYTES].numpy(), expected_packed)
    actual_norm = actual[:, TQ_PACKED_BYTES : TQ_PACKED_BYTES + 2].contiguous().view(torch.float16).float()
    expected_norm_tensor = torch.from_numpy(expected_norm)
    torch.testing.assert_close(actual_norm.flatten(), expected_norm_tensor, rtol=2e-3, atol=0.0)
    assert bool((actual[:, TQ_PACKED_BYTES + 2 :] == 0).all())
    print(f"compress max norm abs error: {(actual_norm.flatten() - expected_norm_tensor).abs().max().item():.8f}")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _pack_sfa_slots(nope: np.ndarray, rope: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vector_norm = np.sqrt(np.sum(nope.astype(np.float64) ** 2, axis=1) + 1e-16)
    unit = nope / vector_norm[:, None].astype(np.float32)
    boundaries = ((TQ_CENTROIDS[:-1] + TQ_CENTROIDS[1:]) * np.float32(0.5)).astype(np.float32)
    nibbles = np.searchsorted(boundaries, unit.ravel(), side="right").reshape(nope.shape).astype(np.uint8)
    reconstructed_unit = TQ_CENTROIDS[nibbles]
    reconstructed_norm = np.sqrt(np.sum(reconstructed_unit.astype(np.float64) ** 2, axis=1) + 1e-16)
    scale = (vector_norm / reconstructed_norm).astype(np.float16)

    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    stored_rope = torch.from_numpy(rope / scale.astype(np.float32)[:, None]).to(torch.bfloat16).contiguous()
    slot = np.zeros((nope.shape[0], TQ_SLOT_BYTES), dtype=np.uint8)
    slot[:, :TQ_PACKED_BYTES] = packed
    slot[:, TQ_PACKED_BYTES:-2] = stored_rope.view(torch.uint8).numpy().reshape(nope.shape[0], -1)
    slot[:, -2:] = scale.view(np.uint8).reshape(nope.shape[0], 2)
    return slot, reconstructed_unit.astype(np.float64), scale, stored_rope.float().numpy().astype(np.float64)


def _reference_sfa(
    query: torch.Tensor,
    reconstructed_unit: np.ndarray,
    scale: np.ndarray,
    stored_rope: np.ndarray,
) -> np.ndarray:
    q = query.float().numpy().astype(np.float64)[0]
    q_nope, q_rope = q[:, :TQ_HEAD_DIM], q[:, TQ_HEAD_DIM:]
    scale_f64 = scale.astype(np.float64)
    scores = scale_f64[None, :] * (q_nope @ reconstructed_unit.T + q_rope @ stored_rope.T) / math.sqrt(TQ_HEAD_DIM)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    values = reconstructed_unit * scale_f64[:, None]
    return probabilities @ values


@torch.inference_mode()
def test_turboquant_sfa_npu_matches_reference() -> None:
    rng = np.random.default_rng(14715)
    physical_block_count = 3
    logical_block_count = 2
    token_count = physical_block_count * TQ_BLOCK_SIZE
    nope = (rng.standard_normal((token_count, TQ_HEAD_DIM)) / math.sqrt(TQ_HEAD_DIM)).astype(np.float32)
    rope = (rng.standard_normal((token_count, TQ_ROPE_HEAD_DIM)) * 0.1).astype(np.float32)
    query = torch.from_numpy((rng.standard_normal((1, 8, TQ_HEAD_DIM + TQ_ROPE_HEAD_DIM)) * 0.1).astype(np.float32)).to(
        torch.bfloat16
    )
    slot, reconstructed_unit, scale, stored_rope = _pack_sfa_slots(nope, rope)
    kv = torch.from_numpy(slot.view(np.int8)).reshape(physical_block_count, TQ_BLOCK_SIZE, 1, TQ_SLOT_BYTES).npu()

    block_table = np.array([[2, 0]], dtype=np.int32)
    logical_indices = rng.permutation(logical_block_count * TQ_BLOCK_SIZE)[:TQ_BLOCK_SIZE].astype(np.int32)
    assert np.unique(logical_indices // TQ_BLOCK_SIZE).size == logical_block_count
    physical_indices = (
        block_table[0, logical_indices // TQ_BLOCK_SIZE] * TQ_BLOCK_SIZE + logical_indices % TQ_BLOCK_SIZE
    )

    actual, _, _ = torch.ops._C_ascend.turboquant_sparse_flash_attention(
        query.npu(),
        kv,
        kv,
        torch.from_numpy(logical_indices).reshape(1, 1, -1).npu(),
        key_dequant_scale=None,
        value_dequant_scale=None,
        block_table=torch.from_numpy(block_table).npu(),
        actual_seq_lengths_query=torch.ones((1,), dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.full((1,), logical_block_count * TQ_BLOCK_SIZE, dtype=torch.int32).npu(),
        scale_value=1.0 / math.sqrt(TQ_HEAD_DIM),
        key_quant_mode=3,
        value_quant_mode=3,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=TQ_ROPE_HEAD_DIM,
        return_softmax_lse=False,
    )

    expected = _reference_sfa(
        query,
        reconstructed_unit[physical_indices],
        scale[physical_indices],
        stored_rope[physical_indices],
    )
    error = np.abs(actual.float().cpu().numpy()[0].astype(np.float64) - expected)
    tolerance = 2**-9 + 2**-9 * np.abs(expected)
    match_ratio = float((error <= tolerance).mean())
    max_abs_error = float(error.max())
    assert match_ratio >= 0.99
    assert max_abs_error <= 0.1
    print(f"SFA match ratio: {match_ratio:.6f}, max abs error: {max_abs_error:.8f}")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
