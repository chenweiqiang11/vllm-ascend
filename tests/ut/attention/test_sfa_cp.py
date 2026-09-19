# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.context_parallel.common_cp import DCPMetadataBuilderMixin
from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)


def test_sfa_dcp_extends_v1_backend() -> None:
    assert issubclass(AscendSFADCPImpl, AscendSFAImpl)
    assert issubclass(
        AscendSFADCPMetadataBuilder,
        AscendSFAMetadataBuilder,
    )
    assert "dcp_context" not in {field.name for field in fields(AscendSFAMetadata)}
    assert "dcp_context" in {field.name for field in fields(AscendSFADCPMetadata)}


def test_sfa_dcp_builder_sizes_replicated_view_from_padded_block_table() -> None:
    def fake_base_init(self, *args, **kwargs) -> None:
        self.dcp_size = 2
        self.kernel_block_size = 128

    kv_cache_spec = SimpleNamespace(block_size=128)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
        ),
        model_config=SimpleNamespace(max_model_len=1024),
    )

    with patch.object(DCPMetadataBuilderMixin, "__init__", new=fake_base_init):
        builder = AscendSFADCPMetadataBuilder(
            kv_cache_spec,
            [],
            vllm_config,
            torch.device("cpu"),
        )

    assert builder.block_table_replicated_view_buf.shape == (5, 8)
    assert builder.arange_buffer.shape == (8,)


def _make_builder(rank: int = 0) -> AscendSFADCPMetadataBuilder:
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.dcp_size = 2
    builder.dcp_rank = rank
    builder.cp_kv_cache_interleave_size = 4
    builder.blocks_per_phys_block = 1
    builder.replicated_view_block_size = 4
    builder.device = torch.device("cpu")
    builder.block_table_replicated_view_buf = torch.empty(
        (4, 8),
        dtype=torch.int32,
    )
    builder.arange_buffer = torch.arange(8, dtype=torch.int32)
    builder.slot_mapping_replicated_view_buf = torch.empty(32, dtype=torch.int32)
    return builder


def test_sfa_dcp_local_sequence_lengths_follow_interleave_layout() -> None:
    seq_lens = torch.tensor([0, 3, 4, 5, 8, 9, 12], dtype=torch.int32)

    rank0 = _make_builder(rank=0)._get_dcp_local_seq_lens(seq_lens)
    rank1 = _make_builder(rank=1)._get_dcp_local_seq_lens(seq_lens)

    torch.testing.assert_close(rank0, torch.tensor([0, 3, 4, 4, 4, 5, 8], dtype=torch.int32))
    torch.testing.assert_close(rank1, torch.tensor([0, 0, 0, 1, 4, 4, 4], dtype=torch.int32))


def test_sfa_dcp_builds_replicated_block_table_view() -> None:
    builder = _make_builder()
    local_block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    seq_lens = torch.tensor([16], dtype=torch.int32)

    replicated = builder._build_block_table_replicated_view(
        local_block_table,
        seq_lens,
    )

    torch.testing.assert_close(
        replicated,
        torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32),
    )


def test_sfa_dcp_updates_dsa_cp_local_slot_mapping_with_padding() -> None:
    builder = _make_builder()
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=6,
        local_start=2,
        local_end_with_pad=5,
        slot_mapping_cp=None,
    )
    metadata = SimpleNamespace(dsa_cp_context=dsa_cp_context)

    builder._update_dsa_cp_slot_mapping_for_dcp(
        metadata,
        dcp_slot_mapping=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        num_input_tokens=4,
    )

    torch.testing.assert_close(
        dsa_cp_context.slot_mapping_cp,
        torch.tensor([12, 13, -1], dtype=torch.int32),
    )


def _make_tq_dcp_decode_case() -> SimpleNamespace:
    impl = MagicMock(spec=AscendSFADCPImpl)
    impl._has_prefill.return_value = False
    impl.kv_lora_rank = 512
    impl.dcp_group = MagicMock()

    gather_context = object()
    ql_nope = torch.randn(2, 1, 4)
    q_pe = torch.randn(2, 1, 2)
    gathered_ql_nope = torch.randn_like(ql_nope)
    gathered_q_pe = torch.randn_like(q_pe)
    topk_indices = torch.tensor([[[0, 1]], [[2, 3]]], dtype=torch.int32)
    gathered_topk = torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]], dtype=torch.int32)
    remapped_topk = torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.int32)
    rotated_query = torch.randn(2, 1, 6)
    attn_out = torch.randn(2, 1, 4, dtype=torch.bfloat16)
    softmax_max = torch.zeros(1, 2, 1)
    softmax_sum = torch.ones(1, 2, 1)
    merged_output = torch.randn_like(attn_out)

    impl._finish_dcp_gather.return_value = (gathered_ql_nope, gathered_q_pe)
    impl._remap_sparse_indices.return_value = remapped_topk
    impl._tq_rotate_query.return_value = rotated_query
    impl._turboquant_sfa.return_value = (attn_out, softmax_max, softmax_sum)
    impl._merge_dcp_outputs.return_value = merged_output

    return SimpleNamespace(
        impl=impl,
        gather_context=gather_context,
        ql_nope=ql_nope,
        q_pe=q_pe,
        kv_cache=(torch.empty(1),),
        topk_indices=topk_indices,
        gathered_topk=gathered_topk,
        remapped_topk=remapped_topk,
        merged_output=merged_output,
        block_table=torch.tensor([[0], [0]], dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([1, 1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([8, 8], dtype=torch.int32),
    )


def test_tq_dcp_decode_without_dsa_cp_context() -> None:
    case = _make_tq_dcp_decode_case()
    dcp_context = SimpleNamespace(
        gather_context=case.gather_context,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    metadata = SimpleNamespace(dcp_context=dcp_context)

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.tq_latent_store.had_inv",
        side_effect=lambda tensor, **_: tensor,
    ) as mock_had_inv:
        result = AscendSFADCPImpl._execute_tq_dcp_sfa(
            case.impl,
            case.ql_nope,
            case.q_pe,
            case.kv_cache,
            case.topk_indices,
            metadata,
            case.actual_seq_lengths_query,
            case.actual_seq_lengths_key,
        )

    case.impl.dcp_group.all_gather.assert_not_called()
    assert case.impl._remap_sparse_indices.call_args.args[0] is case.topk_indices
    tq_args = case.impl._turboquant_sfa.call_args.args
    assert tq_args[2] is case.remapped_topk
    assert tq_args[4] is case.actual_seq_lengths_query
    assert tq_args[5] is case.seq_lens
    assert case.impl._turboquant_sfa.call_args.kwargs == {
        "sparse_mode": 0,
        "return_softmax_lse": True,
    }
    assert case.impl._merge_dcp_outputs.call_args.args[2] is None
    assert dcp_context.gather_context is None
    assert mock_had_inv.call_args.args[0] is case.merged_output
    assert mock_had_inv.call_args.kwargs == {"head_dim": 512}
    torch.testing.assert_close(result, case.merged_output)


def test_tq_dcp_decode_with_dsa_cp_context() -> None:
    case = _make_tq_dcp_decode_case()
    case.impl.dcp_group.all_gather.return_value = case.gathered_topk
    dsa_cp_context = SimpleNamespace()
    cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    dcp_context = SimpleNamespace(
        gather_context=case.gather_context,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
    )
    metadata = SimpleNamespace(
        dcp_context=dcp_context,
        dsa_cp_context=dsa_cp_context,
        cum_query_lens=cum_query_lens,
    )

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.tq_latent_store.had_inv",
        side_effect=lambda tensor, **_: tensor,
    ):
        result = AscendSFADCPImpl._execute_tq_dcp_sfa(
            case.impl,
            case.ql_nope,
            case.q_pe,
            case.kv_cache,
            case.topk_indices,
            metadata,
            case.actual_seq_lengths_query,
            case.actual_seq_lengths_key,
        )

    all_gather_call = case.impl.dcp_group.all_gather.call_args
    torch.testing.assert_close(all_gather_call.args[0], case.topk_indices)
    assert all_gather_call.kwargs == {"dim": 0}
    assert case.impl._remap_sparse_indices.call_args.args[0] is case.gathered_topk
    tq_args = case.impl._turboquant_sfa.call_args.args
    assert tq_args[2] is case.remapped_topk
    assert tq_args[4] is cum_query_lens
    assert tq_args[5] is case.seq_lens
    assert case.impl._merge_dcp_outputs.call_args.args[2] is dsa_cp_context
    assert dcp_context.gather_context is None
    torch.testing.assert_close(result, case.merged_output)
