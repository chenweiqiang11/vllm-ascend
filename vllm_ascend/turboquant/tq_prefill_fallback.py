"""Prefill fallback for TQ_FUSED: dequant the 386B combined slot (active blocks) -> k_had dense (approach B,
no inverse Hadamard) + rope, remapped block_table, for the old npu_sparse_flash_attention. Used only for
prefill (T>1); decode uses the fused op. Matches the fused op's approach-B basis so results are consistent.

Portable: imports tq_latent_store from the same package.
"""
import torch
from vllm_ascend.turboquant import tq_latent_store as t


CHUNK_BLOCKS = 64  # process 64 block_table slots at a time -> 64*128 rows/chunk (~32MB nibbles); avoids OOM


@torch.no_grad()
def dequant_combined_386(cache_i8, block_table, dtype=torch.bfloat16,
                         head_dim=t.HEAD_DIM, rope_head_dim=t.ROPE_HEAD_DIM):
    """cache_i8 [nb,bs,1,fused_slot_size] int8 -> (k_had [A,bs,1,head_dim], rope [A,bs,1,rope_head_dim], remapped_bt).
    k_had = centroid[nibble] * scale, scale=vecNorm/sqrt(sum c^2) precomputed in 2B (Hadamard space, approach B).
    Chunked over CHUNK_BLOCKS block_table slots to avoid OOM at long context / high concurrency (full B*max_blocks
    materialized at once OOMs at 64k+). Gathers by POSITION -- NO unique / NO boolean-mask (aclnnUnique2 &
    aclnnNonzeroV2 both crash on the bigger max-num-seqs>1 profiling dummy); new_bt = positional index."""
    head_dim = int(head_dim)
    rope_head_dim = int(rope_head_dim)
    t._build(cache_i8.device, head_dim=head_dim)
    packed = t.packed_bytes(head_dim)
    rope_bytes = rope_head_dim * t.ROPE_DTYPE_BYTES
    rope_off = packed
    scale_off = rope_off + rope_bytes
    slot_size = scale_off + t.SCALE_BYTES
    dev = cache_i8.device
    bs = cache_i8.shape[1]
    h = cache_i8.shape[2]
    bt_flat = block_table.reshape(-1)                                    # [P], P = B * max_blocks
    P = bt_flat.numel()
    cent = t._CENT.to(dev)

    k_had_parts = []
    rope_parts = []
    for start in range(0, P, CHUNK_BLOCKS):
        end = min(start + CHUNK_BLOCKS, P)
        chunk_idx = bt_flat[start:end].clamp(min=0).to(torch.long)       # -1 padding -> 0 (masked out below)
        blk = cache_i8.index_select(0, chunk_idx).contiguous().view(torch.uint8)  # [chunk,bs,1,fused_slot_size]
        A = end - start
        flat = blk.reshape(A * bs * h, slot_size)
        nbf = flat[:, :packed].float()                                  # packed nibbles; float-arith unpack
        hi = torch.floor(nbf * (1.0 / 16.0))                            # high nibble [0,15] (no >> aclop)
        lo = nbf - hi * 16.0                                            # low nibble [0,15] (no & aclop)
        nibbles = torch.stack([lo, hi], dim=-1).reshape(-1, head_dim).to(torch.long)
        c = cent[nibbles]
        scale = flat[:, scale_off:scale_off + t.SCALE_BYTES].contiguous().view(torch.float16).float().view(-1, 1)
        k_had_parts.append((c * scale).to(dtype).reshape(A, bs, h, head_dim))  # approach-B basis
        rope_parts.append(flat[:, rope_off:scale_off].contiguous().view(torch.bfloat16).reshape(
            A, bs, h, rope_head_dim).to(dtype))
        del nbf, hi, lo, nibbles, c, scale, flat, blk

    k_had = torch.cat(k_had_parts, dim=0)                                # [P,bs,1,head_dim]
    rope = torch.cat(rope_parts, dim=0)                                  # [P,bs,1,rope_head_dim]
    # new_bt: block_table slot i -> local index i in k_had; keep -1 padding (torch.where = select, not nonzero)
    pos = torch.arange(P, device=dev, dtype=block_table.dtype).reshape(block_table.shape)
    new_bt = torch.where(block_table >= 0, pos, block_table)
    return k_had, rope, new_bt
