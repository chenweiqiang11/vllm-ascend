"""Phase B / B2 - TQ4 golden reference for the fused dequant-in-SFA op.

Mirrors kv_quant_sparse_flash_attention's MLA-absorb decode semantics, with the
affine dequant swapped for TQ4 codebook:
  - V = K (MLA absorb): value = the dequant'd nope.
  - QK over nope(Hadamard) + rope; scores * scaleValue; softmax; PV over v(nope).
  - dequant: centroids[nibble] * (1/sqrt(sum(c^2))) * vecNorm.
  - approach B: q is Hadamard-space; output stays Hadamard-space.

Self-contained, focused on the decode case (Sq=1). The real op will be validated
against this at rel 2e-4.
"""
import os

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None
from vllm_ascend.turboquant import tq_latent_store as t


def tq4_dequant_slots(slots_u8, centroids, head_dim=t.HEAD_DIM):
    """slots_u8 [Skv,base_slot_size(head_dim)] uint8 -> [Skv,head_dim] k_nope_had."""
    head_dim = int(head_dim)
    packed = t.packed_bytes(head_dim)
    S = slots_u8.shape[0]
    nb = slots_u8[:, :packed].to(torch.int32)
    lo = nb & 0xF
    hi = (nb >> 4) & 0xF
    nibbles = torch.stack([lo, hi], dim=-1).reshape(S, head_dim)
    cent = centroids[nibbles]
    norm_corr = 1.0 / torch.sqrt((cent * cent).sum(-1, keepdim=True) + 1e-16)
    vec_norm = slots_u8[:, packed:packed + t.SCALE_BYTES].contiguous().view(torch.float16).float().view(S, 1)
    return cent * norm_corr * vec_norm


def golden_tq4(q_nope_had, q_rope, slots_u8, rope_kv, vec_norm_u8, centroids, sparse_idx, scale_value,
               head_dim=t.HEAD_DIM):
    """One decode query (Sq=1), MLA-absorb sparse attention, TQ4 codebook dequant."""
    sl = torch.cat([slots_u8, vec_norm_u8], dim=-1)
    k_nope_had = tq4_dequant_slots(sl[sparse_idx], centroids, head_dim=head_dim)
    k_rope_sel = rope_kv[sparse_idx].float()
    v = k_nope_had
    scores = (q_nope_had.float() @ k_nope_had.T + q_rope.float() @ k_rope_sel.T) * scale_value
    p = torch.softmax(scores, dim=-1)
    return p @ v


if __name__ == "__main__":
    dev = os.environ.get("TQ_GOLDEN_DEVICE", "cpu")
    if dev.startswith("npu") and torch_npu is None:
        raise RuntimeError("TQ_GOLDEN_DEVICE requests NPU but torch_npu is not importable")

    head_dim = int(os.environ.get("TQ_HEAD_DIM", str(t.HEAD_DIM)))
    rope_head_dim = int(os.environ.get("TQ_ROPE_HEAD_DIM", str(t.ROPE_HEAD_DIM)))
    torch.manual_seed(0)
    t._build(dev, head_dim=head_dim)

    H, Skv = 1, 64
    packed = t.packed_bytes(head_dim)
    latent = torch.randn(Skv, head_dim, device=dev).to(torch.bfloat16)
    slot = t.compress(latent, head_dim=head_dim)
    slots_u8 = slot[:, :packed]
    vec_norm_u8 = slot[:, packed:packed + t.SCALE_BYTES]
    rope_kv = torch.randn(Skv, rope_head_dim, device=dev).to(torch.bfloat16)
    q_nope_had = torch.randn(H, head_dim, device=dev)
    q_rope = torch.randn(H, rope_head_dim, device=dev)
    cent = t._CENT.to(dev)
    sparse_idx = torch.arange(Skv, device=dev)
    out = golden_tq4(q_nope_had, q_rope, slots_u8, rope_kv, vec_norm_u8, cent, sparse_idx,
                     1.0 / (head_dim ** 0.5), head_dim=head_dim)
    print(f"[GOLDEN_TQ4] device={dev} head_dim={head_dim} rope_head_dim={rope_head_dim} "
          f"slot={tuple(slot.shape)} out={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"mean={out.mean().item():.5f} std={out.std().item():.5f}")
