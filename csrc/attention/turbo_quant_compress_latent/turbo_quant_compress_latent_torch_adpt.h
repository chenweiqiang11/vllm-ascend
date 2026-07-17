#ifndef TQ_COMPRESS_LATENT_TORCH_ADPT_H
#define TQ_COMPRESS_LATENT_TORCH_ADPT_H

namespace vllm_ascend {

// TurboQuant compress: latent [N,512] fp32 (post-rmsnorm, @ signed-Hadamard, NOT normalized)
// + centroids [16] fp32 -> slot [N,320] uint8 (TQ4 nibbles packed + vecNorm fp16 + pad).
// Hadamard matmul + centroid prep stay in Python (tq_latent_store.compress_kernel); this op
// wraps only the aclnnTurboQuantCompressLatent host launch. SLOT_PAD=320 hardcoded (GLM-5.1).
at::Tensor turbo_quant_compress_latent(const at::Tensor &latent, const at::Tensor &centroids)
{
    int64_t N = latent.size(0);
    at::Tensor slot = at::empty({N, 320}, latent.options().dtype(at::kByte));
    EXEC_NPU_CMD(aclnnTurboQuantCompressLatent, latent, centroids, slot);
    return slot;
}

} // namespace vllm_ascend
#endif
