#ifndef TQ_COMPRESS_LATENT_H_
#define TQ_COMPRESS_LATENT_H_

#include "kernel_operator.h"

using namespace AscendC;

// TurboQuant COMPRESS for the MLA KV latent (reverse of tq_dequant_latent).
// Input  latent_h [N, 512] fp16  (= post-rmsnorm latent already @ signed-Hadamard, NOT normalized)
// Output slot     [N, 320] int8  ([0:256) 512 nibbles packed, [256:258) vecNorm fp16, [258:320) pad)
// Per token: norm=||z|| ; u=z/norm ; nibble[d] = #{midpoint-boundaries <= u[d]} (nearest of 16
// sorted centroids) ; int16[i]=nib[4i]|nib[4i+1]<<4|nib[4i+2]<<8|nib[4i+3]<<12 ; store vecNorm.
// NOTE: HEAD_DIM/SLOT_PAD hardcoded to 512/320 (GLM-5.1 kv_lora_rank=512). De-hardcode is a
// follow-up optimization (see tq4-csrc-dehardcode memory); port as-is for now.
constexpr uint32_t HEAD_DIM     = 512;
constexpr uint32_t SLOT_PAD     = 320;
constexpr uint32_t PACKED_BYTES = 256;   // 512 nibbles / 2
constexpr uint32_t N_CENT       = 16;
constexpr uint32_t ALIGN_BYTES  = 64;

class KernelTurboQuantCompressLatent {
public:
    __aicore__ inline KernelTurboQuantCompressLatent() {}

    __aicore__ inline void Init(GM_ADDR latent, GM_ADDR centroids, GM_ADDR slotOut,
                                uint32_t numTokens, uint32_t tokensPerCore)
    {
        numTokens_ = numTokens;
        uint32_t cid = GetBlockIdx();
        tokStart_ = cid * tokensPerCore;
        tokEnd_ = tokStart_ + tokensPerCore;
        if (tokEnd_ > numTokens_) tokEnd_ = numTokens_;

        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(latent));
        centGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(slotOut));

        uint32_t fp32B = ((HEAD_DIM * sizeof(float)) + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);
        uint32_t slotB = ((SLOT_PAD) + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);
        uint32_t maskB = ((HEAD_DIM / 8) + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);

        pipe_.InitBuffer(inQ_, 1, fp32B);
        pipe_.InitBuffer(outQ_, 1, slotB);
        pipe_.InitBuffer(uBuf_, fp32B);
        pipe_.InitBuffer(nibBuf_, fp32B);
        pipe_.InitBuffer(tmpBuf_, fp32B);
        pipe_.InitBuffer(selBuf_, fp32B);
        pipe_.InitBuffer(packHalfBuf_, ((HEAD_DIM * sizeof(half)) + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1));
        pipe_.InitBuffer(oneBuf_, fp32B);
        pipe_.InitBuffer(redBuf_, fp32B);
        pipe_.InitBuffer(centBuf_, ((N_CENT * sizeof(float)) + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1));
        pipe_.InitBuffer(maskBuf_, maskB);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> cent = centBuf_.Get<float>();
        DataCopy(cent, centGm_, N_CENT);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i + 1 < N_CENT; ++i) {
            bnd_[i] = (cent.GetValue(i) + cent.GetValue(i + 1)) * 0.5f;   // 15 midpoint boundaries
        }
        LocalTensor<float> one = oneBuf_.Get<float>();
        Duplicate(one, 1.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = tokStart_; t < tokEnd_; ++t) {
            CopyIn(t);
            Compute();
            CopyOut(t);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t)
    {
        LocalTensor<float> in = inQ_.AllocTensor<float>();
        Duplicate(in, 0.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        DataCopy(in, latentGm_[(uint64_t)t * HEAD_DIM], HEAD_DIM);
        inQ_.EnQue(in);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<float> in = inQ_.DeQue<float>();        // z = latent @ H (fp32, un-normalized)
        LocalTensor<uint8_t> slot = outQ_.AllocTensor<uint8_t>();
        LocalTensor<float> u = uBuf_.Get<float>();
        LocalTensor<float> nib = nibBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> sel = selBuf_.Get<float>();
        LocalTensor<float> one = oneBuf_.Get<float>();
        LocalTensor<float> red = redBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        Mul(tmp, in, in, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        ReduceSum(red, tmp, red, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        float norm = sqrt(red.GetValue(0) + 1e-16f);
        Duplicate(u, 0.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Muls(u, in, 1.0f / norm, HEAD_DIM);                 // u = z / norm
        PipeBarrier<PIPE_V>();

        Duplicate(nib, 0.0f, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        for (uint32_t b = 0; b + 1 < N_CENT; ++b) {
            CompareScalar(mask, u, bnd_[b], CMPMODE::GE, HEAD_DIM);   // mask = u >= bnd[b]
            PipeBarrier<PIPE_V>();
            Select(sel, mask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, HEAD_DIM);  // 1 where ge
            PipeBarrier<PIPE_V>();
            Add(nib, nib, sel, HEAD_DIM);
            PipeBarrier<PIPE_V>();
        }
        // ---- int4b_t HW pack: nib(0..15, dim order) -> signed s(-8..7) -> half -> int4b_t (low-first) ----
        // s = (nib < 8) ? nib : nib - 16  (two's-complement 4-bit: same 4 bits as nib, signed view).
        // Cast(half -> int4b_t) packs 2 nibbles/byte low-first -> slot[0:256] in dim order.
        LocalTensor<half> packHalf = packHalfBuf_.Get<half>();
        CompareScalar(mask, nib, 8.0f, CMPMODE::LT, HEAD_DIM);                       // mask = nib < 8
        PipeBarrier<PIPE_V>();
        Adds(sel, nib, -16.0f, HEAD_DIM);                                            // nib - 16
        PipeBarrier<PIPE_V>();
        Select(tmp, mask, nib, sel, SELMODE::VSEL_TENSOR_TENSOR_MODE, HEAD_DIM);     // s (float)
        PipeBarrier<PIPE_V>();
        Cast(packHalf, tmp, RoundMode::CAST_RINT, HEAD_DIM);                         // float -> half
        PipeBarrier<PIPE_V>();
        LocalTensor<int4b_t> i4 = slot.ReinterpretCast<int4b_t>();
        Cast(i4, packHalf, RoundMode::CAST_RINT, HEAD_DIM);                          // half -> int4b_t -> slot[0:256]
        PipeBarrier<PIPE_V>();
        half hn = (half)norm;
        uint16_t nbits = *reinterpret_cast<uint16_t*>(&hn);
        slot.SetValue(PACKED_BYTES, (uint8_t)(nbits & 0xff));
        slot.SetValue(PACKED_BYTES + 1, (uint8_t)((nbits >> 8) & 0xff));
        for (uint32_t i = PACKED_BYTES + 2; i < SLOT_PAD; ++i) slot.SetValue(i, (uint8_t)0);

        inQ_.FreeTensor(in);
        outQ_.EnQue(slot);
    }

    __aicore__ inline void CopyOut(uint32_t t)
    {
        LocalTensor<uint8_t> slot = outQ_.DeQue<uint8_t>();
        DataCopy(slotGm_[(uint64_t)t * SLOT_PAD], slot, SLOT_PAD);
        outQ_.FreeTensor(slot);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQ_;
    TQue<QuePosition::VECOUT, 1> outQ_;
    TBuf<TPosition::VECCALC> uBuf_, nibBuf_, tmpBuf_, selBuf_, oneBuf_, redBuf_, centBuf_, maskBuf_;
    TBuf<TPosition::VECCALC> packHalfBuf_;
    GlobalTensor<float> latentGm_;
    GlobalTensor<float> centGm_;
    GlobalTensor<uint8_t> slotGm_;
    uint32_t numTokens_, tokStart_, tokEnd_;
    float bnd_[N_CENT];
};

#endif // TQ_COMPRESS_LATENT_H_
