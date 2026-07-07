// Phase B — baked TQ4 codebook for the fused dequant-in-SFA kernel.
// 16 Lloyd-Max centroids (deterministic seed-0 kmeans, scaled to N(0,1/512)). Baking them as a kernel
// constant means the fused op needs NO new `centroids` input -> the torch_npu binding interface is
// unchanged (B3 ~vanishes). Kernel Init loads TQ4_CENT_SIGNED into a 16-elem UB buffer for the Gather.
//
// centSigned = cent rotated by 8: the int4b unpack gives a SIGNED nibble s in [-8,7]; the lookup index
// is (s+8) in [0,15]; centSigned[s+8] == cent[original_idx]  (orig = s<0 ? s+16 : s). Verified bit-exact
// against tq_latent_store._CENT (test_tq4_golden_dequant.py: dequant cos=1.0).
#ifndef TQ4_CENTROIDS_H_
#define TQ4_CENTROIDS_H_

constexpr int TQ4_N_CENT = 16;

// gather order: centSigned[idx], idx = signed_nibble + 8
__aicore__ inline void Tq4LoadCentSigned(float (&cs)[TQ4_N_CENT]) {
    cs[0]  =  0.0054729f; cs[1]  =  0.0168041f; cs[2]  =  0.0285761f; cs[3]  =  0.0410862f;
    cs[4]  =  0.0549298f; cs[5]  =  0.0710182f; cs[6]  =  0.0911537f; cs[7]  =  0.1203780f;
    cs[8]  = -0.1209128f; cs[9]  = -0.0911112f; cs[10] = -0.0711246f; cs[11] = -0.0551360f;
    cs[12] = -0.0413207f; cs[13] = -0.0287497f; cs[14] = -0.0170049f; cs[15] = -0.0056868f;
}

#endif // TQ4_CENTROIDS_H_
