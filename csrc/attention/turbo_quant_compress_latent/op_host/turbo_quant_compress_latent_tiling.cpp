#include "turbo_quant_compress_latent_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {

constexpr uint32_t MAX_AI_CORES = 40;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* latentShape = context->GetInputShape(0);
    uint32_t numTokens = latentShape->GetStorageShape().GetDim(0);
    if (numTokens < 1) numTokens = 1;

    uint32_t tokensPerCore = (numTokens + MAX_AI_CORES - 1) / MAX_AI_CORES;
    if (tokensPerCore < 1) tokensPerCore = 1;
    uint32_t blockDim = (numTokens + tokensPerCore - 1) / tokensPerCore;

    TurboQuantCompressLatentTilingData tilingData;
    tilingData.set_num_tokens(numTokens);
    tilingData.set_tokens_per_core(tokensPerCore);

    context->SetBlockDim(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantCompressLatent)
    .Tiling(TilingFunc);

} // namespace optiling
