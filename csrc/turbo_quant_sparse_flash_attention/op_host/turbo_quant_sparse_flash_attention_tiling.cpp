/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_sparse_flash_attention_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/turbo_quant_sparse_flash_attention_template_tiling_key.h"
#include "turbo_quant_sparse_flash_attention_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_INT8}},
    {VALUE_NAME,                  {ge::DT_INT8}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}}
};

const std::map<std::string, std::vector<QSFALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {QSFALayout::BSND, QSFALayout::TND}},
    {KEY_NAME,               {QSFALayout::BSND, QSFALayout::TND, QSFALayout::PA_BSND}},
    {VALUE_NAME,             {QSFALayout::BSND, QSFALayout::TND, QSFALayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {QSFALayout::BSND, QSFALayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_INT32, "DT_INT32"},                   // int32 type
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"}                    // dt_variant type
};

struct TurboQuantSparseFlashAttentionCompileInfo {
    int64_t coreNum;
};

static const std::map<QSFALayout, std::vector<QSFAAxis>> QSFA_LAYOUT_AXIS_MAP = {
    {QSFALayout::BSND, {QSFAAxis::B, QSFAAxis::S, QSFAAxis::N, QSFAAxis::D}},
    {QSFALayout::TND, {QSFAAxis::T, QSFAAxis::N, QSFAAxis::D}},
    {QSFALayout::PA_BSND, {QSFAAxis::Bn, QSFAAxis::Bs, QSFAAxis::N, QSFAAxis::D}},
};

static const std::map<QSFALayout, size_t> QSFA_LAYOUT_DIM_MAP = {
    {QSFALayout::BSND, DIM_NUM_FOUR},
    {QSFALayout::TND, DIM_NUM_THREE},
    {QSFALayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string QSFADataTypeToSerialString(ge::DataType type)
{
    const auto qsfaIt = DATATYPE_TO_STRING_MAP.find(type);
    if (qsfaIt != DATATYPE_TO_STRING_MAP.end()) {
        return qsfaIt->second;
    } else {
        OPS_LOG_E("SparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

std::string QSFALayoutToSerialString(QSFALayout layout)
{
    switch (layout) {
        case QSFALayout::BSND: return "BSND";
        case QSFALayout::TND: return "TND";
        case QSFALayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus QSFAMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);     // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAMlaTiling::GetPlatformInfo()
{
    OPS_ERR_IF(qsfaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto qsfaAscendcPlatform = platform_ascendc::PlatformAscendC(qsfaInfo_->platformInfo);
    libapiSize_ = qsfaAscendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = qsfaAscendcPlatform.GetCoreNumAiv();
    aicNum_ = qsfaAscendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QSFAMlaTiling::GenTilingKey()
{
    uint32_t layoutQuery = static_cast<uint32_t>(qsfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(qsfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (qsfaInfo_->kvLayout == QSFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ = GET_TPL_TILING_KEY(0U, pageAttention, layoutQuery, layoutKV, \
        perfMode_ == QSFAPerfMode::V_TEMPLATE_MODE, static_cast<uint32_t>(qsfaInfo_->gSize > 64)); // G大于64时核间切G
    
}

void QSFAMlaTiling::ZeroTensorProcess() const
{
    if (qsfaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计�?
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        qsfaInfo_->s2Size = 1024;
    }
}

void QSFAMlaTiling::InitParams()
{
    perfMode_ = QSFAPerfMode::V_TEMPLATE_MODE;
    coreNum_ = aicNum_;

    headDimAlign_ = Align(qsfaInfo_->qHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对�?
    ZeroTensorProcess();
}

void QSFAMlaTiling::CalcUbBmm()
{
    uint32_t qsfaCubeMSize = qsfaInfo_->gSize * qsfaInfo_->s1Size;
    uint32_t qsfaMaxMSize = mBaseSize_;
    if (qsfaCubeMSize > qsfaMaxMSize) {
        qsfaCubeMSize = qsfaMaxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(qsfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(qsfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = qsfaInfo_->gSize * (headDimAlign_ + qsfaInfo_->ropeHeadDim) * qsfaInfo_->s1Size;
}

void QSFAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void QSFAMlaTiling::CalcInnerSize(uint32_t qsfaS2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && qsfaInfo_->qLayout != QSFALayout::TND) {
        if (qsfaS2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分�?份进行计算，
        } else if (qsfaS2Size > 256 && qsfaS2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (qsfaS2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = qsfaS2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > qsfaS2Size) {
        sInnerSize_ = qsfaS2Size;
    }
    sInnerSizeAlign_ =
        Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对�?
    CheckUbSpace();
}

void QSFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(qsfaInfo_->s2Size);
    InnerSplitParams qsfaInnerSplitParams;
    qsfaInnerSplitParams.s1GBaseSize = qsfaInfo_->gSize;
    tilingData_.innerSplitParams.set_mBaseSize(qsfaInnerSplitParams.s1GBaseSize);

    qsfaInnerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_s2BaseSize(qsfaInnerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void QSFAMlaTiling::Split()
{
    SplitBalanced();
}

void QSFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(qsfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(qsfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(qsfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(qsfaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(qsfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(qsfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(qsfaInfo_->n1Size / qsfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(qsfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(qsfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(qsfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(qsfaInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(qsfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(qsfaInfo_->sparseBlockCount);
    tilingData_.baseParams.set_dSizeVInput(qsfaInfo_->dSizeVInput);
    tilingData_.baseParams.set_headDim(qsfaInfo_->qHeadDim - qsfaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_ropeHeadDim(qsfaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_keyQuantMode(qsfaInfo_->keyQuantMode);
    tilingData_.baseParams.set_valueQuantMode(qsfaInfo_->valueQuantMode);
    tilingData_.baseParams.set_tileSize(qsfaInfo_->tileSize);
    tilingData_.baseParams.set_isActualLenDimsNull(qsfaInfo_->actualQSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_isActualLenDimsKVNull(qsfaInfo_->actualSeqLenFlag ? 0U : 1U);
}

// for flash decode
void QSFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信�?
    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * qsfaInfo_->n2Size * mBaseSize_ * headDimAlign_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信�?sum + max
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * qsfaInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void QSFAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void QSFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void QSFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t QSFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2 * qsfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信�?
}

void QSFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / qsfaInfo_->blockTypeSize); // log和sum的存储空间一致，共需�?份内�?
        workspaceSize_ += (accumOutSize + logSumExpSize) * qsfaInfo_->blockTypeSize;
    }
}

void QSFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void QSFAMlaTiling::GetWorkspaceSize()
{
    uint32_t actCoreNum = coreNum_;
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t qPreProcResElemSize = 0;
    uint32_t softmaxSumElemSize = 4;    // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize; // vec2ResGm
    workspaceSize_ += 4 * 512 * qsfaInfo_->qHeadDim * NUM_BYTES_FLOAT16 * actCoreNum;
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum);
    workspaceSize_ += (size_t)4 * 512 * NUM_BYTES_FLOAT16 * actCoreNum; // O8/O9 sTGm_ s_t [4*s2BaseSize] half/core

    CalcFDWorkSpace(actCoreNum);
}

void QSFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(qsfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
}

ge::graphStatus QSFAMlaTiling::DoOpTiling(QSFATilingInfo *qsfaInfo)
{
    qsfaInfo_ = qsfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTurboQuantSparseFlashAttention(gert::TilingContext *context)
{
    QSFATilingInfo qsfaInfo;
    QSFAInfoParser qsfaInfoParser(context);
    if (qsfaInfoParser.Parse(qsfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QSFATilingCheck tilingChecker(qsfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    QSFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&qsfaInfo);
}

ge::graphStatus TilingPrepareForTurboQuantSparseFlashAttention(gert::TilingParseContext* const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const QSFATilingShapeCompareParam &param, const QSFALayout &layout) const
{
    if (layout == QSFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == QSFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == QSFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OPS_LOG_E(opName_, "layout %s is unsupported", QSFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CompareShape(QSFATilingShapeCompareParam &param,
    const gert::Shape &shape, const QSFALayout &layout, const std::string &name) const
{
    gert::Shape qsfaShapeExpected;
    if (GetExpectedShape(qsfaShapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != qsfaShapeExpected.GetDimNum()) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");

        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != qsfaShapeExpected.GetDim(i)) {
            OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void QSFATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream qsfaOss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        qsfaOss << QSFADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            qsfaOss << ", ";
        }
    }
    OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");
}

ge::graphStatus QSFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *qsfaDesc,
    const std::string &name) const
{
    if (qsfaDesc != nullptr) {
        const auto& qsfaIt = DTYPE_SUPPORT_MAP.find(name);
        OPS_ERR_IF(qsfaIt == DTYPE_SUPPORT_MAP.end(),
            OPS_LOG_E(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &qsfaExpectDtypeList = qsfaIt->second;
        OPS_ERR_IF(std::find(
            qsfaExpectDtypeList.begin(), qsfaExpectDtypeList.end(),
            qsfaDesc->GetDataType()) == qsfaExpectDtypeList.end(),
            LogErrorDtypeSupport(qsfaExpectDtypeList, qsfaDesc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void QSFATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream qsfaOssNum;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        qsfaOssNum << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            qsfaOssNum << ", ";
        }
    }

    OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");
}

template <typename T>
void QSFATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus QSFATilingCheck::CheckDimNumInLayoutSupport(const QSFALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& qsfaDimIt = QSFA_LAYOUT_DIM_MAP.find(layout);
    OPS_ERR_IF(shape->GetStorageShape().GetDimNum() != qsfaDimIt->second,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &qsfaExpectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(qsfaExpectDimNumList.begin(), qsfaExpectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == qsfaExpectDimNumList.end()) {
        LogErrorDimNumSupport(qsfaExpectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void QSFATilingCheck::LogErrorLayoutSupport(const std::vector<QSFALayout> &expectLayoutList,
    const QSFALayout &actualLayout, const std::string &name) const
{
    std::ostringstream qsfaOssLayout;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        qsfaOssLayout << QSFALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            qsfaOssLayout << ", ";
        }
    }
    OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");
}

ge::graphStatus QSFATilingCheck::CheckLayoutSupport(const QSFALayout &actualLayout, const std::string &name) const
{
    const auto& qsfaItLayout = LAYOUT_SUPPORT_MAP.find(name);
    OPS_ERR_IF(qsfaItLayout == LAYOUT_SUPPORT_MAP.end(),
        OPS_LOG_E(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &qsfaExpectLayoutList = qsfaItLayout->second;
    OPS_ERR_IF(std::find(
        qsfaExpectLayoutList.begin(), qsfaExpectLayoutList.end(), actualLayout) == qsfaExpectLayoutList.end(),
        LogErrorLayoutSupport(qsfaExpectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> qsfaQueryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, qsfaQueryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> qsfaKeyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, qsfaKeyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseMode() const
{
    OPS_ERR_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseMode invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OPS_ERR_IF(
        ((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 16) ||
        (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
         static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDequantScaleNotExistence()
{
    if (quantScaleRepoMode_ == 1) {
        OPS_ERR_IF((opParamInfo_.keyDequantScale.tensor == nullptr ||
                     opParamInfo_.valueDequantScale.tensor == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(opName_, "key_dequant_scale and value_dequant_scale invalid"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckParaExistenceMlaAntiquant() const
{
    if (kvLayout_ == QSFALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == QSFALayout::TND) {
        OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsKv invalid"),
                   return ge::GRAPH_FAILED);
    } else if (kvLayout_ == QSFALayout::PA_BSND) {
        OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsKv invalid"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opName_, "blockTable invalid"),
                   return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaAntiquant();
}

ge::graphStatus QSFATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleNotExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

static ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const std::string &name, const char *opName)
{
    if (tensor == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid input");
        return ge::GRAPH_FAILED;
    }
    int64_t qsfaShapeSize = tensor->GetShapeSize();
    if (qsfaShapeSize <= 0) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid input");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(qsfaShapeSize);
    return ge::GRAPH_SUCCESS;
}

void QSFATilingCheck::SetQSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
}

ge::graphStatus QSFATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor != nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OPS_ERR_IF(blockTableBatch != bSize_,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input"),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "invalid input");
            return ge::GRAPH_FAILED;
        }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckTopkShape()
{
    QSFATilingShapeCompareParam qsfaShapeParams;
    qsfaShapeParams.B = bSize_;
    qsfaShapeParams.N = n2Size_;
    qsfaShapeParams.S = s1Size_;
    qsfaShapeParams.D = sparseBlockCount_;
    qsfaShapeParams.T = qTSize_;
    return CompareShape(qsfaShapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus QSFATilingCheck::CheckAttenOutShape()
{
    QSFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = qHeadDim_ - ropeHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQ()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ's dtype invalid");
        return ge::GRAPH_FAILED;
    }

    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ invalid");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensQShape()
{
    uint32_t qsfaShapeSize = 0;
    if (GetActualSeqLenSize(qsfaShapeSize, opParamInfo_.actualSeqLengthsQ.tensor,
        "actualSeqLengthsQ", opName_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qsfaShapeSize != bSize_) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths's dtype invalid");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths invalid");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t qsfaShapeSizeKv = 0;
    if (GetActualSeqLenSize(qsfaShapeSizeKv, opParamInfo_.actualSeqLengths.tensor,
        "actualSeqLengths", opName_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qsfaShapeSizeKv != bSize_) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckMultiParaConsistency()
{
    SetQSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantShape() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShapeSizes() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShapeSparseAndHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantShapeSizes() const
{
    OPS_ERR_IF(bSize_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "batch_size invalid"),
        return ge::GRAPH_FAILED);
        
    OPS_ERR_IF(qTSize_ <= 0 && (qLayout_ == QSFALayout::TND),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "T_size of query invalid"),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "q_head_num invalid"),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(n2Size_ != 1,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "kv_head_num invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ % n2Size_ != 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "q_head_num and kv_head_num invalid"),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "group num invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantShapeSparseAndHeadDim() const
{
    OPS_ERR_IF(sparseBlockSize_ <= 0 || (sparseBlockSize_ & (sparseBlockSize_ - 1)) != 0 || sparseBlockSize_ > 16,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize_ invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qHeadDim_ <= ropeHeadDim_,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "qHeadDim_ invalid"),
        return ge::GRAPH_FAILED);

    uint32_t kvLoraRank = qHeadDim_ - ropeHeadDim_;
    uint32_t tq4SlotBytes = kvLoraRank / 2 + ropeHeadDim_ * NUM_BYTES_BF16 + NUM_BYTES_FLOAT16;
    uint32_t expectedKHeadDim = tq4SlotBytes;
    OPS_ERR_IF(kHeadDim_ != expectedKHeadDim,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "kHeadDim_ invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantLayout() const
{
    const std::vector<std::string> qsfaLayoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OPS_ERR_IF(std::find(qsfaLayoutSupportList.begin(),
        qsfaLayoutSupportList.end(), layoutQuery) == qsfaLayoutSupportList.end(),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "query invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantDtype() const
{
    OPS_ERR_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "query invalid"),
        return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(inputKvType_ != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "key and value invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantAttr() const
{
    OPS_ERR_IF(attentionMode_ != 2, // 2:MLA-absorb
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "attention_mode invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyQuantMode_ != 3, // 3:TQ4 codebook (Phase B fused-in-SFA)
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "key_quant_mode invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(valueQuantMode_ != 3, // 3:TQ4 codebook (Phase B fused-in-SFA)
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "value_quant_mode invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(quantScaleRepoMode_ != 1, // 1:combine
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "quant_scale_repo_mode invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(preTokens_ != INT64_MAX,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "preTokens_ invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(nextTokens_ != INT64_MAX,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "nextTokens_ invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(tileSize_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "tile_size invalid"),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(ropeHeadDim_ <= 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "rope invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"),
        return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % 16 > 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"),
        return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % sparseBlockSize_ > 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMlaAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantAttr() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaAntiquant();
}

ge::graphStatus QSFATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void QSFATilingCheck::Init()
{
    opName_ = qsfaInfo_.opName;
    platformInfo_ = qsfaInfo_.platformInfo;
    opParamInfo_ = qsfaInfo_.opParamInfo;

    bSize_ = qsfaInfo_.bSize;
    n1Size_ = qsfaInfo_.n1Size;
    n2Size_ = qsfaInfo_.n2Size;
    s1Size_ = qsfaInfo_.s1Size;
    s2Size_ = qsfaInfo_.s2Size;
    gSize_ = qsfaInfo_.gSize;
    qHeadDim_ = qsfaInfo_.qHeadDim;
    kHeadDim_ = qsfaInfo_.kHeadDim;
    vHeadDim_ = qsfaInfo_.vHeadDim;
    ropeHeadDim_ = qsfaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = qsfaInfo_.maxBlockNumPerBatch;
    qTSize_ = qsfaInfo_.qTSize;
    kvTSize_ = qsfaInfo_.kvTSize;
    blockSize_ = qsfaInfo_.blockSize;
    sparseBlockCount_ = qsfaInfo_.sparseBlockCount;
    sparseBlockSize_ = qsfaInfo_.sparseBlockSize;

    attentionMode_ = qsfaInfo_.attentionMode;
    keyQuantMode_ = qsfaInfo_.keyQuantMode;
    valueQuantMode_ = qsfaInfo_.valueQuantMode;
    quantScaleRepoMode_ = qsfaInfo_.quantScaleRepoMode;
    tileSize_ = qsfaInfo_.tileSize;
    preTokens_ = qsfaInfo_.preTokens;
    nextTokens_ = qsfaInfo_.nextTokens;

    inputQType_ = qsfaInfo_.inputQType;
    inputKvType_ = qsfaInfo_.inputKvType;
    outputType_ = qsfaInfo_.outputType;

    qLayout_ = qsfaInfo_.qLayout;
    topkLayout_ = qsfaInfo_.topkLayout;
    kvLayout_ = qsfaInfo_.kvLayout;
    outLayout_ = qsfaInfo_.outLayout;

    kvStorageMode_ = qsfaInfo_.kvStorageMode;
    l2CacheSize_ = qsfaInfo_.l2CacheSize;
}

ge::graphStatus QSFATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();

static bool HasAxis(const QSFAAxis &axis, const QSFALayout &layout, const gert::Shape &shape)
{
    const auto& qsfaLayoutIt = QSFA_LAYOUT_AXIS_MAP.find(layout);
    if (qsfaLayoutIt == QSFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<QSFAAxis>& qsfaAxes = qsfaLayoutIt->second;
    const auto& qsfaAxisIt = std::find(qsfaAxes.begin(), qsfaAxes.end(), axis);
    if (qsfaAxisIt == qsfaAxes.end()) {
        return false;
    }

    const auto& qsfaDimIt = QSFA_LAYOUT_DIM_MAP.find(layout);
    if (qsfaDimIt == QSFA_LAYOUT_DIM_MAP.end() || qsfaDimIt->second != shape.GetDimNum()) {
        return false;
    }

    return true;
}

static size_t GetAxisIdx(const QSFAAxis &axis, const QSFALayout &layout)
{
    const std::vector<QSFAAxis>& axes = QSFA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);

    return std::distance(axes.begin(), axisIt);
}

static uint32_t GetAxisNum(const gert::Shape &shape, const QSFAAxis &axis, const QSFALayout &layout)
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : kInvalidDimValue;
}

ge::graphStatus QSFAInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor query invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor query invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor k invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor k invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor value invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.desc == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor value invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor sparseIndices invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor sparseIndices invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor output invalid"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor output invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layoutQuery == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "layoutQuery invalid"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.layoutKV == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "layoutKV invalid"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize invalid"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.scaleValue == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "scaleValue invalid"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseMode invalid"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, "actualSeqLengthsQ", opName_);
}

ge::graphStatus QSFAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "opName invalid");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto qsfaAscendcPlat = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t qsfaAivNum = qsfaAscendcPlat.GetCoreNumAiv();
    uint32_t qsfaAicNum = qsfaAscendcPlat.GetCoreNumAic();
    OPS_ERR_IF(qsfaAicNum == 0 || qsfaAivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);


    qsfaAscendcPlat.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.keyDequantScale.tensor = context_->GetOptionalInputTensor(KEY_DEQUANT_SCALE_INPUT_INDEX);
    opParamInfo_.valueDequantScale.tensor = context_->GetOptionalInputTensor(VALUE_DEQUANT_SCALE_INPUT_INDEX);
}

void QSFAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void QSFAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus QSFAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int64_t>(KEY_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.valueQuantMode = attrs->GetAttrPointer<int64_t>(VALUE_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    opParamInfo_.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
    opParamInfo_.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
    opParamInfo_.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetBatchSize()
{
    // 获取B基准�?
    // 1、非TND�? 以query的batch_size维度为基�?
    // 2、TND�? actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大�?
    if (qLayout_ == QSFALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, QSFAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus QSFAInfoParser::GetQTSize()
{
    // 获取query的T基准�?
    // 1、非TND�? 以query的batch_size维度为基�?
    // 2、TND�? actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大�?
    qTSize_ = (qLayout_ == QSFALayout::TND) ? GetAxisNum(queryShape_, QSFAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKVTSize()
{
    // 获取query的T基准�?
    // 1、非TND�? 以key的batch_size维度为基�?
    // 2、TND�? actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大�?
    kvTSize_ = (kvLayout_ == QSFALayout::TND) ? GetAxisNum(keyShape_, QSFAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetQHeadDim()
{
    // 获取qHeadDim基准�?
    // 以query的D维度为基�?
    qHeadDim_ = GetAxisNum(queryShape_, QSFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKHeadDim()
{
    // 获取kHeadDim基准�?
    // 以key的D维度为基�?
    kHeadDim_ = GetAxisNum(keyShape_, QSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS1Size()
{
    // 获取S1基准�?
    // 1、非TND�? 以query的S维度为基�?
    // 2、TND�? actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == QSFALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, QSFAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, QSFAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == QSFALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准�?
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetKvLayout()
{
    const map<string, QSFALayout> layoutKVMap = {
        {"BSND",        QSFALayout::BSND},
        {"PA_BSND",     QSFALayout::PA_BSND},
        {"TND",         QSFALayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "KV invalid");
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != QSFALayout::PA_BSND && qLayout_ != kvLayout_) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "KV and Q invalid");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == QSFALayout::PA_BSND && keyDimNum != 4U) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "key invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == QSFALayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, QSFAAxis::S, kvLayout_);
    } else if (kvLayout_ == QSFALayout::TND) { // TND
        s2Size_ = GetAxisNum(keyShape_, QSFAAxis::T, kvLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "blockTable invalid");
        return ge::GRAPH_FAILED;
    }
    uint32_t qsfaDimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (qsfaDimNum != DIM_NUM_TWO) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_table invalid");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_table invalid");
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, QSFAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, QSFAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetS2Size()
{
    // 获取S2基准�?
    // 1、BATCH_CONTINUOUS�? 从key的S轴获�?
    // 2、PAGE_ATTENTION�? S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus QSFAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准�?
    // 以value的D维度为基�?
    vHeadDim_ = GetAxisNum(valueShape_, QSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetDSizeKV()
{
    dSizeKV_ = GetAxisNum(keyShape_, QSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准�?
    // layoutQuery: {qLayout, outLayout}
    const std::map<std::string, std::pair<QSFALayout, QSFALayout>> qsfaLayoutMap = {
        {"BSND",        {QSFALayout::BSND,    QSFALayout::BSND}},
        {"TND",         {QSFALayout::TND,     QSFALayout::TND }},
    };

    std::string qsfaLayout(opParamInfo_.layoutQuery);
    auto qsfaLayoutIt = qsfaLayoutMap.find(qsfaLayout);
    if (qsfaLayoutIt != qsfaLayoutMap.end()) {
        qLayout_ = qsfaLayoutIt->second.first;
        outLayout_ = qsfaLayoutIt->second.second;
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "query invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, QSFAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, QSFAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::SetQSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();

    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
}

ge::graphStatus QSFAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSFAInfoParser::GetShapeAndSizeInfo()
{
    SetQSFAShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQHeadDim() ||
        ge::GRAPH_SUCCESS != GetKHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetDSizeKV() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void QSFAInfoParser::GenerateInfo(QSFATilingInfo &qsfaInfo)
{
    qsfaInfo.opName = opName_;
    qsfaInfo.platformInfo = platformInfo_;
    qsfaInfo.opParamInfo = opParamInfo_;

    qsfaInfo.bSize = bSize_;
    qsfaInfo.n1Size = n1Size_;
    qsfaInfo.n2Size = n2Size_;
    qsfaInfo.s1Size = s1Size_;
    qsfaInfo.s2Size = s2Size_;
    qsfaInfo.gSize = gSize_;
    qsfaInfo.qHeadDim = qHeadDim_;
    qsfaInfo.kHeadDim = kHeadDim_;
    qsfaInfo.vHeadDim = vHeadDim_;
    qsfaInfo.qTSize = qTSize_;
    qsfaInfo.kvTSize = kvTSize_;
    qsfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    qsfaInfo.sparseBlockCount = sparseBlockCount_;

    qsfaInfo.inputQType = inputQType_;
    qsfaInfo.inputKvType = inputKvType_;
    qsfaInfo.outputType = outputType_;

    qsfaInfo.kvStorageMode = kvStorageMode_;
    qsfaInfo.l2CacheSize = l2CacheSize_;

    qsfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    qsfaInfo.scaleValue = *opParamInfo_.scaleValue;
    qsfaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    qsfaInfo.blockSize = blockSize_;
    qsfaInfo.blockTypeSize =  sizeof(float);
    qsfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    FillTilingInfoAttrsAndLayouts(qsfaInfo);
}

void QSFAInfoParser::FillTilingInfoAttrsAndLayouts(QSFATilingInfo &qsfaInfo)
{
    qsfaInfo.actualLenDimsQ = actualLenDimsQ_;
    qsfaInfo.actualLenDimsKV = actualLenDimsKV_;
    qsfaInfo.maxActualseq = maxActualseq_;
    
    qsfaInfo.actualQSeqLenFlag = (opParamInfo_.actualSeqLengthsQ.tensor != nullptr);
    qsfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);

    qsfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    qsfaInfo.isSameActualseq = isSameActualseq_;

    qsfaInfo.sparseMode = *opParamInfo_.sparseMode;
    qsfaInfo.attentionMode = *opParamInfo_.attentionMode;
    qsfaInfo.keyQuantMode = *opParamInfo_.keyQuantMode;
    qsfaInfo.valueQuantMode = *opParamInfo_.valueQuantMode;
    qsfaInfo.quantScaleRepoMode = *opParamInfo_.quantScaleRepoMode;
    qsfaInfo.preTokens = *opParamInfo_.preTokens;
    qsfaInfo.nextTokens = *opParamInfo_.nextTokens;
    qsfaInfo.tileSize = *opParamInfo_.tileSize;
    qsfaInfo.ropeHeadDim = *opParamInfo_.ropeHeadDim;

    qsfaInfo.qLayout = qLayout_;
    qsfaInfo.topkLayout = topkLayout_;
    qsfaInfo.kvLayout = kvLayout_;
    qsfaInfo.outLayout = outLayout_;
    uint32_t tileSize = static_cast<uint32_t>(qsfaInfo.tileSize);
    if (qHeadDim_ > qsfaInfo.ropeHeadDim && tileSize > 0) {
        uint32_t kvLoraRank = qHeadDim_ - qsfaInfo.ropeHeadDim;
        qsfaInfo.dSizeVInput = kvLoraRank / 2 + qsfaInfo.ropeHeadDim * NUM_BYTES_BF16 + NUM_BYTES_FLOAT16;
    } else {
        qsfaInfo.dSizeVInput = dSizeKV_;
    }
}

ge::graphStatus QSFAInfoParser::Parse(QSFATilingInfo &qsfaInfo)
{
    if (context_ == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "tiling context invalid");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetShapeAndSizeInfo()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(qsfaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantSparseFlashAttention)
    .Tiling(TilingTurboQuantSparseFlashAttention)
    .TilingParse<TurboQuantSparseFlashAttentionCompileInfo>(TilingPrepareForTurboQuantSparseFlashAttention);
} // namespace optiling
