/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAUSAL_CONV1D_TILING_PLANNER_H
#define CAUSAL_CONV1D_TILING_PLANNER_H

#include "causal_conv1d_tiling_utils.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"

#include <limits>
#include <numeric>

namespace optiling::causal_conv1d_host {

using namespace Ops::Transformer::OpTiling;

// Number of vector lanes for fp16/bf16; channel tiles are rounded up to a multiple of
// this so that every tile (except possibly the last) is a full, alignment-friendly width.
// Must divide MAX_DIM_TILE_SIZE
constexpr int64_t MIN_CHANNELS_PER_TILE = 128;

inline DimTileChoice ChooseChannelTileChoice(int64_t dim)
{
    if (dim <= 0) {
        return {};
    }

    int64_t numChannels = CeilDivInt64(dim, MAX_DIM_TILE_SIZE);
    int64_t channelsPerTile = CeilDivInt64(CeilDivInt64(dim, numChannels), MIN_CHANNELS_PER_TILE) * MIN_CHANNELS_PER_TILE;
    numChannels = CeilDivInt64(dim, channelsPerTile);

    DimTileChoice result;
    result.baseDim = channelsPerTile;
    result.baseDimCnt = numChannels;
    result.gridSize = numChannels;
    return result;
}

inline DimTileChoice ChooseCanonicalUpdateBaseDimChoice(gert::TilingContext *context, int64_t batch, int64_t dim,
                                                        uint32_t coreNum)
{
    if (dim <= 0 || batch <= 0 || coreNum == 0) {
        return {};
    }
    
    DimTileChoice result = ChooseChannelTileChoice(dim);
    result.gridSize *= batch;

    OP_LOGD(context, "DimTile(update) chosen: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld].", result.baseDim,
            result.baseDimCnt, result.gridSize);
    return result;
}

inline int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }

    int64_t tokenCoreBudget = static_cast<int64_t>(coreNum);
    if (fnExecutionPlan == FN_EXECUTION_PLAN_CUTBSD) {
        tokenCoreBudget = std::max<int64_t>(1, tokenCoreBudget / baseDimCnt);
    }
    return tokenCoreBudget;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t batch, int64_t width,
                                                      int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan,
                                                      uint32_t coreNum);

inline TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt,
                                                            FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }

    mapping.tokenBlocksPerCore = CeilDivInt64(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

inline FnTokenSeqRangePlan BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch, int64_t tokenBlockSize,
                                                    int64_t tokenBlockCnt)
{
    FnTokenSeqRangePlan plan;
    if (qslData == nullptr || batch <= 0 || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
        return plan;
    }

    plan.enabled = true;
    plan.rangeCount = tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tokenBlockSize;

        while (seq < batch && qslData[seq + 1] <= tokenStart) {
            ++seq;
        }

        int64_t endSeq = seq;
        while (endSeq < batch && qslData[endSeq] < tokenEnd) {
            ++endSeq;
        }

        plan.tokenTileStartSeq[tokenTileId] = seq;
        plan.tokenTileEndSeq[tokenTileId] = endSeq;
    }
    return plan;
}

inline VarlenTokenTileChoice ChooseUnifiedFnTokenBlockPlan(gert::TilingContext *context,
                                                           const CausalConv1dTilingData &tiling,
                                                           const DimTileChoice &baseDimChoice,
                                                           FnExecutionPlan fnExecutionPlan,
                                                           uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        baseDimChoice.baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return tokenBlockChoice;
    }
    if (tiling.hasNumAcceptedTokens != 0) {
        OP_LOGD(context, "Varlen token tiling disabled: speculative decode still uses the existing seq mapping.");
        return tokenBlockChoice;
    }

    tokenBlockChoice = ChooseFnTokenBlockChoice(tiling.cuSeqlen, tiling.batch, tiling.width, baseDimChoice.baseDimCnt,
                                                fnExecutionPlan, coreNum);

    OP_LOGD(context,
            "FnTokenTile(plan=%ld): cuSeqlen[%ld], baseDimCnt[%ld], tokenBlockSize[%ld], "
            "tokenBlockCnt[%ld], gridSize[%ld].",
            static_cast<int64_t>(fnExecutionPlan), tiling.cuSeqlen, baseDimChoice.baseDimCnt,
            tokenBlockChoice.tokenBlockSize, tokenBlockChoice.tokenBlockCnt, tokenBlockChoice.gridSize);
    return tokenBlockChoice;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t batch, int64_t width,
                                                      int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan,
                                                      uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0 || batch <= 0) {
        return tokenBlockChoice;
    }

    const int64_t avgSeqLen = std::max<int64_t>(1, cuSeqlen / batch);
    // Hard safety floor guaranteeing tokenBlockCnt <= tokenCoreBudget (see function doc).
    const int64_t minBlockSize = std::max<int64_t>(1, CeilDivInt64(cuSeqlen, tokenCoreBudget));

    int64_t numCores = static_cast<int64_t>(coreNum);
    int64_t batchReduced = batch;
    const int64_t gcdCoreBatch = std::gcd(numCores, batchReduced);
    if (gcdCoreBatch > 0) {
        numCores /= gcdCoreBatch;
        batchReduced /= gcdCoreBatch;
    }

    const int64_t depthNumerator = batchReduced * baseDimCnt;
    const int64_t gcdCoreChannels = std::gcd(numCores, baseDimCnt);
    const int64_t uppBnd = (gcdCoreChannels > 0) ? (numCores / gcdCoreChannels) : 1;

    double bestScore = std::numeric_limits<double>::infinity();
    int64_t bestBlockSize = minBlockSize;
    for (int64_t numChunks = 1; numChunks <= uppBnd; ++numChunks) {
        const int64_t depth = CeilDivInt64(depthNumerator * numChunks, numCores);
        const int64_t tokens = CeilDivInt64(avgSeqLen, numChunks);
        const double work = static_cast<double>(tokens + width);
        const double score = static_cast<double>(depth) * work;
        if (score < bestScore) {
            bestScore = score;
            bestBlockSize = std::max<int64_t>(1, CeilDivInt64(avgSeqLen, numChunks));
        }
    }
    bestBlockSize = std::max<int64_t>(bestBlockSize, minBlockSize);

    tokenBlockChoice.enabled = true;
    tokenBlockChoice.tokenBlockSize = bestBlockSize;
    tokenBlockChoice.tokenBlockCnt = CeilDivInt64(cuSeqlen, tokenBlockChoice.tokenBlockSize);
    tokenBlockChoice.gridSize = tokenBlockChoice.tokenBlockCnt * baseDimCnt;
    return tokenBlockChoice;
}

inline FnHostPlan ChooseFnHostPlan(gert::TilingContext *context, const CausalConv1dTilingData &tiling, uint64_t ubSize,
                                   uint32_t coreNum)
{
    FnHostPlan plan;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        tiling.dim <= 0 || coreNum == 0) {
        return plan;
    }

    plan.baseDimChoice = ChooseChannelTileChoice(tiling.dim);
    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }
    plan.baseDimChoice.gridSize = tiling.batch * plan.baseDimChoice.baseDimCnt;

    if (plan.baseDimChoice.baseDimCnt <= 1) {
        plan.caseKind = FN_TILING_CASE_TOKEN_FIRST;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBS;
    } else {
        plan.caseKind = FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBSD;
    }

    plan.tokenBlockChoice =
        ChooseUnifiedFnTokenBlockPlan(context, tiling, plan.baseDimChoice, plan.executionPlan, coreNum);
    if (!plan.tokenBlockChoice.enabled || plan.tokenBlockChoice.tokenBlockSize <= 0 ||
        plan.tokenBlockChoice.tokenBlockCnt <= 0 || plan.tokenBlockChoice.gridSize <= 0) {
        return {};
    }

    plan.tokenCoreMapping = BuildFnTokenCoreMappingChoice(plan.tokenBlockChoice.tokenBlockCnt,
                                                          plan.baseDimChoice.baseDimCnt, plan.executionPlan, coreNum);
    if (plan.tokenCoreMapping.tokenCoreBudget <= 0 || plan.tokenCoreMapping.blockDim <= 0) {
        return {};
    }
    if (plan.tokenCoreMapping.blockDim > static_cast<int64_t>(coreNum)) {
        plan.tokenCoreMapping.blockDim = static_cast<int64_t>(coreNum);
    }
    return plan;
}

} // namespace optiling::causal_conv1d_host

#endif // CAUSAL_CONV1D_TILING_PLANNER_H
