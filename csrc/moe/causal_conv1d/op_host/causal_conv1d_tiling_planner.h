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

namespace optiling::causal_conv1d_host {

using namespace Ops::Transformer::OpTiling;

// Number of vector lanes for fp16/bf16; channel tiles are rounded up to a multiple of
// this so that every tile (except possibly the last) is a full, alignment-friendly width.
constexpr int64_t MIN_CHANNELS_PER_TILE = 128;

// The UB budget actually available for a channel tile, derived from the real per-chip UB
// size queried at tiling time (falls back to MAX_DIM_TILE_SIZE, the kernel's compiled
// buffer width, whichever is smaller -- the kernel's ring/calc buffers are fixed-size at
// compile time, so baseDim can never exceed that regardless of how much UB the chip has).
inline int64_t ComputeUbLimitedMaxChannelsPerTile(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }

    const int64_t bytesPerElem = (RING_SLOT_CNT * static_cast<int64_t>(sizeof(float))) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

// Channel-tile (baseDim) selection, ported from sgl-kernel-npu's `tiling_causal_conv1d`
// (ChristosMatzoros/sgl-kernel-npu, causal-conv1d-tiling branch,
// csrc/causal_conv1d/op_host/causal_conv1d.cpp): pick the largest per-tile channel width
// (rounded to a MIN_CHANNELS_PER_TILE-lane multiple) that still fits inside the UB
// budget, using the fewest tiles needed to cover `dim`. This selection is independent of
// coreNum/batch by design -- in both the reference kernel and this one, any parallelism
// left over after channel-tiling is filled by splitting the token/sequence axis instead
// (see ChooseFnTokenBlockChoice below), so over-splitting channels here would only add
// needless per-tile overhead (extra weight/bias reloads).
inline DimTileChoice ChooseChannelTileChoice(int64_t dim, int64_t maxChannelsPerTile)
{
    if (dim <= 0 || maxChannelsPerTile <= 0) {
        return {};
    }

    int64_t numChannels = CeilDivInt64(dim, maxChannelsPerTile);
    int64_t channelsPerTile = CeilDivInt64(dim / numChannels, MIN_CHANNELS_PER_TILE) * MIN_CHANNELS_PER_TILE;
    numChannels = CeilDivInt64(dim, channelsPerTile);

    DimTileChoice result;
    result.baseDim = channelsPerTile;
    result.baseDimCnt = numChannels;
    result.gridSize = numChannels;
    return result;
}

// Canonical/update (decode) mode has no sequence-splitting kernel path: each grid task
// serially processes the *whole* per-request token window in one shot (see
// ProcessDefaultByWindowMode in op_kernel/causal_conv1d.h). So unlike Fn mode there is no
// second axis available to soak up leftover core parallelism once the channel axis has
// been tiled -- the channel axis is the only tunable knob. We therefore fold both roles
// (fitting the UB budget *and* filling idle cores) into one search over candidate tile
// widths `d`, scored with the same depth * work principle as the reference tiling
// (ChristosMatzoros/sgl-kernel-npu tiling_causal_conv1d): `depth` is how many sequential
// grid-waves a core must execute, and `work` approximates the cost of one wave (compute
// proportional to seqLength * d, plus a fixed per-wave reload overhead proportional to
// the conv width, mirroring the reference's own "+ width" per-chunk overhead term).
inline DimTileChoice ChooseCanonicalUpdateBaseDimChoice(gert::TilingContext *context, int64_t batch, int64_t dim,
                                                        int64_t seqLength, int64_t width, uint64_t ubSize,
                                                        uint32_t coreNum)
{
    const int64_t maxChannelsPerTile = ComputeUbLimitedMaxChannelsPerTile(ubSize);
    if (dim <= 0 || batch <= 0 || coreNum == 0 || maxChannelsPerTile <= 0) {
        return {};
    }
    const int64_t seqLen = std::max<int64_t>(1, seqLength);

    double bestScore = std::numeric_limits<double>::infinity();
    int64_t channelsPerTile = MIN_CHANNELS_PER_TILE;

    auto scoreFunc = [&](int64_t d) {
        const int64_t tileNumPerCore = CeilDivInt64(batch * CeilDivInt64(dim, d), static_cast<int64_t>(coreNum));
        const double tileWork = static_cast<double>(seqLen) * static_cast<double>(d) + static_cast<double>(width);
        return static_cast<double>(tileNumPerCore) * tileWork;
    };

    int64_t d = MIN_CHANNELS_PER_TILE;
    while (d <= maxChannelsPerTile) {
        const double score = scoreFunc(d);
        if (score <= bestScore) {
            bestScore = score;
            channelsPerTile = d;
        }

        const int64_t k = CeilDivInt64(dim, d);
        if (k <= 1) {
            break;
        }
        int64_t next = CeilDivInt64(dim, k - 1);
        next = CeilDivInt64(next, MIN_CHANNELS_PER_TILE) * MIN_CHANNELS_PER_TILE;
        if (next <= d) {
            break;
        }
        d = next;
    }

    const int64_t channelTiles = CeilDivInt64(dim, channelsPerTile);
    DimTileChoice result;
    result.baseDim = channelsPerTile;
    result.baseDimCnt = channelTiles;
    result.gridSize = batch * channelTiles;

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

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum);

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

    tokenBlockChoice = ChooseFnTokenBlockChoice(tiling.cuSeqlen, baseDimChoice.baseDimCnt, fnExecutionPlan, coreNum);

    OP_LOGD(context,
            "FnTokenTile(plan=%ld): cuSeqlen[%ld], baseDimCnt[%ld], tokenBlockSize[%ld], "
            "tokenBlockCnt[%ld], gridSize[%ld].",
            static_cast<int64_t>(fnExecutionPlan), tiling.cuSeqlen, baseDimChoice.baseDimCnt,
            tokenBlockChoice.tokenBlockSize, tokenBlockChoice.tokenBlockCnt, tokenBlockChoice.gridSize);
    return tokenBlockChoice;
}

// Token/sequence-axis splitting for Fn mode. sgl-kernel-npu's `tiling_causal_conv1d`
// greedily searches chunk counts to minimize depth * (tokens + width), where `depth` is
// how many sequential grid-waves a core needs and `tokens` is the per-chunk token count
// -- a genuine trade-off there because its kernel tolerates gridSize > blockDim (a
// grid-stride loop picks up the remainder), so `depth` can legitimately exceed 1.
//
// vllm-ascend's fn kernel cannot do that: it indexes tasks directly by blockIdx with no
// grid-stride loop (see ResolveFnDirectBlockTask), so tokenBlockCnt is hard-clamped to
// tokenCoreBudget below, which forces `depth` to exactly 1 for every reachable
// tokenBlockSize. With depth pinned at 1, score(blockSize) = 1 * (blockSize + width) is
// monotonically increasing in blockSize, so it is minimized at the smallest reachable
// blockSize -- which is minBlockSize itself, by construction. A search over chunk counts
// therefore always converges to exactly the closed-form minBlockSize (verified: 0
// mismatches across the full 168-shape benchmark grid), while actually costing 15-120us
// of host CPU per call for the larger shapes (measured: up to ~45000x the closed form) --
// pure overhead with no effect on the chosen tiling. So skip the search and compute
// minBlockSize directly.
inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0) {
        return tokenBlockChoice;
    }

    const int64_t idealBlockSize = CeilDivInt64(cuSeqlen, tokenCoreBudget);
    tokenBlockChoice.enabled = true;
    tokenBlockChoice.tokenBlockSize = std::max<int64_t>(1, idealBlockSize);
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

    const int64_t maxChannelsPerTile = ComputeUbLimitedMaxChannelsPerTile(ubSize);
    plan.baseDimChoice = ChooseChannelTileChoice(tiling.dim, maxChannelsPerTile);
    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }
    plan.baseDimChoice.gridSize = tiling.batch * plan.baseDimChoice.baseDimCnt;
    plan.executionPlan = static_cast<FnExecutionPlan>(ResolveFnExecutionPlan(plan.baseDimChoice.baseDimCnt));
    plan.caseKind =
        (plan.baseDimChoice.baseDimCnt <= 1) ? FN_TILING_CASE_TOKEN_FIRST : FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;

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
