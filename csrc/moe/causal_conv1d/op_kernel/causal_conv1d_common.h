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

/*!
 * \file causal_conv1d_common.h
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MAX_BLOCK_DIM = 2048;
constexpr int32_t RING_SLOTS = 8;
constexpr int32_t RING_MASK = RING_SLOTS - 1;

__aicore__ inline constexpr int32_t OutRingSlot(int32_t t)
{
    static_assert((RING_SLOTS > 0) && ((RING_SLOTS & (RING_SLOTS - 1)) == 0),
        "RING_SLOTS needs to be positive and a power of two");
    return t & RING_MASK;
}

} // namespace NsCausalConv1dCommon

#endif // CAUSAL_CONV1D_COMMON_H
