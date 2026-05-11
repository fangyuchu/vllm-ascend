/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_GROUPED_MATMUL_K_H
#define SHARED_LIB_IMPL_GROUPED_MATMUL_K_H

// for supporting older gcc, to find the reason
#include <iostream>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_k.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass{

template <class LayoutA, class LayoutB, class LayoutC>
CATLASS_GLOBAL void grouped_matmul_slice_k(GemmCoord problemShape,
                                       uint32_t problemCount,
                                       GM_ADDR gmGroupList, GM_ADDR gmA,
                                       LayoutA layoutA, GM_ADDR gmB,
                                       LayoutB layoutB, GM_ADDR gmC,
                                       LayoutC layoutC) {
  constexpr uint32_t preloadStages = 1;
  constexpr uint32_t l1Stages = 2;
  constexpr uint32_t l0AStages = 4;
  constexpr uint32_t l0BStages = 2;
  constexpr uint32_t l0CStages = 1;
  constexpr bool enableUnitFlag = true;
  constexpr bool enableShuffleK = true;

  using ArchTag = Arch::AtlasA2;
  using DispatchPolicy =
      Gemm::MmadAtlasA2PreloadAsync<preloadStages, l1Stages, l0AStages,
                                    l0BStages, l0CStages, enableUnitFlag,
                                    enableShuffleK>;
  using L1TileShape = GemmShape<128, 256, 256>;
  using L0TileShape = GemmShape<128, 256, 64>;

  using AType = Gemm::GemmType<half, LayoutA>;
  using BType = Gemm::GemmType<half, LayoutB>;
  using CType = Gemm::GemmType<half, LayoutC>;

  using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape,
                                           L0TileShape, AType, BType, CType>;
  using BlockEpilogue = void;
  using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

  // kernel level
  using MatmulKernel =
      Gemm::Kernel::GroupedMatmulSliceK<BlockMmad, BlockEpilogue,
                                        BlockScheduler, int32_t>;

  typename MatmulKernel::Params params{problemShape, problemCount, gmGroupList,
                                       gmA,          layoutA,      gmB,
                                       layoutB,      gmC,          layoutC};

  // call a kernel
  MatmulKernel matmul;
  matmul(params);
}
} // end of namespace Catlass;

#endif  // SHARED_LIB_IMPL_GROUPED_MATMUL_K_H