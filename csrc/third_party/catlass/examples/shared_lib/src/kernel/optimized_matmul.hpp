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

#ifndef SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H
#define SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H

// for supporting older gcc, to find the reason
#include <iostream>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass {
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    // When matrix A is row-major, if the number of rows in matrix A is less than 16,
    // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
    // The situation is similar for matrix B. If the above conditions are met,
    // please uncomment the following and comment out the original matrix A transfer method

    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

constexpr uint32_t alignByByte = 512;
constexpr uint32_t alignByElement = alignByByte / sizeof(half);
using ArchTag = Arch::AtlasA2;
constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool ENABLE_SHUFFLE_K = true;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

using ElementWorkspace = float;

template <class Layout> size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the
    // stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the
    // stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

// common val

template <class Type, bool PADDING> struct PaddingHelper {
    using Layout = typename Type::Layout;
    using Element = typename Type::Element;

    using LayoutPadding = std::conditional_t<std::is_same_v<Layout, layout::RowMajor>, layout::PaddingRowMajor,
                                             layout::PaddingColumnMajor>;
    using ActualType = std::conditional_t<PADDING, Gemm::GemmType<Element, LayoutPadding>, Type>;
    static const uint32_t COMPUTE_LENGTH = 96 * 1024 / sizeof(Element);
    using GlobalPadding = std::conditional_t<
        PADDING, Gemm::Kernel::PaddingMatrixBlockND<ArchTag, Element, Layout, LayoutPadding, COMPUTE_LENGTH>, void>;
    using LayoutW = std::conditional_t<PADDING, LayoutPadding, Layout>;

    CATLASS_DEVICE
    static LayoutW GetLayoutW(uint32_t a, uint32_t b, uint32_t padA, uint32_t padB)
    {
        if constexpr (PADDING) {
            LayoutPadding layoutW = LayoutPadding(a, b, padA, padB);
            return layoutW;

        } else {
            Layout layoutW = Layout(a, b);
            return layoutW;
        }
    }
};

template <class AType, class BType, class CType, bool PADDING_A, bool PADDING_B>
CATLASS_GLOBAL void optimized_matmul(uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC,
                                     GM_ADDR gmWA, GM_ADDR gmWB)
{
    using ArchTag = Arch::AtlasA2;
    AscendC::SetSyncBaseAddr(fftsAddr);

    using LayoutA = typename AType::Layout;
    using LayoutB = typename BType::Layout;
    using LayoutC = typename CType::Layout;
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementC = typename CType::Element;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    using L0TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using BlockEpilogue = void;
    LayoutA layoutA = LayoutA(problemShape.m(), problemShape.k());
    LayoutB layoutB = LayoutB(problemShape.k(), problemShape.n());
    LayoutC layoutC = LayoutC(problemShape.m(), problemShape.n());

    using PaddingHelperA = PaddingHelper<AType, PADDING_A>;
    using LayoutWA = typename PaddingHelperA::LayoutW;
    LayoutWA layoutWA = PaddingHelperA::GetLayoutW(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
    using ActualTypeA = typename PaddingHelperA::ActualType;
    using GlobalPaddingA = typename PaddingHelperA::GlobalPadding;

    using PaddingHelperB = PaddingHelper<BType, PADDING_B>;
    using LayoutWB = typename PaddingHelperB::LayoutW;
    LayoutWB layoutWB = PaddingHelperB::GetLayoutW(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
    using ActualTypeB = typename PaddingHelperB::ActualType;
    using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;

    using TileCopy = TileCopyOpt<ArchTag, ActualTypeA, ActualTypeB, CType>;
    using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, ActualTypeA, ActualTypeB,
                                                CType, void, TileCopy>;
    using MatmulKernel =
        Gemm::Kernel::OptimizedMatmul<GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
    using MatmulParams = typename MatmulKernel::Params;
    MatmulParams params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB};
    MatmulKernel matmul;
    matmul(params);
}
} // namespace Catlass

#endif // SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H