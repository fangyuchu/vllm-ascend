/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/kernel/padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {

template <class PrologueA, class PrologueB, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class OptimizedMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template <class T> struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <> struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayoutWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayoutWB, typename LayoutHelper<PrologueB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;

        // Methods
        CATLASS_DEVICE
        Params()
        {
        }

        CATLASS_DEVICE
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_,
               LayoutA layoutA_,
               GM_ADDR ptrB_,
               LayoutB layoutB_,
               GM_ADDR ptrC_,
               LayoutC layoutC_,
               GM_ADDR ptrWA_,
               LayoutWA layoutWA_,
               GM_ADDR ptrWB_,
               LayoutWB layoutWB_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrWA(ptrWA_),
              layoutWA(layoutWA_),
              ptrWB(ptrWB_),
              layoutWB(layoutWB_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    OptimizedMatmul()
    {
    }

    template <int32_t CORE_TYPE = g_coreType> CATLASS_DEVICE void operator()(Params const &params);

    template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
    }

    /// Executes matmul
    template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockIdxCoord.k() * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockIdCoord);
            }
            MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M, nextBlockIdCoord.k() * L1TileShape::K};
            MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K, nextBlockIdCoord.n() * L1TileShape::N};
            int64_t gmOffsetNextA = params.layoutWA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutWB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutWA, gmB[gmOffsetB], params.layoutWB, gmC[gmOffsetC], params.layoutC,
                      gmA[gmOffsetNextA], gmB[gmOffsetNextB], actualBlockShape, nextActualBlockShape, isFirstBlock,
                      hasNextBlock);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_OPTIMIZED_MATMUL_HPP