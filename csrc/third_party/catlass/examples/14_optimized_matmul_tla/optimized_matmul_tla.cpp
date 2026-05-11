/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/optimized_matmul_tla.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;
using fp16_t = op::fp16_t;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class LayoutWA,
    class LayoutWB,
    class BlockMmad,
    class PaddingA,
    class PaddingB
>
CATLASS_DEVICE
void LaunchMatmulDynamicSwizzle(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, LayoutWA layoutWA,
    GM_ADDR gmWB, LayoutWB layoutWB
)
{
    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmulTla<
            BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB};
        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmulTla<
            BlockMmad, BlockEpilogue, BlockScheduler, PaddingA, PaddingB>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template<class Layout>
CATLASS_DEVICE
auto GetPaddingLayout(Layout layout, uint32_t blockRows, uint32_t blockCols)
{
    if constexpr (std::is_same_v<Layout, layout::RowMajor>) {
        auto shape = MakeShape(MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(
                static_cast<int64_t>(blockCols),
                static_cast<int64_t>(blockRows) * RoundUp(layout.shape(1), blockCols)
            ),
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols)
        );
        return MakeLayout(shape, stride);
    } else {
        auto shape = MakeShape(MakeShape(blockRows, CeilDiv(layout.shape(0), blockRows)),
            MakeShape(blockCols, CeilDiv(layout.shape(1), blockCols)));
        auto stride = MakeStride(
            MakeStride(Int<1>{}, static_cast<int64_t>(blockRows) * blockCols),
            MakeStride(
                static_cast<int64_t>(blockRows),
                RoundUp(layout.shape(0), blockRows) * static_cast<int64_t>(blockCols)
            )
        );
        return MakeLayout(shape, stride);
    }
}

template <
    class LayoutTagA,
    class LayoutTagB,
    class LayoutTagC,
    bool IS_PADDING_A,
    bool IS_PADDING_B
>
CATLASS_GLOBAL
void OptimizedMatmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutTagA tagA,
    GM_ADDR gmB, LayoutTagB tagB,
    GM_ADDR gmC, LayoutTagC tagC,
    GM_ADDR gmWA, GM_ADDR gmWB
)
{
    using ElementA = half;
    using ElementB = half;
    using ElementC = half;
    using ArchTag = Arch::AtlasA2;
    AscendC::SetSyncBaseAddr(fftsAddr);

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);
    using TensorA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), AscendC::TPosition::GM>;
    using TensorB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), AscendC::TPosition::GM>;
    using TensorC = Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), AscendC::TPosition::GM>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutTagA, layout::ColumnMajor> &&
        std::is_same_v<LayoutTagB, layout::ColumnMajor>, Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    using L0TileShape = std::conditional_t<std::is_same_v<LayoutTagA, layout::ColumnMajor> &&
        std::is_same_v<LayoutTagB, layout::ColumnMajor>, Shape<_256, _128, _64>, Shape<_128, _256, _64>>;
    if constexpr (!IS_PADDING_A && !IS_PADDING_B) {
        // no need to padding A and B.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), AscendC::TPosition::GM>;
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, false, false>;
        using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using PaddingA = void;
        using PaddingB = void;
        LaunchMatmulDynamicSwizzle<
            decltype(layoutA), decltype(layoutB), decltype(layoutC), decltype(layoutWA), decltype(layoutWB),
            BlockMmad, PaddingA, PaddingB
        >(problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if constexpr (!IS_PADDING_A && IS_PADDING_B) {
        // no need to padding A, but B needs padding.
        auto layoutWA = MakeLayout(layoutA.shape(), layoutA.stride());
        auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), AscendC::TPosition::GM>;
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, false, true>;
        using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        using PaddingA = void;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        LaunchMatmulDynamicSwizzle<
            decltype(layoutA), decltype(layoutB), decltype(layoutC), decltype(layoutWA), decltype(layoutWB),
            BlockMmad, PaddingA, PaddingB
        >(problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if constexpr (IS_PADDING_A && !IS_PADDING_B) {
        // no need to padding B, but A needs padding.
        auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        auto layoutWB = MakeLayout(layoutB.shape(), layoutB.stride());
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), AscendC::TPosition::GM>;
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, true, false>;
        using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        using PaddingB = void;
        LaunchMatmulDynamicSwizzle<
            decltype(layoutA), decltype(layoutB), decltype(layoutC), decltype(layoutWA), decltype(layoutWB),
            BlockMmad, PaddingA, PaddingB
        >(problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else {
        // Both A and B need padding.
        auto layoutWA = GetPaddingLayout(tagA, get<0>(L1TileShape{}), get<2>(L1TileShape{}));
        auto layoutWB = GetPaddingLayout(tagB, get<2>(L1TileShape{}), get<1>(L1TileShape{}));
        using TensorWA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutWA), AscendC::TPosition::GM>;
        using TensorWB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutWB), AscendC::TPosition::GM>;
        using TileCopy = Gemm::Tile::PaddingPackedTileCopyTla<ArchTag, TensorWA, LayoutTagA, TensorWB, LayoutTagB,
            TensorC, LayoutTagC, void, void, true, true>;
        using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, TensorWA, TensorWB,
            TensorC, void, TileCopy>;
        constexpr const uint32_t computeLengthA = 96 * 1024 / sizeof(ElementA);
        using PaddingA = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorA, TensorWA, computeLengthA>;
        constexpr const uint32_t computeLengthB = 96 * 1024 / sizeof(ElementB);
        using PaddingB = Catlass::Gemm::Kernel::PaddingMatrixBlockND<ArchTag, TensorB, TensorWB, computeLengthB>;
        LaunchMatmulDynamicSwizzle<
            decltype(layoutA), decltype(layoutB), decltype(layoutC), decltype(layoutWA), decltype(layoutWB),
            BlockMmad, PaddingA, PaddingB
        >(problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    }
}

struct Options {
    const std::string HELPER = "14_optimizd_matmul_tla m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

template<class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
        RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    const uint32_t align = 256;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, Shape<_256, _128, _256>, Shape<_128, _256, _256>>;
    size_t sizeWA = GetWorkspaceLen(layoutA, get<0>(L1TileShape{}), get<2>(L1TileShape{})) * sizeof(fp16_t);
    size_t sizeWB = GetWorkspaceLen(layoutB, get<2>(L1TileShape{}), get<1>(L1TileShape{})) * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    if (!isNeedPaddingA && !isNeedPaddingB) {
        constexpr const bool isPaddingA = false;
        constexpr const bool isPaddingB = false;
        OptimizedMatmul<LayoutA, LayoutB, LayoutC, isPaddingA, isPaddingB><<<aicCoreNum, nullptr, stream>>>(
            fftsAddr, options.problemShape,
            deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceWA, deviceWB
        );
    } else if (isNeedPaddingA && !isNeedPaddingB) {
        constexpr const bool isPaddingA = true;
        constexpr const bool isPaddingB = false;
        OptimizedMatmul<LayoutA, LayoutB, LayoutC, isPaddingA, isPaddingB><<<aicCoreNum, nullptr, stream>>>(
            fftsAddr, options.problemShape,
            deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceWA, deviceWB
        );
    } else if (!isNeedPaddingA && isNeedPaddingB) {
        constexpr const bool isPaddingA = false;
        constexpr const bool isPaddingB = true;
        OptimizedMatmul<LayoutA, LayoutB, LayoutC, isPaddingA, isPaddingB><<<aicCoreNum, nullptr, stream>>>(
            fftsAddr, options.problemShape,
            deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceWA, deviceWB
        );
    } else {
        constexpr const bool isPaddingA = true;
        constexpr const bool isPaddingB = true;
        OptimizedMatmul<LayoutA, LayoutB, LayoutC, isPaddingA, isPaddingB><<<aicCoreNum, nullptr, stream>>>(
            fftsAddr, options.problemShape,
            deviceA, layoutA, deviceB, layoutB, deviceC, layoutC, deviceWA, deviceWB
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
