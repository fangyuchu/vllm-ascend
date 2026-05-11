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

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/kernel/gemm.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

using namespace Catlass;

using ScalarType = float;

template <
    class LayoutA,
    class LayoutB,
    class LayoutX
>
CATLASS_GLOBAL
void GemmExample(
    uint64_t fftsAddr,
    ScalarType alpha, ScalarType beta,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmX, LayoutX layoutX,
    GM_ADDR gmWA, LayoutA layoutWA,
    GM_ADDR gmWB, LayoutB layoutWB,
    GM_ADDR gmWorkspace)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    constexpr bool enableABBA = true;
    using GemmBlockDispatchPolicy = Catlass::Gemm::GemmAtlasA2<enableUnitFlag, enableShuffleK, enableABBA>;
    using EpilogueBlockDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2Gemm;
    using AType = Gemm::GemmType<float, LayoutA>;
    using BType = Gemm::GemmType<float, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutX>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = GemmShape<128, 128, 64>;
    using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using DType = XType;
    using ComputeType = CType;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;
    using TileElemWiseAddGemm = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemm = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using TileElemWiseCastD = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using EpilogueBlock = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulsGemm, TileElemWiseCastD, EpilogueTileCopy>;
    typename EpilogueBlock::Params epilogueParams{alpha, beta, gmX, layoutX, gmX, layoutX};
    using GemmKernel = Gemm::Kernel::KernelGemm<GemmBlock, EpilogueBlock>;
    typename GemmKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmWorkspace, gmWA, layoutWA, gmWB, layoutWB, epilogueParams};
    GemmKernel gemm;
    gemm(params);
}

struct Options{
    const std::string HELPER = "15_gemm m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc <= K_INDEX)
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX)
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0) 
    {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(1), align));
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0) 
    {
        return layout;
    }
    return layout::ColumnMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(0), align));
}

size_t GetWorkspaceLen(layout::RowMajor layout)
{
    return layout.shape(0) * layout.stride(0);
}

size_t GetWorkspaceLen(layout::ColumnMajor layout)
{
    return layout.shape(1) * layout.stride(1);
}

bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
{
    return layout1.stride(0) == layout2.stride(0);
}

bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
{
    return layout1.stride(1) == layout2.stride(1);
}

template<class ElementRandom>
void FillRandomScalarData(ElementRandom &scalarData, ElementRandom low, ElementRandom high)
{
    scalarData = static_cast<ElementRandom>(low + (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low));
}

void Run(Options options)
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
    size_t lenX = lenC; 
    
    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);
    size_t sizeX = lenX * sizeof(float);

    const uint32_t align = 128;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutX = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutX layoutX{m, n};
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(float);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(float);

    ScalarType alpha{0};
    ScalarType beta{0};
    FillRandomScalarData(alpha, -1.0f, 1.0f);
    FillRandomScalarData(beta, -1.0f, 1.0f);
    std::vector<float> hostA(lenA);
    std::vector<float> hostB(lenB);
    std::vector<float> hostX(lenX);
    golden::FillRandomData(hostA,  -1.0f, 1.0f);
    golden::FillRandomData(hostB,  -1.0f, 1.0f);
    golden::FillRandomData(hostX,  -1.0f, 1.0f);
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *deviceWA{nullptr};
    if (IsSameStride(layoutWA, layoutA)) 
    {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    uint8_t *deviceWB{nullptr};
    if (IsSameStride(layoutWB, layoutB)) 
    {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    uint8_t *deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));
    
    uint8_t *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GemmExample<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        alpha, beta,
        options.problemShape,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceX, layoutX,
        deviceWA, layoutWA,
        deviceWB, layoutWB,
        gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    
    std::vector<float> hostRes(lenX);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeX, deviceX, sizeX, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostGolden(lenX);
    golden::ComputeGemm(options.problemShape, alpha, beta, hostA, layoutA, hostB, layoutB, hostX, layoutX, hostGolden, layoutX);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m * n);
    if (errorIndices.empty()) 
    {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX));
    if (!IsSameStride(layoutWA, layoutA)) 
    {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (!IsSameStride(layoutWB, layoutB)) 
    {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtFree(gmWorkspace));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if(options.Parse(argc, argv) != 0)
    {
        return -1;
    }
    Run(options);
    return 0;
}