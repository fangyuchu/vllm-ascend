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
#include "catlass/gemv/block/block_gemv.hpp"

#include "catlass/gemv/kernel/kernel_gemv_aic.hpp"
#include "catlass/gemv/tile/tile_copy.hpp"

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"

#include "catlass/layout/layout.hpp"

using namespace Catlass;

using ScalarType = float;

template <
    class LayoutA,
    class LayoutX,
    class LayoutZ
>
CATLASS_GLOBAL 
void GemvAic(
    uint64_t fftsAddr,
    ScalarType alpha, ScalarType beta,
    GemvCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmX, LayoutX layoutX,
    GM_ADDR gmZ, LayoutZ layoutZ,
    GM_ADDR gmWorkspace
) 
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Arch::AtlasA2;
    using LayoutC = layout::RowMajor;   

    // Block level, define BlockGemv
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using L1TileShape = GemvShape<32, 512>;
    using L0TileShape = GemvShape<32, 256>;
    using AType = Gemm::GemmType<float, LayoutA>;
    using XType = Gemm::GemmType<float, LayoutX>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using BiasType = void;
    using TileCopy = Gemv::Tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, CType, BiasType>;
    using TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = Gemv::Block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, CType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = Epilogue::EpilogueAtlasA2Gemv;
    using YType = Gemm::GemmType<float, LayoutZ>;
    using ZType = Gemm::GemmType<float, LayoutZ>;
    using AXType = Gemm::GemmType<float, LayoutZ>; 

    using ComputeType = AXType;
    constexpr uint32_t computeLength = 8192;

    using TileElemWiseAddGemv = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, YType, AXType, ZType>;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, AXType, YType, ZType, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;


    // kernle levels
    using GemvKernel = Gemv::Kernel::KernelGemvAic<BlockGemv, BlockEpilogue>;

    // Prepare params
    typename BlockEpilogue::Params epilogueParams{alpha, beta, gmZ, layoutZ, gmZ, layoutZ};
    typename GemvKernel::Params params{problemShape, gmX, layoutX, gmA, layoutA, gmWorkspace, epilogueParams};

    // call a kernel
    GemvKernel gemv;
    gemv(params);
}

struct Options {
    const std::string HELPER = "20_gemv_aic m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{7};

    Options() = default;

    int Parse(int argc, const char** argv) 
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align) 
{
    if (align == 0) {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
                            RoundUp(layout.shape(1), align));
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align) 
{
    if (align == 0) {
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

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t lenZ = static_cast<size_t>(m) * 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeZ = lenZ * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    size_t sizeWorkspace = lenZ * sizeof(float);

    using LayoutX = layout::VectorLayout;
    using LayoutA = layout::ColumnMajor;
    using LayoutZ = layout::VectorLayout;

    LayoutX layoutX{n};
    LayoutA layoutA{m, n};
    LayoutZ layoutZ{m};

    ScalarType alpha{0};
    ScalarType beta{0};
    FillRandomScalarData(alpha, -1.0f, 1.0f);
    FillRandomScalarData(beta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostX, -1.0f, 1.0f);
    golden::FillRandomData(hostY, -1.0f, 1.0f);
    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeZ, hostY.data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GemvAic<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        alpha, beta,
        options.problemShape,
        deviceA, layoutA,
        deviceX, layoutX,
        deviceZ, layoutZ,
        deviceWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostRes(lenZ);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenZ);

    golden::ComputeGemv(options.problemShape, alpha, beta, hostA, layoutA, hostX, layoutX, hostY, layoutZ, hostGolden, layoutZ);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZ));
    ACL_CHECK(aclrtFree(deviceWorkspace));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}