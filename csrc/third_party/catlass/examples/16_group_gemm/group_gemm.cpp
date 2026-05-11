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
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "helper.hpp"
#include "golden.hpp"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/kernel/group_gemm.hpp"
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
    typename LayoutA,
    typename LayoutB,
    typename LayoutX
>
CATLASS_GLOBAL
void GroupGemm(
    uint32_t problemCount,
    uint64_t fftsAddr,
    GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR ptrProblemShape,
    GM_ADDR gmA, GM_ADDR ptrLayoutA,
    GM_ADDR gmB, GM_ADDR ptrLayoutB,
    GM_ADDR gmX, GM_ADDR ptrLayoutX,
    GM_ADDR gmWA, GM_ADDR ptrLayoutWA,
    GM_ADDR gmWB, GM_ADDR ptrLayoutWB,
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

    using GroupGemmKernel = Gemm::Kernel::KernelGroupGemm<GemmBlock, EpilogueBlock>;
    typename GroupGemmKernel::Params params{problemCount, ptrProblemShape, alpha, beta, gmA, ptrLayoutA, gmB, ptrLayoutB, gmWorkspace, ptrLayoutX, gmWA, ptrLayoutWA, gmWB, ptrLayoutWB, gmX, gmX};

    GroupGemmKernel groupgemm;
    groupgemm(params);
}

struct Options {
    const std::string HELPER = "16_group_gemm groupCnt mlist nlist klist [deviceId]";
    uint32_t groupCnt = 8;
    std::vector<uint32_t> mList;
    std::vector<uint32_t> nList;
    std::vector<uint32_t> kList;
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv) {
        enum ArgsIndex {
            GROUPCNT_INDEX = 1,
            MLIST_INDEX,
            NLIST_INDEX,
            KLIST_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= KLIST_INDEX) 
        {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        groupCnt = std::atoi(argv[GROUPCNT_INDEX]);
        parseList(argv[MLIST_INDEX], mList);
        parseList(argv[NLIST_INDEX], nList);
        parseList(argv[KLIST_INDEX], kList);

        if (mList.size() != groupCnt || nList.size() != groupCnt || kList.size() != groupCnt) 
        {
            std::cerr << "List lengths do not match groupCnt." << std::endl;
            return -1;
        }

        if (argc == ARGS_MAX) 
        {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }

        return 0;
    }

private:
    void parseList(const char* str, std::vector<uint32_t>& list) 
    {
        char* copy = strdup(str);
        char* token = std::strtok(copy, ",");
        while (token != nullptr) {
            list.push_back(std::atoi(token));
            token = std::strtok(nullptr, ",");
        }
        free(copy);
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

void Run(Options& options){
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t groupCnt = options.groupCnt;
    
    const uint32_t align = 128; 
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutX = layout::RowMajor;

    // crate grouped matmul problem shapes and layouts
    std::vector<GemmCoord> problemShapeList(groupCnt);
    std::vector<LayoutA> layoutAList(groupCnt);
    std::vector<LayoutB> layoutBList(groupCnt);
    std::vector<LayoutX> layoutXList(groupCnt);
    std::vector<LayoutA> layoutWAList(groupCnt);
    std::vector<LayoutB> layoutWBList(groupCnt);

    uint64_t allMKCnt = 0;
    uint64_t allKNCnt = 0;
    uint64_t allMNCnt = 0;
    uint64_t allMKCntPadding = 0;
    uint64_t allKNCntPadding = 0;
    for (uint32_t i = 0; i < groupCnt; ++i) 
    {
        problemShapeList[i] = GemmCoord{options.mList[i], options.nList[i], options.kList[i]};
        layoutAList[i] = LayoutA{options.mList[i], options.kList[i]};
        layoutBList[i] = LayoutB{options.kList[i], options.nList[i]};
        layoutXList[i] = LayoutX{options.mList[i], options.nList[i]};
        layoutWAList[i] = GetWorkspaceLayout(layoutAList[i], align);
        layoutWBList[i] = GetWorkspaceLayout(layoutBList[i], align);
        allMKCnt += options.mList[i] * options.kList[i];
        allKNCnt += options.kList[i] * options.nList[i];
        allMNCnt += options.mList[i] * options.nList[i];
        allMKCntPadding += GetWorkspaceLen(layoutWAList[i]);
        allKNCntPadding += GetWorkspaceLen(layoutWBList[i]);
    }
    size_t scalarSize = groupCnt * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(groupCnt);
    std::vector<ScalarType> hostBeta(groupCnt);
    golden::FillRandomData(hostAlpha,  -1.0f, 1.0f);
    golden::FillRandomData(hostBeta,  -1.0f, 1.0f);

    size_t sizeA = allMKCnt * sizeof(float);
    size_t sizeB = allKNCnt * sizeof(float);
    size_t sizeX = allMNCnt * sizeof(float);
    size_t sizeC = allMNCnt * sizeof(float);
    std::vector<float> hostA(allMKCnt);
    std::vector<float> hostB(allKNCnt);
    std::vector<float> hostX(allMNCnt);
    golden::FillRandomData(hostA,  -1.0f, 1.0f);
    golden::FillRandomData(hostB,  -1.0f, 1.0f);
    golden::FillRandomData(hostX,  -1.0f, 1.0f);

    uint8_t* deviceAlpha{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceAlpha), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceAlpha, scalarSize, hostAlpha.data(), scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceBeta{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBeta), scalarSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBeta, scalarSize, hostBeta.data(), scalarSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWA = allMKCntPadding * sizeof(float);
    uint8_t *deviceWA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    size_t sizeWB = allKNCntPadding * sizeof(float);
    uint8_t *deviceWB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));
    
    uint8_t *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *problemShapeListDevice{nullptr};
    size_t sizeProblemShapeList = problemShapeList.size() * sizeof(GemmCoord);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&problemShapeListDevice), sizeProblemShapeList,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(problemShapeListDevice, sizeProblemShapeList,
        problemShapeList.data(), sizeProblemShapeList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutAListDevice{nullptr};
    size_t sizeLayoutAList = layoutAList.size() * sizeof(LayoutA);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutAListDevice), sizeLayoutAList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutAListDevice, sizeLayoutAList,
        layoutAList.data(), sizeLayoutAList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutBListDevice{nullptr};
    size_t sizeLayoutBList = layoutBList.size() * sizeof(LayoutB);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutBListDevice), sizeLayoutBList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutBListDevice, sizeLayoutBList,
        layoutBList.data(), sizeLayoutBList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutXListDevice{nullptr};
    size_t sizeLayoutXList = layoutXList.size() * sizeof(LayoutX);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutXListDevice), sizeLayoutXList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutXListDevice, sizeLayoutXList,
        layoutXList.data(), sizeLayoutXList, ACL_MEMCPY_HOST_TO_DEVICE));
    
    uint8_t *layoutWAListDevice{nullptr};
    size_t sizeLayoutWAList = layoutWAList.size() * sizeof(LayoutA);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutWAListDevice), sizeLayoutWAList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutWAListDevice, sizeLayoutWAList,
        layoutWAList.data(), sizeLayoutWAList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutWBListDevice{nullptr};
    size_t sizeLayoutWBList = layoutWBList.size() * sizeof(LayoutB);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutWBListDevice), sizeLayoutWBList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutWBListDevice, sizeLayoutWBList,
        layoutWBList.data(), sizeLayoutWBList, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GroupGemm<LayoutA, LayoutB, LayoutX><<<aicCoreNum, nullptr, stream>>>(
        groupCnt,
        fftsAddr,
        deviceAlpha, deviceBeta,
        problemShapeListDevice,
        deviceA, layoutAListDevice,
        deviceB, layoutBListDevice,
        deviceX, layoutXListDevice,
        deviceWA, layoutWAListDevice,
        deviceWB, layoutWBListDevice,
        gmWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostRes(allMNCnt);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeX, deviceX, sizeX, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostGolden(allMNCnt);
    golden::ComputeGroupGemm(groupCnt, problemShapeList, hostAlpha, hostBeta, hostA, layoutAList,
        hostB, layoutBList, hostX, layoutXList, hostGolden, layoutXList);
        
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, allMNCnt);
    if (errorIndices.empty()) 
    {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceWA));
    ACL_CHECK(aclrtFree(deviceWB));
    ACL_CHECK(aclrtFree(deviceAlpha));
    ACL_CHECK(aclrtFree(deviceBeta));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutXListDevice));
    ACL_CHECK(aclrtFree(layoutWAListDevice));
    ACL_CHECK(aclrtFree(layoutWBListDevice));
    ACL_CHECK(aclrtFree(gmWorkspace));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv){
    Options options;
    if(options.Parse(argc, argv) != 0){
        return -1;
    }
    Run(options);
    return 0;
}