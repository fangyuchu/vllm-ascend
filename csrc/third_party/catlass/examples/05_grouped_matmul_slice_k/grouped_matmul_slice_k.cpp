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
#include <cstdlib>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_k.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void GroupedMatmul(
    GemmCoord problemShape,
    uint32_t problemCount, GM_ADDR gmGroupList,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
)
{
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 4;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceK<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

    typename MatmulKernel::Params params{
        problemShape, problemCount, gmGroupList, gmA, layoutA, gmB, layoutB, gmC, layoutC
    };

    // call a kernel
    MatmulKernel matmul;
    matmul(params);
}

struct Options {
    const std::string HELPER = "15_grouped_matmul_slice_k group_count m n k [device_id]";

    uint32_t groupCount{1};
    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            GROUP_COUNT_INDEX = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        groupCount = std::atoi(argv[GROUP_COUNT_INDEX]);
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

void Run(Options const &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.groupCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n * problemCount;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    using LayoutA = layout::ColumnMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    const fp16_t fp16_tLower = -5.0;
    const fp16_t fp16_tUpper = 5.0;
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, fp16_tLower, fp16_tUpper);
    golden::FillRandomData<fp16_t>(hostB, fp16_tLower, fp16_tUpper);
    auto groupList = golden::GenerateGroupList<int64_t>(k, problemCount);

    size_t sizeGroupList = problemCount * sizeof(int64_t);
    uint8_t *deviceGroupList{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceGroupList), sizeGroupList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceGroupList, sizeGroupList, groupList.data(), sizeGroupList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    GroupedMatmul<LayoutA, LayoutB, LayoutC><<<aicCoreNum, nullptr, stream>>>(
        options.problemShape,
        problemCount, deviceGroupList,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceC, layoutC);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<GemmCoord> problemShapeList(problemCount);
    std::vector<LayoutA> layoutAList(problemCount);
    std::vector<LayoutB> layoutBList(problemCount);
    std::vector<LayoutC> layoutCList(problemCount);
    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t currentK = (i == 0) ? groupList[0] : (groupList[i] - groupList[i - 1]);
        problemShapeList[i] = GemmCoord{m, n, currentK};
        layoutAList[i] = LayoutA{m, currentK};
        layoutBList[i] = LayoutB{currentK, n};
        layoutCList[i] = LayoutC{m, n};
    }

    std::vector<float> hostGolden(lenC);
    golden::ComputeGroupedMatmul(problemCount, problemShapeList, hostA, layoutAList,
        hostB, layoutBList, hostGolden, layoutCList);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k, groupList, m * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(deviceGroupList));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}