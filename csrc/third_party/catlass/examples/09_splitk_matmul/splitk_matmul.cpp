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
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"
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
void SplitkMatmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWorkspace, uint32_t splitkFactor
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // After the Matmul computation is completed, launch the ReduceAdd kernel to accumulate the partial sums.
    constexpr uint32_t computeLength = 32 * 1024 / sizeof(float);
    using ReduceAdd = Catlass::Gemm::Kernel::ReduceAdd<ArchTag, float, half, computeLength>;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::SplitkMatmul<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

        typename MatmulKernel::Params params{
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWorkspace, splitkFactor
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::SplitkMatmul<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

        typename MatmulKernel::Params params{
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWorkspace, splitkFactor
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

struct Options {
    const std::string HELPER = "09_splitk_matmul m n k [device_id]";

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

// A splitk strategy, which may not be the optimal one.
uint32_t GetSplitkFactor(uint32_t m, uint32_t n, uint32_t k, uint32_t m0, uint32_t n0, uint32_t k0, uint32_t aicCoreNum)
{
    uint32_t maxSplitkFactor;
    if (k <= 1024) {
        // When k is less than or equal to 1024, it can be divided into at most 2 parts.
        maxSplitkFactor = 2;
    } else if (k <= 2048) {
        // When k is less than or equal to 2048, it can be divided into at most 4 parts.
        maxSplitkFactor = 4;
    } else if (k <= 4096) {
        // When k is less than or equal to 4096, it can be divided into at most 8 parts.
        maxSplitkFactor = 8;
    } else {
        // else it can be divided into at most 16 parts.
        maxSplitkFactor = 16;
    }
    uint32_t splitkFactor = 1;
    uint32_t baseTilesCount = CeilDiv(m, m0) * CeilDiv(n, n0);
    splitkFactor = std::min(aicCoreNum / baseTilesCount, maxSplitkFactor);
    // Prevent the split factor form being less than 1
    splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(1));
    if (baseTilesCount < aicCoreNum) {
        while (splitkFactor + 1 <= maxSplitkFactor &&
            CeilDiv(baseTilesCount * splitkFactor, aicCoreNum) >= CeilDiv(baseTilesCount, aicCoreNum) * splitkFactor) {
            splitkFactor += 1;
        }
    }
    // Ensure that splitkFactor is less than the number of base tiels in the k direction.
    splitkFactor = std::min(CeilDiv(k, k0), splitkFactor);
    // If k is very large, splitting k can lead to better cache utilization.
    // If k is greater than 8192.
    if (k > 8192) {
        // split the k direction into at least 2 parts.
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(2));
    }
    // If k is greater than 32768.
    if (k > 32768) {
        // split the k direction into at least 4 parts.
        splitkFactor = std::max(splitkFactor, static_cast<uint32_t>(4));
    }
    return splitkFactor;
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    using L1TileShape = GemmShape<128, 256, 256>;
    // Divide into s parts, using the K-direction tile block as the splitting unit.
    uint32_t splitkFactor = GetSplitkFactor(m, n, k, L1TileShape::M, L1TileShape::N, L1TileShape::K, aicCoreNum);

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;
    size_t lenWorkspace = static_cast<size_t>(m) * n * splitkFactor;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);
    size_t sizeWorkspace = lenWorkspace * sizeof(float);

    layout::RowMajor layoutA{m, k};
    layout::ColumnMajor layoutB{k, n};
    layout::RowMajor layoutC{m, n};

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

    uint8_t *deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));

    SplitkMatmul<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC,
        deviceWorkspace, splitkFactor
    );
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
    ACL_CHECK(aclrtFree(deviceWorkspace));

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
