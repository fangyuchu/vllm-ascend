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
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/layout/layout.hpp"

#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;
using fp16_t = op::fp16_t;

template <
    class LayoutTagA,
    class LayoutTagB,
    class LayoutTagC
>
CATLASS_GLOBAL
void BasicMatmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutTagA tagA,
    GM_ADDR gmB, LayoutTagB tagB,
    GM_ADDR gmC, LayoutTagC tagC
)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = Shape<_128, _256, _256>;
    using L0TileShape = Shape<_128, _256, _64>;

    using ElementA = half;
    using ElementB = half;
    using ElementC = half;

    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    auto layoutC = MakeLayoutFromTag(tagC);

    using TensorA = Tensor<AscendC::GlobalTensor<ElementA>, decltype(layoutA), AscendC::TPosition::GM>;
    using TensorB = Tensor<AscendC::GlobalTensor<ElementB>, decltype(layoutB), AscendC::TPosition::GM>;
    using TensorC = Tensor<AscendC::GlobalTensor<ElementC>, decltype(layoutC), AscendC::TPosition::GM>;
    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, TensorA, LayoutTagA, TensorB, LayoutTagB, TensorC, LayoutTagC>;
    using BlockMmad =
        Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape,
                                    TensorA, TensorB, TensorC, void, TileCopy>;
    using BlockEpilogue = void;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

struct Options {
    const std::string HELPER = "00_basic_matmul m n k [device_id]";

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

    layout::RowMajor layoutA{m, k};
    layout::RowMajor layoutB{k, n};
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

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    BasicMatmul<<<aicCoreNum, nullptr, stream>>>(
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
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
