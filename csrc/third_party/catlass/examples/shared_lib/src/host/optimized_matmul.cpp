#include "kernel/optimized_matmul.hpp"

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "catlass_kernel.h"

namespace CatlassKernel {
using namespace Catlass;
void OptimizedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;

    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using ElementA = bfloat16_t;
    using ElementB = bfloat16_t;
    using ElementC = bfloat16_t;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(ElementA);

    bool isNeedPaddingA = IsNeedPadding(layoutA, alignByElement);
    bool isNeedPaddingB = IsNeedPadding(layoutB, alignByElement);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;

    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    size_t sizeWA = GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(ElementA);
    size_t sizeWB = GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(ElementB);

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    if (isNeedPaddingA && isNeedPaddingB) {
        optimized_matmul<AType, BType, CType, true, true>
            <<<blockNum, nullptr, stream>>>(fftsAddr, problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB);
    }
    if (!isNeedPaddingA && isNeedPaddingB) {
        optimized_matmul<AType, BType, CType, false, true>
            <<<blockNum, nullptr, stream>>>(fftsAddr, problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB);
    }
    if (isNeedPaddingA && !isNeedPaddingB) {
        optimized_matmul<AType, BType, CType, true, false>
            <<<blockNum, nullptr, stream>>>(fftsAddr, problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB);
    }
    if (!isNeedPaddingA && !isNeedPaddingB) {
        optimized_matmul<AType, BType, CType, false, false>
            <<<blockNum, nullptr, stream>>>(fftsAddr, problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB);
    }

    aclrtSynchronizeStream(stream);
    if (isNeedPaddingA) {
        aclrtFree(deviceWA);
    }
    if (isNeedPaddingB) {
        aclrtFree(deviceWB);
    }
}
} // namespace CatlassKernel