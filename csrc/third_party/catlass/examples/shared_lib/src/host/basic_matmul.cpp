#include "kernel/basic_matmul.hpp"

#include <acl/acl.h>

#include "catlass_kernel.h"
#include "common.hpp"

namespace CatlassKernel {
using namespace Catlass;

void BasicMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16 && !kernelInfo.transA &&
        !kernelInfo.transB) {
        using LayoutA = layout::RowMajor;
        using LayoutB = layout::RowMajor;
        using LayoutC = layout::RowMajor;
        using InDType = half;
        using OutDType = half;
        GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
        basic_matmul<LayoutA, LayoutB, LayoutC, InDType, OutDType><<<blockNum, nullptr, stream>>>(
            problemShape, kernelInfo.inputAddr.at(0), kernelInfo.inputAddr.at(1), kernelInfo.outputAddr.at(0));
    }
}
} // namespace CatlassKernel