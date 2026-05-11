
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_CATLASS_KERNEL_H
#define SHARED_LIB_CATLASS_KERNEL_H

#include <acl/acl.h>

#include <vector>

namespace CatlassKernel {

struct KernelInfo {
    enum class GMMSplit : uint32_t { SPLIT_M = 0, SPLIT_K = 1, SPLIT_N = 2 };
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;
    aclDataType outputDataType = aclDataType::ACL_FLOAT16;
    uint32_t g = 1;
    uint32_t b = 1;
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    bool transA = false;
    bool transB = false;
    std::vector<int32_t> groupList;
    GMMSplit split = GMMSplit::SPLIT_M;
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

void BasicMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo);
void GroupedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo);
void OptimizedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo);

}

#endif // SHARED_LIB_CATLASS_KERNEL_H