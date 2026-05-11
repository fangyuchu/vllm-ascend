/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PY_EXT_CATLASS_KERNEL_WRAPPER_H
#define PY_EXT_CATLASS_KERNEL_WRAPPER_H

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "catlass_kernel.h"

namespace CatlassKernelWrapper {
at::Tensor RunBasicMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType);
std::vector<at::Tensor> RunGroupedMatmul(const std::vector<at::Tensor> &mat1, const std::vector<at::Tensor> &mat2,
                                         const std::string &outDType, const bool &splitK);
at::Tensor RunOptimizedMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType);

} // namespace CatlassKernelWrapper

#endif // PY_EXT_CATLASS_KERNEL_WRAPPER_H