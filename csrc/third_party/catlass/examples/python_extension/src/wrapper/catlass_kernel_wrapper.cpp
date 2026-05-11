
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <tiling/platform/platform_ascendc.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"

namespace py = pybind11;
using namespace CatlassKernel;

namespace CatlassKernelWrapper {
torch::Tensor GetOutputTensor(const std::vector<int64_t> &shape, const torch::Dtype dtype) {
    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(dtype).layout(at::kStrided).requires_grad(false).device(torch_npu::utils::get_npu_device_type());
    return at_npu::native::empty_with_format(shape, options, ACL_FORMAT_ND);
}

torch::Dtype TypeStrToTorchDtype(std::string typeStr, torch::Dtype defaultType = torch::kFloat16)
{
    static const std::unordered_map<std::string, torch::Dtype> mapper = {{"float32", torch::kFloat32},
                                                                         {"float16", torch::kFloat16},
                                                                         {"int8", torch::kInt8},
                                                                         {"int32", torch::kInt32},
                                                                         {"bf16", torch::kBFloat16}};
    auto iter = mapper.find(typeStr);
    if (iter == mapper.end()) {
        return defaultType;
    }
    return iter->second;
}

aclDataType TorchDtypeToAclDtype(torch::Dtype torchDtype, aclDataType defaultType = ACL_FLOAT16)
{
    static const std::unordered_map<torch::Dtype, aclDataType> mapper = {{torch::kFloat32, ACL_FLOAT},
                                                                         {torch::kFloat16, ACL_FLOAT16},
                                                                         {torch::kInt8, ACL_INT8},
                                                                         {torch::kInt32, ACL_INT32},
                                                                         {torch::kBFloat16, ACL_BF16}};
    auto iter = mapper.find(torchDtype);
    if (iter == mapper.end()) {
        return defaultType;
    }
    return iter->second;
}

enum class TransposeStatus : uint32_t {
    NO_TRANSPOSE = 0,
    TRANSPOSE = 1,
    NON_CONTINUOUS = 2
};

TransposeStatus GetTransposeStatus(const at::Tensor &mat)
{
    if (mat.is_contiguous()){
        return TransposeStatus::NO_TRANSPOSE;
    }
    std::vector<int64_t> strides = mat.strides().vec();
    std::vector<int64_t> shape = mat.sizes().vec();
    int64_t dimB = shape.at(shape.size() - 1);
    int64_t strideA = strides.at(strides.size() - 2);
    int64_t strideB = strides.at(strides.size() - 1);
    if (strideB == dimB && strideA == 1) {
        return TransposeStatus::TRANSPOSE;
    }
    return TransposeStatus::NON_CONTINUOUS;
}

std::vector<int64_t> InferShape(at::IntArrayRef matAShape, at::IntArrayRef matBShape)
{
    int64_t m = matAShape.at(0);
    int64_t k1 = matAShape.at(1);
    int64_t k2 = matBShape.at(0);
    int64_t n = matBShape.at(1);
    if (k1 != k2) {
        std::stringstream ss;
        ss << "mat1 and mat2 shapes cannot be multiplied";
        ss << "(" << m << "x" << k1 << " and " << k2 << "x" << n << ")";
        throw std::runtime_error(ss.str());
    }
    return {m, n};
}

KernelInfo GetKernelInfo(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo;
    int64_t m = mat1.sizes().at(0);
    int64_t k1 = mat1.sizes().at(1);
    int64_t n = mat2.sizes().at(1);
    kernelInfo.m = m;
    kernelInfo.k = k1;
    kernelInfo.n = n;
    kernelInfo.inputDataType = TorchDtypeToAclDtype(mat1.scalar_type());
    kernelInfo.outputDataType = TorchDtypeToAclDtype(TypeStrToTorchDtype(outDType), kernelInfo.inputDataType);
    TransposeStatus transposeStatus1 = GetTransposeStatus(mat1); 
    TransposeStatus transposeStatus2 = GetTransposeStatus(mat2);
    if(transposeStatus1 == TransposeStatus::NON_CONTINUOUS){
        throw std::runtime_error("mat1 is not contiguous");
    }
    if (transposeStatus2 == TransposeStatus::NON_CONTINUOUS){
        throw std::runtime_error("mat2 is not contiguous");
    }
    kernelInfo.transA = static_cast<bool>(transposeStatus1);
    kernelInfo.transB = static_cast<bool>(transposeStatus2);
    return kernelInfo;
};

template <KernelInfo::GMMSplit SPLIT_TYPE>
KernelInfo GetGroupedKernelInfo(const std::vector<at::Tensor> &mat1, const std::vector<at::Tensor> &mat2,
                                  const std::string &outDType, std::vector<int64_t> &totalSizeList)
{
    if (mat1.size() != mat2.size()) {
        throw std::runtime_error("");
    }
    KernelInfo kernelInfo = GetKernelInfo(mat1.at(0), mat2.at(0), outDType);
    int64_t m;
    int64_t k1;
    int64_t k2;
    int64_t n;
    std::vector<uint32_t> groupedDims;
    int64_t totalSizeA{0};
    int64_t totalSizeB{0};
    int64_t totalSizeC{0};
    size_t inputElemSize = aclDataTypeSize(kernelInfo.inputDataType);
    size_t outputElemSize = aclDataTypeSize(kernelInfo.outputDataType);
    for (size_t i = 0; i < mat1.size(); i++) {
        m = mat1[i].sizes().at(0);
        k1 = mat1[i].sizes().at(1);
        k2 = mat2[i].sizes().at(0);
        n = mat2[i].sizes().at(1);
        if (k1 != k2) {
            throw std::runtime_error("k1 != k2");
        }
        if (n != kernelInfo.n) {
            throw std::runtime_error("n is not equal");
        }
        if constexpr (SPLIT_TYPE == KernelInfo::GMMSplit::SPLIT_M) {
            groupedDims.push_back(static_cast<int32_t>(m));
        }
        if constexpr (SPLIT_TYPE == KernelInfo::GMMSplit::SPLIT_K) {
            if (m != kernelInfo.m) {
                throw std::runtime_error("split k, but m is not equal");
            }
            groupedDims.push_back(static_cast<int32_t>(k1));
        }
        totalSizeA += static_cast<int64_t>(m) * k1 * inputElemSize;
        totalSizeB += static_cast<int64_t>(k2) * n * inputElemSize;
        totalSizeC += static_cast<int64_t>(m) * n * outputElemSize;
    }
    totalSizeList.resize(3);
    totalSizeList[0] = totalSizeA;
    totalSizeList[1] = totalSizeB;
    totalSizeList[2] = totalSizeC;
    kernelInfo.groupList.resize(groupedDims.size());
    std::partial_sum(groupedDims.begin(), groupedDims.end(), kernelInfo.groupList.begin());
    return kernelInfo;
};

at::Tensor RunBasicMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo = GetKernelInfo(mat1, mat2, outDType);
    kernelInfo.inputAddr.resize(2);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(mat1.storage().data()));
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(mat2.storage().data()));
    torch::Dtype outputDataType = TypeStrToTorchDtype(outDType, mat1.scalar_type());
    torch::Tensor result = GetOutputTensor(InferShape(mat1.sizes(), mat2.sizes()), outputDataType);
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr.at(0) = static_cast<uint8_t *>(const_cast<void *>(result.storage().data()));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    BasicMatmul(aicCoreNum, stream, kernelInfo);
    
    return result;
}

std::vector<at::Tensor> RunGroupedMatmul(const std::vector<at::Tensor> &mat1, const std::vector<at::Tensor> &mat2,
                                         const std::string &outDType, const bool &splitK)
{
    KernelInfo kernelInfo;
    std::vector<int64_t> totalSizeList;
    // make grouped list from input shapes
    // and calculate the size of matA, B, C
    if (splitK) {
        kernelInfo = GetGroupedKernelInfo<KernelInfo::GMMSplit::SPLIT_K>(mat1, mat2, outDType, totalSizeList);
        kernelInfo.split = KernelInfo::GMMSplit::SPLIT_K;
    } else {
        kernelInfo = GetGroupedKernelInfo<KernelInfo::GMMSplit::SPLIT_M>(mat1, mat2, outDType, totalSizeList);
        kernelInfo.split = KernelInfo::GMMSplit::SPLIT_M;
    }
    const size_t problemCount = kernelInfo.groupList.size();

    // allocate contiguous memory for matA, B, C
    uint8_t *deviceA{nullptr};
    uint8_t *deviceB{nullptr};
    uint8_t *deviceC{nullptr};
    aclrtMalloc(reinterpret_cast<void **>(&deviceA), totalSizeList[0], ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&deviceB), totalSizeList[1], ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&deviceC), totalSizeList[2], ACL_MEM_MALLOC_HUGE_FIRST);

    // copy non contiguous tensors to allocated contiguous memory, reserve the origin address for kernel execution
    uint8_t *baseDeviceA{deviceA};
    uint8_t *baseDeviceB{deviceB};
    uint8_t *baseDeviceC{deviceC};

    // copy
    for (size_t i = 0; i < problemCount; i++) {
        at::Tensor currentMat1 = mat1[i];
        at::Tensor currentMat2 = mat2[i];
        int64_t currentMat1Size = currentMat1.nbytes();
        int64_t currentMat2Size = currentMat2.nbytes();
        aclrtMemcpy(deviceA, currentMat1Size, currentMat1.storage().data(), currentMat1Size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
        aclrtMemcpy(deviceB, currentMat2Size, currentMat2.storage().data(), currentMat2Size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
        deviceA += currentMat1Size;
        deviceB += currentMat2Size;
    }

    // prepare kernelInfo
    kernelInfo.inputAddr.resize(2);
    kernelInfo.inputAddr[0] = baseDeviceA;
    kernelInfo.inputAddr[1] = baseDeviceB;
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = baseDeviceC;
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // execution
    GroupedMatmul(aicCoreNum, stream, kernelInfo);

    // after execution, the contiguous memory of input will not be used anymore, can be free
    aclrtFree(baseDeviceA);
    aclrtFree(baseDeviceB);

    // allocate output tensor memory
    torch::Dtype outputDataType = TypeStrToTorchDtype(outDType, mat1.at(0).scalar_type());

    // copy
    std::vector<at::Tensor> resultList;
    for (size_t i = 0; i < problemCount; i++) {
        at::Tensor currentMat1 = mat1[i];
        at::Tensor currentMat2 = mat2[i];
        torch::Tensor result = GetOutputTensor(InferShape(currentMat1.sizes(), currentMat2.sizes()), outputDataType);

        resultList.push_back(result);
        int64_t resultSize = result.nbytes();
        aclrtMemcpy(const_cast<void *>(result.storage().data()), resultSize, deviceC, resultSize,
                    ACL_MEMCPY_DEVICE_TO_DEVICE);
        deviceC += resultSize;
    }

    // free
    aclrtFree(baseDeviceC);
    return resultList;
}

at::Tensor RunOptimizedMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo = GetKernelInfo(mat1, mat2, outDType);
    kernelInfo.inputAddr.resize(2);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(mat1.storage().data()));
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(mat2.storage().data()));
    torch::Dtype outputDataType = TypeStrToTorchDtype(outDType, mat1.scalar_type());
    torch::Tensor result = GetOutputTensor(InferShape(mat1.sizes(), mat2.sizes()), outputDataType);
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr.at(0) = static_cast<uint8_t *>(const_cast<void *>(result.storage().data()));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    OptimizedMatmul(aicCoreNum, stream, kernelInfo);
    
    return result;
}
} // namespace Catlass