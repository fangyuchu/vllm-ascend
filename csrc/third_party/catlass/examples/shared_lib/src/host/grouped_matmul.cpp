#include <acl/acl.h>

#include "catlass_kernel.h"
#include "kernel/grouped_matmul_slice_k.hpp"
#include "kernel/grouped_matmul_slice_m.hpp"

namespace CatlassKernel {
using namespace Catlass;
void GroupedMatmul(uint32_t blockNum, aclrtStream stream,
                   KernelInfo kernelInfo) {
  const uint32_t problemCount = kernelInfo.groupList.size();

  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using LayoutC = layout::RowMajor;

  std::vector<int32_t> groupList = kernelInfo.groupList;
  uint32_t m = kernelInfo.m;
  uint32_t n = kernelInfo.n;
  uint32_t k = kernelInfo.k;

  GemmCoord problemShape{m, n, k};
  LayoutA layoutA{m, k};
  LayoutB layoutB{k, n};
  LayoutC layoutC{m, n};

  // layout and shape malloc and copy
  uint8_t *groupListDevice{nullptr};
  typename std::decay<decltype(groupList)>::type::value_type elemGroupList = 0;
  size_t sizeGroupListDevice =
      groupList.size() * sizeof(decltype(elemGroupList));
  aclrtMalloc(reinterpret_cast<void **>(&groupListDevice), sizeGroupListDevice,
              ACL_MEM_MALLOC_HUGE_FIRST);
  aclrtMemcpy(groupListDevice, sizeGroupListDevice, groupList.data(),
              sizeGroupListDevice, ACL_MEMCPY_HOST_TO_DEVICE);

  // execution
  if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_M) {
    grouped_matmul_slice_m<LayoutA, LayoutB, LayoutC>
        <<<blockNum, nullptr, stream>>>(
            problemShape, problemCount, groupListDevice,
            kernelInfo.inputAddr.at(0), layoutA, kernelInfo.inputAddr.at(1),
            layoutB, kernelInfo.outputAddr.at(0), layoutC);
  } else if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_K) {
    grouped_matmul_slice_k<LayoutA, LayoutB, LayoutC>
        <<<blockNum, nullptr, stream>>>(
            problemShape, problemCount, groupListDevice,
            kernelInfo.inputAddr.at(0), layoutA, kernelInfo.inputAddr.at(1),
            layoutB, kernelInfo.outputAddr.at(0), layoutC);
  }
  aclrtFree(groupListDevice);
}
}  // namespace CatlassKernel