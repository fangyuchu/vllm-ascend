# 打包为共享库

有时我们希望在已有的成熟工程中添加模板库算子，实现加速计算的效果，但又不希望大幅度改变构建工程. 为此我们可以将模板库算子编译成共享库，以方便在已有工程中调用.

## 代码结构

```bash
examples/shared_lib
├── include
│   └── catlass_kernel.h            # 头文件
└── src
    ├── common
    │   └── common.hpp          # 公共头文件，预留为多个kernel中的模板函数共用
    ├── host                    # host侧接口
    │   ├── basic_matmul.cpp    
    │   └── ...
    └── kernel                  # kernel侧算子
        ├── basic_matmul.hpp
        └── ...
```

## 编译产物结构

```bash
output/shared_lib
├── include
│   └── catlass_kernel.h # 头文件
└── lib
    ├── libcatlass_kernel.a # 静态链接库
    └── libcatlass_kernel.so # 动态链接库
```

## 使用说明

假设待添加算子为`custom_matmul`.

### 算子kernel实现

分别增加如下文件和代码：

- 在`src/kernel`文件夹中创建`custom_matmul.hpp`，实现算子本身.

```cpp
#include "catlass/catlass.hpp"
// catlass头文件...

using namespace Catlass;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
CATLASS_GLOBAL
void custom_matmul(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
    // 按需定义输入参数...
)
{
    // 使用Catlass api定义算子...
}
```

- 在`src/host`文件夹中创建`custom_matmul.cpp`，实现host接口，该接口负责整理输入后使用`内核调用符`调用算子.

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, ernelInfo kernelInfo) {
    Catlass::GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};
    custom_matmul<<<blockNum, nullptr, stream>>>(problemShape,
        kernelInfo.inputAddr.at(0), layoutA,
        kernelInfo.inputAddr.at(1), layoutB,
        kernelInfo.outputAddr.at(0), layoutC);
}
// ...
```

参数意义如下：

| 参数名       | 类型          | 作用                                                |
| ------------ | ------------- | --------------------------------------------------- |
| `blockNum`   | `uint32_t`    | 设定aiCore个数                                      |
| `stream`     | `aclrtStream` | NPU流                                               |
| `kernelInfo` | `KernelInfo`  | 算子执行的数据地址和输入详细情况，如mnk等维度的大小 |
可根据实际需要自行修改参数.

- 在`include/catlass_kernel.h`中增加`custom_matmul.cpp`中的host入口，以供外部调用.

```cpp
// ...
void CustomMatmul(uint32_t blockNum, aclrtStream stream, ernelInfo kernelInfo);
// ...
```
- 在`CMakeLists.txt`增加`catlass_add_kernel(custom_matmul dav-c220 ${CMAKE_CURRENT_SOURCE_DIR}/src/host/custom_matmul.cpp)`编译命令.

- 如果你增加了多个算子，但又存在相同定义的`模板函数`，这种情况在链接阶段会提示重复符号. 为解决这个问题，你可以将这类函数以`inline`形式存入公共的`common`路径中.

### 编译

```bash
bash scripts/build.sh shared_lib
```

## 注意事项

- 我们目前提供了三种典型算子作为示例：
  - `BasicMatmul`：基本矩阵乘法，并实现了类型模板的实现方法
  - `GroupedMatmul`：分组矩阵乘法，提供分组输入输出示例
  - `OptimizedMatmul`：优化矩阵乘法，提供CV融合的示例
- 本节是算子打包成动态库的一个示例，可根据需要自行扩展功能，并不仅局限于已有的代码.

## 已知问题

> [!NOTE]
> 使用内核调用符调用device侧kernel时，可以在模板参数中传入数据类型，但目前版本编译器暂不支持在内核调用符上使用`bfloat16_t`. 若需要通过模板特化`bfloat16_t`相关的核函数，可参考下面的示例：

```cpp
template<typename T>
CATLASS_DEVICE void real_kernel(...){
    //...
}
template<aclDataType T>
CATLASS_GLOBAL void kernel(...){
    if constexpr (T == ACL_BF16){
        real_kernel<bfloat16_t>(...);
    }
}
void kernel_host(...){
    kernel<ACL_BF16><<<blockNum, nullptr, stream>>>(...);
}
```

即：device侧的特化要在device侧实现.
> [!NOTE]
> 部分算子，如`gemv`，需在编译时指定`--cce-aicore-arch=dav-c220-vec`以标识其为`vector`算子. 对于纯`cube`算子或`CV融合`算子，使用`--cce-aicore-arch=dav-c220`可以直接兼容二者，但无法兼容纯`vector`算子的情况. 然而，目前编译器不支持将多个架构不同的.o链接成一个.so.=

> 开发者需要使用`gemv`的情况下，可参考此项目构建一个单独的`libcatlass_kernel_vec.so`文件.

## 版权声明

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This file is a part of the CANN Open Software.
Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR   PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

## 许可证

[CANN Open Software License Agreement Version 1.0](../../LICENSE)
