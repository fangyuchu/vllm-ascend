# TLA Tensors

这篇文档描述了CATLASS的TLA(Tensor Layout Abstraction)下的`Tensor`。

本质上，`Tensor` （张量）表示一个多维数组。Tensor 抽象了数组元素在内存中的组织方式与存储方式的细节。这使得用户能够编写通用地访问多维数组的算法，并可根据张量的特性（traits）对算法进行特化。如张量的depth、rank、Layout、数据的类型、位置等。

`Tensor` 包含3个模板参数: `BuiltinTensor`、 `Layout`、 `Position`。
关于 `Layout` 的描述, 请参考 [ `Layout` ](./01_layout.md)。

## BuiltinTensor 和 Position

`BuiltinTensor` 为AscendC内的 `GlobalTensor` 或者 `LocalTensor`，`Position` 为AscendC定义的各层级位置。相关使用参考AscendC文档。

## Tensor 构造

当前提供 `MakeTensor` 接口构造`Tensor`， 包含三个模板参数： `BuiltinTensor`、 `Layout`、 `Position`。

有如下两种方式构造：

```cpp
GlobalTensor<float> A = ...;

// 显示指定模板参数
Tensor tensor_8x16 = make_tensor<GlobalTensor<float>, Layout<Shape<uint32_t, Int<16>>, Stride<Int<16>, Int<1>>>, AscendC::TPosition::GM>(A, make_shape(8, Int<16>{}), make_stride(Int<16>{},Int< 1>{}));

// 模板参数自动推导
Tensor tensor_8x16 = make_tensor(A, make_shape(8, Int<16>{}), make_stride(Int<16>{},Int<1>{}), PositionGM{});
```

## Tensors 接口

TLA `Tensor` 提供获取相应特性的接口：

* `.data()`. 返回 `Tensor` 的内存。

* `.layout()`. 返回 `Tensor` 的 `layout`。

* `.shape()`. 返回 `Tensor` 的 `shape`。

* `.stride()`. 返回 `Tensor` 的 `stride`。

## 获取 TileTensor

提供一个 `GetTile` 接口获取 `Tensor` 的一片子tensor，会根据输入坐标对内存进行偏移，根据新的Tile的shape变换Layout（只是逻辑层面的数据组织形式），底层的数据实体不变更。

```cpp
Tensor tensor_8x16 = make_tensor(A, make_shape(8, Int<16>{}), make_stride(Int<16>{},Int<1>{}), PositionGM{});

auto tensor_tile = GetTile(tensor_8x16, MakeCoord(2, 4), MakeShape(4, 8)); // (4,8):(_16,_1):(4,8)
```

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
[CANN Open Software License Agreement Version 1.0](../LICENSE)