# python扩展

为方便开发者使用CATLASS算子，代码仓基于pybind11和torch提供了使用python调用CATLASS算子的示例.

## 代码结构

```bash
python_extension
├── CMakeLists.txt                      # CMake配置文件
├── README.md                           # 说明文档
├── pyproject.toml                      # 项目配置文件
├── setup.py                            # 安装脚本
├── src
│   ├── bindings
│   │   ├── pybind_bindings.cpp         # pybind11绑定文件
│   │   └── torch_bindings.cpp          # torch绑定文件
│   ├── include
│   │   └── wrapper
│   │       └── catlass_kernel_wrapper.h    # wrapper头文件
│   └── wrapper
│       └── catlass_kernel_wrapper.cpp      # catlass算子wrapper文件
├── tests
│   └── test_python_extension.py        # 测试脚本
└── torch_catlass                       
    └── __init__.py                     # 初始化入口，用于打包
```

## 编译产物结构

```bash
output/python_extension
├── libcatlass_torch.so                             # torch动态链接库
└── torch_catlass-0.1.0.20250330120000.cp310-cp310-linux-x86_64.whl  # pybind11动态链接库的wheel包
```

## 使用说明

- 假设你已经在shared_lib中，增加了所需算子的实现和入口.

### pybind接口实现

由于pybind传入参数为at::Tensor而非AscendC中的GM地址指针，所以需要对python侧传来的数据进行处理.
主要步骤为根据输入tensor的信息填充运行信息参数，申请输出内存.
此部分较为灵活，与算子本身参数较为相关，可参考已有的BasicMatmul实现.

### 编译

各部分代码完成后：

- 使用`bash scripts/build.sh python_extension`编译pybind扩展.
- 使用`bash scripts/build.sh torch_library`编译torch扩展.

编译环境如下：

| 依赖              | 版本               | 安装指令                      |
| ----------------- | ------------------ | ----------------------------- |
| `build-essential` | gcc版本为7.5以上(建议使用9以上版本)     | `apt install build-essential` |
| `cmake`           | `>=3.16`           | `apt install cmake`           |
| `CANN`            | `>=8.0.0.alpha002` | run包安装                     |
| `pybind11`        | 无要求             | `pip install pybind11`        |
| `pybind11-stubgen` | 无要求             | `pip install pybind11-stubgen` |
| `torch`           | 无要求(建议使用2.1+)             | `pip install torch`           |
| `torch_npu`       | 无要求(最新post版本)             | `pip install torch_npu`       |

- 虽然`torch`的版本没有要求，可沿用现有环境，但是为保证能够正确调用`torch_npu`的接口，`torch_npu`需要安装大版本下的**最新post**版本。你可以根据`CANN`版本和`torch`版本，在[Ascend/pytorch](https://gitee.com/ascend/pytorch)查询适合的`torch_npu`版本.

### 安装

- 对于torch扩展，你只需要在使用算子前增加如下代码加载：

```python
torch.ops.load_library("output/python_extension/libcatlass_torch.so")
```

- 对于pybind扩展，编译产物即为一个wheel包，执行`pip install torch_catlass-xxxxx.whl`即可.

### 运行

```python
import torch_catlass
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class CatlassTest(TestCase):
    def test_basic_matmul(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        result = torch_catlass.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
    def test_basic_matmul_torch_lib(self):
        a = torch.ones((2, 3)).to(torch.float16).npu()
        b = torch.ones((3, 4)).to(torch.float16).npu()
        torch.ops.load_library("../../output/python_extension/libcatlass_torch.so") # 确保加载正确路径
        result = torch.ops.CatlassTorch.basic_matmul(a, b, "float16")
        golden = torch.mm(a, b)
        self.assertRtolEqual(result, golden)
        
if __name__ == "__main__":
    run_tests()
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

[CANN Open Software License Agreement Version 1.0](../../LICENSE)
