# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import re
import subprocess
import unittest
from typing import List, Type

CMAKE_BINARY_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "build", "bin")
CMAKE_EXAMPLES_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "examples")


class CatlassExampleTest(unittest.TestCase):
    def run_case(self, executable_name: str, args: List):
        args = [str(arg) for arg in args]

        ret = subprocess.run(
            [os.path.join(CMAKE_BINARY_PATH, executable_name)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for error_log_line in ret.stderr.decode().splitlines():
            acl_match = re.match(
                r"^.*aclError:.*([1-9][0-9]{5})", error_log_line)
            rt_match = re.match(
                r"^.*rtError:.*([1-9][0-9]{5})", error_log_line)
            compare_match = re.match(
                r"^.*Compare failed. Error count: :.*([1-9][0-9]{5})", error_log_line)
            acl_code = 0 if acl_match is None else int(acl_match.group(1))
            rt_code = 0 if rt_match is None else int(rt_match.group(1))
            compare_code = 0 if compare_match is None else int(
                compare_match.group(1))
            self.assertEqual(
                acl_code, 0, f"There is an ACL error: {acl_code}")
            self.assertEqual(rt_code, 0, f"There is an RT error: {rt_code}")
            self.assertEqual(compare_code, 0,
                             f"There is a compare error: {compare_code}")
        self.assertEqual(
            ret.returncode, 0, f"Return code is not zero: {ret.returncode}")

    def test_19_mla(self):
        case_base = [str(i) for i in [1, 1, 128, 16, 16, 128]]
        case_py = case_base + ["half"]
        ret = subprocess.run(["python", os.path.join(
            CMAKE_EXAMPLES_PATH, "19_mla", "gen_data.py")]+case_py)
        case_cpp = case_base + ["--dtype", "half", "--datapath",
                                os.path.join(CMAKE_EXAMPLES_PATH, "19_mla", "data")]
        self.run_case("19_mla", case_cpp)


normal_cases = ["00_basic_matmul 256 512 1024 0",
                "01_batched_matmul 5 256 512 1024 0",
                "02_grouped_matmul_slice_m 128 512 1024 2048 0",
                "03_matmul_add 256 512 1024 0",
                "04_padding_matmul 256 512 1024 0",
                "05_grouped_matmul_slice_k 128 512 1024 2048 0",
                "06_optimized_matmul 256 512 1024 0",
                "07_grouped_matmul_slice_m_per_token_dequant_moe 128 512 1024 2048 0",
                "08_grouped_matmul 128 512 1024 2048 0",
                "09_splitk_matmul 256 512 1024 0",
                "10_grouped_matmul_slice_m_per_token_dequant 128 512 1024 2048 0",
                "11_grouped_matmul_slice_k_per_token_dequant 128 512 1024 2048 0",
                "12_quant_matmul 256 512 1024 0",
                "13_basic_matmul_tla 256 512 1024 0",
                "14_optimized_matmul_tla 256 512 1024 0",
                "15_gemm 256 512 1024 0",
                "16_group_gemm 3 '128,256,512' '256,512,128' '512,256,128' 0",
                "17_gemv_aiv 256 512 0",
                "18_gemv_aic 256 512 0"]


def set_case(case: str):
    case_splited = case.split()
    case_executable_name = case_splited[0]
    case_args = case_splited[1:]

    def __(self: Type[CatlassExampleTest]):
        self.run_case(case_executable_name, case_args)

    setattr(CatlassExampleTest, f"test_{case_executable_name}", __)


for normal_case in normal_cases:
    set_case(normal_case)
if __name__ == '__main__':
    unittest.main()
