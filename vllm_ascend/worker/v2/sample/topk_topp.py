# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
import torch_npu
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits on NPU.

    The upstream Triton ``_topk_topp_kernel`` cannot be lowered by the
    Ascend Triton backend (ConvertTritonIRToLinalgIR fails), so keep the
    vLLM v2 sampler away from it. Use the fused CANN op on A2/A3 and the
    upstream PyTorch fallback elsewhere.
    """
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]:
        if k is None and p is None:
            return logits
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)
    return apply_top_k_top_p_pytorch(logits, k, p)
