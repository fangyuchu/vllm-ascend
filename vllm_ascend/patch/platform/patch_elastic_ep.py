# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Allow --enable-elastic-ep on Ascend NPU.
Upstream lists "elastic expert parallelism" as unsupported by the V2
model runner, which fails validation on Ascend where the V2 runner
does support elastic EP.
.. note::
   This patch must be imported **before** the V2 model runner
   compatibility check runs (``VllmConfig._get_v2_model_runner_unsupported_features``
   is consulted during ``_validate_v2_model_runner``), otherwise the
   upstream unsupported-feature list is used and ``--enable-elastic-ep``
   is rejected for the V2 model runner.
"""

from vllm.config.vllm import VllmConfig

# ---------------------------------------------------------------------------
# Patch VllmConfig._get_v2_model_runner_unsupported_features so that
# elastic expert parallelism is not reported as unsupported by the V2
# model runner.
#
# Upstream considers elastic EP unsupported by the V2 model runner
# (vllm/config/vllm.py:2185-2186), so with --enable-elastic-ep the
# V2 model runner either falls back to V1 or raises during
# _validate_v2_model_runner().  On Ascend the V2 model runner supports
# elastic EP, so drop that entry from the unsupported list.
# ---------------------------------------------------------------------------
_original_get_v2_model_runner_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features


def _patched_get_v2_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _original_get_v2_model_runner_unsupported_features(self)
    # Ascend's V2 model runner supports elastic EP, so remove it from
    # the upstream unsupported list.
    if "elastic expert parallelism" in unsupported:
        unsupported.remove("elastic expert parallelism")
    return unsupported


VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_v2_model_runner_unsupported_features
