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

Upstream requires ``enable_eplb=True`` when ``enable_elastic_ep=True``,
and ``enable_eplb`` is gated by ``current_platform.is_cuda_alike()``.
Temporary / flag-based overrides of ``is_cuda_alike`` are ineffective
inside pydantic v2's compiled Rust ``SchemaValidator``.

The only approach that works is to permanently override
``is_cuda_alike()`` on the ``NPUPlatform`` class to return ``True``
when the call-site is inside ``_validate_parallel_config`` (detected
via stack inspection) AND the ParallelConfig instance being validated
has ``enable_elastic_ep=True``.  All other call-sites delegate to the
original implementation, so normal platform detection is unaffected.

.. note::
   This patch must be imported **before** any ``ParallelConfig`` is
   constructed, otherwise ``enable_eplb`` / ``enable_elastic_ep``
   inference will not take effect during validation.
"""

import logging
import sys

from vllm.config.parallel import ParallelConfig, EPLBConfig
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-infer enable_eplb=True from enable_elastic_ep=True.
# ---------------------------------------------------------------------------
_original_init = ParallelConfig.__init__


def _patched_init(self, **data: object):
    if data.get("enable_elastic_ep", False):
        data["enable_eplb"] = True
        existing = data.get("eplb_config")
        if existing is None:
            data["eplb_config"] = EPLBConfig(use_async=False)
        elif isinstance(existing, EPLBConfig):
            existing.use_async = False
        elif isinstance(existing, dict):
            existing["use_async"] = False
    _original_init(self, **data)


ParallelConfig.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Override NPUPlatform.is_cuda_alike permanently.  Only the immediate
# caller frame is checked — sys._getframe(1) is O(1) vs inspect.stack().
# ---------------------------------------------------------------------------
_npu_cls = type(current_platform)
_original_is_cuda_alike = _npu_cls.is_cuda_alike

_VALIDATE_METHOD = "_validate_parallel_config"

if not hasattr(ParallelConfig, _VALIDATE_METHOD):
    logger.warning(
        "ParallelConfig.%s not found. The elastic EP patch may be "
        "ineffective after a vLLM upgrade.",
        _VALIDATE_METHOD,
    )


def _patched_is_cuda_alike(self) -> bool:
    caller_frame = sys._getframe(1)
    if caller_frame.f_code.co_name == _VALIDATE_METHOD:
        caller_self = caller_frame.f_locals.get("self")
        if caller_self is not None and getattr(
            caller_self, "enable_elastic_ep", False
        ):
            return True
    return _original_is_cuda_alike(self)


_npu_cls.is_cuda_alike = _patched_is_cuda_alike
