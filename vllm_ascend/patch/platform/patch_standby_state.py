#
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
#
# Patch target: vllm.distributed.elastic_ep.standby_state.create_standby_groups
#
# When the vLLM EngineCore performs an elastic expert-parallel (EP) expansion,
# it builds "standby" communication groups (world / dp / ep / eplb) for the new
# set of ranks. Ascend additionally needs MC2 and dynamic-EPLB standby groups,
# which upstream vLLM does not create.
#
# This patch wraps the upstream function: it first runs the original logic to
# build the upstream standby groups, then calls
# `vllm_ascend.distributed.elastic_ep.standby_state.create_ascend_standby_groups`
# to build the Ascend-specific standby groups.

import vllm.distributed.elastic_ep.standby_state as _standby_state_pkg

from vllm_ascend.distributed.elastic_ep.standby_state import create_ascend_standby_groups

# Keep a reference to the original upstream function so the wrapper below can
# delegate to it.
_orig_create_standby_groups = _standby_state_pkg.create_standby_groups


def _ascend_create_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    use_all2all: bool,
    enable_eplb: bool = True,
    backend: str | None = None,
) -> None:
    # First run the original upstream logic to build the world / dp / ep / eplb
    # standby groups.
    _orig_create_standby_groups(
        new_dp_size,
        new_world_size_across_dp,
        master_ip,
        coord_store_port,
        use_all2all,
        enable_eplb,
        backend,
    )

    # Then create the Ascend-specific MC2 and dynamic-EPLB standby groups.
    create_ascend_standby_groups(
        new_dp_size,
        new_world_size_across_dp,
        master_ip,
        coord_store_port,
        backend,
    )


# Replace the upstream function so every standby-group creation also sets up
# the Ascend-specific groups.
_standby_state_pkg.create_standby_groups = _ascend_create_standby_groups
