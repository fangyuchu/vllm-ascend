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
# Patch target: vllm.distributed.stateless_coordinator
#
# Replace stateless_init_torch_distributed_process_group and
# stateless_destroy_torch_distributed_process_group with HCCL-aware
# wrappers that register / clean up PyTorch's global _world pg_map.
# Also replace CudaCommunicator with NPUCommunicator.

import vllm.distributed.stateless_coordinator as _stateless_coordinator
from torch.distributed import ProcessGroup, Store
from torch.distributed.distributed_c10d import BackendConfig, _world

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

# Keep references to original functions for use inside the wrappers.
_orig_stateless_init = _stateless_coordinator.stateless_init_torch_distributed_process_group
_orig_stateless_destroy = _stateless_coordinator.stateless_destroy_torch_distributed_process_group


def _ascend_stateless_init_pg(**kwargs) -> ProcessGroup | tuple[ProcessGroup, Store]:
    if kwargs.get("return_store", False):
        pg, store = _orig_stateless_init(**kwargs)
    else:
        pg = _orig_stateless_init(**kwargs)

    if kwargs["backend"] == "hccl":
        backend = "hccl"
        prefix_store = pg.get_group_store()
        group_name = pg.group_name
        backend_config = BackendConfig(backend)

        _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
        _world.pg_map[pg] = (backend, prefix_store)
        _world.pg_names[pg] = group_name
        _world.pg_backend_config[pg] = str(backend_config)

        if "WORLD" in group_name:
            _world.default_pg = pg

    if kwargs.get("return_store", False):
        return pg, store
    else:
        return pg


def _ascend_stateless_destroy_pg(pg: ProcessGroup) -> None:
    _orig_stateless_destroy(pg)

    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


_stateless_coordinator.stateless_init_torch_distributed_process_group = _ascend_stateless_init_pg
_stateless_coordinator.stateless_destroy_torch_distributed_process_group = _ascend_stateless_destroy_pg
_stateless_coordinator.CudaCommunicator = NPUCommunicator
