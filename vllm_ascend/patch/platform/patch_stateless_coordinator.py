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
# Patch vllm.distributed.stateless_coordinator: use NPUCommunicator and
# register HCCL/gloo stateless groups into torch's global ``_world``.

import vllm.distributed.stateless_coordinator as _stateless_coordinator
from torch.distributed import ProcessGroup, Store
from torch.distributed.distributed_c10d import BackendConfig, _world

# Keep references to original functions for use inside the wrappers.
_orig_stateless_init = _stateless_coordinator.stateless_init_torch_distributed_process_group
_orig_stateless_destroy = _stateless_coordinator.stateless_destroy_torch_distributed_process_group


# Register HCCL/gloo stateless groups into torch's ``_world`` on init so the
# standard torch.distributed APIs can find them.
def _ascend_stateless_init_pg(**kwargs) -> ProcessGroup | tuple[ProcessGroup, Store]:
    # Call the original helper first to create the stateless group.
    if kwargs.get("return_store", False):
        pg, store = _orig_stateless_init(**kwargs)
    else:
        pg = _orig_stateless_init(**kwargs)

    # Stateless groups are not registered by the upstream helper; register
    # them into torch's global ``_world`` state so torch.distributed APIs
    # work. gloo registration is required for the async EPLB communicator
    # (TorchDistGlooStagedEplbCommunicator) which issues batch_isend_irecv on
    # the stateless cpu_group during elastic EP.
    if kwargs["backend"] in ("hccl", "gloo"):
        backend = kwargs["backend"]
        prefix_store = pg.get_group_store()
        group_name = pg.group_name
        backend_config = BackendConfig(backend)

        # Each rank of this stateless group maps 1:1 to itself (rank i in
        # the group is global rank i).
        _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
        _world.pg_map[pg] = (backend, prefix_store)
        _world.pg_names[pg] = group_name
        _world.pg_backend_config[pg] = str(backend_config)

        # The WORLD group is used as torch's default process group.
        if "WORLD" in group_name:
            _world.default_pg = pg

    if kwargs.get("return_store", False):
        return pg, store
    else:
        return pg


# Mirror the registration above: drop the group from ``_world`` on destroy.
def _ascend_stateless_destroy_pg(pg: ProcessGroup) -> None:
    _orig_stateless_destroy(pg)

    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


_stateless_coordinator.stateless_init_torch_distributed_process_group = _ascend_stateless_init_pg
_stateless_coordinator.stateless_destroy_torch_distributed_process_group = _ascend_stateless_destroy_pg

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

_stateless_coordinator.CudaCommunicator = NPUCommunicator
