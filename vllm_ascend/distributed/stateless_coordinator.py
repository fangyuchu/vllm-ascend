# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register the stateless EPLB group's CPU (gloo) process group in ``_world``.

Upstream stateless PGs are deliberately absent from ``_world``, so the
module-level ``torch.distributed`` APIs used by the HIXL EPLB session and the
dynamic-EPLB P2P transfer cannot resolve them. Register *after* the coordinator
(and its device communicator) is built, or ``DeviceCommunicatorBase`` would
misclassify the group as stateful. Only the gloo group needs it; unregistration
is paired in ``AscendElasticEPScalingExecutor._destroy_retired_groups``.
"""

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import BackendConfig, _world
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


def _register_pg(pg: ProcessGroup, backend: str) -> None:
    """Register a stateless PG into ``_world`` (rank i maps to global rank i)."""
    _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
    _world.pg_map[pg] = (backend, pg.get_group_store())
    _world.pg_names[pg] = pg.group_name
    _world.pg_backend_config[pg] = str(BackendConfig(backend))


def _unregister_pg(pg: ProcessGroup) -> None:
    """Mirror ``_register_pg``: drop the group from ``_world``."""
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


def register_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Register the coordinator's gloo CPU group into ``_world``; pair with the unregister below."""
    if coordinator.cpu_group is not None:
        _register_pg(coordinator.cpu_group, "gloo")


def unregister_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Mirror ``register_stateless_coordinator_pgs`` on destruction."""
    if coordinator.cpu_group is not None:
        _unregister_pg(coordinator.cpu_group)
