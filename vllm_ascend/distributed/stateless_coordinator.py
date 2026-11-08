# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register stateless groups after their device communicator is constructed.

Gloo registration supports EPLB's torch APIs; HCCL registration additionally
allows torch_npu fault recovery to find the device groups. Unregistration is
paired with group retirement.
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


def register_stateless_coordinator_pgs(coordinator: StatelessGroupCoordinator | None) -> None:
    """Register CPU and device groups after coordinator construction."""
    if not isinstance(coordinator, StatelessGroupCoordinator):
        return
    if coordinator.cpu_group is not None:
        _register_pg(coordinator.cpu_group, "gloo")
    if coordinator.device_group is not None:
        _register_pg(coordinator.device_group, coordinator.backend)


def unregister_stateless_coordinator_pgs(coordinator: StatelessGroupCoordinator | None) -> None:
    """Mirror ``register_stateless_coordinator_pgs`` on destruction."""
    if not isinstance(coordinator, StatelessGroupCoordinator):
        return
    if coordinator.cpu_group is not None:
        _unregister_pg(coordinator.cpu_group)
    if coordinator.device_group is not None:
        _unregister_pg(coordinator.device_group)
