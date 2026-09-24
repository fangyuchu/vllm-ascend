# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register the stateless EPLB group in torch's global ``_world``.

Upstream ``stateless_init_torch_distributed_process_group`` creates
process groups that are deliberately not registered in torch's global
``_world`` state, so ``torch.distributed`` module-level APIs cannot
resolve them. Ascend needs this for the stateless EPLB group's CPU
(gloo) process group, which is consumed by the gloo staged EPLB
communicator (``dist.get_global_rank`` + ``batch_isend_irecv``) and by
the dynamic-EPLB P2P transfer (global ranks) during elastic EP.

Only the CPU (gloo) group is registered. The world/dp/ep groups talk
through coordinator methods (PyHccl / TCP store) and never consult
``_world``, and the EPLB device group is only used through
``ProcessGroup`` methods, so registering them would only widen the
blast radius.

Ordering constraint: ``DeviceCommunicatorBase.__init__`` treats a
``cpu_group`` as stateless iff it is absent from ``_world.pg_map``.
Registration must therefore happen *after* the coordinator (and its
device communicator) has been constructed, or the communicator would be
misclassified as stateful and take the ``dist.get_rank(group=...)``
path. Call sites:
- ``NPUWorker._init_worker_distributed_environment`` for the startup
  EPLB group, and
- ``AscendElasticEPScalingExecutor.prepare_reconfiguration`` for the
  standby EPLB group created on scale-up preparation.
Unregistration is paired in
``AscendElasticEPScalingExecutor._destroy_retired_groups``.
"""

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import BackendConfig, _world
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


def _register_pg(pg: ProcessGroup, backend: str) -> None:
    """Register a stateless PG into torch's global ``_world``.

    Each rank of a stateless group maps 1:1 to itself (rank i in the
    group is global rank i).
    """
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
    """Register a stateless coordinator's CPU (gloo) group into ``_world``.

    Call right after the coordinator is created; pair with
    ``unregister_stateless_coordinator_pgs`` when it is destroyed.
    """
    if coordinator.cpu_group is not None:
        _register_pg(coordinator.cpu_group, "gloo")


def unregister_stateless_coordinator_pgs(
    coordinator: StatelessGroupCoordinator,
) -> None:
    """Mirror ``register_stateless_coordinator_pgs`` on destruction."""
    if coordinator.cpu_group is not None:
        _unregister_pg(coordinator.cpu_group)
