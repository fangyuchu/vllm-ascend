# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Register stateless process groups in torch's global ``_world``.

Upstream ``stateless_init_torch_distributed_process_group`` creates
process groups that are not registered in torch's global ``_world``
state, so ``torch.distributed`` module-level APIs cannot find them.
``NPUPlatform`` registers the HCCL device groups and gloo CPU groups
into ``_world`` on creation (and removes them on destroy) via the
platform lifecycle hooks. This is required by e.g. ``broadcast``/
``send``/``recv`` global-rank translation on the stateless world/dp/ep
groups and by the async EPLB communicator issuing ``batch_isend_irecv``
on the stateless gloo group during elastic EP.
"""

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import BackendConfig, _world


def _register_pg(pg: ProcessGroup, backend: str) -> None:
    """Register a stateless PG into torch's global ``_world``.

    Each rank of a stateless group maps 1:1 to itself (rank i in the
    group is global rank i).
    """
    _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
    _world.pg_map[pg] = (backend, pg.get_group_store())
    _world.pg_names[pg] = pg.group_name
    _world.pg_backend_config[pg] = str(BackendConfig(backend))

    # The WORLD group is used as torch's default process group.
    if "WORLD" in (pg.group_name or ""):
        _world.default_pg = pg


def _unregister_pg(pg: ProcessGroup) -> None:
    """Mirror ``_register_pg``: drop the group from ``_world``."""
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)
