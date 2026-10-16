import torch
from vllm.distributed.parallel_state import (
    _init_stateless_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import get_cached_tcp_store_client

_STANDBY_MC2: StatelessGroupCoordinator | None = None


def create_ascend_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
) -> None:
    global _STANDBY_MC2

    assert new_world_size_across_dp == torch.distributed.get_world_size() * new_dp_size
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size

    all_ranks = torch.arange(new_world_size_across_dp).reshape(-1, new_dp_size * pp_size * tp_size)
    group_ranks = all_ranks.unbind(0)
    standby_mc2_ranks = [x.tolist() for x in group_ranks]

    # Always stateless: elastic EP needs dynamic membership.
    _STANDBY_MC2 = _init_stateless_group(
        standby_mc2_ranks,
        "mc2",
        master_ip,
        backend,
        coord_store=coord_store,
    )


def pop_ascend_standby_groups() -> StatelessGroupCoordinator | None:
    """Return the standby MC2 group and clear the standby state."""
    global _STANDBY_MC2
    result = _STANDBY_MC2
    _STANDBY_MC2 = None
    return result
