import torch
from vllm.distributed.parallel_state import (
    _init_stateless_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import get_cached_tcp_store_client

_STANDBY_MC2: StatelessGroupCoordinator | None = None


def get_standby_mc2_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_MC2


def _mc2_group_ranks(
    world_size: int,
    dp_size: int,
    pp_size: int,
    pcp_size: int,
    tp_size: int,
) -> list[list[int]]:
    all_ranks = torch.arange(world_size).reshape(
        -1,
        dp_size,
        pp_size,
        pcp_size,
        tp_size,
    )
    group_ranks = all_ranks.transpose(1, 2).reshape(-1, dp_size * pcp_size * tp_size).unbind(0)
    return [ranks.tolist() for ranks in group_ranks]


def create_ascend_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    backend: str | None = None,
) -> None:
    global _STANDBY_MC2

    assert new_world_size_across_dp == torch.distributed.get_world_size() * new_dp_size
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size
    pcp_size = get_pcp_group().world_size

    standby_mc2_ranks = _mc2_group_ranks(
        new_world_size_across_dp,
        new_dp_size,
        pp_size,
        pcp_size,
        tp_size,
    )
    # The standby MC2 group is always stateless: it is only built for elastic
    # EP scaling, so new ranks must be able to join the topology dynamically.
    _STANDBY_MC2 = _init_stateless_group(
        standby_mc2_ranks,
        "mc2",
        master_ip,
        backend,
        coord_store=coord_store,
    )


def pop_ascend_standby_groups() -> dict:
    """Return all standby groups and clear the standby state."""
    global _STANDBY_MC2
    result = dict(
        mc2=_STANDBY_MC2,
    )
    _STANDBY_MC2 = None
    return result
