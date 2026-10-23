# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import TYPE_CHECKING

from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

logger = init_logger(__name__)

_WORKER_DIST_INIT_READY_KEY_PREFIX = "eep_new_worker_dist_init_ready"


def _operation_scope(parallel_config: "ParallelConfig") -> str:
    """Return a per-reconfiguration namespace shared by old and new ranks."""
    coord_store_port = parallel_config._coord_store_port
    if not coord_store_port:
        raise RuntimeError("Elastic EP scale-up Worker readiness requires an active coordination store.")
    return str(coord_store_port)


def worker_dist_init_ready_key(operation_scope: str, global_rank: int) -> str:
    return f"{_WORKER_DIST_INIT_READY_KEY_PREFIX}/{operation_scope}/{global_rank}"


def new_worker_dist_init_ready_keys(
    parallel_config: "ParallelConfig",
    old_dp_size: int,
    new_dp_size: int,
    worker_world_size: int,
) -> list[str]:
    operation_scope = _operation_scope(parallel_config)
    return [
        worker_dist_init_ready_key(
            operation_scope,
            dp_rank * worker_world_size + worker_rank,
        )
        for dp_rank in range(new_dp_size)
        if dp_rank not in (parallel_config._eep_existing_dp_ranks or list(range(old_dp_size)))
        for worker_rank in range(worker_world_size)
    ]


def publish_worker_dist_init_ready(parallel_config: "ParallelConfig", worker_rank: int) -> None:
    """Publish that a scale-up Worker is about to enter distributed init."""
    global_rank = parallel_config.data_parallel_rank * parallel_config.world_size
    global_rank += worker_rank
    operation_scope = _operation_scope(parallel_config)
    store = get_cached_tcp_store_client(
        parallel_config.data_parallel_master_ip,
        parallel_config._coord_store_port,
    )
    store.set(
        worker_dist_init_ready_key(operation_scope, global_rank),
        b"1",
    )
    logger.info(
        "[Elastic EP scale-up] Worker reached distributed init: "
        "operation_scope=%s, global_rank=%s, dp_rank=%s, worker_rank=%s",
        operation_scope,
        global_rank,
        parallel_config.data_parallel_rank,
        worker_rank,
    )
