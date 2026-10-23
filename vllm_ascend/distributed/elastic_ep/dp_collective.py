# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Worker-owned serving progress, shared with EngineCore without executor changes."""

from contextvars import ContextVar
from functools import wraps
from multiprocessing.reduction import ForkingPickler

import torch
import torch.distributed as dist
from vllm.distributed import get_dp_group
from vllm.v1.worker.gpu import dp_utils

_CURRENT_STATE: ContextVar[torch.Tensor | None] = ContextVar("ascend_dp_collective_state", default=None)


def needs_worker_progress(config) -> bool:
    parallel = config.parallel_config
    return bool(
        parallel.enable_elastic_ep
        and parallel.enable_fault_tolerance
        and config.use_v2_model_runner
        and parallel.tensor_parallel_size
        == parallel.pipeline_parallel_size
        == parallel.prefill_context_parallel_size
        == 1
    )


class WorkerDPCollectiveState:
    def __init__(self) -> None:
        self.tensor = torch.zeros(2, dtype=torch.int64).share_memory_()

    def share(self) -> bytes:
        # Normal RPC tensor serialization copies bytes. Preserve multiprocessing
        # storage handles inside an opaque payload instead.
        return bytes(ForkingPickler.dumps(self.tensor))


def open_worker_state(metadata: bytes) -> torch.Tensor:
    return ForkingPickler.loads(metadata)


def track_dp_collectives(execute_model):
    @wraps(execute_model)
    def wrapped(self, *args, **kwargs):
        state = getattr(self, "_elastic_ep_dp_state", None)
        token = _CURRENT_STATE.set(state.tensor if state is not None else None)
        try:
            return execute_model(self, *args, **kwargs)
        finally:
            _CURRENT_STATE.reset(token)

    return wrapped


class _ServingDPCollectives:
    """Extend only the metadata collective issued by vLLM's DP dispatch helper."""

    def __getattr__(self, name):
        return getattr(dist, name)

    def all_reduce(self, tensor, *args, **kwargs):
        state = _CURRENT_STATE.get()
        if state is None:
            return dist.all_reduce(tensor, *args, **kwargs)
        entered = int(state[0].item()) + 1
        state[0] = entered
        metadata = torch.cat((tensor, torch.zeros_like(tensor[:1])))
        metadata[-1, get_dp_group().rank_in_group] = entered
        result = dist.all_reduce(metadata, *args, **kwargs)
        tensor.copy_(metadata[:-1])
        completed = int(metadata[-1].max().item())
        state[0] = completed
        state[1] = completed
        return result


def install_dp_collective_tracking() -> None:
    if not isinstance(dp_utils.dist, _ServingDPCollectives):
        dp_utils.dist = _ServingDPCollectives()
