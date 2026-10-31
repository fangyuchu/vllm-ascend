# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.distributed.eplb.eplb_state import EplbModelState, _commit_eplb_maps
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_ep_group,
    get_tp_group,
)
from vllm.distributed.utils import set_gloo_backend_timeout
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.sentinel.eplb_redistribute import (
    mark_dead_expert_slots_inplace,
    redistribute_expert_placement,
)
from vllm.v1.worker.sentinel.gpu_worker_sentinel import (
    WorkerSentinel as GPUWorkerSentinel,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.eplb.eplb_state import refresh_model_routing_tables
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.sentinel.eplb_quant_schemes import DRAFTER_SUFFIX_MAP
from vllm_ascend.worker.sentinel.eplb_redistribute import (
    build_orig_to_dense_rank_table,
    densify_routing_table_physical_ids,
    reload_experts_from_disk,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


_sentinel: "WorkerSentinel | None" = None


def fault_barrier_wrapper(func: Callable):
    """Barrier between device faults and the async step loop.

    On the first device-touching fault (e.g. an EP-group allreduce failing on a
    dead peer that poisons the model stream) it quarantines the worker and
    resets the device immediately, before any tensor teardown on the broken
    stream can std::terminate the process. While quarantined, wrapped methods
    short-circuit with an empty output so in-flight async steps drain safely
    without re-hitting the device; retry lifts the quarantine only after the
    groups are rebuilt.
    """

    def wrapped(self, *args, **kwargs):
        sentinel = _sentinel
        if sentinel is None:
            return func(self, *args, **kwargs)
        if sentinel.worker_faulted:
            return EMPTY_MODEL_RUNNER_OUTPUT
        try:
            return func(self, *args, **kwargs)
        except SystemExit:
            raise
        except Exception as exc:
            sentinel.worker_faulted = True
            logger.warning(
                "[FT] Quarantining %s after fault: %s",
                getattr(self, "rank", "worker"),
                exc,
            )
            try:
                sentinel.reset_device()
            except Exception:
                logger.exception(
                    "[FT] self device reset failed on %s.",
                    getattr(self, "rank", "worker"),
                )
            return EMPTY_MODEL_RUNNER_OUTPUT

    return wrapped


class WorkerSentinel(GPUWorkerSentinel):
    """Per-worker sentinel for fault tolerance on Ascend NPU.

    Handles commands dispatched from EngineCoreSentinel via collective_rpc,
    including device restart and DP group re-initialization on retry, and
    MC2 elastic_info masking + expert redistribution on scale_down.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        global _sentinel
        self.device = device
        self.worker = worker
        # Set once a device-touching method faults, to keep this worker off the
        # device until FT recovery rebuilds the groups.
        self.worker_faulted = False
        _sentinel = self

    def query_mask(self, ft_request: FaultToleranceRequest) -> dict:
        """Report the dead-rank mask (upstream convention: 0=live, 1=dead)."""
        return {"mask": get_ep_all2all_manager().query_active_mask().tolist()}

    def reset_device(self) -> None:
        NPUPlatform.set_device(self.device)
        torch_npu.npu.stop_device(self.device.index)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()

    def retry(self, ft_request: FaultToleranceRequest):
        # Reset first so hung device collectives are aborted, then run the
        # base flow and lift the quarantine after the groups are rebuilt.
        self.reset_device()
        super().retry(ft_request)
        self.worker_faulted = False

    def init_num_local_experts(self) -> None:
        """Record the per-rank physical expert slot count after model load."""
        if self.worker.model_runner.eplb_state is None:
            return
        eplb_model_state = self._eplb_model_state()
        num_local_experts = eplb_model_state.physical_to_logical_map.shape[1] // get_ep_group().world_size
        get_ep_all2all_manager().set_num_local_physical_experts(num_local_experts)

    def scale_down(self, ft_request: FaultToleranceRequest):
        """Scale down over the surviving DP ranks, reusing the upstream flow.

        ``super().scale_down`` runs the deterministic dead-rank masking and
        expert redistribution, dispatching ``retry`` / ``_redistribute_experts``
        to the Ascend overrides below. Ascend adds its platform preconditions
        and a dummy-batch runnability check on top.
        """
        self._validate_scale_down_preconditions()
        super().scale_down(ft_request)

        # Verify the redistributed model is runnable before reporting healthy.
        parallel_config = self.worker.parallel_config
        timeout = timedelta(seconds=self.worker.parallel_config.fault_tolerance_config.engine_recovery_timeout_sec)
        if parallel_config.data_parallel_size > 1:
            set_gloo_backend_timeout(get_dp_group().cpu_group, timeout)
        if parallel_config.tensor_parallel_size > 1:
            set_gloo_backend_timeout(get_tp_group().cpu_group, timeout)
        self.worker.execute_dummy_batch()
        self.activate_cpu_group_timeouts(ft_request)
        torch.npu.synchronize()

    def _validate_scale_down_preconditions(self) -> None:
        if not self.worker.use_v2_model_runner:
            raise ValueError("[FT] scale_down on Ascend NPU requires the v2 model runner.")
        model_runner = self.worker.model_runner
        eplb_config = self.worker.parallel_config.eplb_config
        if model_runner.eplb_state is None or eplb_config.num_redundant_experts <= 0:
            raise ValueError(
                "[FT] scale_down requires EPLB with num_redundant_experts > 0 to re-host the dead rank's experts."
            )
        self._validate_drafter_eplb_coverage(model_runner)
        ascend_config = get_ascend_config()
        if ascend_config.enable_fused_mc2:
            raise ValueError(
                "[FT] scale_down is not supported with enable_fused_mc2: the "
                "fused dispatch_ffn_combine operators take no elastic_info."
            )
        if ascend_config.enable_mc2_hierarchy_comm:
            raise ValueError(
                "[FT] scale_down (elastic_info) is mutually exclusive with mc2 hierarchy comm (comm_alg='hierarchy')."
            )
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise ValueError(
                "[FT] scale_down requires npu_moe_distribute_dispatch_v2 "
                "(aclnn V3+); please upgrade the CANN/torch_npu version."
            )

    def _redistribute_experts(self, dead_ep_ranks: set[int]) -> None:
        """Redistribute experts onto the surviving slots after scale-down.

        Overrides the upstream flow so reassigned weights reload through the
        Ascend reloader: the upstream reloader writes via ``model.load_weights``,
        which cannot produce Ascend's runtime expert layout (transpose / NZ /
        per-slot lists / quant scales). The shared redistribution steps (mark
        dead slots, steal spare slots, rebuild the logical maps) are kept; on
        top of that, refreshes the Ascend kernel-facing routing tables into
        the densified id space and shrinks the MC2 physical-expert width.

        Each EPLB-registered model (main model and drafter) is redistributed
        against its own placement and reloaded from its own checkpoint; results
        are never shared across models.
        """
        model_runner = self.worker.model_runner

        models: list[tuple[torch.nn.Module, EplbModelState, str, dict[str, str] | None]] = [
            (model_runner.model, self._eplb_model_state(), self.worker.vllm_config.model_config.model, None)
        ]
        drafter = self._registered_drafter(model_runner)
        if drafter is not None:
            draft_model, draft_state = drafter
            # A drafter may have its own checkpoint; reload from that one.
            models.append((draft_model, draft_state, self._drafter_checkpoint_path(), DRAFTER_SUFFIX_MAP))

        ep_world_size = get_ep_group().world_size
        for model, model_state, model_path, suffix_map in models:
            p2l = model_state.physical_to_logical_map
            num_logical = model_state.logical_replica_count.shape[1]
            num_local_experts = p2l.shape[1] // ep_world_size

            # Both mutate p2l in place; the apply below derives l2p/lrc from it.
            mark_dead_expert_slots_inplace(p2l, dead_ep_ranks, num_local_experts)
            reassignments = redistribute_expert_placement(p2l, num_logical, num_local_experts)

            logger.info(
                "[FT] %s expert redistribution: moe_layers=%d, num_logical=%d, reassignments=%d",
                type(model).__name__,
                p2l.shape[0],
                num_logical,
                len(reassignments),
            )
            self._apply_placement(model, model_state, model_path, reassignments, suffix_map=suffix_map)

    def _apply_placement(
        self,
        model: torch.nn.Module,
        model_state: EplbModelState,
        model_path: str,
        reassignments: set[tuple[int, int]],
        suffix_map: dict[str, str] | None = None,
    ) -> None:
        """Commit a model's redistributed placement in place so captured graphs stay valid."""
        p2l = model_state.physical_to_logical_map
        _commit_eplb_maps(model_state, p2l.cpu())
        # Keep the model-side expert map: slot ownership is unchanged, and the
        # map is physical-id indexed, so p2l (logical ids) must not be written into it.
        if reassignments:
            reload_experts_from_disk(model, model_path, reassignments, suffix_map=suffix_map)
        refresh_model_routing_tables(model_state)
        self._densify_routing_tables(model_state)

    @staticmethod
    def _drafter_model(model_runner) -> torch.nn.Module | None:
        """The speculative drafter the runner loaded, if any."""
        return getattr(getattr(model_runner, "speculator", None), "model", None)

    def _drafter_eplb_state(self, model_runner) -> EplbModelState | None:
        """The drafter's EPLB state; None unless EPLB registered its MoE part."""
        draft_moe = get_mixture_of_experts_model(self._drafter_model(model_runner))
        if draft_moe is None:
            return None
        for model_state in model_runner.eplb_state.model_states.values():
            if model_state.model is draft_moe:
                return model_state
        return None

    def _registered_drafter(self, model_runner) -> tuple[torch.nn.Module, EplbModelState] | None:
        """Return the speculative drafter and its EPLB state, if EPLB has one."""
        model_state = self._drafter_eplb_state(model_runner)
        if model_state is None:
            return None
        return self._drafter_model(model_runner), model_state

    def _drafter_checkpoint_path(self) -> str:
        """Checkpoint directory the drafter was loaded from (may differ from the target's)."""
        speculative_config = getattr(self.worker.vllm_config, "speculative_config", None)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        if draft_model_config is not None:
            return draft_model_config.model
        return self.worker.vllm_config.model_config.model

    def _validate_drafter_eplb_coverage(self, model_runner) -> None:
        """Reject a MoE drafter EPLB did not register (e.g. V4.1 DSpark).

        It would keep routing to the dead rank after the group shrinks.
        """
        drafter = self._drafter_model(model_runner)
        has_moe = get_mixture_of_experts_model(drafter) is not None or bool(getattr(drafter, "moe_layers", None))
        if has_moe and self._drafter_eplb_state(model_runner) is None:
            raise ValueError(
                "[FT] scale_down does not support a MoE drafter that EPLB did not register "
                "(e.g. a V4.1 DSpark draft, or a draft with its own EPLB disabled): its "
                "expert placement would keep pointing at the dead rank."
            )

    def _densify_routing_tables(self, eplb_model_state) -> None:
        """Renumber the kernel-facing routing tables into the densified id space.

        The MC2 kernels consume the routing tables in the densified id space,
        so the kernel-facing values are renumbered in place after the refresh.
        The dead set is cumulative (accumulated across recovery rounds), so it
        is derived from the manager's mask rather than this round's ranks.
        """
        p2l = eplb_model_state.physical_to_logical_map
        ep_world_size = get_ep_group().world_size
        num_local_experts = p2l.shape[1] // ep_world_size
        active_mask = get_ep_all2all_manager().query_active_mask()
        dead_ranks = {rank for rank, is_dead in enumerate(active_mask.tolist()) if is_dead}
        orig_to_dense_rank = build_orig_to_dense_rank_table(ep_world_size, dead_ranks)
        for layer in eplb_model_state.model.moe_layers:
            routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
            if routing_table is not None:
                densify_routing_table_physical_ids(routing_table, orig_to_dense_rank, num_local_experts)
