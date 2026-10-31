# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Track Worker progress and coordinate Ascend Elastic EP prepare drains."""

from functools import wraps

import torch
from vllm.distributed.utils import cpu_all_reduce_
from vllm.logger import init_logger
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore
from vllm.v1.engine.core_client import DPAsyncMPClient

from vllm_ascend.distributed.elastic_ep.dp_collective import (
    install_dp_collective_tracking,
    needs_worker_progress,
    open_worker_state,
)

_original_init = EngineCore.__init__
_original_step = EngineCore.step
_original_step_with_batch_queue = EngineCore.step_with_batch_queue
_original_execute_dummy_batch = EngineCore.execute_dummy_batch
_original_has_global_unfinished_reqs = DPEngineCoreProc._has_global_unfinished_reqs
_original_eep_commit_pause_mode = DPAsyncMPClient._eep_commit_pause_mode
logger = init_logger(__name__)


def _get_prepare_controller(engine):
    state = getattr(engine, "eep_scaling_state", None)
    return state.prepare_controller if state is not None else None


@wraps(_original_eep_commit_pause_mode)
def _eep_commit_pause_mode(self):
    coordinator = self.external_eep_coordinator
    prepared = coordinator.prepared_scale if coordinator is not None else None
    if prepared is not None and prepared.scale_up:
        # Load the Worker implementation only when an external operation is prepared.
        from vllm_ascend.distributed.elastic_ep.elastic_execute import AscendElasticEPScalingExecutor

        key = AscendElasticEPScalingExecutor._capture_key(prepared.bootstrap.operation_id, "enabled")
        store = prepared.reconfig_store
        # Use this operation's actual decision, not just the MegaMoe config.
        # Old graphs and KV cache survive commit; pause batches, not whole requests.
        if store.check([key]) and store.get(key) == b"1":
            return "keep"
    return _original_eep_commit_pause_mode(self)


@wraps(_original_init)
def _init_with_worker_progress(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    config = self.vllm_config
    if needs_worker_progress(config):
        metadata = self.model_executor.collective_rpc("get_elastic_ep_dp_collective_state")
        self._ascend_dp_collective_states = [open_worker_state(handle) for handle in metadata]


@wraps(_original_step)
def _step(self):
    controller = _get_prepare_controller(self)
    if controller is not None and controller.draining:
        return {}, False
    return _original_step(self)


@wraps(_original_step_with_batch_queue)
def _step_with_batch_queue(self):
    controller = _get_prepare_controller(self)
    if controller is None or not controller.draining:
        return _original_step_with_batch_queue(self)

    batch_queue = self.batch_queue
    if not batch_queue or not controller.poll_pending_batch(batch_queue[-1][0]):
        return None, False

    # Process only an already submitted batch; draining never schedules another.
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        self.capture_iteration_details(scheduler_output) as iteration_details,
        self.log_error_detail(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")
    self._process_aborts_queue()
    outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    self._attach_iteration_details(outputs, iteration_details)
    return outputs, False


@wraps(_original_execute_dummy_batch)
def _execute_dummy_batch(self):
    controller = _get_prepare_controller(self)
    if controller is not None and controller.draining:
        return controller.step_dummy_batch()
    return _original_execute_dummy_batch(self)


@wraps(_original_has_global_unfinished_reqs)
def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
    if not self.vllm_config.parallel_config.enable_elastic_ep:
        return _original_has_global_unfinished_reqs(self, local_unfinished)

    controller = _get_prepare_controller(self)
    if controller is not None and controller.draining:
        return True
    self.step_counter += 1
    if self.step_counter != 1 and self.step_counter % self.dp_sync_interval != 0:
        return True

    # Every elastic NPU rank sends three fields, even before it receives PREPARE.
    # Piggyback readiness on normal DP sync so peers stop at the same boundary.
    tensor = torch.tensor(
        [int(local_unfinished), int(self.pending_pause), int(controller is not None and controller.is_ready())],
        dtype=torch.int32,
        device="cpu",
    )
    cpu_all_reduce_(tensor, self.dp_group, timeout=self.vllm_config.parallel_config.cpu_distributed_timeout)
    dp_size = self.dp_group.size()
    pause_count = tensor[1].item()
    prepare_count = tensor[2].item()
    if prepare_count == dp_size:
        assert controller is not None
        controller.begin()

    if pause_count == dp_size:
        self.ignore_start_dp_wave = True
        self.pending_pause = False
        logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")
    elif self.enable_fault_tolerance and self.ft_sentinel.coordinator_disabled and not self.ignore_start_dp_wave:
        # After pause consensus, let outstanding requests drain and stop stepping.
        return True
    return tensor[0].item() > 0 or pause_count % dp_size != 0 or prepare_count > 0


EngineCore.__init__ = _init_with_worker_progress
EngineCore.step = _step
EngineCore.step_with_batch_queue = _step_with_batch_queue
EngineCore.execute_dummy_batch = _execute_dummy_batch
DPEngineCoreProc._has_global_unfinished_reqs = _has_global_unfinished_reqs
DPAsyncMPClient._eep_commit_pause_mode = _eep_commit_pause_mode
install_dp_collective_tracking()
