# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend prepare drain at a common serving-Worker communication boundary."""

from concurrent.futures import Future
from typing import Any

import torch
from vllm.distributed.elastic_ep.elastic_state import ScaleUpExistingEngineState
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger

from vllm_ascend.distributed.elastic_ep.readiness import new_worker_dist_init_ready_keys

logger = init_logger(__name__)


class AscendPrepareDrain:
    def __init__(self, scaling_state) -> None:
        self.state = scaling_state
        self._prepare_workers = scaling_state._prepare_workers
        self._workers_prepared = False
        self._capture_enabled: bool | None = None
        # Extend only this old-rank operation; other platforms keep the base path.
        scaling_state._prepare_workers = self.prepare_workers
        self.draining = False
        self._quiesced = False
        self._catch_up = False
        self._result_future: Future[Any] | None = None
        self._dummy_future: Future[Any] | None = None

    @property
    def engine(self):
        return self.state.engine_core

    @property
    def _operation_id(self) -> str:
        request = self.state.reconfig_request
        return request.operation_id if request is not None else ""

    def prepare_workers(self) -> bool:
        if not self._workers_prepared:
            if not self.state._ensure_new_dp_group():
                return False
            # The background wait owns the cached store connection lock.
            if self.state._prepare_future is None and not self._new_workers_dist_init_ready():
                return False
            if not self._prepare_workers():
                return False
            self._workers_prepared = True
        return not self._uses_precommit_graph_capture() or self.progress()

    def _new_workers_dist_init_ready(self) -> bool:
        state = self.state
        ready_keys = new_worker_dist_init_ready_keys(
            state.new_parallel_config,
            state.old_dp_group.size(),
            state.new_parallel_config.data_parallel_size,
            state.new_parallel_config.world_size,
        )
        store = get_cached_tcp_store_client(
            state.new_parallel_config.data_parallel_master_ip,
            state.new_parallel_config._coord_store_port,
        )
        if not store.check(ready_keys):
            return False
        logger.info_once(
            "[Elastic EP scale-up] All new Workers reached distributed init; "
            "starting standby communication group creation"
        )
        return True

    def _uses_precommit_graph_capture(self) -> bool:
        if self._capture_enabled is None:
            results = self.state._collective_rpc(
                "elastic_ep_execute", args=("supports_precommit_graph_capture", self._operation_id)
            )
            enabled_values = {bool(result) for result in results}
            if len(enabled_values) != 1:
                raise RuntimeError(f"Workers disagreed on pre-commit graph capture support: {results}")
            self._capture_enabled = enabled_values.pop()
        return self._capture_enabled

    def is_ready(self) -> bool:
        state = self.state
        return (
            state.scale_type == "scale_up"
            and state.worker_type == "existing"
            and state.state is ScaleUpExistingEngineState.PREPARE
            and self._workers_prepared
            and self._capture_enabled is True
            and not self.draining
        )

    def begin(self) -> None:
        assert self.is_ready()
        self.draining = True
        self._catch_up = False
        logger.info(
            "[Elastic EP] All old ranks reached prepare consensus; starting inference drain: "
            "operation_id=%s, dp_step=%s",
            self._operation_id,
            self.engine.step_counter,
        )

    def _worker_progress(self) -> tuple[int, int]:
        snapshots = []
        for tensor in self.engine._ascend_dp_collective_states:
            while True:
                entered = int(tensor[0].item())
                completed = int(tensor[1].item())
                if entered == int(tensor[0].item()) and completed <= entered:
                    snapshots.append((entered, completed))
                    break
        if not snapshots or any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
            raise RuntimeError(f"Local Workers have inconsistent DP collective progress: {snapshots}")
        return snapshots[0]

    def poll_pending_batch(self, future: Future[Any]) -> bool:
        if self._result_future is None:
            # Multiproc FutureWrapper receives replies lazily in result().
            # Only this oldest batch may consume the executor's reply queue.
            self._result_future = self.state._prepare_executor.submit(future.result)
        if not self._result_future.done():
            return False
        completed = self._result_future
        self._result_future = None
        completed.result()
        return True

    def step_dummy_batch(self) -> None:
        if self._catch_up and self._dummy_future is None and not self.engine.model_executor.is_sleeping:
            self._dummy_future = self.state._prepare_executor.submit(self.engine.model_executor.execute_dummy_batch)

    def _progress_boundary(self) -> bool:
        if self._dummy_future is not None and self._dummy_future.done():
            self._dummy_future.result()
            self._dummy_future = None
        batch_queue = self.engine.batch_queue
        # Count submitted dummy work even before its Worker publishes an epoch.
        pending = bool(batch_queue) or self._dummy_future is not None
        entered, completed = self._worker_progress()
        local = torch.tensor([entered, completed, int(pending)], dtype=torch.int64)
        group = self.state.old_dp_group
        gathered = [torch.empty_like(local) for _ in range(group.size())]
        torch.distributed.all_gather(gathered, local, group=group)
        states = [value.tolist() for value in gathered]
        target = max(value[0] for value in states)
        self._catch_up = not pending and entered < target
        if self._catch_up:
            self.engine.engines_running = True
        if all(started == target and finished == target and queued == 0 for started, finished, queued in states):
            self._catch_up = False
            return True
        logger.info_once(
            "[Elastic EP] Waiting for a common DP batch boundary: local_epoch=%s/%s, "
            "target_epoch=%s, local_pending_batches=%s, catch_up_dummy=%s",
            completed,
            entered,
            target,
            len(batch_queue) if batch_queue is not None else 0,
            self._catch_up,
        )
        return False

    def progress(self) -> bool:
        if not self.draining:
            self.engine.engines_running = True
            return False
        if not self._quiesced:
            if not self._progress_boundary():
                return False
            self._quiesced = True
        try:
            self.state._collective_rpc("elastic_ep_execute", args=("finalize_precommit_prepare", self._operation_id))
        finally:
            # Resume every old rank with the same EngineCore sync cadence.
            self.engine.step_counter = 0
            self.engine.engines_running = True
            self._catch_up = False
            self.draining = False
        return True
