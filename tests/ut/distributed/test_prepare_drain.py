# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import sync_cudagraph_and_dp_padding

from vllm_ascend.distributed.elastic_ep.dp_collective import (
    WorkerDPCollectiveState,
    install_dp_collective_tracking,
    open_worker_state,
    track_dp_collectives,
)
from vllm_ascend.distributed.elastic_ep.prepare_drain import AscendPrepareDrain


@pytest.mark.parametrize(
    "ready,counts,running",
    [
        (None, [0, 0, 0], False),
        (None, [0, 0, 2], True),
        (True, [0, 0, 3], True),
        (False, [0, 1, 0], True),
        (False, [0, 3, 0], False),
        (False, [1, 0, 0], True),
    ],
)
def test_prepare_consensus_uses_normal_dp_sync_even_before_local_prepare(monkeypatch, ready, counts, running):
    from vllm_ascend.patch.platform import patch_elastic_ep

    controller = None if ready is None else SimpleNamespace(draining=False, is_ready=lambda: ready, begin=MagicMock())
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(enable_elastic_ep=True)),
        eep_scaling_state=SimpleNamespace(prepare_controller=controller),
        step_counter=0,
        dp_sync_interval=16,
        dp_group=SimpleNamespace(size=lambda: 3),
        pending_pause=counts[1] > 0,
        ignore_start_dp_wave=False,
        enable_fault_tolerance=False,
    )

    def reduce(tensor, **kwargs):
        assert tensor.tolist() == [0, int(engine.pending_pause), int(bool(ready))]
        assert kwargs["group"] is engine.dp_group
        tensor.copy_(torch.tensor(counts))

    monkeypatch.setattr(patch_elastic_ep.torch.distributed, "all_reduce", reduce)
    assert patch_elastic_ep._has_global_unfinished_reqs(engine, False) is running
    assert engine.step_counter == 1
    assert engine.ignore_start_dp_wave is (counts[1] == 3)
    if controller is not None:
        assert controller.begin.call_count == int(counts[2] == 3)


def test_prepare_waits_for_worker_startup_without_rechecking_a_busy_store(monkeypatch):
    state = SimpleNamespace(
        _ensure_new_dp_group=MagicMock(return_value=False),
        _prepare_workers=MagicMock(side_effect=[False, True]),
        _prepare_future=None,
        _collective_rpc=MagicMock(return_value=[False]),
        reconfig_request=SimpleNamespace(operation_id="restore"),
        old_dp_group=SimpleNamespace(size=lambda: 3),
        new_parallel_config=SimpleNamespace(
            data_parallel_size=4,
            world_size=2,
            _eep_existing_dp_ranks=[1, 2, 3],
            data_parallel_master_ip="127.0.0.1",
            _coord_store_port=1234,
        ),
    )
    store = MagicMock()
    store.check.return_value = False
    monkeypatch.setattr(
        "vllm_ascend.distributed.elastic_ep.prepare_drain.get_cached_tcp_store_client", lambda *args: store
    )
    drain = AscendPrepareDrain(state)
    assert not state._prepare_workers()
    store.check.assert_not_called()
    state._ensure_new_dp_group.return_value = True
    assert not state._prepare_workers()
    drain._prepare_workers.assert_not_called()
    store.check.assert_called_once_with(
        ["eep_new_worker_dist_init_ready/1234/0", "eep_new_worker_dist_init_ready/1234/1"]
    )
    store.check.return_value = True
    assert not state._prepare_workers()
    state._collective_rpc.assert_not_called()
    state._prepare_future = Future()
    store.check.side_effect = AssertionError("background wait owns the connection")
    assert state._prepare_workers()
    assert drain._prepare_workers.call_count == 2
    assert not drain.draining
    assert not drain._uses_precommit_graph_capture()
    state._collective_rpc.assert_called_once()


@pytest.mark.parametrize("rank", [0, 3])
def test_local_capture_preserves_dp_shape_without_collectives_or_serving_epochs(rank):
    descriptor = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=2)
    manager = MagicMock()
    manager.dispatch.return_value = descriptor
    progress = WorkerDPCollectiveState()
    progress.tensor.fill_(17)
    runner = SimpleNamespace(_elastic_ep_dp_state=progress)
    run = track_dp_collectives(lambda self, **kwargs: sync_cudagraph_and_dp_padding(**kwargs))
    install_dp_collective_tracking()
    with (
        skip_dp_coordination(),
        patch(
            "vllm.v1.worker.gpu.dp_utils.get_dp_group",
            return_value=SimpleNamespace(cpu_group=object()),
        ),
        patch(
            "vllm.v1.worker.gpu.dp_utils.get_tp_group",
            side_effect=AssertionError("TP group accessed"),
        ),
        patch(
            "vllm.v1.worker.gpu.dp_utils.dist.all_reduce",
            side_effect=AssertionError("DP all-reduce called"),
        ),
    ):
        desc, sync = run(
            runner,
            cudagraph_manager=manager,
            desired_batch_desc=descriptor,
            num_tokens=7,
            num_reqs=2,
            uniform_token_count=None,
            dp_size=4,
            dp_rank=rank,
            parallel_config=SimpleNamespace(enable_fault_tolerance=True, tensor_parallel_size=2),
        )
        assert desc == descriptor
        assert sync.num_tokens_across_dp.tolist() == [8, 8, 8, 8]
        assert not sync.eager
        assert progress.tensor.tolist() == [17, 17]


def test_worker_progress_storage_is_shared_without_executor_tensor_serialization():
    import pickle

    owner = WorkerDPCollectiveState()
    observer = open_worker_state(pickle.loads(pickle.dumps(owner.share())))
    owner.tensor[0] = 7
    assert observer.tolist() == [7, 0]
    owner.tensor[1] = 7
    assert observer.tolist() == [7, 7]


@pytest.mark.parametrize("elastic", [False, True])
def test_worker_progress_attaches_once_after_engine_initialization(monkeypatch, elastic):
    from vllm_ascend.patch.platform import patch_elastic_ep

    owner = WorkerDPCollectiveState()
    initialized = []
    monkeypatch.setattr(patch_elastic_ep, "_original_init", lambda self: initialized.append(True))

    def share(*args):
        assert initialized
        return [owner.share()]

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(
                enable_elastic_ep=elastic,
                enable_fault_tolerance=True,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
        ),
        model_executor=SimpleNamespace(collective_rpc=MagicMock(side_effect=share)),
    )
    patch_elastic_ep._init_with_worker_progress(engine)
    if elastic:
        engine.model_executor.collective_rpc.assert_called_once_with("get_elastic_ep_dp_collective_state")
        owner.tensor.fill_(13)
        assert engine._ascend_dp_collective_states[0].tolist() == [13, 13]
    else:
        engine.model_executor.collective_rpc.assert_not_called()


def test_serving_collective_aligns_joiner_epoch_and_preserves_failed_progress(monkeypatch):
    from vllm_ascend.distributed.elastic_ep import dp_collective

    progress = WorkerDPCollectiveState()
    runner = SimpleNamespace(_elastic_ep_dp_state=progress)
    install_dp_collective_tracking()
    monkeypatch.setattr(dp_collective, "get_dp_group", lambda: SimpleNamespace(rank_in_group=0))

    def reduce(tensor, **kwargs):
        assert tensor.shape == (7, 2)
        assert progress.tensor.tolist() == [1, 0]
        tensor[-1, 1] = 50

    monkeypatch.setattr(dp_collective.dist, "all_reduce", reduce)

    @track_dp_collectives
    def run(self):
        tensor = torch.zeros((6, 2), dtype=torch.int32)
        dp_collective.dp_utils.dist.all_reduce(tensor, group=object())
        return tensor

    assert run(runner).shape == (6, 2)
    assert progress.tensor.tolist() == [50, 50]
    monkeypatch.setattr(dp_collective.dist, "all_reduce", MagicMock(side_effect=RuntimeError("collective failed")))
    with pytest.raises(RuntimeError, match="collective failed"):
        run(runner)
    assert progress.tensor.tolist() == [51, 50]


@pytest.fixture
def drain_state(monkeypatch):
    from vllm.distributed.elastic_ep.elastic_state import ScaleUpExistingEngineState

    with ThreadPoolExecutor(max_workers=1) as pool:
        progress = torch.tensor([5, 5], dtype=torch.int64)
        engine = SimpleNamespace(
            batch_queue=deque(),
            _ascend_dp_collective_states=[progress],
            model_executor=SimpleNamespace(is_sleeping=False, execute_dummy_batch=MagicMock()),
            engines_running=False,
            step_counter=16,
        )
        state = SimpleNamespace(
            engine_core=engine,
            scale_type="scale_up",
            worker_type="existing",
            state=ScaleUpExistingEngineState.PREPARE,
            _prepare_workers=MagicMock(return_value=True),
            _ensure_new_dp_group=MagicMock(return_value=True),
            _prepare_future=None,
            _prepare_executor=pool,
            reconfig_request=SimpleNamespace(operation_id="restore"),
            old_dp_group=SimpleNamespace(size=lambda: 2),
            _collective_rpc=MagicMock(return_value=[True]),
        )
        drain = AscendPrepareDrain(state)
        monkeypatch.setattr(drain, "_new_workers_dist_init_ready", lambda: True)
        assert not state._prepare_workers()
        state._collective_rpc.reset_mock()
        yield drain, state, progress


def test_pending_result_wait_keeps_engine_available_for_boundary_exchange(drain_state, monkeypatch):
    from vllm_ascend.patch.platform import patch_elastic_ep

    drain, state, _ = drain_state
    engine = state.engine_core
    engine.eep_scaling_state = SimpleNamespace(prepare_controller=drain)
    engine.capture_iteration_details = lambda output: nullcontext(None)
    engine.log_error_detail = lambda output: nullcontext()
    engine._process_aborts_queue = MagicMock()
    engine._attach_iteration_details = MagicMock()
    engine.scheduler = MagicMock()
    pending = Future()
    scheduled = object()
    engine.batch_queue.append((pending, scheduled, Future()))
    exchanges = []

    def gather(outputs, local, **kwargs):
        exchanges.append(local.tolist())
        for output in outputs:
            output.copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    assert not drain.progress()  # Wait for old-rank consensus before draining.
    drain.begin()
    try:
        assert patch_elastic_ep._step_with_batch_queue(engine) == (None, False)
        assert len(engine.batch_queue) == 1
        assert not state._prepare_workers()  # Keep exchanging while result() waits.
        assert exchanges == [[5, 5, 1]]
        state._collective_rpc.assert_not_called()
    finally:
        pending.set_result({})
        drain._result_future.result(timeout=2)
    assert patch_elastic_ep._step_with_batch_queue(engine) == (engine.scheduler.update_from_output.return_value, False)
    engine.scheduler.schedule.assert_not_called()
    engine.scheduler.update_from_output.assert_called_once_with(scheduled, {})
    engine._process_aborts_queue.assert_called_once()
    assert not engine.batch_queue
    assert state._prepare_workers()
    drain._prepare_workers.assert_called_once_with()
    state._collective_rpc.assert_called_once()
    assert not drain.draining and state.engine_core.engines_running
    assert state.engine_core.step_counter == 0


def test_catch_up_dummy_wait_does_not_block_or_submit_twice(drain_state, monkeypatch):
    from vllm_ascend.patch.platform import patch_elastic_ep

    drain, state, progress = drain_state
    state.engine_core.eep_scaling_state = SimpleNamespace(prepare_controller=drain)
    release = Event()
    state.engine_core.model_executor.execute_dummy_batch = MagicMock(side_effect=lambda: release.wait(2))
    local_states = []

    def gather(outputs, local, **kwargs):
        local_states.append(local.tolist())
        outputs[0].copy_(local)
        outputs[1].copy_(torch.tensor([6, 6, 0]))

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    drain.begin()
    assert not drain.progress()
    patch_elastic_ep._execute_dummy_batch(state.engine_core)
    submitted = drain._dummy_future
    try:
        assert not drain.progress()
        patch_elastic_ep._execute_dummy_batch(state.engine_core)
        assert drain._dummy_future is submitted
        assert local_states[-1] == [5, 5, 1]
        state._collective_rpc.assert_not_called()
    finally:
        release.set()
        submitted.result(timeout=2)
    progress.fill_(6)
    assert drain.progress()
    state.engine_core.model_executor.execute_dummy_batch.assert_called_once()
    state._collective_rpc.assert_called_once()


def test_fresh_elastic_group_supports_fault_tolerant_dp_sync():
    # Elastic groups bypass GroupCoordinator.__init__, but must expose the
    # same FT topology state before the first profiling/model step.
    with (
        patch("vllm.distributed.parallel_state._WORLD", None),
        patch("vllm.distributed.stateless_coordinator.get_ip", return_value="127.0.0.1"),
        patch(
            "vllm.distributed.stateless_coordinator.allocate_group_ports",
            return_value=([12301, 12302, 12303], []),
        ),
        patch("vllm.distributed.stateless_coordinator.stateless_init_torch_distributed_process_group"),
        patch("vllm.distributed.stateless_coordinator.StatelessProcessGroup.create"),
    ):
        group = StatelessGroupCoordinator([[0, 1]], 0, "gloo", False, MagicMock(), group_name="dp")

    descriptor = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=1)
    with (
        patch("vllm.v1.worker.gpu.dp_utils.get_dp_group", return_value=group),
        patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce"),
    ):
        sync_cudagraph_and_dp_padding(
            None,
            descriptor,
            num_tokens=1,
            num_reqs=1,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            parallel_config=SimpleNamespace(enable_fault_tolerance=True, tensor_parallel_size=1),
        )
