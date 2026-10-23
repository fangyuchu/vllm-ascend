from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.distributed.test_mega_moe_capture import make_capture_state
from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager
from vllm_ascend.distributed.elastic_ep.elastic_execute import (
    AscendElasticEPScalingExecutor,
    _match_peer_parameters,
    setup_moe_comm_and_quant_method,
)
from vllm_ascend.quantization.quant_type import QuantType


def _executor_and_worker():
    executor = object.__new__(AscendElasticEPScalingExecutor)
    parallel_config = SimpleNamespace(
        world_size=2,
        data_parallel_size=2,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_master_port=29500,
        _data_parallel_master_port_list=[29500, 29501],
        _coord_store_port=1234,
        _data_parallel_master_rank=0,
        _eep_existing_dp_ranks=None,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=parallel_config,
            lora_config=None,
        ),
    )
    executor.worker_ref = lambda: worker
    executor.reconfig_request = None
    return executor, worker


def _capture_mapping_executor(monkeypatch, width, dead_ranks):
    executor, worker = _executor_and_worker()
    mapping = torch.arange(width).reshape(1, width)
    model_state = SimpleNamespace(
        physical_to_logical_map=mapping,
        logical_replica_count=torch.ones(1, 2),
    )
    worker.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(compute_hash=lambda: "model"),
        eplb_state=SimpleNamespace(model_states={"model": model_state}),
    )
    worker.device = torch.device("cpu")
    executor._mega_moe_precommit_capture = True
    executor._capture_old_dp_size = 4 - len(dead_ranks)
    executor.reconfig_request = SimpleNamespace(
        new_data_parallel_size=4,
        existing_dp_ranks=[rank for rank in range(4) if rank not in dead_ranks],
    )
    module = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    monkeypatch.setattr(f"{module}.get_dp_group", lambda: SimpleNamespace(world_size=4, dead_dp_ranks=dead_ranks))
    monkeypatch.setattr(f"{module}.get_standby_dp_group", lambda: object())
    broadcast = MagicMock()
    monkeypatch.setattr(f"{module}.upstream_elastic_execute.broadcast_expert_mapping", broadcast)
    return executor, model_state, broadcast


@pytest.mark.parametrize(
    "width,dead_ranks,expected",
    [
        (8, {2}, [0, 1, 2, 3, 0, 1, 6, 7]),
        (16, {2}, list(range(8)) + list(range(4)) + list(range(12, 16))),
        (8, {1, 3}, [0, 1, 0, 1, 4, 5, 4, 5]),
        (8, {0}, [2, 3, 2, 3, 4, 5, 6, 7]),
        (8, {0, 2, 3}, [2, 3, 2, 3, 2, 3, 2, 3]),
    ],
)
def test_capture_broadcast_preserves_survivors_and_serving_mapping(monkeypatch, width, dead_ranks, expected):
    executor, state, broadcast = _capture_mapping_executor(monkeypatch, width, dead_ranks)
    original = state.physical_to_logical_map
    snapshot = original.clone()

    executor.broadcast_expert_mapping()

    sent = broadcast.call_args.kwargs["physical_to_logical"]
    torch.testing.assert_close(sent, torch.tensor([expected]))
    assert state.physical_to_logical_map is original
    torch.testing.assert_close(original, snapshot)


def test_mega_restore_preserves_full_capacity_eplb_storage(monkeypatch):
    executor, worker = _executor_and_worker()
    mapping = torch.arange(8).reshape(1, 8)
    load_pass = torch.ones(1, 8)
    load_window = torch.ones(2, 1, 8)
    state = SimpleNamespace(
        physical_to_logical_map=mapping,
        physical_to_logical_map_buffer=mapping,
        expert_load_pass=load_pass,
        expert_load_pass_buffer=load_pass,
        expert_load_window=load_window,
        logical_to_physical_map=torch.zeros(1, 4, 2),
        logical_replica_count=torch.ones(1, 4),
        num_unpadded_tokens_tensors=[],
    )
    model = MagicMock()
    model.modules.return_value = [SimpleNamespace(moe_config=SimpleNamespace(num_local_experts=2, num_experts=8))]
    eplb = MagicMock(model_states={"model": state})
    worker.model_runner = SimpleNamespace(
        get_model=lambda: model, eplb_state=eplb, model_config=SimpleNamespace(compute_hash=lambda: "model")
    )
    worker.vllm_config.parallel_config.eplb_config = SimpleNamespace(num_redundant_experts=0)
    executor._capture_bootstrap_mapping = mapping.clone()
    executor._prepared_eplb_communicator = object()
    module = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    monkeypatch.setattr(f"{module}.is_moe_layer", lambda _: True)
    monkeypatch.setattr(f"{module}.set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(f"{module}.refresh_model_routing_tables", MagicMock())

    executor._restore_capture_scale_up_model_state(4)

    assert state.physical_to_logical_map.data_ptr() == mapping.data_ptr()
    assert state.expert_load_pass.data_ptr() == load_pass.data_ptr()
    assert state.expert_load_window is load_window
    eplb.update_mapping.assert_called_once_with(worker.model_runner.model_config, executor._capture_bootstrap_mapping)


@pytest.mark.parametrize("has_quant_group_name", [True, False], ids=["existing-rank", "new-rank"])
def test_setup_moe_comm_initializes_groups_in_same_order(monkeypatch, has_quant_group_name):
    """All ranks must enter lazy HCCL initialization in the same order."""
    initialized_groups = []

    def initialize_group(name):
        if name not in initialized_groups:
            initialized_groups.append(name)
        return f"new-{name}-group"

    backend = MagicMock()
    backend.get_hccl_comm_name.side_effect = lambda _: initialize_group("MC2")
    device_group = MagicMock()
    device_group._get_backend.return_value = backend
    mc2_group = SimpleNamespace(device_group=device_group, rank_in_group=3)
    quant_method = SimpleNamespace()
    if has_quant_group_name:
        quant_method.moe_all_to_all_group_name = "old-MC2-group"
    module = SimpleNamespace(
        routed_experts=SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_method)),
        moe_config=SimpleNamespace(ep_size=4),
    )

    comm_module = "vllm_ascend.ops.fused_moe.moe_comm_method"
    monkeypatch.setattr(f"{comm_module}._MoECommMethods", {})
    monkeypatch.setattr(f"{comm_module}.AlltoAllCommImpl", lambda _: initialize_group("EP"))
    monkeypatch.setattr(f"{comm_module}.AllGatherCommImpl", MagicMock())
    monkeypatch.setattr(f"{comm_module}.MC2CommImpl", lambda _: initialize_group("MC2"))
    monkeypatch.setattr(f"{comm_module}.FusedMC2CommImpl", lambda _: initialize_group("MC2"))
    monkeypatch.setattr("vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group", lambda: mc2_group)

    setup_moe_comm_and_quant_method(module)

    assert initialized_groups == ["EP", "MC2"]
    if has_quant_group_name:
        assert quant_method.moe_all_to_all_group_name == "new-MC2-group"


def test_target_ep_is_ready_before_new_rank_moe_setup(monkeypatch):
    """New-rank AlltoAll setup must not need old ranks already in update_ctx."""
    executor, _ = _executor_and_worker()
    materialized = set()
    phase = "warmup"
    ep_device_group = MagicMock()

    def get_ep_name(rank):
        assert rank == 3
        if "EP" not in materialized:
            assert phase == "warmup", "New rank waits on EP while old ranks wait on MC2 update_ctx"
            materialized.add("EP")
        return "target-ep"

    ep_device_group._get_backend.return_value.get_hccl_comm_name.side_effect = get_ep_name
    ep_group = SimpleNamespace(device_group=ep_device_group, cpu_group=object(), rank_in_group=3, world_size=4)
    module_name = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    monkeypatch.setattr(f"{module_name}.torch.distributed.barrier", MagicMock())
    monkeypatch.setattr(f"{module_name}.torch.npu.synchronize", MagicMock())
    # Passing no DP group also checks we do not initialize an unused DP device group.
    executor._warm_target_groups(None, ep_group)
    phase = "old_ranks_in_update_ctx"

    comm_module = "vllm_ascend.ops.fused_moe.moe_comm_method"
    monkeypatch.setattr(f"{comm_module}._MoECommMethods", {})
    monkeypatch.setattr(f"{comm_module}.AlltoAllCommImpl", lambda _: get_ep_name(3))
    monkeypatch.setattr(f"{comm_module}.AllGatherCommImpl", MagicMock())
    monkeypatch.setattr(f"{comm_module}.MC2CommImpl", MagicMock())
    monkeypatch.setattr(f"{comm_module}.FusedMC2CommImpl", MagicMock())
    module = SimpleNamespace(
        routed_experts=SimpleNamespace(quant_method=SimpleNamespace(quant_method=None)),
        moe_config=SimpleNamespace(ep_size=4),
    )
    setup_moe_comm_and_quant_method(module)
    assert materialized == {"EP"}


def test_match_peer_parameters_is_deterministic_on_mismatch():
    parameters = [object(), object(), object()]

    sender = _match_peer_parameters(
        ["z", "a", "sender_only"],
        parameters,
        ["receiver_only", "a", "z"],
    )
    receiver_parameters = [object(), object(), object()]
    receiver = _match_peer_parameters(
        ["receiver_only", "a", "z"],
        receiver_parameters,
        ["z", "a", "sender_only"],
    )

    assert sender == [parameters[1], parameters[0]]
    assert receiver == [receiver_parameters[1], receiver_parameters[2]]


def _mega_restore_executor(monkeypatch, dead_ranks=(2,)):
    executor, worker = _executor_and_worker()
    parallel = worker.vllm_config.parallel_config
    parallel.world_size = 1
    parallel.data_parallel_size = 4
    parallel.enable_fault_tolerance = True
    comm = SimpleNamespace(
        mega_moe_capture_state=make_capture_state(),
        mega_moe_quant_type=QuantType.W8A8,
        token_dispatcher=SimpleNamespace(refresh_hccl_group=MagicMock()),
    )
    module = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    moe_config = SimpleNamespace(num_experts=128, num_local_experts=64)
    moe = SimpleNamespace(moe_config=moe_config)
    worker.get_model = lambda: SimpleNamespace(modules=lambda: [moe])
    monkeypatch.setattr(f"{module}.is_moe_layer", lambda candidate: candidate is moe)
    monkeypatch.setattr(f"{module}.get_moe_comm_method", lambda kind, config: comm if config is moe_config else None)
    monkeypatch.setattr(f"{module}.get_dp_group", lambda: SimpleNamespace(world_size=4, dead_dp_ranks=set(dead_ranks)))
    monkeypatch.setattr(f"{module}.get_mc2_group", lambda: SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]))
    return executor, worker, comm


@pytest.mark.parametrize("rank_order", [[0, 1, 2, 3], [0, 1, 3, 2]])
def test_mega_middle_restore_exchanges_backend_and_preserves_mc2_slots(monkeypatch, rank_order):
    executor, _, _ = _mega_restore_executor(monkeypatch)
    monkeypatch.setattr(
        "vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group",
        lambda: SimpleNamespace(world_size=4, ranks=rank_order),
    )
    values = {}
    store = SimpleNamespace(set=values.__setitem__, get=values.__getitem__)
    monkeypatch.setattr(executor, "_capture_store", lambda request: store)
    request = SimpleNamespace(new_data_parallel_size=4, operation_id="mega-restore")
    assert executor._publish_capture_decision(request, old_dp_size=3)
    assert executor._mega_moe_precommit_capture
    assert executor._capture_mc2_rank_order == rank_order
    new_executor, _ = _executor_and_worker()
    monkeypatch.setattr(new_executor, "_capture_store", lambda request: store)
    assert new_executor._read_capture_decision(request)
    assert new_executor._mega_moe_precommit_capture
    assert new_executor._capture_old_dp_size == 3
    assert new_executor._capture_mc2_rank_order == rank_order


@pytest.mark.parametrize("existing", [True, False])
def test_mega_commit_preserves_graphs_and_connects_fault_manager(monkeypatch, existing):
    executor, worker, comm = _mega_restore_executor(monkeypatch)
    state = comm.mega_moe_capture_state
    if not existing:
        state.buffer.rank_id = 2
        state.prepare_capture(topk=4)
    else:
        state.buffer.update_mask_buffer(2, True)
    addresses = (state.buffer.context.data_ptr(), state.buffer.mask_buffer.data_ptr(), state.rank_map.data_ptr())
    executor._mega_moe_precommit_capture_done = True
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    group = SimpleNamespace(world_size=4, ranks=[0, 1, 3, 2])
    module = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    monkeypatch.setattr(f"{module}.get_mc2_group", lambda: group)
    monkeypatch.setattr(f"{module}.get_ep_group", lambda: SimpleNamespace(ranks=[0, 1, 2, 3]))
    monkeypatch.setattr(f"{module}.get_ep_all2all_manager", lambda: manager)
    calls = []
    for name in (
        "_resume_async_eplb_from_bootstrap",
        "_set_eplb_suppressed",
        "_start_group_cleanup",
        "_synchronize_capture_context_rendezvous",
    ):
        monkeypatch.setattr(executor, name, MagicMock(side_effect=lambda *args, n=name: calls.append(n)))
    monkeypatch.setattr(executor, "_switch_and_prepare_capture_restore_scale_up", MagicMock(return_value=(object(),)))
    monkeypatch.setattr(executor, "_activate_ascend_standby_groups", MagicMock(return_value=object()))
    monkeypatch.setattr(executor, "_release_cuda_graphs", MagicMock(side_effect=AssertionError("old graphs released")))
    monkeypatch.setattr(executor, "warm_and_capture", MagicMock(side_effect=AssertionError("old graphs recaptured")))
    worker.worker_sentinel = SimpleNamespace(init_num_local_experts=MagicMock())
    executor._commit_mega_moe_restore(existing)
    assert not state.capture_active
    assert not state.buffer.mask_buffer.any()
    assert state.rank_map.tolist() == [0, 1, 3, 2]
    assert addresses == (
        state.buffer.context.data_ptr(),
        state.buffer.mask_buffer.data_ptr(),
        state.rank_map.data_ptr(),
    )
    assert calls.index("_synchronize_capture_context_rendezvous") < calls.index("_resume_async_eplb_from_bootstrap")
    manager.update_mask(3)
    assert state.buffer.mask_buffer.tolist() == [0, 0, 1, 0]
    assert manager.query_active_mask().tolist() == [0, 0, 0, 1]
    assert executor._switch_and_prepare_capture_restore_scale_up.call_count == int(existing)


@pytest.mark.parametrize("is_async", [False, True])
def test_mega_resume_preserves_eplb_execution_mode(monkeypatch, is_async):
    executor, worker, _ = _mega_restore_executor(monkeypatch)
    executor._mega_moe_precommit_capture = True
    state = SimpleNamespace(is_async=is_async, expert_rearrangement_step=50, start_async_loop=MagicMock())
    worker.model_runner = SimpleNamespace(eplb_state=state)
    executor._resume_async_eplb_from_bootstrap()
    assert state.is_async is is_async
    assert state.expert_rearrangement_step == 0
    assert state.start_async_loop.call_count == int(is_async)


@pytest.mark.parametrize("fast_capture,fail", [(False, False), (True, False), (True, True)])
def test_new_rank_capture_skips_dp_only_within_warmup(monkeypatch, fast_capture, fail):
    from vllm.v1.worker.dp_utils import should_skip_dp_coordination

    executor, _, comm = _mega_restore_executor(monkeypatch)
    executor._mega_moe_precommit_capture = fast_capture
    comm.mega_moe_capture_state.prepare_capture(topk=4)
    executor._mega_moe_precommit_capture_done = False
    calls = []

    def warmup():
        assert should_skip_dp_coordination() is fast_capture
        calls.append("warmup")

    def capture():
        assert should_skip_dp_coordination()
        assert comm.mega_moe_capture_state.buffer.mask_buffer.tolist() == [0, 1, 1, 1]
        calls.append("capture")
        if fail:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(executor, "warmup_local_kernels", warmup)
    monkeypatch.setattr(executor, "warm_and_capture", capture)
    with pytest.raises(RuntimeError, match="capture failed") if fail else nullcontext():
        executor.warmup_new_worker()
    assert calls == (["warmup", "capture"] if fast_capture else ["warmup"])
    assert not should_skip_dp_coordination()
    assert executor._mega_moe_precommit_capture_done is (fast_capture and not fail)
    # Restore remote peers only at coordinated commit, never when capture ends.
    assert comm.mega_moe_capture_state.buffer.mask_buffer.tolist() == [0, 1, 1, 1]
