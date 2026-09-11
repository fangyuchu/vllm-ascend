from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

from vllm_ascend.distributed.elastic_ep.elastic_execute import (
    AscendElasticEPScalingExecutor,
    _match_peer_parameters,
    setup_moe_comm_and_quant_method,
)
from vllm_ascend.distributed.elastic_ep.standby_state import _mc2_group_ranks


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
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=parallel_config,
            lora_config=None,
        ),
    )
    executor.worker_ref = lambda: worker
    return executor, worker


def _v3_mapping_executor(monkeypatch, width, local_experts, dead_ranks):
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
    moe = SimpleNamespace(moe_config=SimpleNamespace(num_local_experts=local_experts))
    worker.get_model = lambda: SimpleNamespace(modules=lambda: [moe])
    worker.device = torch.device("cpu")
    executor._v3_precommit_capture = True
    executor._v3_old_dp_size = 4 - len(dead_ranks)
    executor.reconfig_request = SimpleNamespace(new_data_parallel_size=executor._v3_old_dp_size + 1)
    module = "vllm_ascend.distributed.elastic_ep.elastic_execute"
    monkeypatch.setattr(f"{module}.is_moe_layer", lambda layer: layer is moe)
    monkeypatch.setattr(f"{module}.get_dp_group", lambda: SimpleNamespace(world_size=4, dead_dp_ranks=dead_ranks))
    monkeypatch.setattr(f"{module}.get_standby_dp_group", lambda: object())
    broadcast = MagicMock()
    monkeypatch.setattr(f"{module}.upstream_elastic_execute.broadcast_expert_mapping", broadcast)
    return executor, model_state, broadcast


@pytest.mark.parametrize(
    "width,dead_ranks,expected",
    [
        (8, {2}, [0, 1, 2, 3, 6, 7, 0, 1]),
        (16, {2}, list(range(8)) + list(range(12, 16)) + list(range(4))),
        (8, {1, 3}, [0, 1, 4, 5, 0, 1]),
        (8, set(), list(range(8)) + [0, 1]),
    ],
)
def test_v3_broadcast_preserves_survivors_and_serving_mapping(monkeypatch, width, dead_ranks, expected):
    executor, state, broadcast = _v3_mapping_executor(monkeypatch, width, 2, dead_ranks)
    original = state.physical_to_logical_map
    snapshot = original.clone()

    executor.broadcast_expert_mapping()

    sent = broadcast.call_args.kwargs["physical_to_logical"]
    torch.testing.assert_close(sent, torch.tensor([expected]))
    assert state.physical_to_logical_map is original
    torch.testing.assert_close(original, snapshot)


@pytest.mark.parametrize(
    "width,local_experts,dead_ranks,error",
    [
        (8, 0, {2}, "must be positive"),
        (8, -1, {2}, "must be positive"),
        (7, 2, {2}, "not aligned to local expert capacity"),
        (12, 2, {2}, "cannot be partitioned evenly"),
        (12, 2, set(), "cannot be partitioned evenly"),
        (8, 2, {4}, "invalid dead DP ranks"),
    ],
)
def test_v3_broadcast_rejects_invalid_capacity_before_transfer(monkeypatch, width, local_experts, dead_ranks, error):
    executor, _, broadcast = _v3_mapping_executor(monkeypatch, width, local_experts, dead_ranks)
    with pytest.raises((ValueError, RuntimeError), match=error):
        executor.broadcast_expert_mapping()
    broadcast.assert_not_called()


def test_prepare_reconfiguration_adds_ascend_standby_group():
    executor, _ = _executor_and_worker()
    request = SimpleNamespace(
        new_data_parallel_size=3,
        new_data_parallel_master_ip="127.0.0.1",
        coord_store_port=1234,
        operation_id="scale-1",
    )

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "prepare_reconfiguration",
            side_effect=lambda *_: setattr(executor, "_target_global_rank", 2),
        ) as upstream_prepare,
        patch.object(
            executor,
            "_publish_v3_capture_decision",
            return_value=False,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_dp_group",
            return_value=SimpleNamespace(world_size=2),
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.create_ascend_standby_groups"
        ) as create_ascend_groups,
    ):
        executor.prepare_reconfiguration(request, use_all2all=True)

    upstream_prepare.assert_called_once_with(request, True)
    create_ascend_groups.assert_called_once_with(
        new_dp_size=3,
        new_world_size_across_dp=6,
        master_ip="127.0.0.1",
        coord_store_port=1234,
        create_v3_capture_dp=False,
        new_global_rank=2,
    )


def test_receive_expert_mapping_preserves_new_upstream_return_type():
    executor, _ = _executor_and_worker()
    mapping = torch.tensor([[0, 1]])
    executor._setup_moe_comm_and_quant_method = MagicMock()

    with patch.object(
        ElasticEPScalingExecutor,
        "receive_expert_mapping",
        return_value=mapping,
    ):
        result = executor.receive_expert_mapping()

    assert result is mapping
    executor._setup_moe_comm_and_quant_method.assert_called_once_with()


def test_prepare_new_worker_uses_ascend_weight_transfer():
    executor, _ = _executor_and_worker()
    events = []
    request = SimpleNamespace(operation_id="scale-2")

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "prepare_new_worker",
            side_effect=lambda _: events.append("upstream"),
        ) as upstream_prepare,
        patch.object(
            executor,
            "_use_ascend_transfer_impl",
            return_value=nullcontext(),
        ) as use_ascend_transfer,
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.create_ascend_standby_groups",
            side_effect=lambda **_: events.append("ascend_mc2"),
        ) as create_ascend_groups,
        patch.object(
            executor,
            "_read_v3_capture_decision",
            return_value=False,
        ),
    ):
        result = executor.prepare_new_worker(request)

    use_ascend_transfer.assert_called_once_with(include_expert_weights=False)
    upstream_prepare.assert_called_once_with(request)
    create_ascend_groups.assert_called_once_with(
        new_dp_size=2,
        new_world_size_across_dp=4,
        master_ip="127.0.0.1",
        coord_store_port=1234,
        create_v3_capture_dp=False,
    )
    assert events == ["upstream", "ascend_mc2"]
    assert result == "scale-2"


def test_new_worker_activates_ascend_mc2_before_commit():
    executor, _ = _executor_and_worker()
    events = []

    with (
        patch.object(
            executor,
            "_activate_ascend_standby_groups",
            side_effect=lambda: events.append("activate_mc2"),
        ) as activate_groups,
        patch.object(
            ElasticEPScalingExecutor,
            "commit_scale_up",
            side_effect=lambda _: events.append("upstream_commit"),
        ) as upstream_commit,
    ):
        executor.commit_scale_up(is_existing_worker=False)

    activate_groups.assert_called_once_with()
    upstream_commit.assert_called_once_with(False)
    assert events == ["activate_mc2", "upstream_commit"]


def test_new_worker_recovers_external_operation_id_from_store():
    executor, _ = _executor_and_worker()
    store = MagicMock()
    store.check.return_value = True
    store.get.return_value = b"scale-3"

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "prepare_new_worker",
        ),
        patch.object(
            executor,
            "_use_ascend_transfer_impl",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_cached_tcp_store_client",
            return_value=store,
        ),
        patch.object(
            executor,
            "_read_v3_capture_decision",
            return_value=False,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.create_ascend_standby_groups",
        ),
    ):
        result = executor.prepare_new_worker()

    assert result == "scale-3"
    assert executor.reconfig_request.operation_id == "scale-3"
    store.check.assert_called_once_with(["elastic_ep/external/current_epoch"])


def test_switch_and_prepare_retires_previous_mc2_group():
    executor, worker = _executor_and_worker()
    worker.model_runner = MagicMock()
    worker.model_runner.model.modules.return_value = []
    retired_groups = (MagicMock(),)
    standby_mc2 = MagicMock()
    executor._setup_moe_comm_and_quant_method = MagicMock()

    with (
        patch.object(
            ElasticEPScalingExecutor,
            "switch_and_prepare",
            return_value=retired_groups,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.pop_ascend_standby_groups",
            return_value={"mc2": standby_mc2},
        ),
        patch("vllm_ascend.distributed.elastic_ep.elastic_execute._replace_ascend_active_groups") as replace_groups,
    ):
        result = executor.switch_and_prepare()

    assert result == (*retired_groups, replace_groups.return_value)
    replace_groups.assert_called_once_with(mc2=standby_mc2)


def test_setup_moe_comm_refreshes_deferred_quant_group_name():
    backend = MagicMock()
    backend.get_hccl_comm_name.return_value = "new-mc2-group"
    device_group = MagicMock()
    device_group._get_backend.return_value = backend
    mc2_group = SimpleNamespace(device_group=device_group, rank_in_group=3)
    quant_method = SimpleNamespace(moe_all_to_all_group_name="")
    module = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=quant_method),
        ),
        moe_config=MagicMock(),
    )

    with (
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group",
            return_value=mc2_group,
        ),
        patch("vllm_ascend.distributed.elastic_ep.elastic_execute.setup_moe_comm_method") as setup_moe_comm_method,
    ):
        setup_moe_comm_and_quant_method(module)

    assert quant_method.moe_all_to_all_group_name == "new-mc2-group"
    backend.get_hccl_comm_name.assert_called_once_with(3)
    setup_moe_comm_method.assert_called_once_with(module.moe_config)


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


def test_mc2_standby_ranks_match_dp_ep_layout():
    assert _mc2_group_ranks(
        world_size=4,
        dp_size=2,
        pp_size=2,
        pcp_size=1,
        tp_size=1,
    ) == [[0, 2], [1, 3]]


def test_v3_scale_down_uses_graph_preserving_switch():
    executor, _ = _executor_and_worker()
    executor.perform_scale_down_eplb_reshuffle = MagicMock()
    executor._switch_v3_scale_down_survivor = MagicMock()

    with patch.object(
        executor,
        "_can_preserve_v3_scale_down",
        return_value=True,
    ):
        executor.commit_scale_down(new_dp_size=1, removing=False)

    executor.perform_scale_down_eplb_reshuffle.assert_called_once_with(1)
    executor._switch_v3_scale_down_survivor.assert_called_once_with(1)


def test_non_v3_scale_down_keeps_upstream_behavior():
    executor, _ = _executor_and_worker()
    with (
        patch.object(
            executor,
            "_can_preserve_v3_scale_down",
            return_value=False,
        ),
        patch.object(
            ElasticEPScalingExecutor,
            "commit_scale_down",
        ) as upstream_commit,
    ):
        executor.commit_scale_down(new_dp_size=1, removing=True)

    upstream_commit.assert_called_once_with(1, True)


def test_v3_graph_preserving_scale_down_allows_async_eplb():
    executor, worker = _executor_and_worker()
    worker.model_runner = SimpleNamespace(
        eplb_state=SimpleNamespace(is_async=True),
    )

    with (
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3",
            True,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group",
            return_value=SimpleNamespace(world_size=2),
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_v3_elastic_info",
            return_value=torch.zeros(8, dtype=torch.int32),
        ),
        patch.object(executor, "_current_v3_dispatchers", return_value=[object()]),
    ):
        assert executor._can_preserve_v3_scale_down(new_dp_size=1)


@pytest.mark.parametrize(
    "physical,active,width,expected",
    [
        ([0, 1, 2, 3], [0, 1, 3], 1, [0, 1, 3, 2]),
        ([0, 1, 2, 3], [0, 1, 2], 1, [0, 1, 2, 3]),
        ([0, 1, 3, 2], [0, 2, 3], 1, [0, 3, 2, 1]),
        (list(range(8)), [0, 1, 3], 2, [0, 1, 2, 3, 6, 7, 4, 5]),
    ],
)
def test_v3_restore_keeps_captured_mc2_rank_ids(physical, active, width, expected):
    order = AscendElasticEPScalingExecutor._build_v3_mc2_rank_order(physical, active, width)
    assert order == expected


def test_v3_middle_rank_restore_publishes_capture_decision_and_mc2_order():
    executor, worker = _executor_and_worker()
    worker.vllm_config.parallel_config.world_size = 1
    worker.vllm_config.parallel_config.data_parallel_size = 3
    request = SimpleNamespace(new_data_parallel_size=4, operation_id="restore")
    store = MagicMock()
    values = {}
    store.set.side_effect = values.__setitem__
    store.get.side_effect = values.__getitem__
    with (
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3",
            True,
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_dp_group",
            return_value=SimpleNamespace(world_size=4, dead_dp_ranks={2}),
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_mc2_group",
            return_value=SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]),
        ),
        patch(
            "vllm_ascend.distributed.elastic_ep.elastic_execute.get_v3_elastic_info",
            return_value=torch.tensor([1, 3, 0, 192]),
        ),
        patch.object(executor, "_v3_capture_store", return_value=store),
    ):
        assert executor._is_v3_scale_up_candidate(request)
        assert executor._publish_v3_capture_decision(request, old_dp_size=3)
        assert executor._v3_mc2_rank_order == [0, 1, 3, 2]
        new_executor, _ = _executor_and_worker()
        with patch.object(new_executor, "_v3_capture_store", return_value=store):
            assert new_executor._read_v3_capture_decision(request)
            assert new_executor._v3_mc2_rank_order == [0, 1, 3, 2]
