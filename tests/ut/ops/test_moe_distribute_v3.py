# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoETokenDispatchInput,
)
from vllm_ascend.ops.fused_moe.moe_distribute_v3 import (
    MoeDistributeV3Adapter,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2


class _FakeMoeDistributeBuffer:
    get_low_latency_ccl_buffer_size = MagicMock(return_value=4096)
    instances = []

    def __init__(self, group, *, ccl_buffer_size, comm_alg):
        self.group = group
        self.ccl_buffer_size = SimpleNamespace(value=ccl_buffer_size)
        self.comm_alg = comm_alg
        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.instances.append(self)

    def npu_low_latency_dispatch(self, **kwargs):
        self.dispatch_kwargs = kwargs
        hidden_states = kwargs["x"]
        num_tokens = hidden_states.shape[0]
        return (
            hidden_states + 1,
            torch.ones(num_tokens),
            torch.arange(num_tokens, dtype=torch.int32),
            torch.ones(kwargs["num_experts"], dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            None,
        )

    def npu_low_latency_combine(self, **kwargs):
        self.combine_kwargs = kwargs
        return kwargs["ori_x"] + 2

    def update_ctx(self, group):
        self.group = group


@pytest.fixture(autouse=True)
def _reset_fake_buffers():
    _FakeMoeDistributeBuffer.instances.clear()
    _FakeMoeDistributeBuffer.get_low_latency_ccl_buffer_size.reset_mock()


def _token_dispatch_input() -> MoETokenDispatchInput:
    return MoETokenDispatchInput(
        hidden_states=torch.randn(4, 8),
        topk_weights=torch.rand(4, 2),
        topk_ids=torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            dtype=torch.int64,
        ),
        routing=MoeRouterInput(
            expert_map=torch.arange(4),
            global_redundant_expert_num=0,
            mc2_mask=torch.ones(4, dtype=torch.bool),
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(),
    )


def test_v3_dispatch_and_combine_forward_operator_arguments():
    device_group = object()
    mc2_group = SimpleNamespace(
        device_group=device_group,
        rank_in_group=0,
        world_size=2,
    )
    elastic_info = torch.tensor([0, 2, 0, 4, 0, 1, 0, 1], dtype=torch.int32)
    adapter = MoeDistributeV3Adapter(max_tokens_per_rank=16)
    token_input = _token_dispatch_input()

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_distribute_v3.get_mc2_group",
            return_value=mc2_group,
        ),
        patch.object(
            MoeDistributeV3Adapter,
            "_load_buffer_cls",
            return_value=_FakeMoeDistributeBuffer,
        ),
        patch.object(
            MoeDistributeV3Adapter,
            "_register_stateless_group_rank",
            return_value=nullcontext(),
        ),
        patch.object(MoeDistributeV3Adapter, "_is_graph_mode", return_value=True),
        patch(
            "vllm_ascend.ops.fused_moe.moe_distribute_v3.get_v3_elastic_info",
            return_value=elastic_info,
        ),
    ):
        dispatched = adapter.dispatch(token_input, moe_expert_num=4, quant_mode=0)
        combined = adapter.combine(
            dispatched.hidden_states,
            dispatched.combine_metadata,
            moe_expert_num=4,
        )

    buffer = _FakeMoeDistributeBuffer.instances[0]
    assert len(_FakeMoeDistributeBuffer.instances) == 1
    assert buffer.dispatch_kwargs["topk_idx"].dtype == torch.int32
    assert buffer.dispatch_kwargs["elastic_info"] is elastic_info
    assert buffer.dispatch_kwargs["x_active_mask"] is token_input.routing.mc2_mask
    assert dispatched.combine_metadata.ori_x is token_input.hidden_states
    assert buffer.combine_kwargs["ori_x"] is token_input.hidden_states
    torch.testing.assert_close(combined, token_input.hidden_states + 2)


def test_v3_rejects_unsupported_communication_quant_mode():
    adapter = MoeDistributeV3Adapter(max_tokens_per_rank=16)
    with pytest.raises(RuntimeError, match="quant_mode 0 or 2"):
        adapter.dispatch(_token_dispatch_input(), moe_expert_num=4, quant_mode=4)


def test_v3_updates_buffer_context_when_elastic_info_changes():
    device_group = object()
    mc2_group = SimpleNamespace(
        device_group=device_group,
        rank_in_group=0,
        world_size=2,
    )
    elastic_info = torch.tensor(
        [0, 2, 0, 4, 0, 1, 0, 1],
        dtype=torch.int32,
    )
    adapter = MoeDistributeV3Adapter(max_tokens_per_rank=16)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.moe_distribute_v3.get_mc2_group",
            return_value=mc2_group,
        ),
        patch.object(
            MoeDistributeV3Adapter,
            "_load_buffer_cls",
            return_value=_FakeMoeDistributeBuffer,
        ),
        patch.object(
            MoeDistributeV3Adapter,
            "_register_stateless_group_rank",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.moe_distribute_v3.get_v3_elastic_info",
            return_value=elastic_info,
        ),
    ):
        adapter.prepare_for_shape(
            hidden_size=8,
            topk=2,
            moe_expert_num=4,
            dtype=torch.float32,
            device="cpu",
        )
        elastic_info[0] = 1
        assert adapter.update_ctx_to_mc2_group(mc2_group)
        target_group = SimpleNamespace(device_group=object(), rank_in_group=0, world_size=2)
        assert adapter.update_ctx_to_mc2_group(target_group)
        # Serving still sees the old global group before the commit switch.
        adapter.prepare_for_shape(
            hidden_size=8,
            topk=2,
            moe_expert_num=4,
            dtype=torch.float32,
            device="cpu",
        )
        with pytest.raises(RuntimeError, match="captured ep_rank_id"):
            adapter.update_ctx_to_mc2_group(SimpleNamespace(device_group=object(), rank_in_group=1, world_size=2))

    assert len(_FakeMoeDistributeBuffer.instances) == 1
    assert _FakeMoeDistributeBuffer.instances[0].group is target_group.device_group


def test_v3_captured_dummy_routing_stops_even_with_permuted_or_faulted_topology():
    adapter = MoeDistributeV3Adapter(max_tokens_per_rank=16)
    info = torch.tensor([1, 1, 0, 2, -1, 0, 1, -1], dtype=torch.int32)
    topk = torch.tensor([[3, 2], [2, 3]], dtype=torch.int32)
    with patch(
        "vllm_ascend.distributed.elastic_ep.v3_capture.get_v3_capture_session",
        return_value=object(),
    ):
        adapter._remap_topk_ids_for_capture(topk, info)
        graph = torch.jit.trace(
            lambda ids: adapter._remap_topk_ids_for_capture(ids, info),
            topk,
            check_trace=False,
        )
    assert graph(topk).tolist() == [[0, 1], [1, 0]]
    adapter.finish_capture()
    # Rank translation remains enabled after commit and after subsequent FT.
    assert info[0] == 1
    torch.testing.assert_close(graph(topk), topk)
    info[3] = 1
    torch.testing.assert_close(graph(topk), topk)


def test_dispatcher_keeps_captured_physical_expert_capacity():
    dispatcher = object.__new__(TokenDispatcherWithMC2)
    dispatcher.v3_adapter = MagicMock()
    dispatcher._initial_v3_moe_expert_num = None
    dispatcher._ensure_v3_identity_elastic_info = MagicMock()
    dispatcher._get_dispatch_quant_mode = MagicMock(return_value=0)
    token_input = _token_dispatch_input()

    TokenDispatcherWithMC2.token_dispatch(dispatcher, token_input)
    smaller_input = replace(
        token_input,
        routing=replace(token_input.routing, expert_map=torch.arange(3)),
    )
    TokenDispatcherWithMC2.token_dispatch(dispatcher, smaller_input)

    assert dispatcher.v3_adapter.dispatch.call_args_list[0].args[1] == 4
    assert dispatcher.v3_adapter.dispatch.call_args_list[1].args[1] == 4


@pytest.mark.parametrize("created_in_inference_mode", [False, True])
def test_ft_elastic_info_tracks_live_capacity_without_reallocating(created_in_inference_mode):
    from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager

    with torch.inference_mode(created_in_inference_mode):
        manager = _NpuAll2AllManager(4, torch.device("cpu"))
        info = manager.get_elastic_info()
    address = info.data_ptr()
    assert info.is_inference() == created_in_inference_mode
    manager.set_num_local_physical_experts(4)
    assert info[:4].tolist() == [0, 4, 0, 16]
    manager.update_mask(1)
    assert info.tolist() == [1, 3, 0, 12, 0, -1, 1, 2, 0, 2, 3, -1]
    manager.update_mask(2)
    assert manager.get_elastic_info().data_ptr() == address
    assert info.tolist() == [1, 2, 0, 8, 0, -1, -1, 1, 0, 3, -1, -1]


def test_v3_initialization_uses_ft_local_expert_capacity():
    from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager

    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    dispatcher = object.__new__(TokenDispatcherWithMC2)
    dispatcher._ft_enabled = True
    dispatcher.ep_world_size = 4
    with (
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_v3_elastic_info", return_value=None),
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_all2all_manager", return_value=manager),
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.set_v3_elastic_info_from_ep") as publish,
    ):
        dispatcher._ensure_v3_identity_elastic_info(256, torch.device("cpu"))
    assert publish.call_args.args[0] is manager.get_elastic_info()
    manager.update_mask(2)
    assert manager.get_elastic_info()[:4].tolist() == [1, 3, 0, 192]
