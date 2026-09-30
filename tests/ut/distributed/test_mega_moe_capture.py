from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.test_moe_fault_tolerance import FakeSymmBuffer
from vllm_ascend.distributed.elastic_ep.mega_moe_capture import MegaMoeCaptureState, ep_to_mc2_ranks


def make_capture_state(rank_map=None):
    buffer = FakeSymmBuffer()
    buffer.num_experts = 16
    buffer.context = torch.zeros(8, dtype=torch.int32)
    buffer.clean_mask_buffer()
    buffer.rank_id = 0
    buffer.ccl_buffer_size = 512
    buffer.topo_type = 0
    buffer.rank_num_per_server = 4
    buffer.update_group = Mock(side_effect=lambda group: buffer.context.add_(1))
    return MegaMoeCaptureState(buffer, 4, rank_map or [0, 1, 2, 3])


def test_middle_rank_routing_and_capture_switch_preserve_storage():
    state = make_capture_state([0, 1, 3, 2])
    ids = torch.tensor([[0, 4, 8, 12], [3, 7, 11, 15]])
    expected = torch.tensor([[0, 4, 12, 8], [3, 7, 15, 11]], dtype=torch.int32)
    torch.testing.assert_close(state.route(ids), expected)
    state.prepare_capture([2], topk=4)
    addresses = (state.buffer.mask_buffer.data_ptr(), state.rank_map.data_ptr(), state.capture_active.data_ptr())
    captured = state.route(ids)
    assert state.buffer.mask_buffer.tolist() == [1, 1, 0, 1]
    assert ((captured >= 8) & (captured < 12)).all()
    assert all(row.unique().numel() == 4 for row in captured)
    state.finish_capture()
    torch.testing.assert_close(state.route(ids), expected)
    assert not state.buffer.mask_buffer.any()
    assert addresses == (
        state.buffer.mask_buffer.data_ptr(),
        state.rank_map.data_ptr(),
        state.capture_active.data_ptr(),
    )
    # The previously new worker becomes a survivor in another restore cycle.
    state.set_rank_map([0, 3, 2, 1])
    torch.testing.assert_close(state.route(ids)[0], torch.tensor([0, 12, 8, 4], dtype=torch.int32))
    assert state.rank_map.data_ptr() == addresses[1]
    assert not state.capture_active


def test_capture_routes_to_multiple_noncontiguous_new_slots():
    state = make_capture_state()
    state.prepare_capture([1, 3], topk=8)
    ids = state.route(torch.zeros((4, 8), dtype=torch.int32))
    assert set(ids.flatten().tolist()) == set(range(4, 8)) | set(range(12, 16))
    assert all(row.unique().numel() == 8 for row in ids)
    assert state.buffer.mask_buffer.tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize("active,topk", [([], 1), ([4], 1), ([1, 1], 1), ([1], 5)])
def test_invalid_capture_topology_does_not_change_mask(active, topk):
    state = make_capture_state()
    with pytest.raises(ValueError, match="distinct experts"):
        state.prepare_capture(active, topk)
    assert not state.buffer.mask_buffer.any()
    assert state.capture_experts is None


@pytest.mark.parametrize("order", [[0, 1, 1, 3], [0, 1, 2], [0, 1, 2, 4]])
def test_rank_map_requires_permutation(order):
    state = make_capture_state()
    with pytest.raises(ValueError, match="permutation"):
        state.set_rank_map(order)


def test_ep_to_mc2_mapping_preserves_middle_hole():
    assert ep_to_mc2_ranks([0, 1, 2, 3], [0, 1, 3, 2]) == [0, 1, 3, 2]
    with pytest.raises(ValueError, match="rank sets"):
        ep_to_mc2_ranks([0, 1, 2, 3], [0, 1, 2, 4])


@pytest.mark.parametrize("change", [None, "context", "mask", "size", "topology"])
def test_update_context_checks_captured_contract(monkeypatch, change):
    state = make_capture_state()
    buffer = state.buffer
    buffer.update_mask_buffer(2, True)
    old_context = buffer.context.data_ptr()
    group = SimpleNamespace(world_size=4, rank_in_group=0, device_group=object())
    monkeypatch.setattr(
        "vllm_ascend.distributed.elastic_ep.mega_moe_capture.register_stateless_group_rank",
        lambda *args: nullcontext(),
    )

    def update(_):
        if change == "context":
            buffer.context = buffer.context.clone()
        elif change == "mask":
            buffer.mask_buffer = buffer.mask_buffer.clone()
        elif change == "size":
            buffer.ccl_buffer_size += 1
        elif change == "topology":
            buffer.topo_type += 1
        else:
            buffer.context.add_(1)

    buffer.update_group = Mock(side_effect=update)
    if change is not None:
        with pytest.raises(RuntimeError, match="changed captured"):
            state.update_context(group)
    else:
        state.update_context(group)
        assert buffer.context.data_ptr() == old_context
        assert buffer.context.tolist() == [1] * 8
        assert buffer.mask_buffer.tolist() == [0, 0, 1, 0]
        assert not buffer.ccl.any()


@pytest.mark.parametrize("size,rank", [(5, 0), (4, 1)])
def test_update_context_rejects_capacity_or_slot_change_before_collectives(size, rank):
    state = make_capture_state()
    with pytest.raises(ValueError, match="preserve physical"):
        state.update_context(SimpleNamespace(world_size=size, rank_in_group=rank))
    state.buffer.update_group.assert_not_called()
