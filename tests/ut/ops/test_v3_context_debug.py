# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops.fused_moe import v3_context_debug as debug


def test_metadata_snapshot_never_reads_tensor_contents():
    tensor = torch.ones(8, dtype=torch.int32)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device read")):
        result = debug._tensor_snapshot(tensor, read_device=False)
    assert result["ptr"] == hex(tensor.data_ptr())
    assert "sha256" not in result


def test_context_layout_and_unknown_abi():
    tensor = torch.zeros(2052, dtype=torch.int32)
    tensor[:8] = torch.tensor([3, 4, -1, 1, 16, 2, 0, 0], dtype=torch.int32)
    snapshot = debug._tensor_snapshot(tensor, read_device=True)
    assert snapshot["kfc_layout_candidate"] == {
        "ep_rank": 3,
        "ranks_per_server": 4,
        "kfc_context_addr": "0x1ffffffff",
        "nonzero_rank_buffers": {"0": "0x200000010"},
    }
    assert "kfc_layout_candidate" not in debug._tensor_snapshot(tensor[:8], read_device=True)


def test_cached_pre_reset_snapshot_and_inplace_change(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FT_REPLAY_DEBUG", "2")
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    adapter = SimpleNamespace(
        _buffer=SimpleNamespace(context=torch.ones(8, dtype=torch.int32), group_name="test"),
        _buffer_rank=0,
        _buffer_key=(1, 4),
        _elastic_info_signature=(0, 4),
        _capture_active=None,
    )
    elastic = torch.tensor([0, 4], dtype=torch.int32)
    debug.trace_adapter_context(adapter, "baseline", elastic, read_device=True)
    baseline = adapter._ft_context_last_snapshot
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device read")):
        debug.trace_adapter_context(adapter, "before_reset", elastic, read_device=False)
    assert adapter._ft_context_last_snapshot is baseline
    adapter._buffer.context[0] = 2
    elastic[1] = 3
    debug.trace_adapter_context(adapter, "after_reset", elastic, read_device=True)
    result = adapter._ft_context_last_snapshot
    assert result["context_contents_changed"]
    assert result["elastic_info_contents_changed"]
    assert not result["context_ptr_changed"]
    assert not result["elastic_info_ptr_changed"]
    # The diagnostic did not rewrite runtime state.
    assert adapter._buffer.context[0] == 2
    assert elastic.tolist() == [0, 3]


def test_disabled_or_capturing_skips_snapshots(monkeypatch):
    spy = Mock(side_effect=AssertionError("unexpected snapshot"))
    monkeypatch.setattr(debug, "_tensor_snapshot", spy)
    monkeypatch.setenv("VLLM_ASCEND_FT_REPLAY_DEBUG", "0")
    debug.trace_adapter_context(None, "disabled", None, read_device=True)
    monkeypatch.setenv("VLLM_ASCEND_FT_REPLAY_DEBUG", "2")
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    debug.trace_adapter_context(None, "capture", None, read_device=True)
    spy.assert_not_called()
