# SPDX-License-Identifier: Apache-2.0
"""Recovery diagnostics must not wait for replay before submitting updates."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.worker.v2 import aclgraph_utils as graphs


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_recovery_sync_follows_graph_updates_and_only_runs_once(monkeypatch, mode):
    calls = []
    manager = object.__new__(graphs.ModelAclGraphManager)
    manager.ft_replay_debug = mode
    manager.vllm_config = None
    manager.model_runner = SimpleNamespace(attn_groups=[], model_state=SimpleNamespace(attn_metadata={}))
    manager.update_stream = SimpleNamespace(
        wait_stream=lambda _: calls.append("wait"),
        synchronize=lambda: calls.append("update_sync"),
    )
    stream = SimpleNamespace(synchronize=lambda: calls.append("compute_sync"))
    monkeypatch.setattr(graphs, "UpdatableGraph", SimpleNamespace)
    graph = SimpleNamespace(
        tasks=[],
        resolve_tasks=lambda _: calls.append("resolve") or (),
        update=lambda *args, **kwargs: calls.append("update"),
    )
    desc = graphs.BatchExecutionDescriptor(cg_mode=graphs.CUDAGraphMode.FULL, num_tokens=1, num_reqs=1)
    manager.graphs = {desc: graph}
    monkeypatch.setattr(graphs, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(graphs, "_get_graph_update_backend", lambda _: object())
    monkeypatch.setattr(graphs, "use_updatable_graph", lambda _: True)
    monkeypatch.setattr(graphs.torch.npu, "current_stream", lambda: stream)
    result = object()
    monkeypatch.setattr(graphs.ModelCudaGraphManager, "run_fullgraph", lambda *_: calls.append("replay") or result)

    assert manager.run_fullgraph(desc) is result
    expected = ["resolve", "wait", "replay", "update"]
    assert calls == expected + (["update_sync", "compute_sync"] if mode == 2 else [])
    assert manager.ft_replay_debug == 0
    calls.clear()
    assert manager.run_fullgraph(desc) is result
    assert calls == expected


def test_failed_graph_update_does_not_enter_diagnostic_sync(monkeypatch):
    manager = object.__new__(graphs.ModelAclGraphManager)
    manager.ft_replay_debug = 2
    manager.vllm_config = None
    manager.model_runner = SimpleNamespace(attn_groups=[], model_state=SimpleNamespace(attn_metadata={}))
    manager.update_stream = MagicMock()
    monkeypatch.setattr(graphs, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(graphs, "_get_graph_update_backend", lambda _: object())
    monkeypatch.setattr(graphs, "use_updatable_graph", lambda _: True)
    monkeypatch.setattr(manager, "_updatable_graph_replay", MagicMock(side_effect=RuntimeError("update failed")))
    with pytest.raises(RuntimeError, match="update failed"):
        manager.run_fullgraph(SimpleNamespace(num_tokens=1))
    manager.update_stream.synchronize.assert_not_called()
    assert manager.ft_replay_debug == 0
