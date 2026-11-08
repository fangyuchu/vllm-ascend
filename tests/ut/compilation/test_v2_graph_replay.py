# SPDX-License-Identifier: Apache-2.0
"""Graph replay must submit updates without first waiting for replay."""

from contextlib import nullcontext
from types import SimpleNamespace

from vllm_ascend.worker.v2 import aclgraph_utils as graphs


def test_replay_submits_graph_updates_without_synchronizing(monkeypatch):
    calls = []
    manager = object.__new__(graphs.ModelAclGraphManager)
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
    assert calls == expected
    calls.clear()
    assert manager.run_fullgraph(desc) is result
    assert calls == expected
