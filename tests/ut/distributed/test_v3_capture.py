# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.distributed.elastic_ep.v3_capture import (
    V3CaptureDPSyncSession,
)


class _Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return self.values[key]


def _group(rank: int):
    store = _Store()
    return SimpleNamespace(
        rank_in_group=rank,
        world_size=4,
        cpu_group=object(),
        tcp_store_group=SimpleNamespace(store=store),
    )


def test_capture_session_uses_operation_scoped_keys():
    group = _group(rank=2)
    session = V3CaptureDPSyncSession(group, "scale-17", old_dp_size=2)

    session.notify_all_reduce()
    session.notify_all_reduce()
    session.mark_capture_done()

    assert group.tcp_store_group.store.values == {
        "v3_capture/scale-17/step/0": b"1",
        "v3_capture/scale-17/step/1": b"1",
        "v3_capture/scale-17/done/2": b"2",
    }


def test_existing_companion_matches_steps_until_all_new_ranks_finish():
    group = _group(rank=0)
    store = group.tcp_store_group.store
    store.set("v3_capture/scale-18/step/0", b"1")
    store.set("v3_capture/scale-18/step/1", b"1")
    store.set("v3_capture/scale-18/done/2", b"2")
    store.set("v3_capture/scale-18/done/3", b"2")
    session = V3CaptureDPSyncSession(group, "scale-18", old_dp_size=2)

    with patch("vllm_ascend.distributed.elastic_ep.v3_capture.dist.all_reduce") as all_reduce:
        steps = session.run_existing_rank_companion()

    assert steps == 2
    assert all_reduce.call_count == 2
