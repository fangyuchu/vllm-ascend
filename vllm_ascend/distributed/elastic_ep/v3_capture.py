# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Operation-scoped DP synchronization for concurrent V3 graph capture."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.v1.worker.gpu.dp_utils import override_dp_sync_group

_CAPTURE_SESSION: ContextVar[V3CaptureDPSyncSession | None] = ContextVar(
    "ascend_v3_capture_session",
    default=None,
)


@dataclass
class V3CaptureDPSyncSession:
    """Synchronize new-rank capture collectives with serving companions.

    Store keys are namespaced by the external reconfiguration operation ID, so
    repeated scale cycles cannot consume stale markers from an earlier cycle.
    """

    group: StatelessGroupCoordinator
    operation_id: str
    old_dp_size: int
    timeout_s: float = 600.0
    sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("V3 capture DP sync requires a non-empty operation_id")
        if not 0 < self.old_dp_size < self.group.world_size:
            raise ValueError(
                "V3 capture DP sync requires a scale-up topology: "
                f"old_dp_size={self.old_dp_size}, "
                f"new_dp_size={self.group.world_size}"
            )

    @property
    def _store(self):
        return self.group.tcp_store_group.store

    @property
    def _prefix(self) -> str:
        return f"v3_capture/{self.operation_id}"

    def _step_key(self, sequence: int) -> str:
        return f"{self._prefix}/step/{sequence}"

    def _done_key(self, rank: int) -> str:
        return f"{self._prefix}/done/{rank}"

    @property
    def _error_key(self) -> str:
        return f"{self._prefix}/error"

    @property
    def new_ranks(self) -> range:
        return range(self.old_dp_size, self.group.world_size)

    def notify_all_reduce(self) -> None:
        """Publish the next collective before the new rank enters it."""
        self._store.set(self._step_key(self.sequence), b"1")
        self.sequence += 1

    @contextmanager
    def activate_for_capture(self) -> Iterator[None]:
        """Route DP metadata collectives through the target topology."""
        token = _CAPTURE_SESSION.set(self)
        try:
            with override_dp_sync_group(
                self.group.cpu_group,
                self.notify_all_reduce,
            ):
                yield
        finally:
            _CAPTURE_SESSION.reset(token)

    def mark_capture_done(self) -> None:
        if self.group.rank_in_group not in self.new_ranks:
            raise RuntimeError("Only a new rank can mark V3 capture complete")
        self._store.set(
            self._done_key(self.group.rank_in_group),
            str(self.sequence).encode(),
        )

    def mark_capture_failed(self, error: BaseException) -> None:
        self._store.set(self._error_key, repr(error).encode())

    def _all_new_ranks_done(self) -> bool:
        return self._store.check([self._done_key(rank) for rank in self.new_ranks])

    def run_existing_rank_companion(self) -> int:
        """Match every DP metadata all-reduce issued during new-rank capture."""
        if self.group.rank_in_group >= self.old_dp_size:
            raise RuntimeError("Only an existing rank can run a capture companion")

        tensor = torch.zeros(
            4,
            self.group.world_size,
            dtype=torch.int32,
            device="cpu",
        )
        start = time.monotonic()
        while True:
            if self._store.check([self._error_key]):
                error = self._store.get(self._error_key).decode()
                raise RuntimeError(f"New-rank V3 graph capture failed: {error}")
            step_key = self._step_key(self.sequence)
            if self._store.check([step_key]):
                dist.all_reduce(tensor, group=self.group.cpu_group)
                tensor.zero_()
                self.sequence += 1
                continue

            if self._all_new_ranks_done():
                step_counts = {int(self._store.get(self._done_key(rank)).decode()) for rank in self.new_ranks}
                if step_counts != {self.sequence}:
                    raise RuntimeError(
                        "V3 capture ranks completed with inconsistent DP "
                        f"collective counts: companion={self.sequence}, "
                        f"new_ranks={sorted(step_counts)}"
                    )
                return self.sequence

            if time.monotonic() - start > self.timeout_s:
                raise TimeoutError(
                    "Timed out waiting for V3 capture DP collective "
                    f"operation={self.operation_id}, sequence={self.sequence}"
                )
            time.sleep(0.02)


def get_v3_capture_session() -> V3CaptureDPSyncSession | None:
    return _CAPTURE_SESSION.get()
