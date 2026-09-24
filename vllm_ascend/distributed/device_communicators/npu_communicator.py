#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

logger = init_logger(__name__)


def _accelerator_synchronize() -> None:
    # torch.accelerator APIs raise on CPU-only builds; the mask manager is
    # unit-testable without a device, so guard the synchronize.
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


class _NpuAll2AllManager:
    """All2All-manager adapter for the CANN MegaMoe backend.

    Mirrors the upstream ``All2AllManagerBase`` mask API consumed by the
    Elastic EP executor. The MegaMoe symmetric buffer's device-side rank
    mask is the communication-layer expression of EP membership: captured
    ACL graphs replay against the same buffer and re-read the mask, so a
    membership change becomes a data update on a long-lived buffer instead
    of a comm-object swap (which would invalidate every captured graph).

    All mutable state lives at class level on purpose, mirroring upstream
    ``NixlEPAll2AllManager._buffer``: manager instances are created per
    process-group communicator and are replaced when Elastic EP switches
    groups, but the mask binding must outlive those switches so that
    graphs captured before the switch stay valid after it.
    """

    # Class-level shared state: survives process-group switches.
    _mega_moe_buffer: Any = None
    _ep_to_mc2: tuple[int, ...] = ()
    _ep_world_size: int = 0
    _dead: set[int] = set()
    _mask: torch.Tensor | None = None

    def __init__(self, ep_world_size: int | None = None):
        if ep_world_size is not None and ep_world_size > 0:
            _NpuAll2AllManager.bind_ep_world_size(ep_world_size)

    @classmethod
    def bind_ep_world_size(cls, ep_world_size: int) -> None:
        if cls._ep_world_size == 0:
            cls._ep_world_size = ep_world_size
        elif cls._ep_world_size != ep_world_size:
            # Coordinators created later (e.g. a smaller standby EP group
            # built for scale-down) must not shrink the bound space: mask
            # indices and the captured graphs stay in the original rank
            # space of the buffer that was bound at startup.
            logger.debug(
                "Ignoring EP world size %d (bound space stays %d).",
                ep_world_size,
                cls._ep_world_size,
            )

    @property
    def uses_mega_moe(self) -> bool:
        return _NpuAll2AllManager._mega_moe_buffer is not None

    @classmethod
    def reset_for_test(cls) -> None:
        cls._mega_moe_buffer = None
        cls._ep_to_mc2 = ()
        cls._ep_world_size = 0
        cls._dead = set()
        cls._mask = None

    @torch.inference_mode()
    def bind_mega_moe_buffer(self, buffer, ep_to_mc2: list[int]) -> None:
        """Attach MegaMoe's stable rank mask before graph capture.

        Must run during warmup, i.e. before any ACL graph is captured: a
        graph captured while ``mask_buffer`` is unbound cannot honor a
        mask that only appears after ranks are removed. The CPU ``_dead``
        set stays authoritative; binding propagates any already-recorded
        dead ranks into the device mask.
        """
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if ep_world_size == 0:
            ep_world_size = getattr(buffer, "ep_world_size", len(ep_to_mc2))
            _NpuAll2AllManager.bind_ep_world_size(ep_world_size)
        if sorted(ep_to_mc2) != list(range(ep_world_size)):
            raise ValueError(
                "Elastic EP graph reuse requires the EP and MC2 rank sets "
                f"to match (got ep_to_mc2={ep_to_mc2}, ep_world_size={ep_world_size})."
            )
        if self.uses_mega_moe:
            # Re-binding must never resurrect an already dead peer.
            return
        buffer.clean_mask_buffer()
        for rank in sorted(_NpuAll2AllManager._dead):
            buffer.update_mask_buffer(ep_to_mc2[rank], True)
        _NpuAll2AllManager._ep_to_mc2 = tuple(ep_to_mc2)
        _NpuAll2AllManager._mega_moe_buffer = buffer
        logger.info(
            "Bound MegaMoe rank-mask buffer for %d EP ranks before graph capture.",
            ep_world_size,
        )

    @torch.inference_mode()
    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Mark an EP rank dead/alive on both the CPU set and device mask."""
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < ep_world_size:
            raise ValueError(f"EP rank must be in [0, {ep_world_size}), got {rank}.")
        if self.uses_mega_moe:
            if not masked and rank in _NpuAll2AllManager._dead:
                # Recovery must stop outstanding device work before unmasking.
                self.clean_buffers()
            _NpuAll2AllManager._mega_moe_buffer.update_mask_buffer(_NpuAll2AllManager._ep_to_mc2[rank], masked)
            _accelerator_synchronize()
        if masked:
            _NpuAll2AllManager._dead.add(rank)
        else:
            _NpuAll2AllManager._dead.discard(rank)

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        # MC2/MegaMoe kernels do not detect faults themselves; faults
        # surface as aborted ops. Masking is driven by the executor.
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        """EP-order liveness mask (0 = alive, 1 = dead), stable address.

        Refreshed from the authoritative CPU ``_dead`` set on every call
        so a tensor captured once in a graph keeps observing the latest
        membership.
        """
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if ep_world_size == 0:
            return torch.zeros(1, dtype=torch.bool, device="cpu")
        if _NpuAll2AllManager._mask is None or _NpuAll2AllManager._mask.numel() < ep_world_size:
            _NpuAll2AllManager._mask = torch.zeros(ep_world_size, dtype=torch.int32)
        mask = _NpuAll2AllManager._mask[:ep_world_size]
        mask.fill_(0)
        for rank in _NpuAll2AllManager._dead:
            mask[rank] = 1
        return mask

    @torch.inference_mode()
    def clean_buffers(self) -> None:
        """Clear fused communication flags after a membership change.

        Keeps dead ranks masked; only the local control flags are reset so
        a recovery (unmask) does not observe stale pre-fault state.
        """
        if self.uses_mega_moe:
            _NpuAll2AllManager._mega_moe_buffer.get_local_buffer_tensor(torch.uint8).zero_()
            _accelerator_synchronize()

    # Elastic-EP hooks added to All2AllManagerBase upstream. NPU does not
    # have an incremental "connect ranks" primitive for the MegaMoe buffer
    # (the buffer is created by a collective handshake over fixed
    # membership), so staged sizing is not supported: masking is driven
    # explicitly by AscendElasticEPScalingExecutor on the scale-down
    # reuse path.
    def stage_ep_size(self) -> None:
        pass

    def commit_ep_size(self) -> None:
        pass

    @contextmanager
    def mask_remote_ranks(self) -> Iterator[None]:
        """Temporarily mask every peer rank (self stays alive).

        Used to isolate a warmup/dummy run from peers that are not
        participating, e.g. when a new worker captures its graphs. No-op
        unless the MegaMoe buffer has been bound.
        """
        buffer = _NpuAll2AllManager._mega_moe_buffer
        ep_world_size = _NpuAll2AllManager._ep_world_size
        if buffer is None or ep_world_size == 0:
            yield
            return
        ep_to_mc2 = _NpuAll2AllManager._ep_to_mc2
        # Mask every EP rank: replay/dummy traffic stays local. The caller
        # is responsible for only using this around local-only execution.
        # The writes stay in inference mode (the mask buffer is an
        # inference tensor); the caller's body must NOT be wrapped, so the
        # context only covers the mask updates themselves.
        peers = [ep_to_mc2[r] for r in range(ep_world_size)]
        with torch.inference_mode():
            for mc2_rank in peers:
                buffer.update_mask_buffer(mc2_rank, True)
        _accelerator_synchronize()
        try:
            yield
        finally:
            with torch.inference_mode():
                for mc2_rank in peers:
                    # Restore only ranks that are not recorded dead.
                    ep_rank = ep_to_mc2.index(mc2_rank)
                    if ep_rank not in _NpuAll2AllManager._dead:
                        buffer.update_mask_buffer(mc2_rank, False)
            _accelerator_synchronize()


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all,
        )
        self.device = torch.npu.current_device()

        # Create pyhccl_comm handle for batch_transfer_weights in elastic_ep
        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(group=tcp_store_group, device=self.device, warmup=False)

        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager(self.world_size)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.pyhccl_comm is not None:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()
            input_size = input_.size()
            # Use concat-style all-gather: stack-style has torch.compile
            # compatibility issues (pytorch/pytorch#138795).
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            # All-gather.
            output_tensor = self.pyhccl_comm.all_gather(input_, output_tensor)
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
            )
            return output_tensor
        else:
            return super().all_gather(input_, dim)

    def destroy(self):
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def batch_isend_irecv(self, p2p_ops: list):
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyHccl communicator found")
