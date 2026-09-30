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

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.utils import StatelessProcessGroup

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class _NpuAll2AllManager:
    """All2All-manager adapter for MC2 fault tolerance.

    Owns the dead-rank mask, encoded into the ``elastic_info`` tensor consumed
    by the MC2 dispatch/combine operators. The public interface mirrors the
    upstream All2AllManagerBase mask API.
    """

    # MC2 kernels do not detect faults themselves; the mask is written
    # host-side by FT recovery, so a per-step query can never observe one.
    support_fault_tolerance = False

    def __init__(self, ep_world_size: int, device: torch.device | None = None) -> None:
        self._ep_world_size = ep_world_size
        self._device = device
        self._dead: set[int] = set()
        self._num_local_experts: int = 0
        self._mega_moe_buffer = None
        self._ep_to_mc2: tuple[int, ...] = ()

        # elastic_info layout: [is_scaling_down, dense ep world size,
        #  shared_expert_rank_num, num_physical_experts] + table1(orig->dense)
        #  + table2(dense->orig). num_physical_experts is derived from the
        #  dead set and num_local_experts on every rebuild.
        size = 4 + 2 * ep_world_size
        self._elastic_info_host = torch.zeros(size, dtype=torch.int32)
        if device is None:
            device = torch.device("npu", torch.npu.current_device())
        self._device = device
        self._elastic_info: torch.Tensor | None = None

    def stage_ep_size(self) -> None:
        # Ascend stages communication groups in AscendElasticEPScalingExecutor;
        # this mask manager has no connections to stage.
        pass

    def commit_ep_size(self) -> None:
        # The Ascend executor switches groups and updates backend masks.
        # Keep the upstream default no-op here to avoid changing them twice.
        pass

    @property
    def uses_mega_moe(self) -> bool:
        return self._mega_moe_buffer is not None

    @torch.inference_mode()
    def bind_mega_moe_buffer(self, buffer, ep_to_mc2: list[int]) -> None:
        """Attach the active fused backend before graph capture.

        The manager remains the CPU source of truth for faults. MegaMoe uses
        full-width physical expert ids, whereas MC2 uses elastic_info's dense
        ids. Binding explicitly avoids importing the MoE stack here.
        """
        if buffer.ep_world_size != self._ep_world_size or sorted(ep_to_mc2) != list(range(self._ep_world_size)):
            raise ValueError("MegaMoe fault tolerance requires matching EP and MC2 rank sets.")
        if self.uses_mega_moe:
            if buffer is not self._mega_moe_buffer or tuple(ep_to_mc2) != self._ep_to_mc2:
                raise RuntimeError("Cannot replace a captured MegaMoe fault-tolerance buffer or its rank order.")
            return
        for method in ("clean_mask_buffer", "update_mask_buffer", "get_local_buffer_tensor"):
            if not callable(getattr(buffer, method, None)):
                raise RuntimeError(
                    f"MegaMoe fault tolerance requires SymmBuffer.{method}; "
                    "install matching CANN operators and cann_ops_transformer with PR #9368 support."
                )
        # Do not swallow failures: a graph captured with mask_buffer=None
        # cannot start honoring a newly allocated mask after a fault.
        buffer.clean_mask_buffer()
        for rank in sorted(self._dead):
            buffer.update_mask_buffer(ep_to_mc2[rank], True)
        self._ep_to_mc2 = tuple(ep_to_mc2)
        self._mega_moe_buffer = buffer

    @torch.inference_mode()
    def update_mask(self, rank: int, masked: bool = True) -> None:
        """Mark an EP rank dead/alive and rebuild elastic_info in place."""
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < self._ep_world_size:
            raise ValueError(f"EP rank must be in [0, {self._ep_world_size}), got {rank}.")
        if not isinstance(masked, bool):
            raise TypeError("masked must be a bool")
        if self.uses_mega_moe:
            if not masked and rank in self._dead:
                # Recovery runs only after outstanding device work is stopped.
                # Old completion flags must not survive re-enabling a peer.
                self.clean_buffers()
            self._mega_moe_buffer.update_mask_buffer(self._ep_to_mc2[rank], masked)
        if masked:
            self._dead.add(rank)
        else:
            self._dead.discard(rank)
        self._rebuild_elastic_info()

    def query_active_mask(self) -> torch.Tensor:
        """Per-EP-rank mask (1=dead, 0=live) as a CPU tensor, matching the
        upstream mask-buffer convention.

        Built on CPU on purpose: this is called while a fault is being
        probed, when the NPU may be hung — any device op would fail.
        """
        mask = torch.zeros(self._ep_world_size, dtype=torch.int32)
        for rank in self._dead:
            mask[rank] = 1
        return mask

    def query_fault(self) -> torch.Tensor:
        # MC2 has no in-kernel fault detection; faults surface as aborted ops.
        return torch.tensor(False)

    @torch.inference_mode()
    def clean_buffers(self) -> None:
        """Clear fused communication flags after device reset, keeping dead ranks masked."""
        if self.uses_mega_moe:
            self._mega_moe_buffer.get_local_buffer_tensor(torch.uint8).zero_()

    def get_elastic_info(self) -> torch.Tensor:
        """The device elastic_info tensor for the next MC2 dispatch/combine."""
        if self._elastic_info is None:
            self._elastic_info = self._elastic_info_host.to(self._device)
        return self._elastic_info

    def set_num_local_physical_experts(self, num_local_experts: int) -> None:
        """Record the physical expert slots per EP rank."""
        self._num_local_experts = num_local_experts
        self._rebuild_elastic_info()

    @torch.inference_mode()
    def _rebuild_elastic_info(self) -> None:
        """Rebuild elastic_info from the dead set into the existing device
        tensor (never reallocates, so captured graphs stay valid).

        The device tensor may have been created during inference-mode warmup.
        """

        world_size = self._ep_world_size
        alive = sorted(set(range(world_size)) - self._dead)
        num_physical_experts = len(alive) * self._num_local_experts
        table1 = torch.full((world_size,), -1, dtype=torch.int32)
        table1[alive] = torch.arange(len(alive), dtype=torch.int32)
        table2 = torch.full((world_size,), -1, dtype=torch.int32)
        table2[: len(alive)] = torch.tensor(alive, dtype=torch.int32)
        self._elastic_info_host.copy_(
            torch.cat(
                [
                    torch.tensor([int(bool(self._dead)), len(alive), 0, num_physical_experts], dtype=torch.int32),
                    table1,
                    table2,
                ]
            )
        )
        if self._elastic_info is not None:
            self._elastic_info.copy_(self._elastic_info_host, non_blocking=True)


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
            use_all2all=use_all2all,
        )
        if tcp_store_group is not None:
            # StatelessGroupCoordinator passes its logical device index here.
            # Under Ray, the worker may already be bound to a different NPU,
            # especially when Ray does not rewrite ASCEND_RT_VISIBLE_DEVICES.
            # HCCL requires every communicator rank to use the device selected
            # by its worker, so prefer the active NPU for stateless groups.
            self.device = torch.device(f"npu:{torch.npu.current_device()}")
        else:
            self.device = device or torch.device(f"npu:{torch.npu.current_device()}")

        # Create pyhccl_comm handle for batch_transfer_weights in elastic_ep
        self.pyhccl_comm: PyHcclCommunicator | None = None
        if self.world_size > 1 and tcp_store_group is not None:
            self.pyhccl_comm = PyHcclCommunicator(
                group=tcp_store_group,
                device=self.device,
                warmup=False,
            )

        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        # Only the EP group's instance is ever looked up (via the upstream
        # get_ep_all2all_manager()); the rest stay dormant.
        self.all2all_manager = _NpuAll2AllManager(self.world_size, self.device)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().all_gather(input_, dim)

        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size,
            dtype=input_.dtype,
            device=input_.device,
        )
        pyhccl_comm.all_gather(input_.contiguous(), output_tensor)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :])

    def destroy(self) -> None:
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.destroy()
            self.pyhccl_comm = None

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().broadcast(tensor, src)
        pyhccl_comm.broadcast(tensor, src)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().send(tensor, dst)
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        pyhccl_comm.send(tensor, dst)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            return super().recv(size, dtype, src)
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pyhccl_comm.recv(tensor, src)
        return tensor

    def batch_isend_irecv(self, p2p_ops: list, stream=None) -> None:
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is None or pyhccl_comm.disabled:
            raise ValueError("No PyHccl communicator found")
        if stream is None:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            pyhccl_comm.batch_isend_irecv(p2p_ops, stream=stream)
