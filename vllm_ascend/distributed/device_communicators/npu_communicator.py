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
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    # Elastic-EP hooks that upstream calls unconditionally. NPU routes MoE
    # communication through MC2/all_gather and never reuses kernels across EP
    # rank changes, so these are no-ops.
    def stage_ep_size(self) -> None:
        pass

    def commit_ep_size(self) -> None:
        pass


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
        self.all2all_manager = _NpuAll2AllManager()

    def destroy(self):
        if self.pyhccl_comm is not None:
            self.pyhccl_comm.close()
            self.pyhccl_comm = None

    def batch_isend_irecv(self, p2p_ops: list):
        pyhccl_comm = self.pyhccl_comm
        if pyhccl_comm is not None and not pyhccl_comm.disabled:
            pyhccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyHccl communicator found")
