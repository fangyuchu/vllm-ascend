# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import torch
from torch.distributed import P2POp
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator, TorchDistNcclEplbCommunicator

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class HcclEplbCommunicator(TorchDistNcclEplbCommunicator):
    """Torch-distributed EPLB transfers over the HCCL device group."""

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        # Ascend keeps each expert in an independent persistent tensor. The
        # upstream profile collective expects every weight entry to be one
        # stacked tensor, so reserve HCCL buffers during actual P2P transfers.
        return False


class PyHcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyHcclCommunicator using HcclSend/HcclRecv."""

    def __init__(
        self,
        pyhccl_comm: PyHcclCommunicator,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        self._pyhccl_comm = pyhccl_comm
        self._stream = stream
        self._p2p_ops: list[P2POp] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        for tensor in tensors:
            op = object.__new__(P2POp)
            op.op = torch.distributed.isend
            op.tensor = tensor
            op.group_peer = dst_rank
            op.tag = expert_id
            self._p2p_ops.append(op)

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        for tensor in tensors:
            op = object.__new__(P2POp)
            op.op = torch.distributed.irecv
            op.tensor = tensor
            op.group_peer = src_rank
            op.tag = expert_id
            self._p2p_ops.append(op)

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        self._p2p_ops.sort(key=lambda op: op.tag)
        try:
            self._pyhccl_comm.batch_isend_irecv(self._p2p_ops, self._stream)
        finally:
            self._p2p_ops.clear()

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        # Ascend keeps each expert in an independent persistent tensor. The
        # upstream profile collective expects every weight entry to be one
        # stacked tensor, so reserve HCCL buffers during actual P2P transfers.
        return False
