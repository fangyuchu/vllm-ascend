# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable

import msgspec
import torch
import torch_npu
import zmq
from vllm.config import ParallelConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.worker.worker_base import WorkerBase

from vllm_ascend.platform import NPUPlatform

logger = init_logger(__name__)
_GLOBAL_PAUSE_EVENT = threading.Event()


def get_pause_event() -> threading.Event:
    global _GLOBAL_PAUSE_EVENT
    return _GLOBAL_PAUSE_EVENT


class NPUWorkerSentinel(BaseSentinel):
    def __init__(
        self,
        parallel_config: ParallelConfig,
        clear_input_batch_callback: Callable,
        device: torch.device,
        worker_cmd_addr: str,
        worker: WorkerBase,
    ):
        self.worker = worker
        self.dp_rank = parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity_str = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(parallel_config, f"{self.dp_rank}_{identity_str}", identity_str.encode())
        self.device = device
        self.data_parallel_master_ip = parallel_config.data_parallel_master_ip
        self.data_parallel_master_port = parallel_config.data_parallel_master_port
        self.dp_size = parallel_config.data_parallel_size
        torch.accelerator.set_device_index(self.device)

        self.clear_input_batch_callback = clear_input_batch_callback
        self.engine_core_cmd_socket = make_zmq_socket(
            self.ctx,
            worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=self.identity,
        )

        threading.Thread(target=self.run, daemon=True, name="WorkerSentinelThread").start()

    def run(self):
        # Wait for fault tolerance instructions from EngineCoreSentinel
        while not self.sentinel_dead:
            self.poll_and_execute_upstream_cmd()

    def poll_and_execute_upstream_cmd(self):
        """
        Receive and execute a command from upstream sentinel and send back
        the execution result.
        """
        try:
            _, msg = self.engine_core_cmd_socket.recv_multipart()
            ft_request = msgspec.msgpack.decode(msg, type=FaultToleranceRequest)
            ft_result = self._execute_cmd(ft_request)
            msg_bytes = msgspec.msgpack.encode(ft_result)
            self.engine_core_cmd_socket.send_multipart([b"", msg_bytes])
        except zmq.ZMQError:
            logger.info("Socket closed, terminating.")
            self.sentinel_dead = True

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        self.pause_event.set()
        NPUPlatform.set_device(self.device)
        result = torch_npu.npu.stop_device(self.device.index)
        if result == 0:
            logger.info("npu stop device %s succeeded", self.device.index)
            return FaultToleranceResult(ft_request.request_id, True)
        elif result == 1:
            logger.info("npu stop device %s failed", self.device.index)
            return FaultToleranceResult(ft_request.request_id, False)
        else:
            raise ValueError(f"Unexpected return value from stop_device: {result}")

    def retry(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        self.clear_input_batch_callback()
        get_pause_event().clear()
        NPUPlatform.set_device(self.device)
        torch_npu.npu.restart_device(self.device.index)
        get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            ft_request.params["new_stateless_dp_group_port"],
            self.dp_rank,
            self.dp_size,
            backend="gloo",
        )
        return FaultToleranceResult(ft_request.request_id, True)

    def shutdown(self):
        close_sockets([self.engine_core_cmd_socket])
        super().shutdown()

    def descale(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        get_pause_event().clear()
        exclude_ep_ranks = ft_request.params["exclude_ep_ranks"]
        vllm_config_update_dict = ft_request.params["vllm_config_update_dict"]
        NPUPlatform.set_device(self.device)
        torch_npu.npu.restart_device(self.device.index)
        self.clear_input_batch_callback()
        # comm_groups = get_all_model_groups()
        # for group in comm_groups:
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()
        self.worker.dp_descale(exclude_ep_ranks, vllm_config_update_dict)
        self.worker.execute_dummy_batch()
        return FaultToleranceResult(ft_request.request_id, True)
