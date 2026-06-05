# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import msgspec
import torch
import torch_npu
import zmq
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.distributed.utils import get_cached_tcp_store_client, stateless_init_torch_distributed_process_group
from vllm.logger import logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.worker.worker_base import WorkerBase

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_elastic_info
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.sentinel.scale_down import ScaleDownHelper

_GLOBAL_PAUSE_EVENT = threading.Event()


def get_pause_event() -> threading.Event:
    global _GLOBAL_PAUSE_EVENT
    return _GLOBAL_PAUSE_EVENT


def evaluate_pause_condition() -> None:
    if get_pause_event().is_set():
        raise RuntimeError("Worker is paused as a fault was detected on another rank.")


class NPUWorkerSentinel(BaseSentinel):
    def __init__(
        self,
        parallel_config: ParallelConfig,
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
        get_pause_event().set()
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
        self._clean_worker_state()
        get_pause_event().clear()
        NPUPlatform.set_device(self.device)
        torch_npu.npu.restart_device(self.device.index)
        torch_npu.distributed.reinit_process_group(None, False)
        get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            ft_request.params["new_stateless_dp_group_port"],
            self.dp_rank,
            self.dp_size,
            backend="gloo",
            gloo_timeout_seconds=self.worker.vllm_config.parallel_config.fault_tolerance_config.gloo_comm_timeout,
        )

        return FaultToleranceResult(ft_request.request_id, True)

    def shutdown(self):
        close_sockets([self.engine_core_cmd_socket])
        super().shutdown()

    def scale_down(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        get_pause_event().clear()
        exclude_ep_ranks = ft_request.params["exclude_ep_ranks"]
        vllm_config_update_dict = ft_request.params["vllm_config_update_dict"]
        self._coord_store_port = ft_request.params["coord_store_port"]
        store = get_cached_tcp_store_client(self.data_parallel_master_ip, self._coord_store_port)
        NPUPlatform.set_device(self.device)
        torch_npu.npu.restart_device(self.device.index)
        self._clean_worker_state()
        torch_npu.distributed.reinit_process_group(None, False)
        torch.npu.synchronize()
        self.scale_down_worker(exclude_ep_ranks, vllm_config_update_dict, store)
        self.worker.execute_dummy_batch()
        return FaultToleranceResult(ft_request.request_id, True)

    def scale_down_worker(self, excluded_ep_ranks: list[int], scale_down_config, coord_store):
        """
        Reconfigure data-parallel (DP) layout and MoE expert placement after
        excluding one or more DP ranks (e.g., due to failure).

        Args:
            excluded_ep_ranks: EP ranks to exclude from service.
            scale_down_config: Dict containing rank_mapping and new parallel config.
            coord_store: TCP store for coordinating group reinitialization.
        """
        assert self.worker.vllm_config.parallel_config.enable_fault_tolerance is True, "enable_fault_tolerance is False"
        if not self.worker.model_loaded:
            raise RuntimeError("model has not been loaded yet")

        rank_mapping = scale_down_config.get("rank_mapping")
        assert rank_mapping is not None
        assert isinstance(rank_mapping, dict)

        new_dp_rank = rank_mapping[self.worker.parallel_config.data_parallel_rank]

        num_logical_expert = self.worker.num_logical_expert

        enable_d2d_rebalance = (
            self.worker.vllm_config.parallel_config.fault_tolerance_config.enable_fault_tolerance_rebalance
        )
        if self.worker.model_runner.shared_dict["moe_load"] is None or torch.all(
            self.worker.model_runner.shared_dict["moe_load"][0] == 0
        ):
            enable_d2d_rebalance = False

        scale_down_helper = ScaleDownHelper(self.worker.vllm_config, self.worker.model_runner, self.worker.quant)
        # Currently,only TP=1 is supported.Therefore excluded_dp_ranks = excluded_ep_ranks
        # TODO: In scenarios TP>1,the logic for converting from
        #  excluded_ep_ranks to excluded_dp_ranks needs to be added
        excluded_dp_ranks = excluded_ep_ranks

        # Phase 1: Expert distribution recalculation
        experts_to_load = scale_down_helper.get_expert_distribution_after_scale_down(
            excluded_dp_ranks, enable_d2d_rebalance, new_dp_rank
        )
        num_add_experts_per_rank = self.worker.model_runner.shared_dict["num_add_experts_per_rank"]

        if num_add_experts_per_rank > 0:
            # use_mask_mc2 is False
            raise RuntimeError("only support mask mc2")

        # Phase 2: Expert weight reloading
        saved_weights = scale_down_helper.load_expert_weights_to_cpu(experts_to_load, self.worker.weight_name_to_tensor)
        scale_down_helper.reload_expert_weights(experts_to_load, saved_weights)

        # Phase 3：EPLB adaptor update
        if get_ascend_config().eplb_config.dynamic_eplb:
            scale_down_helper.update_eplb_adaptor_info(num_add_experts_per_rank, new_dp_rank)

        # Phase 4: Log2phy map generation
        if enable_d2d_rebalance:
            all_layer_log2phy = scale_down_helper.d2d_transmission_for_scaling_down()
        else:
            all_layer_log2phy = scale_down_helper.gen_all_layer_log2phy(new_dp_rank)

        self.worker.global_experts_distribution = self.worker.model_runner.eplb_process.worker.local2global(
            self.worker.model_runner.shared_dict["expert_maps"]
        )

        # Phase 5: Configuration and state update
        old_ep_size = len(self.worker.ep2dp_map)
        scale_down_helper.update_parallel_config(scale_down_config)
        self.dp_size = self.worker.vllm_config.parallel_config.data_parallel_size
        self.dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank
        self.worker.model_runner.dp_size = self.dp_size
        self.worker.model_runner.dp_rank = self.dp_rank
        logger.info(
            f"ep2dp_map is {self.worker.ep2dp_map} "
            f"excluded_dp_ranks is {excluded_dp_ranks} "
            f"rank_mapping is {rank_mapping}"
        )
        self.worker.ep2dp_map = scale_down_helper.update_ep2dp_map(
            self.worker.ep2dp_map, excluded_dp_ranks, rank_mapping
        )
        elastic_info = get_elastic_info()
        num_new_phy_experts = (self.worker.model_runner.shared_dict["expert_maps"][0] != -1).sum().item()
        scale_down_helper.update_elastic_info(elastic_info, num_new_phy_experts, old_ep_size, self.worker.ep2dp_map)

        # Phase 6: Communication group reinitialization
        scale_down_helper.destroy_comm_group()
        with set_current_vllm_config(self.worker.vllm_config):
            scale_down_helper.init_dp_cpu_group(coord_store, "stateless")

        # Phase 7: MoE reconfiguration
        scale_down_helper.reconfigure_moe(num_logical_expert, num_new_phy_experts, all_layer_log2phy)

    def _clean_worker_state(self):
        self.worker.model_runner.execute_model_state = None
        self.worker.model_runner.kv_connector_output = None
        input_batch = self.worker.model_runner.input_batch
        cached_req_ids = input_batch.req_id_to_index.keys()
        for req_id in list(cached_req_ids):
            input_batch.remove_request(req_id)
        input_batch.condense()
        input_batch.refresh_metadata()
        input_batch.req_prompt_embeds.clear()
        self.worker.model_runner.async_output_copy_stream = torch.cuda.Stream()
        self.worker.model_runner.prepare_inputs_event = torch.Event()
