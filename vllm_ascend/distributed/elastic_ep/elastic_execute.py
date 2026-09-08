# Adapted from vLLM's elastic_execute.py with Ascend-specific changes:
# NPU/ACL graphs, quantized weight transfer, MC2 comm groups, PyHccl EPLB.

import gc
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.distributed import P2POp
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import create_standby_groups, get_standby_eplb_group
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.parallel_state import (
    GroupCoordinator,
    _replace_ascend_active_groups,
    get_mc2_group,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

_PATCH_LOCK = threading.Lock()


def ascend_batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    # Ascend HCCL P2P weight transfer. Replaces upstream batch_transfer_weights via
    # monkey-patch. Differs from upstream: collects params from __dict__/AttentionImplBase,
    # negotiates param names via TCP store, skips contiguous() (HCCL native support).
    device_comm = dp_group.device_communicator
    tcp_store_group = dp_group.tcp_store_group
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            if isinstance(weight, torch.Tensor):
                expert_weights_set.add(weight.data_ptr())
            else:
                expert_weights_set.update(w.data_ptr() for w in weight)

    state_dict = model.state_dict()
    all_params = []
    all_params_ptrs = set()
    all_params_name = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        ptr = param.data_ptr()
        if ptr not in all_params_ptrs and ptr not in expert_weights_set:
            if param.device.type == "npu":
                all_params.append(param.data)
                all_params_ptrs.add(ptr)
                all_params_name.append(name)

    def handle_sub_module(submodule, submodule_name):
        for attr_name, attr_value in submodule.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                data_ptr = attr_value.data_ptr()
                if data_ptr not in all_params_ptrs and data_ptr not in expert_weights_set:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    if len(all_params_name) != len(peer_rank_all_params_name):
        common = list(set(all_params_name) & set(peer_rank_all_params_name))
        ids = [all_params_name.index(name) for name in common]
        all_params = [param for idx, param in enumerate(all_params) if idx in ids]

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        op = object.__new__(P2POp)
        if is_sender:
            op.op = torch.distributed.isend
            op.tensor = param
        else:
            op.op = torch.distributed.irecv
            op.tensor = param
        op.group_peer = peer_rank
        p2p_ops.append(op)

    device_comm.batch_isend_irecv(p2p_ops)


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    if isinstance(
        quant_method := getattr(module.routed_experts.quant_method, "quant_method", None),
        AscendW8A8DynamicFusedMoEMethod,
    ):
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    @contextmanager
    def _use_ascend_transfer_impl(self):
        with patch(
            "vllm.distributed.elastic_ep.elastic_execute.batch_transfer_weights", new=ascend_batch_transfer_weights
        ):
            yield

    def prepare_reconfiguration(self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool) -> None:
        # Ascend-specific variant of the upstream preparation. It mirrors
        # ElasticEPScalingExecutor.prepare_reconfiguration step by step, with
        # the Ascend MC2 standby group created after the upstream
        # world/dp/ep/eplb groups and before transfer_weights (upstream runs
        # the warm-up at that point instead; see _warm_target_groups for why
        # it is skipped on Ascend).
        #
        # The placement is load-bearing: every stateless-group rendezvous
        # completes only when ALL members have joined, so both sides must
        # create the groups in the SAME relative order. The new worker's
        # boot creates its initial groups as world -> dp -> ep -> eplb -> mc2
        # (ensure_model_parallel_initialized, then init_ascend_model_parallel)
        # and only reaches prepare_new_worker after that. Creating the MC2
        # standby group before the upstream groups, or after the weight
        # transfer, deadlocks the scale-up:
        #   existing @ mc2/world rendezvous  <->  new worker @ world/mc2,
        # or existing @ transfer_weights  <->  new worker @ mc2 rendezvous.
        self._wait_for_group_cleanup()
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        old_dp_size = get_dp_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        world_size = parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        create_standby_groups(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
            use_all2all=use_all2all,
            enable_eplb=parallel_config.enable_eplb,
        )
        create_ascend_standby_groups(
            new_dp_size=new_dp_size,
            new_world_size_across_dp=new_world_size_across_dp,
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )
        self.stage_standby_moe_quant_methods()
        self._prepare_eplb_communicator(get_standby_eplb_group())
        if new_dp_size > old_dp_size:
            self.transfer_weights(old_dp_size, new_dp_size)

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().transfer_weights(old_dp_size=old_dp_size, new_dp_size=new_dp_size)

    def _release_cuda_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        reset_graph_params()

        mgr = self.worker.model_runner.cudagraph_manager
        if mgr is not None:
            mgr.graphs.clear()
            mgr._graphs_captured = False
            # NPU graph pools cache old allocations; a fresh pool is
            # required before re-capture (NPUCachingAllocator.cpp:2106).
            mgr.pool = current_platform.graph_pool_handle()
            if hasattr(mgr, "capture_sizes"):
                capture_sizes = mgr.capture_sizes
                if self.worker.model_runner.use_aclgraph:
                    set_graph_params(capture_sizes)
                    if self.worker.model_runner.speculative_config:
                        set_draft_graph_params(capture_sizes)

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def switch_and_remove(self) -> None:
        super().switch_and_remove()
        retired_mc2 = _replace_ascend_active_groups(mc2=None)
        if retired_mc2 is not None:
            retired_mc2.destroy()

    def switch_and_prepare(self) -> tuple[GroupCoordinator | None, ...]:
        retired_groups = super().switch_and_prepare()
        retired_mc2 = _replace_ascend_active_groups(**pop_ascend_standby_groups())
        self.worker.model_runner.dp_size = self.worker.parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = self.worker.parallel_config.data_parallel_rank
        moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
        self._setup_moe_comm_and_quant_method()
        if retired_mc2 is not None:
            # The retired MC2 group joins the upstream retired groups so the
            # executor's async cleanup thread destroys it after the switch.
            return (*retired_groups, retired_mc2)
        return retired_groups

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def prepare_new_worker(self) -> None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().prepare_new_worker()

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        # No-op on Ascend: the dp/ep device_group carries no HCCL traffic
        # (DP sync on cpu_group, MoE on MC2, EPLB on gloo), so skip the warm.
        return

    def warmup_local_kernels(self) -> None:
        pass

    def warm_and_capture(self) -> None:
        # No need to save/clear/restore the KV-cache block tables like the
        # upstream warm_and_capture: the V2 runner's dummy attention uses
        # all-zero block tables (reserved null block) and PAD_SLOT_ID slot
        # mappings, so real KV-cache blocks are never written during the
        # dummy run.
        runner = self.worker.model_runner
        self._release_cuda_graphs()
        unlock_workspace()
        runner._dummy_run(runner.max_num_tokens, is_profile=True, skip_eplb=True)
        self.worker.compile_or_warm_up_model()
        lock_workspace()

    def _setup_moe_comm_and_quant_method(self) -> None:
        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)
