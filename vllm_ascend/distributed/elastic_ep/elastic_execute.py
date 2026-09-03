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
from vllm.distributed.parallel_state import GroupCoordinator
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
        # Reuse the upstream implementation to build the world / dp / ep / eplb
        # standby groups and run the merged preparation steps (staged MoE quant
        # methods, EPLB communicator on the standby group, weight transfer for
        # scale-up, and target-group warmup), then create the Ascend-specific
        # MC2 standby group.
        super().prepare_reconfiguration(reconfig_request, use_all2all)
        create_ascend_standby_groups(
            new_dp_size=reconfig_request.new_data_parallel_size,
            new_world_size_across_dp=(
                self.worker.vllm_config.parallel_config.world_size * reconfig_request.new_data_parallel_size
            ),
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )

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
        # The retired MC2 group is destroyed on the async executor like the
        # upstream retired groups. The (collective) destroy must finish before
        # this removing worker shuts down.
        retired_ascend_groups = _replace_ascend_active_groups(mc2=None)
        self._start_group_cleanup(retired_ascend_groups)
        self._wait_for_group_cleanup()

    def switch_and_prepare(self) -> tuple[GroupCoordinator | None, ...]:
        retired_groups = super().switch_and_prepare()
        # The Ascend MC2 group is retired alongside the upstream groups; it is
        # appended here so the caller's background cleanup destroys it together
        # with the upstream retired groups instead of blocking the commit.
        retired_ascend_groups = _replace_ascend_active_groups(**pop_ascend_standby_groups())
        self.worker.model_runner.dp_size = self.worker.parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = self.worker.parallel_config.data_parallel_rank
        moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
        self._setup_moe_comm_and_quant_method()
        return (*retired_groups, *retired_ascend_groups)

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def prepare_new_worker(self) -> None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().prepare_new_worker()

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
