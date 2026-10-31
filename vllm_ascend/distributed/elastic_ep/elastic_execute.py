# Adapted from vLLM's elastic_execute.py with Ascend-specific changes:
# NPU/ACL graphs, quantized weight transfer, MC2 groups, and torch-gloo EPLB.

import gc
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn as nn
import vllm.distributed.elastic_ep.elastic_execute as upstream_elastic_execute
from torch.distributed import P2POp
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import set_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import (
    get_standby_dp_group,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.ascend_forward_context import MoECommType, _is_decode_only_node
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.mega_moe_capture import MegaMoeCaptureState, ep_to_mc2_ranks
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    get_standby_mc2_group,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.eplb.eplb_state import refresh_model_routing_tables
from vllm_ascend.distributed.parallel_state import (
    _replace_ascend_active_groups,
    get_mc2_group,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    get_moe_comm_method,
    setup_moe_comm_method,
)
from vllm_ascend.ops.fused_moe.moe_utils import _get_cann_mega_moe_quant_settings
from vllm_ascend.quantization.quant_type import QuantType

_PATCH_LOCK = threading.Lock()

_BOOTSTRAP_EXPERT_PARAMETER_PREFIX = "__elastic_bootstrap_expert__."


def ascend_batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    stream=None,
    include_expert_weights: bool = False,
) -> None:
    # Ascend HCCL P2P weight transfer. Replaces upstream batch_transfer_weights via
    # monkey-patch. Differs from upstream: collects params from __dict__/AttentionImplBase,
    # negotiates parameter names via the TCP store.
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
            if attr_name.endswith("expert_map"):
                continue
            if isinstance(attr_value, torch.Tensor):
                data_ptr = attr_value.data_ptr()
                if data_ptr not in all_params_ptrs and data_ptr not in expert_weights_set:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    # Static quantization uses plain tensor attributes, including input scale
    # reciprocals and offsets, which are absent from state_dict().
    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    expert_params: list[torch.Tensor] = []
    expert_param_ptrs: set[int] = set()
    if include_expert_weights:
        for weight_group in expert_weights:
            for weight in weight_group:
                tensors = (weight,) if isinstance(weight, torch.Tensor) else weight
                for tensor in tensors:
                    ptr = tensor.data_ptr()
                    if ptr not in expert_param_ptrs:
                        param = tensor.data
                        expert_params.append(param)
                        expert_param_ptrs.add(ptr)
        for index, param in enumerate(expert_params):
            all_params.append(param)
            all_params_name.append(f"{_BOOTSTRAP_EXPERT_PARAMETER_PREFIX}{index:08d}")

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
    transfer_stream = stream or torch.npu.current_stream()
    p2p_ops = []
    for param in all_params:
        op = object.__new__(P2POp)
        op.op = torch.distributed.isend if is_sender else torch.distributed.irecv
        op.tensor = param
        op.group_peer = peer_rank
        p2p_ops.append(op)
    device_comm.batch_isend_irecv(p2p_ops, stream=transfer_stream)


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    # Initialize EP and MC2 in the same order on existing and new ranks;
    # scale-up quant methods may not have a cached MC2 group name yet.
    setup_moe_comm_method(module.moe_config)
    quant_method = getattr(module.routed_experts.quant_method, "quant_method", None)
    if hasattr(quant_method, "moe_all_to_all_group_name"):
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    def _is_mega_moe_restore(self, request: ReconfigureDistributedRequest) -> bool:
        config = self.worker.vllm_config
        parallel = config.parallel_config
        comm = self._get_fused_mc2_comm()
        state = getattr(comm, "mega_moe_capture_state", None)
        if not isinstance(state, MegaMoeCaptureState):
            return False
        dp_group = get_dp_group()
        return bool(
            request.operation_id
            and parallel.enable_fault_tolerance
            and parallel.tensor_parallel_size
            == parallel.pipeline_parallel_size
            == parallel.prefill_context_parallel_size
            == 1
            and config.lora_config is None
            and getattr(config, "speculative_config", None) is None
            and comm.mega_moe_quant_type == QuantType.W8A8
            and upstream_elastic_execute.get_active_dp_size(dp_group) < request.new_data_parallel_size
            and request.new_data_parallel_size == state.buffer.ep_world_size == get_mc2_group().world_size
            and bool(getattr(dp_group, "dead_dp_ranks", ()))
            and callable(getattr(state.buffer, "update_group", None))
        )

    def _get_fused_mc2_comm(self):
        model = self.worker.get_model()
        module = next(module for module in model.modules() if is_moe_layer(module))
        return get_moe_comm_method(MoECommType.FUSED_MC2, module.moe_config)

    def _mega_moe_comm(self):
        comm = self._get_fused_mc2_comm()
        if not isinstance(getattr(comm, "mega_moe_capture_state", None), MegaMoeCaptureState):
            raise RuntimeError("MegaMoe restore requires graph-stable routing and a live symmetric buffer")
        return comm

    def start_async(self, execute_method: str, *args, **kwargs) -> str:
        """Arm EPLB suppression before background MegaMoe preparation starts.

        ``start_async`` runs synchronously on the worker command loop between
        model executions. Setting the flag here ensures every old rank sees it
        before the next serving step can reach periodic EPLB rearrangement.
        """
        suppression_armed = False
        if (
            execute_method == "prepare_reconfiguration"
            and args
            and isinstance(args[0], ReconfigureDistributedRequest)
            and self._is_mega_moe_restore(args[0])
        ):
            self._set_eplb_suppressed(True)
            self._capture_eplb_suppression_operation_id = args[0].operation_id
            suppression_armed = True
        try:
            return super().start_async(execute_method, *args, **kwargs)
        except BaseException:
            if suppression_armed:
                self._set_eplb_suppressed(False)
                self._capture_eplb_suppression_operation_id = None
            raise

    @staticmethod
    def _capture_key(operation_id: str, name: str) -> str:
        return f"elastic_capture/{operation_id}/bootstrap/{name}"

    @staticmethod
    def _capture_store(reconfig_request: ReconfigureDistributedRequest):
        return get_cached_tcp_store_client(
            reconfig_request.new_data_parallel_master_ip,
            reconfig_request.coord_store_port,
        )

    def _synchronize_capture_mapping_setup(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        """Wait for every target rank before initializing target MC2 communication."""
        # Fast restore requires TP=PP=PCP=1 and preserves DP identities.
        # Old ranks arrive after draining; new ranks arrive after mapping setup.
        operation_id = reconfig_request.operation_id
        rank = self.worker.vllm_config.parallel_config.data_parallel_rank
        store = self._capture_store(reconfig_request)
        ready_keys = [
            self._capture_key(operation_id, f"mapping/ready/{rank}")
            for rank in range(reconfig_request.new_data_parallel_size)
        ]
        store.set(self._capture_key(operation_id, f"mapping/ready/{rank}"), b"1")
        store.wait(ready_keys)
        logger.info("[Elastic EP] MegaMoe mapping rendezvous completed: rank=%s", rank)

    def _publish_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        old_dp_size: int,
    ) -> bool:
        self._mega_moe_precommit_capture = self._is_mega_moe_restore(reconfig_request)
        self._capture_bootstrap_mapping = None
        enabled = self._mega_moe_precommit_capture
        store = self._capture_store(reconfig_request)
        store.set(
            self._capture_key(reconfig_request.operation_id, "enabled"),
            b"1" if enabled else b"0",
        )
        store.set(
            self._capture_key(reconfig_request.operation_id, "old_dp_size"),
            str(old_dp_size).encode(),
        )
        self._capture_old_dp_size = old_dp_size
        return enabled

    def _read_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
    ) -> bool:
        if not reconfig_request.operation_id:
            self._mega_moe_precommit_capture = False
            return False
        store = self._capture_store(reconfig_request)
        enabled = store.get(self._capture_key(reconfig_request.operation_id, "enabled")) == b"1"
        self._mega_moe_precommit_capture = enabled
        self._mega_moe_precommit_capture_done = False
        self._capture_old_dp_size = int(
            store.get(
                self._capture_key(
                    reconfig_request.operation_id,
                    "old_dp_size",
                )
            ).decode()
        )
        return enabled

    @contextmanager
    def _use_ascend_transfer_impl(
        self,
        *,
        include_expert_weights: bool | None = None,
    ):
        original_transfer = upstream_elastic_execute.batch_transfer_weights
        if getattr(self, "_weight_transfer_stream", None) is not None:
            raise RuntimeError("An Ascend weight transfer is already active")
        if include_expert_weights is None:
            include_expert_weights = bool(getattr(self, "_mega_moe_precommit_capture", False))
        self._weight_transfer_stream = torch.npu.Stream()
        upstream_elastic_execute.batch_transfer_weights = partial(
            ascend_batch_transfer_weights,
            stream=self._weight_transfer_stream,
            include_expert_weights=include_expert_weights,
        )
        try:
            yield
        finally:
            # Drain queued P2P work before restoring the upstream transfer function.
            try:
                self._weight_transfer_stream.synchronize()
            finally:
                upstream_elastic_execute.batch_transfer_weights = original_transfer
                self._weight_transfer_stream = None

    def _synchronize_weight_transfer(self) -> None:
        stream = getattr(self, "_weight_transfer_stream", None)
        if stream is None:
            torch.npu.synchronize()
            return
        stream.synchronize()

    def supports_precommit_graph_capture(self, operation_id: str) -> bool:
        request = getattr(self, "reconfig_request", None)
        return bool(
            getattr(self, "_mega_moe_precommit_capture", False)
            and request is not None
            and request.operation_id == operation_id
        )

    @staticmethod
    def _materialize_hccl_group(
        group: StatelessGroupCoordinator,
        group_name: str,
    ) -> None:
        """Initialize a lazy HCCL communicator with every group rank present."""
        torch.distributed.barrier(group=group.cpu_group)
        device_group = group.device_group
        backend = device_group._get_backend(torch.device("npu"))
        comm_name = backend.get_hccl_comm_name(group.rank_in_group)
        torch.npu.synchronize()
        torch.distributed.barrier(group=group.cpu_group)
        logger.info(
            "[Elastic EP] Materialized %s HCCL communicator: rank=%s/%s, comm_name=%s",
            group_name,
            group.rank_in_group,
            group.world_size,
            comm_name,
        )

    def materialize_new_communication_groups(self) -> None:
        """Materialize the target MC2 communicator before MegaMoe graph capture."""
        mc2_group = get_standby_mc2_group()
        if mc2_group is None:
            mc2_group = get_mc2_group()
        self._materialize_hccl_group(mc2_group, "MC2")

    def _set_eplb_suppressed(self, suppressed: bool) -> None:
        """Pause the complete EPLB controller during MegaMoe reconfiguration."""
        self.worker.model_runner.eep_eplb_suppressed = suppressed
        ep_group = get_ep_group()
        if ep_group.rank_in_group == 0:
            logger.info(
                "[Elastic EP] EPLB %s MegaMoe elastic scaling transition",
                "disabled during" if suppressed else "re-enabled after",
            )

    def _drain_async_eplb(self) -> None:
        """Finish an already-started async EPLB cycle before MegaMoe collectives.

        ``eep_eplb_suppressed`` prevents the serving loop from starting another
        EPLB cycle, but a cycle that was scheduled just before suppression may
        still be transferring expert weights. All existing ranks call this
        method only after serving collectives have been drained, so the EPLB
        group's coordinated stop cannot race with MegaMoe context update.
        """
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is not None:
            eplb_state.drain_async()

    @staticmethod
    def _synchronize_capture_context_rendezvous(
        mc2_group: StatelessGroupCoordinator,
        role: str,
    ) -> None:
        """Confirm every target rank completed MegaMoe context creation/update."""
        torch.npu.synchronize()
        torch.distributed.barrier(group=mc2_group.cpu_group)
        logger.info(
            "[Elastic EP] MegaMoe context rendezvous completed: role=%s, rank=%s/%s",
            role,
            mc2_group.rank_in_group,
            mc2_group.world_size,
        )

    def broadcast_expert_mapping(self) -> None:
        if not getattr(self, "_mega_moe_precommit_capture", False):
            super().broadcast_expert_mapping()
            return

        standby_dp_group = get_standby_dp_group()
        if standby_dp_group is None:
            raise RuntimeError("Standby DP group is not initialized")

        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("MegaMoe pre-commit capture requires EPLB state")

        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        physical_to_logical = model_state.physical_to_logical_map
        request = self.reconfig_request
        assert request is not None
        dp_group = get_dp_group()
        old_dp_size = upstream_elastic_execute.get_active_dp_size(dp_group)
        assert old_dp_size == self._capture_old_dp_size
        assert request.new_data_parallel_size == dp_group.world_size
        existing, new = self.scale_up_members(old_dp_size, request.new_data_parallel_size)

        # FT preserves physical slots. Clone the live table and fill only the
        # holes with the actual weight sender's mapping; serving stays untouched.
        bootstrap_mapping = physical_to_logical.clone()
        rank_slots = bootstrap_mapping.view(physical_to_logical.shape[0], dp_group.world_size, -1)
        for rank in new:
            source = upstream_elastic_execute.scale_up_source_rank(existing, new, rank)
            rank_slots[:, rank].copy_(rank_slots[:, source])
        self._capture_bootstrap_mapping = bootstrap_mapping
        upstream_elastic_execute.broadcast_expert_mapping(
            physical_to_logical=bootstrap_mapping,
            dp_group=standby_dp_group,
            src_rank=self._mapping_source_rank(),
            device=self.worker.device,
        )

    def prepare_reconfiguration(self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool) -> None:
        # Let upstream prepare world / DP / EP / EPLB and weight transfer, then
        # add the Ascend-specific MC2 standby group used at commit time.
        old_dp_size = upstream_elastic_execute.get_active_dp_size(get_dp_group())
        suppression_prearmed = (
            getattr(self, "_capture_eplb_suppression_operation_id", None) == reconfig_request.operation_id
        )
        use_precommit_capture = self._publish_capture_decision(
            reconfig_request,
            old_dp_size,
        )
        if use_precommit_capture and not suppression_prearmed:
            # Normal inference continues throughout pre-commit preparation.
            # Stop the complete EPLB controller so no load collective, async
            # commit, or weight rearrangement can overlap the target MegaMoe/MC2
            # context update.
            self._set_eplb_suppressed(True)
            self._capture_eplb_suppression_operation_id = reconfig_request.operation_id
        elif not use_precommit_capture and suppression_prearmed:
            self._set_eplb_suppressed(False)
            self._capture_eplb_suppression_operation_id = None
        super().prepare_reconfiguration(reconfig_request, use_all2all)
        if use_precommit_capture:
            # Standby groups and weight transfer are safe on the dedicated
            # preparation thread. Mapping broadcast, lazy MC2 creation, and
            # MegaMoe symmetric-buffer update_group issue target-topology NPU
            # collectives and are finalized separately after EngineCore has
            # drained serving collectives on every old rank.
            return

        create_ascend_standby_groups(
            new_dp_size=reconfig_request.new_data_parallel_size,
            new_world_size_across_dp=(
                self.worker.vllm_config.parallel_config.world_size * reconfig_request.new_data_parallel_size
            ),
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )

    def stage_standby_moe_quant_methods(self, all2all_manager) -> None:
        # MegaMoe keeps the quant methods referenced by the existing graphs.
        if not self._mega_moe_precommit_capture:
            super().stage_standby_moe_quant_methods(all2all_manager)

    def finalize_precommit_prepare(self, operation_id: str) -> None:
        """Finalize MegaMoe target collectives while old-rank inference is drained."""
        reconfig_request = self.reconfig_request
        if reconfig_request is None:
            raise RuntimeError("Missing MegaMoe pre-commit reconfiguration request")
        if reconfig_request.operation_id != operation_id:
            raise RuntimeError(
                f"MegaMoe pre-commit operation mismatch: expected={reconfig_request.operation_id}, got={operation_id}"
            )
        if not getattr(self, "_mega_moe_precommit_capture", False):
            raise RuntimeError("MegaMoe pre-commit finalization was not prepared")

        self._drain_async_eplb()
        # Pair with receive_expert_mapping on the independently launched
        # ranks. This broadcast must not overlap the active serving
        # metadata all-reduce on only a subset of old ranks.
        self.broadcast_expert_mapping()
        self._synchronize_capture_mapping_setup(reconfig_request)
        create_ascend_standby_groups(
            new_dp_size=reconfig_request.new_data_parallel_size,
            new_world_size_across_dp=(
                self.worker.vllm_config.parallel_config.world_size * reconfig_request.new_data_parallel_size
            ),
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
        )
        # Lazy HCCL construction and MegaMoe context update are both
        # target-group collectives. Keep them in the same drained phase so
        # old and new ranks enter them in one deterministic order.
        self.materialize_new_communication_groups()
        group = get_standby_mc2_group()
        comm = self._mega_moe_comm()
        comm.mega_moe_capture_state.update_context(group)
        self._synchronize_capture_context_rendezvous(group, "existing MegaMoe")

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().transfer_weights(
                old_dp_size=old_dp_size,
                new_dp_size=new_dp_size,
            )

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

    def _activate_ascend_standby_groups(self):
        retired_mc2 = _replace_ascend_active_groups(**pop_ascend_standby_groups())
        moe_modules = [module for module in self.worker.model_runner.model.modules() if is_moe_layer(module)]
        for module in moe_modules:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
        return retired_mc2

    def switch_and_prepare(self):
        retired_groups = super().switch_and_prepare()
        retired_mc2 = self._activate_ascend_standby_groups()
        self._setup_moe_comm_and_quant_method()
        return (*retired_groups, retired_mc2) if retired_mc2 is not None else retired_groups

    @staticmethod
    def _get_capture_num_local_experts(moe_modules: list[nn.Module]) -> int:
        if not moe_modules:
            raise RuntimeError("MegaMoe Elastic EP requires at least one MoE layer")
        num_local_experts = int(moe_modules[0].moe_config.num_local_experts)
        if num_local_experts <= 0:
            raise RuntimeError(f"MegaMoe restore requires a positive local expert capacity, got {num_local_experts}")
        if any(int(module.moe_config.num_local_experts) != num_local_experts for module in moe_modules[1:]):
            raise RuntimeError(
                "MegaMoe restore requires every MoE layer to preserve the same number of local physical expert slots"
            )
        return num_local_experts

    def _restore_capture_scale_up_model_state(self, target_ep_size: int) -> None:
        """Restore the original physical EPLB capacity on existing ranks."""
        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("MegaMoe Elastic EP scale-up requires EPLB state")

        model = model_runner.get_model()
        moe_modules = [module for module in model.modules() if is_moe_layer(module)]
        num_local_experts = self._get_capture_num_local_experts(moe_modules)
        num_physical_experts = num_local_experts * target_ep_size
        for module in moe_modules:
            if int(module.moe_config.num_experts) != num_physical_experts:
                raise RuntimeError(
                    "MegaMoe restore requires the captured MoE physical capacity "
                    "to remain unchanged: "
                    f"captured={module.moe_config.num_experts}, "
                    f"target={num_physical_experts}"
                )

        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        assert model_state.physical_to_logical_map.shape[1] == num_physical_experts
        bootstrap_mapping = getattr(self, "_capture_bootstrap_mapping", None)
        if bootstrap_mapping is None:
            raise RuntimeError("MegaMoe scale-up cannot commit without a bootstrap expert mapping")
        if bootstrap_mapping.shape != model_state.physical_to_logical_map.shape:
            raise RuntimeError(
                "MegaMoe bootstrap mapping shape does not match restored capacity: "
                f"bootstrap={tuple(bootstrap_mapping.shape)}, "
                f"restored={tuple(model_state.physical_to_logical_map.shape)}"
            )

        # MegaMoe FT preserves the full physical width, including load history.
        # Restoring the same topology requires no storage expansion.
        assert model_state.expert_load_window.shape[-1] == num_physical_experts
        assert model_state.expert_load_pass.shape[1] == num_physical_experts

        num_logical_experts = model_state.logical_replica_count.shape[1]
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.eplb_config.num_redundant_experts = num_physical_experts - num_logical_experts
        # FT preserves capacity, weight storage and layer-state bindings.
        # Install the restored placement without re-registering those objects.

        # The new ranks received exact copies of their source ranks' expert
        # tensors during background preparation. Commit the matching mapping
        # only after the target groups become active, then refresh the
        # graph-stable Ascend routing tables in place.
        eplb_state.update_mapping(model_config, bootstrap_mapping)
        refresh_model_routing_tables(model_state)

        if self._prepared_eplb_communicator is None:
            raise RuntimeError("Standby EPLB communicator was not prepared")
        eplb_state.update_communicator(
            model_config,
            self._prepared_eplb_communicator,
        )
        self._prepared_eplb_communicator = None
        logger.info(
            "[Elastic EP] Restored MegaMoe model state without graph recapture: "
            "ep_size=%s, local_physical_experts=%s, "
            "physical_experts=%s",
            target_ep_size,
            num_local_experts,
            num_physical_experts,
        )

    def _switch_and_prepare_capture_restore_scale_up(self):
        """Commit target groups while preserving existing MegaMoe ACL graphs."""
        reconfig_request = self.reconfig_request
        if reconfig_request is None:
            raise RuntimeError("Missing Elastic EP reconfiguration request")

        retired_groups = _replace_active_groups(**pop_standby_groups())
        self._update_parallel_config_from_request()
        target_ep_size = get_ep_group().world_size
        self._restore_capture_scale_up_model_state(target_ep_size)
        return retired_groups

    def _resume_async_eplb_from_bootstrap(self) -> None:
        """Resume periodic async EPLB after installing a valid topology.

        Calling ``rearrange()`` here would still perform load aggregation and
        snapshot creation on the serving thread, even in async mode.  The
        bootstrap weights and mapping already form a correct target topology,
        so only restart the common EPLB cadence.  A later regular serving step
        will trigger the next rearrangement and hand its per-layer transfers to
        the async worker outside the Elastic EP pause window.
        """
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("MegaMoe Elastic EP scale-up requires EPLB state")
        eplb_state.expert_rearrangement_step = 0
        if eplb_state.is_async:
            eplb_state.start_async_loop()

    def commit_scale_up(self, is_existing_worker: bool) -> None:
        if getattr(self, "_mega_moe_precommit_capture", False):
            self._commit_mega_moe_restore(is_existing_worker)
            return

        if not is_existing_worker:
            # New workers already use the new upstream DP/EP groups and do not
            # run switch_and_prepare. Install the MC2 group they created in
            # prepare_new_worker before expert mapping initializes MoE comms.
            self._activate_ascend_standby_groups()
        super().commit_scale_up(is_existing_worker)
        sentinel = getattr(self.worker, "worker_sentinel", None)
        if sentinel is not None:
            sentinel.init_num_local_experts()

    def _update_parallel_config_from_request(self) -> None:
        request = self.reconfig_request
        if request is None:
            raise RuntimeError("Missing Elastic EP reconfiguration request")
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.data_parallel_size = request.new_data_parallel_size
        if request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = request.new_data_parallel_rank
        if request.new_data_parallel_rank_local != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = request.new_data_parallel_master_port
        parallel_config._data_parallel_master_port_list = request.new_data_parallel_master_port_list
        parallel_config._coord_store_port = request.coord_store_port
        parallel_config._data_parallel_master_rank = request.data_parallel_master_rank
        parallel_config._eep_existing_dp_ranks = request.existing_dp_ranks

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        # MoE setup constructs the AlltoAll dispatcher even on the MegaMoe path.
        # Its get_hccl_comm_name() lazily initializes the target EP group.
        # Pair that initialization on old and new ranks here, before old
        # ranks enter MC2 update_ctx while new ranks construct their buffers.
        # DP synchronization uses the CPU group and needs no device warmup.
        assert ep_group is not None
        self._materialize_hccl_group(ep_group, "EP")

    def prepare_new_worker(
        self,
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ) -> None:
        parallel_config = self.worker.vllm_config.parallel_config
        use_precommit_capture = bool(reconfig_request is not None and self._read_capture_decision(reconfig_request))
        with (
            _PATCH_LOCK,
            self._use_ascend_transfer_impl(
                include_expert_weights=use_precommit_capture,
            ),
        ):
            super().prepare_new_worker(reconfig_request)
        if not use_precommit_capture:
            # Existing workers create this group after their upstream
            # preparation. The new worker follows prepare_new_worker instead,
            # so it must join the same stateless MC2 creation here.
            create_ascend_standby_groups(
                new_dp_size=parallel_config.data_parallel_size,
                new_world_size_across_dp=(parallel_config.world_size * parallel_config.data_parallel_size),
                master_ip=parallel_config.data_parallel_master_ip,
                coord_store_port=parallel_config._coord_store_port,
            )
            return

        assert reconfig_request is not None

        # Receive the mapping without installing MoE communication methods yet:
        # get_hccl_comm_name must not run until every target MC2 rank enters the
        # materialization rendezvous below.
        mapping = super().receive_expert_mapping()
        self.worker.model_runner.setup_eplb_from_mapping(mapping)
        self._set_eplb_suppressed(True)
        self._capture_eplb_suppression_operation_id = reconfig_request.operation_id

        self._synchronize_capture_mapping_setup(reconfig_request)

        # New ranks install the final-size group early so graph capture can use
        # it before commit. The mapping rendezvous above ensures all target
        # ranks enter this stateless group only after device-side mapping setup.
        create_ascend_standby_groups(
            new_dp_size=parallel_config.data_parallel_size,
            new_world_size_across_dp=(parallel_config.world_size * parallel_config.data_parallel_size),
            master_ip=parallel_config.data_parallel_master_ip,
            coord_store_port=parallel_config._coord_store_port,
        )
        self._activate_ascend_standby_groups()

        self.materialize_new_communication_groups()

        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        if not moe_modules:
            raise RuntimeError("MegaMoe pre-commit capture requires at least one MoE layer")

        self._prepare_new_mega_moe_capture(moe_modules)

    def warmup_new_worker(self) -> None:
        if not getattr(self, "_mega_moe_precommit_capture", False):
            return
        # Remote ranks stay masked while new ranks capture without serving DP sync.
        with skip_dp_coordination():
            self.warm_and_capture()
        self._mega_moe_precommit_capture_done = True

    def _prepare_new_mega_moe_capture(self, moe_modules: list[nn.Module]) -> None:
        self._setup_moe_comm_and_quant_method()
        comm = self._get_fused_mc2_comm()
        scheme = moe_modules[0].routed_experts.quant_method.quant_method
        if scheme.quant_type != QuantType.W8A8:
            raise RuntimeError("MegaMoe restore capture currently requires W8A8")
        mode, dtype, _ = _get_cann_mega_moe_quant_settings(scheme.quant_type)
        comm.mega_moe_quant_type = scheme.quant_type
        comm.mega_moe_symm_buffer = comm._init_mega_moe_symm_buffer(
            mode, dtype, is_decode_only_node=_is_decode_only_node(self.worker.vllm_config)
        )
        group = get_mc2_group()
        comm.mega_moe_capture_state.prepare_capture(moe_modules[0].moe_config.experts_per_token)
        comm.mega_moe_symm_buffer.get_local_buffer_tensor(torch.uint8).zero_()
        self._synchronize_capture_context_rendezvous(group, "new MegaMoe")

    def _commit_mega_moe_restore(self, is_existing_worker: bool) -> None:
        completed = False
        try:
            comm = self._mega_moe_comm()
            if is_existing_worker:
                retired = self._switch_and_prepare_capture_restore_scale_up()
                retired_mc2 = self._activate_ascend_standby_groups()
                retired = (*retired, retired_mc2) if retired_mc2 is not None else retired
            else:
                if not getattr(self, "_mega_moe_precommit_capture_done", False):
                    raise RuntimeError("New-rank MegaMoe capture has not completed")
                retired = None

            group = get_mc2_group()
            state = comm.mega_moe_capture_state
            rank_map = ep_to_mc2_ranks(get_ep_group().ranks, group.ranks)
            state.set_rank_map(rank_map)
            # Keep the original FusedMC2CommImpl, weights and graphs alive.
            comm.token_dispatcher.refresh_hccl_group()
            get_ep_all2all_manager().bind_mega_moe_buffer(state.buffer, rank_map)
            state.buffer.get_local_buffer_tensor(torch.uint8).zero_()
            self._synchronize_capture_context_rendezvous(group, "MegaMoe clear flags")
            state.finish_capture()
            sentinel = getattr(self.worker, "worker_sentinel", None)
            if sentinel is not None:
                sentinel.init_num_local_experts()
            self._synchronize_capture_context_rendezvous(group, "MegaMoe commit")
            self._resume_async_eplb_from_bootstrap()
            self._set_eplb_suppressed(False)
            self._capture_eplb_suppression_operation_id = None
            self._capture_bootstrap_mapping = None
            if retired is not None:
                self._start_group_cleanup(retired)
            completed = True
            logger.info(
                "[Elastic EP] MegaMoe restore committed; existing graphs preserved, ep_size=%s", group.world_size
            )
        finally:
            if not completed:
                logger.warning("[Elastic EP] Keeping EPLB suppressed because MegaMoe restore did not complete")

    def warm_and_capture(self) -> None:
        # No need to save/clear/restore the KV-cache block tables like the
        # upstream warm_and_capture: the V2 runner's dummy attention uses
        # all-zero block tables (reserved null block) and PAD_SLOT_ID slot
        # mappings, so real KV-cache blocks are never written during the
        # dummy run.
        runner = self.worker.model_runner
        self._release_cuda_graphs()
        unlock_workspace()
        try:
            runner._dummy_run(
                runner.max_num_tokens,
                is_profile=True,
                skip_eplb=True,
            )
            self.worker.compile_or_warm_up_model()
        finally:
            lock_workspace()

    def _setup_moe_comm_and_quant_method(self) -> None:
        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)
