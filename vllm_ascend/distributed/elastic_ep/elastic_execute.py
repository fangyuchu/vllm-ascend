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
from vllm.platforms import current_platform
from vllm.utils import is_moe_layer
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    get_standby_mc2_group,
    get_standby_v3_capture_dp_group,
    pop_ascend_standby_groups,
    pop_standby_v3_capture_dp_group,
)
from vllm_ascend.distributed.elastic_ep.v3_capture import (
    V3CaptureDPSyncSession,
)
from vllm_ascend.distributed.eplb.state import refresh_model_routing_tables
from vllm_ascend.distributed.parallel_state import (
    _detach_ascend_active_groups,
    _replace_ascend_active_groups,
    get_mc2_group,
    get_v3_elastic_info,
    remap_v3_elastic_info,
    set_v3_elastic_info,
    set_v3_elastic_info_from_ep,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    get_moe_comm_method,
    setup_moe_comm_method,
)
from vllm_ascend.ops.fused_moe.moe_distribute_v3 import (
    update_moe_distribute_v3_contexts,
)

_PATCH_LOCK = threading.Lock()

_V3_BOOTSTRAP_EXPERT_PARAMETER_PREFIX = "__v3_bootstrap_expert__."


def _match_peer_parameters(
    parameter_names: list[str],
    parameters: list[torch.Tensor],
    peer_parameter_names: list[str],
) -> list[torch.Tensor]:
    if parameter_names == peer_parameter_names:
        return parameters
    parameters_by_name = dict(zip(parameter_names, parameters, strict=True))
    common_names = sorted(parameters_by_name.keys() & set(peer_parameter_names))
    return [parameters_by_name[name] for name in common_names]


@torch.inference_mode()
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
                        expert_params.append(tensor.data)
                        expert_param_ptrs.add(ptr)
        for index, param in enumerate(expert_params):
            all_params.append(param)
            all_params_name.append(f"{_V3_BOOTSTRAP_EXPERT_PARAMETER_PREFIX}{index:08d}")

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    local_expert_names = {name for name in all_params_name if name.startswith(_V3_BOOTSTRAP_EXPERT_PARAMETER_PREFIX)}
    peer_expert_names = {
        name for name in peer_rank_all_params_name if name.startswith(_V3_BOOTSTRAP_EXPERT_PARAMETER_PREFIX)
    }
    if local_expert_names != peer_expert_names:
        raise RuntimeError(
            "V3 bootstrap expert weights do not match between peers: "
            f"local={len(local_expert_names)}, peer={len(peer_expert_names)}"
        )

    all_params = _match_peer_parameters(
        all_params_name,
        all_params,
        peer_rank_all_params_name,
    )

    assert len(all_params) > 0
    transfer_stream = stream or torch.npu.current_stream()
    with torch.npu.stream(transfer_stream):
        p2p_ops = []
        for param in all_params:
            # HCCL P2P transfers flat memory and does not honor tensor strides.
            transfer_param = param.contiguous()
            op = object.__new__(P2POp)
            op.op = torch.distributed.isend if is_sender else torch.distributed.irecv
            op.tensor = transfer_param
            op.group_peer = peer_rank
            p2p_ops.append(op)
            if transfer_param is not param:
                device_comm.batch_isend_irecv(
                    p2p_ops,
                    stream=transfer_stream,
                )
                p2p_ops.clear()
                if not is_sender:
                    param.copy_(transfer_param)

        if p2p_ops:
            device_comm.batch_isend_irecv(
                p2p_ops,
                stream=transfer_stream,
            )


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    quant_method = getattr(module.routed_experts.quant_method, "quant_method", None)
    if hasattr(quant_method, "moe_all_to_all_group_name"):
        try:
            device_group = get_mc2_group().device_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    @staticmethod
    def _build_v3_mc2_rank_order(
        physical_ranks: list[int],
        active_dp_ranks: list[int],
        world_size_per_dp: int,
    ) -> list[int]:
        """Keep survivors in their captured MC2 slots; fill holes with new ranks."""
        if sorted(physical_ranks) != list(range(len(physical_ranks))):
            raise ValueError("V3 MC2 ranks must cover the physical world")
        dense_dp_ranks = {rank: dense for dense, rank in enumerate(active_dp_ranks)}
        new_ranks = iter(range(len(active_dp_ranks) * world_size_per_dp, len(physical_ranks)))
        result = []
        for rank in physical_ranks:
            dp_rank, local_rank = divmod(rank, world_size_per_dp)
            if dp_rank in dense_dp_ranks:
                result.append(dense_dp_ranks[dp_rank] * world_size_per_dp + local_rank)
            else:
                result.append(next(new_ranks))
        return result

    def _is_v3_scale_up_candidate(
        self,
        reconfig_request: ReconfigureDistributedRequest,
    ) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        return bool(
            envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            and reconfig_request.operation_id
            and reconfig_request.new_data_parallel_size > parallel_config.data_parallel_size
            and parallel_config.pipeline_parallel_size == 1
            and self.worker.vllm_config.lora_config is None
        )

    def start_async(self, execute_method: str, *args, **kwargs) -> str:
        """Arm EPLB suppression before background V3 preparation starts.

        ``start_async`` runs synchronously on the worker command loop between
        model executions. Setting the flag here ensures every old rank sees it
        before the next serving step can reach periodic EPLB rearrangement.
        """
        suppression_armed = False
        if (
            execute_method == "prepare_reconfiguration"
            and args
            and isinstance(args[0], ReconfigureDistributedRequest)
            and self._is_v3_scale_up_candidate(args[0])
        ):
            self._set_eplb_suppressed(True)
            self._v3_eplb_suppression_operation_id = args[0].operation_id
            suppression_armed = True
        try:
            return super().start_async(execute_method, *args, **kwargs)
        except BaseException:
            if suppression_armed:
                self._set_eplb_suppressed(False)
                self._v3_eplb_suppression_operation_id = None
            raise

    @staticmethod
    def _v3_capture_key(operation_id: str, name: str) -> str:
        return f"v3_capture/{operation_id}/bootstrap/{name}"

    @staticmethod
    def _scale_up_source_dp_rank(
        old_dp_size: int,
        new_dp_size: int,
        new_dp_rank: int,
    ) -> int:
        """Return the old DP rank whose weights initialize a new DP rank."""
        if not 0 <= new_dp_rank - old_dp_size < new_dp_size - old_dp_size:
            raise ValueError(
                "new_dp_rank must identify a newly added rank: "
                f"old_dp_size={old_dp_size}, new_dp_size={new_dp_size}, "
                f"new_dp_rank={new_dp_rank}"
            )
        num_new_workers = new_dp_size - old_dp_size
        new_worker_idx = new_dp_rank - old_dp_size
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size
        larger_sender_span = remainder * (num_dst_per_sender + 1)
        if new_worker_idx < larger_sender_span:
            return new_worker_idx // (num_dst_per_sender + 1)
        return remainder + (new_worker_idx - larger_sender_span) // num_dst_per_sender

    @classmethod
    def _build_v3_bootstrap_mapping(
        cls,
        physical_to_logical: torch.Tensor,
        num_local_physical_experts: int,
        old_dp_size: int,
        new_dp_size: int,
    ) -> torch.Tensor:
        """Build a target-size mapping matching background weight cloning.

        Every new DP worker is initialized from one existing DP worker by
        ``transfer_weights``. Copy the corresponding mapping chunks as well,
        so the target topology is immediately valid without modifying any
        serving expert weights. Normal async EPLB can optimize this safe
        bootstrap placement after the topology commits.
        """
        if old_dp_size <= 0 or new_dp_size <= old_dp_size:
            raise ValueError(
                f"V3 bootstrap mapping requires scale-up: old_dp_size={old_dp_size}, new_dp_size={new_dp_size}"
            )
        mapping_width = physical_to_logical.shape[1]
        if mapping_width % num_local_physical_experts != 0:
            raise RuntimeError(
                "Expert mapping is not aligned to local expert capacity: "
                f"mapping_width={mapping_width}, "
                f"num_local_experts={num_local_physical_experts}"
            )
        old_ep_size = mapping_width // num_local_physical_experts
        if old_ep_size % old_dp_size != 0:
            raise RuntimeError(
                "Expert ranks cannot be partitioned evenly across old DP ranks: "
                f"old_ep_size={old_ep_size}, old_dp_size={old_dp_size}"
            )
        ep_ranks_per_dp = old_ep_size // old_dp_size
        new_ep_size = new_dp_size * ep_ranks_per_dp
        bootstrap_mapping = torch.empty(
            (physical_to_logical.shape[0], new_ep_size * num_local_physical_experts),
            dtype=physical_to_logical.dtype,
            device=physical_to_logical.device,
        )
        bootstrap_mapping[:, :mapping_width].copy_(physical_to_logical)

        for new_dp_rank in range(old_dp_size, new_dp_size):
            source_dp_rank = cls._scale_up_source_dp_rank(
                old_dp_size,
                new_dp_size,
                new_dp_rank,
            )
            for ep_offset in range(ep_ranks_per_dp):
                source_ep_rank = source_dp_rank * ep_ranks_per_dp + ep_offset
                target_ep_rank = new_dp_rank * ep_ranks_per_dp + ep_offset
                source_begin = source_ep_rank * num_local_physical_experts
                target_begin = target_ep_rank * num_local_physical_experts
                bootstrap_mapping[:, target_begin : target_begin + num_local_physical_experts].copy_(
                    physical_to_logical[
                        :,
                        source_begin : source_begin + num_local_physical_experts,
                    ]
                )
        return bootstrap_mapping

    @staticmethod
    def _v3_capture_store(reconfig_request: ReconfigureDistributedRequest):
        return get_cached_tcp_store_client(
            reconfig_request.new_data_parallel_master_ip,
            reconfig_request.coord_store_port,
        )

    def _synchronize_v3_mapping_setup(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        *,
        is_existing_worker: bool,
    ) -> None:
        """Rendezvous after new ranks finish device-side EPLB mapping setup."""
        operation_id = reconfig_request.operation_id
        if not operation_id:
            raise RuntimeError("V3 mapping synchronization requires operation_id")

        old_dp_size = getattr(self, "_v3_old_dp_size", None)
        if old_dp_size is None:
            raise RuntimeError("V3 mapping synchronization requires old_dp_size")

        parallel_config = self.worker.vllm_config.parallel_config
        world_size_per_dp = parallel_config.world_size
        old_world_size = old_dp_size * world_size_per_dp
        new_world_size = reconfig_request.new_data_parallel_size * world_size_per_dp
        ready_keys = [
            self._v3_capture_key(operation_id, f"mapping/ready/{rank}")
            for rank in range(old_world_size, new_world_size)
        ]
        ack_keys = [self._v3_capture_key(operation_id, f"mapping/ack/{rank}") for rank in range(old_world_size)]
        if is_existing_worker:
            worker_global_rank = getattr(self, "_target_global_rank", None)
            if worker_global_rank is None:
                raise RuntimeError(
                    "Existing V3 mapping synchronization requires the target "
                    "global rank computed during standby-group preparation"
                )
        else:
            # A new Worker starts directly with its target DP rank, so its
            # current config already describes the target topology.
            worker_global_rank = parallel_config.data_parallel_rank * world_size_per_dp + self.worker.rank
        store = self._v3_capture_store(reconfig_request)

        if is_existing_worker:
            if worker_global_rank >= old_world_size:
                raise RuntimeError(
                    "Existing V3 worker rank is outside the old world: "
                    f"rank={worker_global_rank}, old_world_size={old_world_size}"
                )
            # Do not enter target HCCL initialization while any new rank is
            # still issuing device-side EPLB mapping allocations.
            store.wait(ready_keys)
            store.set(
                self._v3_capture_key(
                    operation_id,
                    f"mapping/ack/{worker_global_rank}",
                ),
                b"1",
            )
        else:
            if worker_global_rank < old_world_size:
                raise RuntimeError(
                    "New V3 worker rank is inside the old world: "
                    f"rank={worker_global_rank}, old_world_size={old_world_size}"
                )
            store.set(
                self._v3_capture_key(
                    operation_id,
                    f"mapping/ready/{worker_global_rank}",
                ),
                b"1",
            )

        # Every target rank enters MC2 creation only after every old Worker
        # acknowledged that all new-rank mappings are ready.
        store.wait(ack_keys)
        logger.info(
            "[Elastic EP] V3 mapping rendezvous completed: role=%s, rank=%s, old_world_size=%s, new_world_size=%s",
            "existing" if is_existing_worker else "new",
            worker_global_rank,
            old_world_size,
            new_world_size,
        )

    def _publish_v3_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
        old_dp_size: int,
    ) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        target_ep_size = (
            reconfig_request.new_data_parallel_size
            * parallel_config.tensor_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        elastic_info = get_v3_elastic_info()
        has_inactive_ranks = False
        if elastic_info is not None:
            elastic_info_cpu = elastic_info.detach().cpu()
            has_inactive_ranks = bool(elastic_info_cpu[0].item()) and (int(elastic_info_cpu[1].item()) < target_ep_size)
        enabled = bool(
            envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            and reconfig_request.operation_id
            and reconfig_request.new_data_parallel_size > old_dp_size
            and parallel_config.pipeline_parallel_size == 1
            and self.worker.vllm_config.lora_config is None
            and get_mc2_group().world_size == target_ep_size
            and has_inactive_ranks
        )
        store = self._v3_capture_store(reconfig_request)
        if enabled:
            dp_group = get_dp_group()
            dead_ranks = set(getattr(dp_group, "dead_dp_ranks", ()))
            active_ranks = [rank for rank in range(dp_group.world_size) if rank not in dead_ranks]
            self._v3_mc2_rank_order = self._build_v3_mc2_rank_order(
                get_mc2_group().ranks, active_ranks, parallel_config.world_size
            )
            store.set(
                self._v3_capture_key(reconfig_request.operation_id, "mc2_rank_order"),
                ",".join(map(str, self._v3_mc2_rank_order)).encode(),
            )
        store.set(
            self._v3_capture_key(reconfig_request.operation_id, "enabled"),
            b"1" if enabled else b"0",
        )
        store.set(
            self._v3_capture_key(reconfig_request.operation_id, "old_dp_size"),
            str(old_dp_size).encode(),
        )
        self._v3_precommit_capture = enabled
        self._v3_old_dp_size = old_dp_size
        return enabled

    def _read_v3_capture_decision(
        self,
        reconfig_request: ReconfigureDistributedRequest,
    ) -> bool:
        if not reconfig_request.operation_id:
            self._v3_precommit_capture = False
            return False
        store = self._v3_capture_store(reconfig_request)
        enabled = store.get(self._v3_capture_key(reconfig_request.operation_id, "enabled")) == b"1"
        self._v3_precommit_capture = enabled
        if enabled:
            self._v3_mc2_rank_order = [
                int(rank)
                for rank in store.get(self._v3_capture_key(reconfig_request.operation_id, "mc2_rank_order"))
                .decode()
                .split(",")
            ]
        self._v3_old_dp_size = int(
            store.get(
                self._v3_capture_key(
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
            include_expert_weights = bool(getattr(self, "_v3_precommit_capture", False))
        self._weight_transfer_stream = torch.npu.Stream()
        upstream_elastic_execute.batch_transfer_weights = partial(
            ascend_batch_transfer_weights,
            stream=self._weight_transfer_stream,
            include_expert_weights=include_expert_weights,
        )
        try:
            yield
        finally:
            # Also drain the stream on exceptional exits before temporary
            # contiguous transfer tensors leave scope.
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
            getattr(self, "_v3_precommit_capture", False)
            and request is not None
            and request.operation_id == operation_id
        )

    def _make_v3_capture_session(
        self,
        operation_id: str,
    ) -> V3CaptureDPSyncSession:
        group = get_standby_v3_capture_dp_group()
        if group is None:
            raise RuntimeError("V3 capture DP group is not initialized")
        return V3CaptureDPSyncSession(
            group=group,
            operation_id=operation_id,
            old_dp_size=self._v3_old_dp_size,
        )

    def run_new_rank_capture_companion(self, operation_id: str) -> None:
        session = self._make_v3_capture_session(operation_id)
        steps = session.run_existing_rank_companion()
        self._v3_capture_companion_done = True
        if session.group.rank_in_group == 0:
            print(
                f"[Elastic EP] V3 capture companion completed: operation={operation_id}, steps={steps}",
                flush=True,
            )

    def capture_new_rank_graphs(self, operation_id: str) -> None:
        session = self._make_v3_capture_session(operation_id)
        try:
            self.warmup_local_kernels()
            with session.activate_for_capture():
                self.warm_and_capture()
        except BaseException as error:
            session.mark_capture_failed(error)
            raise
        session.mark_capture_done()
        self._v3_precommit_capture_done = True

    def _set_v3_target_elastic_info(self, elastic_info: torch.Tensor, *, allow_shape_change: bool = False) -> None:
        set_v3_elastic_info(
            remap_v3_elastic_info(elastic_info, self._v3_mc2_rank_order),
            allow_shape_change=allow_shape_change,
        )

    def _set_v3_identity_elastic_info(self) -> None:
        current = get_v3_elastic_info()
        if current is None:
            raise RuntimeError("V3 elastic_info is not initialized")
        world_size = get_mc2_group().world_size
        if current.numel() != 4 + 2 * world_size:
            raise RuntimeError(
                "V3 elastic_info shape does not match the committed MC2 group: "
                f"numel={current.numel()}, mc2_size={world_size}"
            )
        rank_table = torch.arange(
            world_size,
            dtype=torch.int32,
            device=current.device,
        )
        moe_module = next(
            (module for module in self.worker.get_model().modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        moe_expert_num = moe_module.moe_config.num_experts
        self._set_v3_target_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [0, world_size, 0, moe_expert_num],
                        dtype=torch.int32,
                        device=current.device,
                    ),
                    rank_table,
                    rank_table,
                )
            ).contiguous()
        )

    def _set_v3_old_active_elastic_info_for_target_mc2(
        self,
        mc2_group: StatelessGroupCoordinator,
    ) -> None:
        """Remap existing ranks into the target MC2 rank space.

        A fault-recovery scale-down preserves the original physical MC2
        communicator, so its elastic_info may contain holes for dead ranks.
        The target EP group assigns survivors dense ranks, while MC2 keeps
        their captured physical slots. Translate the dense survivor mask back
        into those slots before serving resumes with the target context.
        """
        current = get_v3_elastic_info()
        if current is None:
            raise RuntimeError("V3 elastic_info is not initialized")
        reconfig_request = self.reconfig_request
        old_dp_size = getattr(self, "_v3_old_dp_size", None)
        if reconfig_request is None or old_dp_size is None:
            raise RuntimeError("V3 target MC2 remapping requires the active reconfiguration request and old DP size")

        new_dp_size = reconfig_request.new_data_parallel_size
        target_ep_size = mc2_group.world_size
        if target_ep_size % new_dp_size != 0:
            raise RuntimeError(
                f"Target MC2 size must divide evenly by target DP size: mc2={target_ep_size}, dp={new_dp_size}"
            )
        if current.numel() != 4 + 2 * target_ep_size:
            raise RuntimeError(
                "V3 elastic_info shape does not match the target MC2 group: "
                f"numel={current.numel()}, mc2_size={target_ep_size}"
            )

        ep_ranks_per_dp = target_ep_size // new_dp_size
        old_active_ep_size = old_dp_size * ep_ranks_per_dp
        if not 0 < old_active_ep_size < target_ep_size:
            raise RuntimeError(
                "Invalid old-active V3 topology for target MC2: "
                f"active_ep_size={old_active_ep_size}, "
                f"target_ep_size={target_ep_size}"
            )

        moe_module = next(
            (module for module in self.worker.get_model().modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = int(moe_module.moe_config.num_local_experts)

        orig_to_dense = torch.full(
            (target_ep_size,),
            -1,
            dtype=torch.int32,
            device=current.device,
        )
        dense_to_orig = torch.full_like(orig_to_dense, -1)
        active_ranks = torch.arange(
            old_active_ep_size,
            dtype=torch.int32,
            device=current.device,
        )
        orig_to_dense[:old_active_ep_size] = active_ranks
        dense_to_orig[:old_active_ep_size] = active_ranks
        self._set_v3_target_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [
                            1,
                            old_active_ep_size,
                            0,
                            old_active_ep_size * num_local_experts,
                        ],
                        dtype=torch.int32,
                        device=current.device,
                    ),
                    orig_to_dense,
                    dense_to_orig,
                )
            ).contiguous()
        )
        logger.info(
            "[Elastic EP] Installed old-active V3 topology for target MC2: rank=%s/%s, active_ep_size=%s",
            mc2_group.rank_in_group,
            target_ep_size,
            old_active_ep_size,
        )

    @staticmethod
    def _current_v3_dispatchers() -> list:
        dispatchers = []
        for comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2):
            comm_method = get_moe_comm_method(comm_type)
            if comm_method is None:
                continue
            dispatcher = comm_method.token_dispatcher
            if getattr(dispatcher, "v3_adapter", None) is not None:
                dispatchers.append(dispatcher)
        return dispatchers

    def _cleanup_v3_capture_group(self) -> None:
        group = pop_standby_v3_capture_dp_group()
        if group is not None:
            group.destroy()

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
        """Materialize the target MC2 communicator before V3 graph capture."""
        mc2_group = get_standby_mc2_group()
        if mc2_group is None:
            mc2_group = get_mc2_group()
        self._materialize_hccl_group(mc2_group, "MC2")

    def _set_eplb_suppressed(self, suppressed: bool) -> None:
        """Pause the complete EPLB controller during V3 reconfiguration."""
        self.worker.model_runner.eep_eplb_suppressed = suppressed
        ep_group = get_ep_group()
        if ep_group.rank_in_group == 0:
            logger.info(
                "[Elastic EP] EPLB %s V3 elastic scaling transition",
                "disabled during" if suppressed else "re-enabled after",
            )

    def _drain_async_eplb(self) -> None:
        """Finish an already-started async EPLB cycle before V3 collectives.

        ``eep_eplb_suppressed`` prevents the serving loop from starting another
        EPLB cycle, but a cycle that was scheduled just before suppression may
        still be transferring expert weights. All existing ranks call this
        method only after serving collectives have been drained, so the EPLB
        group's coordinated stop cannot race with MC2 ``update_ctx``.
        """
        eplb_state = self.worker.model_runner.eplb_state
        if eplb_state is not None:
            eplb_state.drain_async()

    @staticmethod
    def _synchronize_v3_context_rendezvous(
        mc2_group: StatelessGroupCoordinator,
        role: str,
    ) -> None:
        """Confirm every target rank completed V3 context creation/update."""
        torch.npu.synchronize()
        torch.distributed.barrier(group=mc2_group.cpu_group)
        logger.info(
            "[Elastic EP] V3 context rendezvous completed: role=%s, rank=%s/%s",
            role,
            mc2_group.rank_in_group,
            mc2_group.world_size,
        )

    def _update_existing_v3_contexts(self) -> None:
        """Pair old-rank ``update_ctx`` with new-rank buffer construction."""
        mc2_group = get_standby_mc2_group()
        if mc2_group is None:
            raise RuntimeError("Target MC2 group is unavailable for V3 context update")
        logger.info(
            "[Elastic EP] Updating existing-rank V3 context: rank=%s/%s",
            mc2_group.rank_in_group,
            mc2_group.world_size,
        )
        self._set_v3_old_active_elastic_info_for_target_mc2(mc2_group)
        updated = update_moe_distribute_v3_contexts(mc2_group)
        if updated != 1:
            raise RuntimeError("Existing rank did not update its MoeDistribute V3 context")
        self._synchronize_v3_context_rendezvous(mc2_group, "existing")

    def broadcast_expert_mapping(self) -> None:
        if not getattr(self, "_v3_precommit_capture", False):
            super().broadcast_expert_mapping()
            return

        standby_dp_group = get_standby_dp_group()
        if standby_dp_group is None:
            raise RuntimeError("Standby DP group is not initialized")

        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("V3 pre-commit capture requires EPLB state")

        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        physical_to_logical = model_state.physical_to_logical_map
        moe_module = next(
            (module for module in self.worker.get_model().modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 pre-commit capture requires at least one MoE layer")

        # Graph-preserving FT scale-down keeps the original physical mapping
        # width, including dead-rank chunks, while serving continues with a
        # dense survivor topology. Compact only the temporary bootstrap input;
        # mutating model_state here would race with ongoing inference.
        num_local_physical_experts = int(moe_module.moe_config.num_local_experts)
        if num_local_physical_experts <= 0:
            raise ValueError(f"num_local_physical_experts must be positive: {num_local_physical_experts}")
        if physical_to_logical.shape[1] % num_local_physical_experts != 0:
            raise RuntimeError(
                "Active expert mapping is not aligned to local expert capacity: "
                f"mapping_width={physical_to_logical.shape[1]}, "
                f"num_local_experts={num_local_physical_experts}"
            )
        num_logical_experts = model_state.logical_replica_count.shape[1]
        reconfig_request = self.reconfig_request
        old_dp_size = getattr(self, "_v3_old_dp_size", None)
        if reconfig_request is None or old_dp_size is None:
            raise RuntimeError("V3 bootstrap mapping requires the active reconfiguration request and old DP size")
        active_dp_group = get_dp_group()
        dead_dp_ranks = set(getattr(active_dp_group, "dead_dp_ranks", ()))
        compacted_dp_size = upstream_elastic_execute.get_active_dp_size(active_dp_group)
        physical_ep_size = physical_to_logical.shape[1] // num_local_physical_experts
        if physical_ep_size % active_dp_group.world_size != 0:
            raise RuntimeError(
                "Physical expert ranks cannot be partitioned evenly across "
                "the graph-preserved DP group: "
                f"physical_ep_size={physical_ep_size}, "
                f"physical_dp_size={active_dp_group.world_size}"
            )
        if compacted_dp_size != old_dp_size:
            raise RuntimeError(
                "V3 bootstrap mapping active DP size mismatch: "
                f"compacted_dp_size={compacted_dp_size}, "
                f"old_dp_size={old_dp_size}, "
                f"physical_dp_size={active_dp_group.world_size}, "
                f"dead_dp_ranks={sorted(dead_dp_ranks)}"
            )
        active_mapping = upstream_elastic_execute.compact_active_expert_tensor(physical_to_logical, active_dp_group)
        if dead_dp_ranks:
            logger.info(
                "[Elastic EP] Compacted V3 bootstrap mapping from physical "
                "DP size %d to active DP size %d; dead_dp_ranks=%s",
                active_dp_group.world_size,
                compacted_dp_size,
                sorted(dead_dp_ranks),
            )
        bootstrap_mapping = self._build_v3_bootstrap_mapping(
            active_mapping,
            num_local_physical_experts,
            old_dp_size,
            reconfig_request.new_data_parallel_size,
        )
        self._v3_bootstrap_mapping = bootstrap_mapping
        upstream_elastic_execute.broadcast_expert_mapping(
            physical_to_logical=bootstrap_mapping,
            num_local_physical_experts=num_local_physical_experts,
            num_logical_experts=num_logical_experts,
            dp_group=standby_dp_group,
            src_rank=0,
            device=self.worker.device,
        )

    def prepare_reconfiguration(self, reconfig_request: ReconfigureDistributedRequest, use_all2all: bool) -> None:
        # Let upstream prepare world / DP / EP / EPLB and weight transfer, then
        # add the Ascend-specific MC2 standby group used at commit time.
        old_dp_size = upstream_elastic_execute.get_active_dp_size(get_dp_group())
        suppression_prearmed = getattr(self, "_v3_eplb_suppression_operation_id", None) == reconfig_request.operation_id
        use_v3_capture = self._publish_v3_capture_decision(
            reconfig_request,
            old_dp_size,
        )
        if use_v3_capture and not suppression_prearmed:
            # Normal inference continues throughout pre-commit preparation.
            # Stop the complete EPLB controller so no load collective, async
            # commit, or weight rearrangement can overlap the target V3/MC2
            # context update.
            self._set_eplb_suppressed(True)
            self._v3_eplb_suppression_operation_id = reconfig_request.operation_id
        elif not use_v3_capture and suppression_prearmed:
            self._set_eplb_suppressed(False)
            self._v3_eplb_suppression_operation_id = None
        super().prepare_reconfiguration(reconfig_request, use_all2all)
        if use_v3_capture:
            # Standby groups and weight transfer are safe on the dedicated
            # preparation thread. Mapping broadcast, lazy MC2 creation, and
            # MoeDistributeBuffer.update_ctx issue target-topology NPU
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
            create_v3_capture_dp=False,
            new_global_rank=self._target_global_rank,
        )

    def finalize_precommit_prepare(self, operation_id: str) -> None:
        """Finalize V3 target collectives while old-rank inference is drained."""
        reconfig_request = self.reconfig_request
        if reconfig_request is None:
            raise RuntimeError("Missing V3 pre-commit reconfiguration request")
        if reconfig_request.operation_id != operation_id:
            raise RuntimeError(
                f"V3 pre-commit operation mismatch: expected={reconfig_request.operation_id}, got={operation_id}"
            )
        if not getattr(self, "_v3_precommit_capture", False):
            raise RuntimeError("V3 pre-commit finalization was not prepared")

        self._drain_async_eplb()
        # Pair with receive_expert_mapping on the independently launched
        # ranks. This broadcast must not overlap the active serving
        # metadata all-reduce on only a subset of old ranks.
        self.broadcast_expert_mapping()
        self._synchronize_v3_mapping_setup(
            reconfig_request,
            is_existing_worker=True,
        )
        create_ascend_standby_groups(
            new_dp_size=reconfig_request.new_data_parallel_size,
            new_world_size_across_dp=(
                self.worker.vllm_config.parallel_config.world_size * reconfig_request.new_data_parallel_size
            ),
            master_ip=reconfig_request.new_data_parallel_master_ip,
            coord_store_port=reconfig_request.coord_store_port,
            create_v3_capture_dp=True,
            new_global_rank=self._target_global_rank,
            mc2_rank_order=self._v3_mc2_rank_order,
        )
        # Lazy HCCL construction and V3 context update are both
        # target-group collectives. Keep them in the same drained phase so
        # old and new ranks enter them in one deterministic order.
        self.materialize_new_communication_groups()
        self._update_existing_v3_contexts()

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
        parallel_config = self.worker.vllm_config.parallel_config
        self.worker.model_runner.dp_size = parallel_config.data_parallel_size
        self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank
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
    def _restore_v3_tensor_capacity(
        tensor: torch.Tensor,
        target_size: int,
        tensor_name: str,
    ) -> torch.Tensor:
        """Restore a scale-down view without changing its storage address."""
        current_size = tensor.shape[-1]
        if current_size == target_size:
            return tensor
        if current_size > target_size:
            raise RuntimeError(
                f"Cannot restore {tensor_name} to a smaller capacity: current={current_size}, target={target_size}"
            )

        target_shape = (*tensor.shape[:-1], target_size)
        strides = tensor.stride()
        if any(stride < 0 for stride in strides):
            raise RuntimeError(f"Cannot restore {tensor_name} with negative strides: {strides}")
        final_storage_index = tensor.storage_offset()
        for size, stride in zip(target_shape, strides, strict=True):
            if size > 0:
                final_storage_index += (size - 1) * stride
        storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
        if final_storage_index >= storage_elements:
            raise RuntimeError(
                f"V3 graph-preserving restore cannot expand {tensor_name} "
                "without reallocating its captured storage: "
                f"current={current_size}, target={target_size}, "
                f"storage_elements={storage_elements}"
            )

        restored = tensor.as_strided(
            target_shape,
            strides,
            tensor.storage_offset(),
        )
        restored[..., current_size:].zero_()
        return restored

    @staticmethod
    def _get_v3_num_local_experts(moe_modules: list[nn.Module]) -> int:
        if not moe_modules:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = int(moe_modules[0].moe_config.num_local_experts)
        if num_local_experts <= 0:
            raise RuntimeError(f"V3 restore requires a positive local expert capacity, got {num_local_experts}")
        if any(int(module.moe_config.num_local_experts) != num_local_experts for module in moe_modules[1:]):
            raise RuntimeError(
                "V3 restore requires every MoE layer to preserve the same number of local physical expert slots"
            )
        return num_local_experts

    def _restore_v3_scale_up_model_state(self, target_ep_size: int) -> None:
        """Restore the original physical EPLB capacity on existing ranks."""
        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("V3 Elastic EP scale-up requires EPLB state")

        model = model_runner.get_model()
        moe_modules = [module for module in model.modules() if is_moe_layer(module)]
        num_local_experts = self._get_v3_num_local_experts(moe_modules)
        num_physical_experts = num_local_experts * target_ep_size
        for module in moe_modules:
            if int(module.moe_config.num_experts) != num_physical_experts:
                raise RuntimeError(
                    "V3 restore requires the captured MoE physical capacity "
                    "to remain unchanged: "
                    f"captured={module.moe_config.num_experts}, "
                    f"target={num_physical_experts}"
                )

        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        physical_to_logical = model_state.physical_to_logical_map
        current_num_physical_experts = physical_to_logical.shape[1]
        if current_num_physical_experts > num_physical_experts:
            raise RuntimeError(
                "V3 restore target is smaller than the active expert map: "
                f"current={current_num_physical_experts}, "
                f"target={num_physical_experts}"
            )
        if current_num_physical_experts < num_physical_experts:
            expanded_physical_to_logical = torch.full(
                (physical_to_logical.shape[0], num_physical_experts),
                -1,
                dtype=physical_to_logical.dtype,
                device=physical_to_logical.device,
            )
            expanded_physical_to_logical[:, :current_num_physical_experts].copy_(physical_to_logical)
            model_state.physical_to_logical_map = expanded_physical_to_logical

        bootstrap_mapping = getattr(self, "_v3_bootstrap_mapping", None)
        if bootstrap_mapping is None:
            raise RuntimeError("V3 scale-up cannot commit without a bootstrap expert mapping")
        if bootstrap_mapping.shape != model_state.physical_to_logical_map.shape:
            raise RuntimeError(
                "V3 bootstrap mapping shape does not match restored capacity: "
                f"bootstrap={tuple(bootstrap_mapping.shape)}, "
                f"restored={tuple(model_state.physical_to_logical_map.shape)}"
            )

        # Scale-down keeps slices of the original full-capacity load tensors.
        # Recover full views of that storage so existing ACL graphs keep the
        # same captured addresses while EPLB regains the target topology.
        model_state.expert_load_pass = self._restore_v3_tensor_capacity(
            model_state.expert_load_pass,
            num_physical_experts,
            "expert_load_pass",
        )
        model_state.expert_load_window = self._restore_v3_tensor_capacity(
            model_state.expert_load_window,
            num_physical_experts,
            "expert_load_window",
        )

        num_logical_experts = model_state.logical_replica_count.shape[1]
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.eplb_config.num_redundant_experts = num_physical_experts - num_logical_experts
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                model_state.expert_load_pass,
                model_state.logical_to_physical_map,
                model_state.logical_replica_count,
            )
            eplb_state._propagate_shared_tensors(
                model,
                model_state.num_unpadded_tokens_tensors,
            )
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )

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
        # The preserved graph still references the live quant methods. The
        # staged replacements are for the generic graph-recapture path.
        self._staged_moe_quant_methods.clear()
        logger.info(
            "[Elastic EP] Restored V3 model state without graph recapture: "
            "ep_size=%s, local_physical_experts=%s, "
            "physical_experts=%s, previously_active_experts=%s",
            target_ep_size,
            num_local_experts,
            num_physical_experts,
            current_num_physical_experts,
        )

    def _switch_and_prepare_v3_restore_scale_up(self):
        """Commit target groups while preserving existing V3 ACL graphs."""
        reconfig_request = self.reconfig_request
        if reconfig_request is None:
            raise RuntimeError("Missing Elastic EP reconfiguration request")

        retired_groups = _replace_active_groups(**pop_standby_groups())
        self._update_parallel_config_from_request()
        target_ep_size = get_ep_group().world_size
        self._restore_v3_scale_up_model_state(target_ep_size)
        return retired_groups

    def _activate_v3_identity_context(self) -> None:
        """Activate all EP ranks through the graph-preserving MC2 permutation."""
        self._set_v3_identity_elastic_info()
        for dispatcher in self._current_v3_dispatchers():
            dispatcher.v3_adapter.finish_capture()
        mc2_group = get_mc2_group()
        updated = update_moe_distribute_v3_contexts(mc2_group)
        if updated != 1:
            raise RuntimeError("Committed rank did not activate its MoeDistribute V3 context")
        self._synchronize_v3_context_rendezvous(mc2_group, "commit")

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
            raise RuntimeError("V3 Elastic EP scale-up requires EPLB state")
        if not eplb_state.is_async:
            raise RuntimeError("V3 graph-preserving scale-up requires asynchronous EPLB")
        eplb_state.expert_rearrangement_step = 0
        eplb_state.start_async_loop()

    def commit_scale_up(self, is_existing_worker: bool) -> None:
        if getattr(self, "_v3_precommit_capture", False):
            commit_completed = False
            try:
                if is_existing_worker:
                    if not getattr(self, "_v3_capture_companion_done", False):
                        raise RuntimeError("V3 capture companion has not completed")
                    captured_dispatchers = self._current_v3_dispatchers()
                    retired_groups = self._switch_and_prepare_v3_restore_scale_up()
                    retired_mc2 = self._activate_ascend_standby_groups()
                    if retired_mc2 is not None:
                        retired_groups = (*retired_groups, retired_mc2)
                    for dispatcher in captured_dispatchers:
                        dispatcher.refresh_hccl_group()
                else:
                    if not getattr(self, "_v3_precommit_capture_done", False):
                        raise RuntimeError("New-rank V3 graph capture has not completed")
                    retired_groups = None

                # Existing buffers currently carry the old-active mask while
                # new buffers carry the capture-only new-rank mask. update_ctx
                # is a collective, so all committed ranks switch together.
                sentinel = getattr(self.worker, "worker_sentinel", None)
                if sentinel is not None:
                    sentinel.init_num_local_experts()
                self._activate_v3_identity_context()

                # The bootstrap mapping already matches the expert weights
                # cloned to every new rank during background preparation, so
                # inference can resume without an in-place reshard. Do not call
                # rearrange() in this pause window: even async rearrange still
                # aggregates loads and creates its snapshot on this thread.
                # Restart the common cadence and let a regular serving step
                # schedule the next load-optimized asynchronous placement.
                self._resume_async_eplb_from_bootstrap()
                self._set_eplb_suppressed(False)
                self._v3_eplb_suppression_operation_id = None
                self._v3_bootstrap_mapping = None
                commit_completed = True
                if retired_groups is not None:
                    self._start_group_cleanup(retired_groups)
                self._cleanup_v3_capture_group()
                return
            finally:
                if not commit_completed:
                    logger.warning("[Elastic EP] Keeping EPLB suppressed because V3 scale-up commit did not complete")

        if not is_existing_worker:
            # New workers already use the new upstream DP/EP groups and do not
            # run switch_and_prepare. Install the MC2 group they created in
            # prepare_new_worker before expert mapping initializes MoE comms.
            self._activate_ascend_standby_groups()
        super().commit_scale_up(is_existing_worker)
        sentinel = getattr(self.worker, "worker_sentinel", None)
        if sentinel is not None:
            sentinel.init_num_local_experts()

    def _can_preserve_v3_scale_down(self, new_dp_size: int) -> bool:
        parallel_config = self.worker.vllm_config.parallel_config
        eplb_state = self.worker.model_runner.eplb_state
        old_dp_size = parallel_config.data_parallel_size
        physical_ep_size = get_mc2_group().world_size
        expected_physical_ep_size = (
            old_dp_size * parallel_config.tensor_parallel_size * parallel_config.prefill_context_parallel_size
        )
        elastic_info = get_v3_elastic_info()
        return bool(
            envs_ascend.VLLM_ASCEND_ENABLE_MOE_DISTRIBUTE_V3
            and 0 < new_dp_size < old_dp_size
            and parallel_config.tensor_parallel_size == 1
            and parallel_config.pipeline_parallel_size == 1
            and parallel_config.prefill_context_parallel_size == 1
            and self.worker.vllm_config.lora_config is None
            and physical_ep_size == expected_physical_ep_size
            and elastic_info is not None
            and elastic_info.numel() == 4 + 2 * physical_ep_size
            # EPLB collectives use the independent live-rank EPLB group. The
            # larger physical EP/MC2 group may therefore remain captured while
            # async EPLB continues on the surviving ranks.
            and eplb_state is not None
            and self._current_v3_dispatchers()
        )

    def _set_v3_active_elastic_info(self, new_dp_size: int) -> None:
        current = get_v3_elastic_info()
        if current is None:
            raise RuntimeError("V3 elastic_info is not initialized")
        physical_ep_size = get_mc2_group().world_size
        active_ep_size = new_dp_size
        if not 0 < active_ep_size < physical_ep_size:
            raise RuntimeError(
                f"Invalid V3 scale-down size: active_ep_size={active_ep_size}, physical_ep_size={physical_ep_size}"
            )
        moe_module = next(
            (module for module in self.worker.model_runner.get_model().modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = moe_module.moe_config.num_local_experts
        orig_to_dense = torch.full(
            (physical_ep_size,),
            -1,
            dtype=torch.int32,
            device=current.device,
        )
        dense_to_orig = torch.full_like(orig_to_dense, -1)
        active_ranks = torch.arange(
            active_ep_size,
            dtype=torch.int32,
            device=current.device,
        )
        orig_to_dense[:active_ep_size] = active_ranks
        dense_to_orig[:active_ep_size] = active_ranks
        set_v3_elastic_info_from_ep(
            torch.cat(
                (
                    torch.tensor(
                        [
                            1,
                            active_ep_size,
                            0,
                            active_ep_size * num_local_experts,
                        ],
                        dtype=torch.int32,
                        device=current.device,
                    ),
                    orig_to_dense,
                    dense_to_orig,
                )
            ).contiguous()
        )

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
        self.worker.model_runner.dp_size = request.new_data_parallel_size
        self.worker.model_runner.dp_rank = parallel_config.data_parallel_rank

    def _commit_v3_scale_down_model_state(self, new_dp_size: int) -> None:
        model_runner = self.worker.model_runner
        eplb_state = model_runner.eplb_state
        if eplb_state is None:
            raise RuntimeError("V3 Elastic EP scale-down requires EPLB state")
        model = model_runner.get_model()
        model_config = model_runner.model_config
        model_state = eplb_state.model_states[model_config.compute_hash()]
        moe_module = next(
            (module for module in model.modules() if is_moe_layer(module)),
            None,
        )
        if moe_module is None:
            raise RuntimeError("V3 Elastic EP requires at least one MoE layer")
        num_local_experts = moe_module.moe_config.num_local_experts
        num_physical_experts = new_dp_size * num_local_experts
        num_logical_experts = model_state.logical_replica_count.shape[1]
        if num_physical_experts < num_logical_experts:
            raise RuntimeError(
                "Scale-down leaves fewer physical than logical experts: "
                f"physical={num_physical_experts}, logical={num_logical_experts}"
            )

        model_state.expert_load_pass = model_state.expert_load_pass[:, :num_physical_experts]
        model_state.expert_load_window = model_state.expert_load_window[:, :, :num_physical_experts]
        parallel_config = self.worker.vllm_config.parallel_config
        parallel_config.eplb_config.num_redundant_experts = num_physical_experts - num_logical_experts
        model.expert_weights = []
        with set_current_vllm_config(self.worker.vllm_config):
            model.set_eplb_state(
                model_state.expert_load_pass,
                model_state.logical_to_physical_map,
                model_state.logical_replica_count,
            )
            eplb_state._propagate_shared_tensors(
                model,
                model_state.num_unpadded_tokens_tensors,
            )
            model.update_physical_experts_metadata(
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_local_experts,
            )

        if self._prepared_eplb_communicator is None:
            raise RuntimeError("Standby EPLB communicator was not prepared")
        eplb_state.update_communicator(
            model_config,
            self._prepared_eplb_communicator,
        )
        self._prepared_eplb_communicator = None
        self._staged_moe_quant_methods.clear()

        for module in model.modules():
            if not is_moe_layer(module):
                continue
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()

    def _switch_v3_scale_down_survivor(self, new_dp_size: int) -> None:
        standby_groups = pop_standby_groups()
        unused_standby_ep = standby_groups["ep"]
        standby_groups["ep"] = get_ep_group()
        retired_groups = list(_replace_active_groups(**standby_groups))
        # Keep the original physical EP group. The newly-created, smaller EP
        # group is unused by the captured V3 graph and can be retired instead.
        retired_groups[1] = unused_standby_ep

        standby_mc2 = pop_ascend_standby_groups()["mc2"]
        self._update_parallel_config_from_request()
        self._commit_v3_scale_down_model_state(new_dp_size)
        self._set_v3_active_elastic_info(new_dp_size)
        for dispatcher in self._current_v3_dispatchers():
            dispatcher.refresh_hccl_group()
        self._start_group_cleanup(tuple(retired_groups) + (standby_mc2,))
        logger.info(
            "[Elastic EP] V3 scale-down preserved physical EP/MC2 and NPU graphs: active_dp=%d, physical_ep=%d",
            new_dp_size,
            get_mc2_group().world_size,
        )

    def _switch_v3_scale_down_removed_rank(self) -> None:
        retired_groups = _replace_active_groups(
            world=None,
            dp=None,
            ep=None,
            eplb=None,
            node_count=None,
        )
        _detach_ascend_active_groups()
        # Match survivor cleanup order for the old groups, but intentionally
        # keep the physical EP/MC2 communicators alive on survivor ranks.
        self._destroy_retired_groups(
            (
                retired_groups[0],
                None,
                retired_groups[2],
                retired_groups[3],
            )
        )

    def commit_scale_down(self, new_dp_size: int, removing: bool) -> None:
        if not self._can_preserve_v3_scale_down(new_dp_size):
            super().commit_scale_down(new_dp_size, removing)
            return

        self.perform_scale_down_eplb_reshuffle(new_dp_size)
        if removing:
            self._switch_v3_scale_down_removed_rank()
        else:
            self._switch_v3_scale_down_survivor(new_dp_size)

    def receive_expert_mapping(self) -> torch.Tensor:
        mapping = super().receive_expert_mapping()
        self._setup_moe_comm_and_quant_method()
        return mapping

    def _warm_target_groups(self, dp_group, ep_group) -> None:
        # Ascend uses CPU DP sync, Gloo EPLB and a separate MC2 group.
        pass

    def prepare_new_worker(
        self,
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ) -> str | None:
        parallel_config = self.worker.vllm_config.parallel_config
        if reconfig_request is None:
            coord_store = get_cached_tcp_store_client(
                parallel_config.data_parallel_master_ip,
                parallel_config._coord_store_port,
            )
            current_epoch_key = "elastic_ep/external/current_epoch"
            if coord_store.check([current_epoch_key]):
                operation_id = coord_store.get(current_epoch_key).decode()
                reconfig_request = ReconfigureDistributedRequest(
                    new_data_parallel_size=parallel_config.data_parallel_size,
                    new_data_parallel_rank=parallel_config.data_parallel_rank,
                    new_data_parallel_rank_local=(parallel_config.data_parallel_rank_local),
                    new_data_parallel_master_ip=(parallel_config.data_parallel_master_ip),
                    new_data_parallel_master_port=(parallel_config.data_parallel_master_port),
                    new_data_parallel_master_port_list=(parallel_config._data_parallel_master_port_list),
                    coord_store_port=parallel_config._coord_store_port,
                    operation_id=operation_id,
                )
                self.reconfig_request = reconfig_request
        use_v3_capture = bool(reconfig_request is not None and self._read_v3_capture_decision(reconfig_request))
        with (
            _PATCH_LOCK,
            self._use_ascend_transfer_impl(
                include_expert_weights=use_v3_capture,
            ),
        ):
            super().prepare_new_worker(reconfig_request)
        if not use_v3_capture:
            # Existing workers create this group after their upstream
            # preparation. The new worker follows prepare_new_worker instead,
            # so it must join the same stateless MC2 creation here.
            create_ascend_standby_groups(
                new_dp_size=parallel_config.data_parallel_size,
                new_world_size_across_dp=(parallel_config.world_size * parallel_config.data_parallel_size),
                master_ip=parallel_config.data_parallel_master_ip,
                coord_store_port=parallel_config._coord_store_port,
                create_v3_capture_dp=False,
            )
            return reconfig_request.operation_id if reconfig_request is not None else None

        if reconfig_request is None:
            raise RuntimeError("V3 pre-commit capture requires a reconfiguration request")

        # Receive the mapping without installing MoE communication methods yet:
        # get_hccl_comm_name must not run until every target MC2 rank enters the
        # materialization rendezvous below.
        mapping = super().receive_expert_mapping()
        self.worker.model_runner.setup_eplb_from_mapping(mapping)
        self._set_eplb_suppressed(True)
        self._v3_eplb_suppression_operation_id = reconfig_request.operation_id

        self._synchronize_v3_mapping_setup(
            reconfig_request,
            is_existing_worker=False,
        )

        # New ranks install the final-size group early so graph capture can use
        # it before commit. The mapping rendezvous above ensures all target
        # ranks enter this stateless group only after device-side mapping setup.
        create_ascend_standby_groups(
            new_dp_size=parallel_config.data_parallel_size,
            new_world_size_across_dp=(parallel_config.world_size * parallel_config.data_parallel_size),
            master_ip=parallel_config.data_parallel_master_ip,
            coord_store_port=parallel_config._coord_store_port,
            create_v3_capture_dp=True,
            mc2_rank_order=self._v3_mc2_rank_order,
        )
        self._activate_ascend_standby_groups()

        self.materialize_new_communication_groups()

        moe_modules = [module for module in self.worker.get_model().modules() if is_moe_layer(module)]
        if not moe_modules:
            raise RuntimeError("V3 pre-commit capture requires at least one MoE layer")

        new_ep_size = get_mc2_group().world_size
        new_dp_size = parallel_config.data_parallel_size
        if new_ep_size % new_dp_size != 0:
            raise RuntimeError(
                f"MC2 size must divide evenly by DP size for V3 capture: mc2={new_ep_size}, dp={new_dp_size}"
            )
        old_ep_size = self._v3_old_dp_size * (new_ep_size // new_dp_size)
        active_ranks = list(range(old_ep_size, new_ep_size))
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        table_orig_to_dense = torch.full(
            (new_ep_size,),
            -1,
            dtype=torch.int32,
            device=self.worker.device,
        )
        table_dense_to_orig = torch.full_like(table_orig_to_dense, -1)
        table_orig_to_dense[active_ranks] = torch.arange(
            len(active_ranks),
            dtype=torch.int32,
            device=self.worker.device,
        )
        table_dense_to_orig[: len(active_ranks)] = torch.tensor(
            active_ranks,
            dtype=torch.int32,
            device=self.worker.device,
        )
        self._set_v3_target_elastic_info(
            torch.cat(
                (
                    torch.tensor(
                        [
                            1,
                            len(active_ranks),
                            0,
                            len(active_ranks) * num_local_experts,
                        ],
                        dtype=torch.int32,
                        device=self.worker.device,
                    ),
                    table_orig_to_dense,
                    table_dense_to_orig,
                )
            ).contiguous(),
            allow_shape_change=True,
        )
        self._setup_moe_comm_and_quant_method()

        moe_comm_method = get_moe_comm_method(MoECommType.MC2)
        if moe_comm_method is None:
            raise RuntimeError("MC2 communication method is not initialized")
        dispatcher = moe_comm_method.token_dispatcher
        first_moe_config = moe_modules[0].moe_config
        target_mc2_group = get_mc2_group()
        logger.info(
            "[Elastic EP] Creating new-rank V3 context: rank=%s/%s",
            target_mc2_group.rank_in_group,
            target_mc2_group.world_size,
        )
        prepared = dispatcher.prepare_v3_buffer(
            hidden_size=first_moe_config.hidden_dim,
            moe_expert_num=first_moe_config.num_experts,
            topk=first_moe_config.experts_per_token,
            dtype=self.worker.vllm_config.model_config.dtype,
            device=self.worker.device,
        )
        if not prepared:
            raise RuntimeError("Failed to prepare MoeDistribute V3 buffer")
        self._synchronize_v3_context_rendezvous(target_mc2_group, "new")
        return reconfig_request.operation_id

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
