import socket
import struct
from contextlib import contextmanager
from copy import copy
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_npu
from torch.distributed.distributed_c10d import _set_pg_timeout
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tp_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.model_loader import get_model_loader

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (
    get_dynamic_eplb_group,
    set_elastic_info,
)
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
else:
    NPUModelRunner = None

_PORTS_FMT = "!2I"
# TODO: Refactor scale_down.py - use descaler object instead of NpuWorker attrs to streamline code
STORE_KEY = ""

BASE_WEIGHT_SUFFIXES = {"down_proj.weight", "up_proj.weight", "gate_proj.weight"}
QUANT_WEIGHT_SUFFIXES = {
    "down_proj.weight_offset",
    "up_proj.weight_offset",
    "gate_proj.weight_offset",
    "down_proj.weight_scale",
    "up_proj.weight_scale",
    "gate_proj.weight_scale",
}
# model_type → MTP weight path template mapping
_MTP_WEIGHT_PATH_TEMPLATES: dict[frozenset[str], str] = {
    frozenset(
        {
            "nemotron_h",
            "nemotron_h_mtp",
            "qwen3_next",
            "qwen3_next_mtp",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_mtp",
            "exaone_moe",
            "exaone_moe_mtp",
        }
    ): "mtp.layers.{idx}.mlp.experts.{eid}.{suffix}",
    frozenset({"glm_moe_dsa"}): "model.layers.{layer_id}.mlp.experts.{eid}.{suffix}",
    frozenset({"longcat_flash", "longcat_flash_mtp"}): (
        "model.mtp.layers.{idx}.transformer_layer.mlp.experts.{eid}.{suffix}"
    ),
    frozenset({"ernie4_5_moe", "ernie_mtp"}): "model.mtp_block.0.mlp.experts.{eid}.{suffix}",
}


def _append_mtp_copies(main_list: list, num_mtp_layers: int) -> None:
    """Append MTP layer copies to main_list by cyclic modulo indexing."""
    if num_mtp_layers <= 0 or not main_list:
        return
    num_main = len(main_list)
    for mtp_idx in range(num_mtp_layers):
        main_idx = mtp_idx % num_main
        item = main_list[main_idx]
        main_list.append(item.clone() if hasattr(item, "clone") else copy(item))


def _resolve_mtp_weight_prefix(prefix: str, experts_saved_weights: dict[str, torch.Tensor], probe_key: str) -> str:
    """Resolve GLM-5 MTP mtp_block prefix mismatch.

    The checkpoint raw weight names and module.layer_name may differ;
    automatically toggle between prefix with/without mtp_block.
    """
    if probe_key in experts_saved_weights:
        return prefix
    if "mtp_block" in prefix:
        return prefix.replace(".mtp_block.mlp.", ".mlp.", 1)
    return prefix.replace(".mlp.", ".mtp_block.mlp.", 1)


def _get_mtp_weight_path(
    model_type: str,
    mtp_local_idx: int,
    layer_id: int,
    expert_id: int,
    suffix: str,
) -> str:
    """Look up the MTP weight path template for the given model_type."""
    for model_types, template in _MTP_WEIGHT_PATH_TEMPLATES.items():
        if model_type in model_types:
            return template.format(
                idx=mtp_local_idx,
                layer_id=layer_id,
                eid=expert_id,
                suffix=suffix,
            )
    return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"


def distribute_experts(global_num_expert: int, ep_size: int) -> dict[int, list[int]]:
    distribution = {}
    base = global_num_expert // ep_size
    remainder = global_num_expert % ep_size

    start_index = 0
    for rank in range(ep_size):
        num = base + (1 if rank < remainder else 0)
        expert_ids = list(range(start_index, start_index + num))
        distribution[rank] = expert_ids
        start_index += num
    return distribution


def gen_global_log2phy_map(
    num_logical_experts: int, num_npu: int, redundant_expert_list: list[int]
) -> dict[int, list[int]]:
    num_redundant_experts = len(redundant_expert_list)
    assert (num_logical_experts + num_redundant_experts) % num_npu == 0, (
        "the physical expert count must evenly divide across NPUs"
    )
    num_phy_exp_per_npu = (num_logical_experts + num_redundant_experts) // num_npu

    # How many physical experts per NPU after placing redundancy
    exp_distribution_without_redundancy = distribute_experts(num_logical_experts, num_npu)
    num_routed_experts_list = []
    num_redundant_experts_list = []
    for rank in range(num_npu):
        num_routed_experts_list.append(len(exp_distribution_without_redundancy[rank]))
        num_redundant_experts_list.append(num_phy_exp_per_npu - len(exp_distribution_without_redundancy[rank]))

    # Mapping: logical expert -> list of physical expert IDs assigned
    global_log2phy_map: dict[int, list[int]] = {log_expert_id: [] for log_expert_id in range(num_logical_experts)}
    log_experts_iter = iter(range(num_logical_experts))

    global_pos = 0
    re_exp_assign_map = [[exp_id, False] for exp_id in redundant_expert_list]

    for rank in range(num_npu):
        local_expert_map = []
        for _ in range(num_routed_experts_list[rank]):
            expert_id = next(log_experts_iter)
            global_log2phy_map[expert_id].insert(0, global_pos)
            global_pos += 1
            local_expert_map.append(expert_id)

        for _ in range(num_redundant_experts_list[rank]):
            success = False
            for i in range(len(re_exp_assign_map)):
                eid, assigned = re_exp_assign_map[i]
                if assigned:
                    continue
                if eid in local_expert_map:
                    continue
                global_log2phy_map[eid].append(global_pos)
                global_pos += 1
                local_expert_map.append(eid)
                re_exp_assign_map[i][1] = True
                success = True
                break
            if not success:
                raise RuntimeError(
                    "expert placement aborted. The distribution of redundant experts cannot"
                    "satisfy the requirement that physical replicas of each logical expert are properly replicated."
                )
    return global_log2phy_map


def init_global_expert_distribution(global_log2phy_map: dict[int, list[int]], ep_size: int) -> dict[int, list[int]]:
    num_phy_experts = sum(map(len, global_log2phy_map.values()))
    num_phy_exp_per_npu = num_phy_experts // ep_size
    global_expert_distribution = {i: [-1 for _ in range(num_phy_exp_per_npu)] for i in range(ep_size)}
    for log_eid, phy_expert_pos in global_log2phy_map.items():
        for pos in phy_expert_pos:
            rank = pos // num_phy_exp_per_npu
            local_pos = pos - rank * num_phy_exp_per_npu
            global_expert_distribution[rank][local_pos] = log_eid
    return global_expert_distribution


def generate_redundant_expert_ids(num_experts: int, ep_size: int, num_redundant_experts: int) -> list[int]:
    assert num_redundant_experts % ep_size == 0
    experts_per_ep_group = num_experts // ep_size
    redundant_per_group = num_redundant_experts // ep_size
    redundant_ids = []
    for rank in range(ep_size):
        start_id = rank * experts_per_ep_group
        for i in range(redundant_per_group):
            redundant_ids.append(start_id + i)
    return redundant_ids


def init_ep2dp_map(dp_size: int, tp_size: int) -> dict[int, int]:
    ep2dp_map = {}
    for dp_rank in range(dp_size):
        ep_start = dp_rank * tp_size
        ep_end = (dp_rank + 1) * tp_size
        for ep_rank in range(ep_start, ep_end):
            ep2dp_map[ep_rank] = dp_rank
    return ep2dp_map


def update_ep2dp_map(
    ep2dp_map: dict[int, int],
    excluded_dp_ranks: list[int],
    rank_mapping: dict[int, int],
) -> dict[int, int]:
    for old_ep_rank, dp_rank in ep2dp_map.items():
        if dp_rank != -1:
            if dp_rank in excluded_dp_ranks:
                ep2dp_map[old_ep_rank] = -1
            else:
                ep2dp_map[old_ep_rank] = rank_mapping[dp_rank]
    return ep2dp_map


def update_parallel_config(original_config: VllmConfig, update_config: dict[str, int]) -> None:
    required_keys = {
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_rank",
        "data_parallel_master_port",
    }
    missing_keys = required_keys - set(update_config.keys())
    if missing_keys:
        raise ValueError(f"update parallel config failed missing keys: {missing_keys}")

    original_config.parallel_config.data_parallel_size = update_config["data_parallel_size"]
    original_config.parallel_config.data_parallel_size_local = update_config["data_parallel_size_local"]
    original_config.parallel_config.data_parallel_rank = update_config["data_parallel_rank"]
    original_config.parallel_config.data_parallel_master_port = update_config["data_parallel_master_port"]


def init_elastic_info(
    ep_size: int,
    phy_experts_num: int,
    share_expert_rank_num: int = 0,
):
    # Basic configuration (first 4 parameters)
    # Meaning: whether to scale down (0 = no scale down), actual number of ranks after scale down
    # reduction (=ep_size), number of ranks for shared experts,number of MoE experts
    is_scaled_down = 0
    base_config = torch.tensor([is_scaled_down, ep_size, share_expert_rank_num, phy_experts_num], dtype=torch.int32)

    # Table1: epRankID -> localEpRankId(-1 indicates invalid）
    table1 = torch.arange(0, ep_size, dtype=torch.int32)
    # Table2: localEpRankId -> epRankID(-1 indicates padding）
    table2 = torch.arange(0, ep_size, dtype=torch.int32)

    elastic_info = torch.cat([base_config, table1, table2], dim=0).npu().contiguous()
    elastic_info.requires_grad_(False)
    set_elastic_info(elastic_info)


def update_elastic_info(
    elastic_info: torch.Tensor,
    expert_num: int,
    raw_ep_size: int,
    ep2dp: dict[int, int],
    share_expert_num: int = 0,
) -> None:
    if elastic_info is None:
        elastic_info = torch.full((4 + 2 * raw_ep_size,), -1, dtype=torch.int32).npu().contiguous()
    raw_ep_ranks = sorted(ep2dp.keys())
    valid_ep_ranks = [ep for ep in raw_ep_ranks if ep2dp[ep] != -1]
    scaled_down_ep_size = len(valid_ep_ranks)
    is_scaled_down = 1 if scaled_down_ep_size < raw_ep_size else 0

    # Table1: epRankID -> localEpRankId(-1 indicates invalid）
    table1 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
    for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
        table1[ep_rank] = local_ep_rank

    # Table2: localEpRankId -> epRankID(-1 indicates padding）
    table2 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
    for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
        if local_ep_rank < scaled_down_ep_size:
            table2[local_ep_rank] = ep_rank

    # update elastic_info
    new_elastic_info_cpu = torch.cat(
        [
            torch.tensor([is_scaled_down, scaled_down_ep_size, share_expert_num, expert_num], dtype=torch.int32),
            table1,
            table2,
        ],
        dim=0,
    )
    elastic_info.copy_(new_elastic_info_cpu)
    set_elastic_info(elastic_info)


def dynamic_merge_view(
    target_tensor: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    dim_size1 = tensor1.shape[dim]
    dim_size2 = tensor2.shape[dim]
    total_dim_size = dim_size1 + dim_size2

    non_dim_shapes = [s for i, s in enumerate(tensor1.shape) if i != dim]
    for i, s in enumerate(tensor2.shape):
        if i != dim and s != non_dim_shapes[i if i < dim else i - 1]:
            expected = non_dim_shapes[i if i < dim else i - 1]
            raise ValueError(f"size mismatch on non merged dimension {i}：tensor1={expected} vs tensor2={s}")
    if target_tensor.shape[dim] != total_dim_size:
        raise ValueError(f"target tensor on dim {dim} must be {dim_size1}+{dim_size2}={total_dim_size}")

    top_view = target_tensor.narrow(dim, 0, dim_size1)
    bottom_view = target_tensor.narrow(dim, dim_size1, dim_size2)

    top_view.copy_(tensor1)
    bottom_view.copy_(tensor2)

    return target_tensor


class ScaleDownHelper:
    """Orchestrates the full scale-down workflow for fault tolerance.

    Encapsulates: expert distribution recalculation, weight reloading,
    communication group reinitialization, MoE reconfiguration, and
    elastic info updates.
    """

    def __init__(self, vllm_config: VllmConfig, model_runner, quant: bool):
        self.vllm_config = vllm_config
        self.model_runner = model_runner
        self.quant = quant

    def get_expert_distribution_after_scale_down(self, excluded_dp_ranks, enable_d2d_rebalance, rank):
        """Wake up EPLB worker and get the new expert distribution for this rank."""
        model_runner = self.model_runner
        eplb_updator = model_runner.eplb_updator
        model_runner.shared_dict["scale_down"] = True
        model_runner.shared_dict["enable_d2d_after_failure"] = enable_d2d_rebalance
        model_runner.shared_dict["excluded_dp_ranks"] = excluded_dp_ranks
        expert_maps = model_runner.shared_dict["expert_maps"]
        if expert_maps is None or (expert_maps.shape == (1, 1, 1) and not expert_maps.any()):
            model_runner.shared_dict["expert_maps"] = self._get_global_expert_map()

        eplb_updator.wakeup_eplb_worker()
        eplb_updator.update_info_all = eplb_updator.eplb_process.block_update_q.get()
        need_load_h2d = model_runner.shared_dict["need_load_h2d"]

        experts_to_load = []
        for layer_id in range(len(need_load_h2d)):
            per_layer = need_load_h2d[layer_id][rank].copy()
            experts_to_load.append(per_layer)

        return experts_to_load

    def _get_global_expert_map(self):
        """Collect global expert maps from all model layers."""
        model_runner = self.model_runner
        model = model_runner.get_model()
        num_dense_layers = getattr(model.config, "first_k_dense_replace", 0)
        num_moe_layers = model.config.num_hidden_layers - num_dense_layers
        all_layer_global_expert_map = []
        for layer_id in range(num_moe_layers):
            map_cpu = model.model.layers[num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
        num_mtp_layers = self._get_mtp_num_layers()
        if num_mtp_layers > 0:
            drafter = getattr(model_runner, "drafter", None)
            if drafter is not None and hasattr(drafter, "model"):
                mtp_model = drafter.model
                for mtp_layer_idx in range(num_mtp_layers):
                    mtp_layer = mtp_model.model.layers[mtp_layer_idx]
                    map_cpu = mtp_layer.mlp.experts.global_expert_map.cpu()
                    all_layer_global_expert_map.append(map_cpu)
        return torch.stack(all_layer_global_expert_map)

    def _get_mtp_num_layers(self) -> int:
        if not self._is_mtp_speculative():
            return 0

        draft = getattr(self.vllm_config.speculative_config, "draft_model_config", None)
        if draft is not None:
            num = draft.get_total_num_hidden_layers()
            if num > 0:
                return num

        hf = self.vllm_config.model_config.hf_config
        for attr in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "n_predict"):
            num = getattr(hf, attr, None)
            if num:
                return num

        raise RuntimeError("MTP layer count not found in model config; unsupported model configuration.")

    def _is_mtp_speculative(self) -> bool:
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        if spec_config is None:
            return False
        method = getattr(spec_config, "method", None)
        return method == "mtp" or (isinstance(method, str) and method.endswith("_mtp"))

    def load_expert_weights_to_cpu(self, experts_to_load, weight_name_to_tensor) -> dict[str, torch.Tensor]:
        """Load specified expert weights from disk into CPU memory"""

        weight_suffixes = BASE_WEIGHT_SUFFIXES.union(QUANT_WEIGHT_SUFFIXES) if self.quant else BASE_WEIGHT_SUFFIXES

        def _generate_expert_weight_name(layer_id: int, expert_id: int, suffix: str) -> str:
            """Generate the full parameter name for a single expert weight."""
            if layer_id < num_hidden_layers:
                return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"

            mtp_local_idx = layer_id - num_hidden_layers
            if mtp_local_idx >= num_mtp_layers:
                raise RuntimeError(
                    f"MTP layer index out of range: layer_id={layer_id}, "
                    f"mtp_local_idx={mtp_local_idx}, num_mtp_layers={num_mtp_layers}"
                )

            model_type = getattr(self.vllm_config.model_config.hf_config, "model_type", "")
            return _get_mtp_weight_path(model_type, mtp_local_idx, layer_id, expert_id, suffix)

        num_dense_layers = getattr(self.model_runner.get_model().config, "first_k_dense_replace", 0)
        num_hidden_layers = getattr(self.model_runner.get_model().config, "num_hidden_layers", 0)
        num_mtp_layers = self._get_mtp_num_layers()
        num_main_moe_layers = num_hidden_layers - num_dense_layers

        weights_to_save = set()
        for index, per_layer_experts in enumerate(experts_to_load):
            layer_id = index + num_dense_layers
            if per_layer_experts:
                for pos, expert_id in per_layer_experts:
                    for suffix in weight_suffixes:
                        weights_to_save.add(_generate_expert_weight_name(layer_id, expert_id, suffix))

        # MTP layers
        if num_mtp_layers > 0:
            for mtp_idx in range(num_mtp_layers):
                layer_id = num_hidden_layers + mtp_idx
                data_idx = num_main_moe_layers + mtp_idx
                if data_idx < len(experts_to_load) and experts_to_load[data_idx]:
                    for pos, expert_id in experts_to_load[data_idx]:
                        for suffix in weight_suffixes:
                            weights_to_save.add(_generate_expert_weight_name(layer_id, expert_id, suffix))

        saved_expert_weights = {}
        for weight_name in weights_to_save:
            weight_tensor = weight_name_to_tensor[weight_name]
            if weight_tensor.ndim >= 2:
                weight_tensor = weight_tensor.transpose(0, 1).contiguous()
            if any(weight_name.endswith(suffix) for suffix in QUANT_WEIGHT_SUFFIXES):
                weight_tensor = torch.squeeze(weight_tensor)
            saved_expert_weights[weight_name] = weight_tensor

        return saved_expert_weights

    def reload_expert_weights(self, experts_to_load, saved_weights: dict[str, torch.Tensor]) -> None:
        """Load saved expert weights from CPU into the FusedMoE modules."""

        def _load_single_expert(expert_id: int, target_index: int):
            raw_prefix = f"{module.layer_name}.{expert_id}"
            prefix = _resolve_mtp_weight_prefix(raw_prefix, saved_weights, f"{raw_prefix}.gate_proj.weight")
            w1_key = f"{prefix}.gate_proj.weight"
            w2_key = f"{prefix}.down_proj.weight"
            w3_key = f"{prefix}.up_proj.weight"
            w1_weight = saved_weights[w1_key]
            w2_weight = saved_weights[w2_key]
            w3_weight = saved_weights[w3_key]
            if get_ascend_config().eplb_config.dynamic_eplb:
                device = module.w2_weight_list[target_index].device
                w2_expert_data = module.w2_weight_list[target_index]
                w13_expert_data = module.w13_weight_list[target_index]
            else:
                device = module.w2_weight.device
                w2_expert_data = module.w2_weight[target_index]
                w13_expert_data = module.w13_weight[target_index]
            module._load_w2(
                expert_data=w2_expert_data,
                shard_dim=1,
                loaded_weight=w2_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=w13_expert_data,
                shard_dim=1,
                shard_id="w1",
                loaded_weight=w1_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=w13_expert_data,
                shard_dim=1,
                shard_id="w3",
                loaded_weight=w3_weight.to(device),
                tp_rank=module.tp_rank,
            )

            if self.quant:
                prefix = _resolve_mtp_weight_prefix(raw_prefix, saved_weights, f"{raw_prefix}.gate_proj.weight_scale")
                s1_key = f"{prefix}.gate_proj.weight_scale"
                s2_key = f"{prefix}.down_proj.weight_scale"
                s3_key = f"{prefix}.up_proj.weight_scale"
                o1_key = f"{prefix}.gate_proj.weight_offset"
                o2_key = f"{prefix}.down_proj.weight_offset"
                o3_key = f"{prefix}.up_proj.weight_offset"
                w1_weight_scale = saved_weights[s1_key].to(device)
                w2_weight_scale = saved_weights[s2_key].to(device)
                w3_weight_scale = saved_weights[s3_key].to(device)
                w1_weight_offset = saved_weights[o1_key].to(device)
                w2_weight_offset = saved_weights[o2_key].to(device)
                w3_weight_offset = saved_weights[o3_key].to(device)
                module.w2_weight_offset.data[target_index].copy_(w2_weight_offset)
                dynamic_merge_view(module.w13_weight_offset.data[target_index], w1_weight_offset, w3_weight_offset)
                if get_ascend_config().eplb_config.dynamic_eplb:
                    module.w2_weight_scale_list[target_index].copy_(w2_weight_scale)
                    dynamic_merge_view(
                        module.w13_weight_scale_fp32_list[target_index], w1_weight_scale, w3_weight_scale
                    )
                else:
                    module.w2_weight_scale[target_index].copy_(w2_weight_scale)
                    dynamic_merge_view(module.w13_weight_scale_fp32[target_index], w1_weight_scale, w3_weight_scale)

        cur_layer_id = 0
        for module in self.model_runner.get_model().modules():
            if isinstance(module, FusedMoE):
                if experts_to_load[cur_layer_id] is not None:
                    for slot_pos, expert_id in experts_to_load[cur_layer_id]:
                        _load_single_expert(expert_id=expert_id, target_index=slot_pos)

                cur_layer_id += 1

        draft_model = getattr(getattr(self.model_runner, "drafter", None), "model", None)
        if draft_model is not None:
            for module in draft_model.modules():
                if isinstance(module, FusedMoE):
                    if cur_layer_id < len(experts_to_load) and experts_to_load[cur_layer_id] is not None:
                        for slot_pos, expert_id in experts_to_load[cur_layer_id]:
                            _load_single_expert(expert_id=expert_id, target_index=slot_pos)
                    cur_layer_id += 1

    def update_eplb_adaptor_info(self, num_add_experts_per_rank, rank):
        model_runner = self.model_runner
        model_runner.eplb_adaptor.rank_id = rank
        model_runner.eplb_adaptor.model.clear_all_moe_loads()
        model_runner.shared_dict["moe_load"] = None
        model_runner.eplb_updator.cur_iterations = 0

        if num_add_experts_per_rank > 0:
            model_runner.eplb_adaptor.init_buffer_tensor(num_add_experts_per_rank)

        model_runner.eplb_adaptor.init_expert_param_per_layer()
        cur_deployment = model_runner.shared_dict["expert_maps"]
        for layer_id in range(cur_deployment.shape[0]):
            model_runner.eplb_adaptor.do_clone_update_expert_map(layer_id, cur_deployment[layer_id][rank])

    def gen_all_layer_log2phy(self, rank):
        all_layer_log2phy = []
        cur_deployment = self.model_runner.shared_dict["expert_maps"]
        for layer_id in range(cur_deployment.shape[0]):
            cur_layer_log2phy_map = generate_log2phy_map(cur_deployment[layer_id], rank)
            all_layer_log2phy.append(cur_layer_log2phy_map)

        num_mtp_layers = self._get_mtp_num_layers() or 0
        _append_mtp_copies(all_layer_log2phy, num_mtp_layers)

        return all_layer_log2phy

    def d2d_transmission_for_scaling_down(self):
        eplb_loader = self.model_runner.eplb_loader
        eplb_adaptor = self.model_runner.eplb_adaptor
        eplb_updator = self.model_runner.eplb_updator

        all_layer_log2phy_map = []

        for info in eplb_updator.update_info_all:
            (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = info
            log2phy_map_this_rank = torch.from_numpy(np.array(log2phy_map))
            all_layer_log2phy_map.append(log2phy_map_this_rank)
            eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(np.array(updated_expert_map))
            eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info,
                expert_recv_info,
                updated_expert_map_this_rank,
                layer_id + eplb_adaptor.num_dense_layers,
            )
            reqs = []
            eplb_loader.asyn_expert_weight_transfer(reqs)
            eplb_loader.update_expert_map_and_weight(reqs)

        eplb_updator.update_info_all.clear()

        num_mtp_layers = self._get_mtp_num_layers() or 0
        _append_mtp_copies(all_layer_log2phy_map, num_mtp_layers)

        torch_npu.npu.synchronize()

        return all_layer_log2phy_map

    def update_parallel_config(self, update_config: dict[str, int]) -> None:
        update_parallel_config(self.vllm_config, update_config)

    def update_ep2dp_map(self, ep2dp_map, excluded_dp_ranks, rank_mapping):
        return update_ep2dp_map(ep2dp_map, excluded_dp_ranks, rank_mapping)

    def update_elastic_info(self, elastic_info, expert_num, raw_ep_size, ep2dp, share_expert_num: int = 0):
        update_elastic_info(elastic_info, expert_num, raw_ep_size, ep2dp, share_expert_num)

    def destroy_comm_group(self) -> None:
        get_dp_group().destroy_cpu_group()
        if get_ascend_config().eplb_config.dynamic_eplb:
            get_dynamic_eplb_group().destroy_cpu_group()

    def init_dp_cpu_group(self, coord_store, group_type="normal") -> None:
        init_dp_cpu_group_impl(self.vllm_config, coord_store, group_type)

    def reconfigure_moe(self, num_logical_expert, num_new_phy_experts, all_layer_log2phy):
        reconfigure_moe(
            self.model_runner,
            self.vllm_config,
            num_logical_expert,
            num_new_phy_experts,
            all_layer_log2phy,
        )


def init_dp_cpu_group_impl(vllm_config: VllmConfig, coord_store, group_type="normal") -> None:
    """Initialize DP CPU group using TCP store for port coordination."""
    listen_sockets = []
    ports = []
    if vllm_config.parallel_config.data_parallel_rank == 0:
        for i in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((vllm_config.parallel_config.data_parallel_master_ip, 0))
            sock.listen()
            listen_sockets.append(sock)
            ports.append(sock.getsockname()[1])
        coord_store.set(STORE_KEY, struct.pack(_PORTS_FMT, *ports))
    else:
        ports = list(struct.unpack(_PORTS_FMT, coord_store.get(STORE_KEY)))
        listen_sockets = []

    eplb_port, dp_port = ports
    if get_ascend_config().eplb_config.dynamic_eplb:
        get_dynamic_eplb_group().cpu_group = stateless_init_torch_distributed_process_group(
            vllm_config.parallel_config.data_parallel_master_ip,
            eplb_port,
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
            listen_socket=listen_sockets[0] if listen_sockets else None,
            backend="gloo",
            gloo_timeout_seconds=vllm_config.parallel_config.fault_tolerance_config.gloo_comm_timeout,
        )
        get_dynamic_eplb_group().group_type = group_type

    # TODO: Temporarily hardcode the port value for debugging. Will replace with get_open_port().
    get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
        vllm_config.parallel_config.data_parallel_master_ip,
        dp_port,
        vllm_config.parallel_config.data_parallel_rank,
        vllm_config.parallel_config.data_parallel_size,
        backend="gloo",
        listen_socket=listen_sockets[1] if listen_sockets else None,
    )
    get_dp_group().group_type = group_type
    timeout = timedelta(seconds=vllm_config.parallel_config.fault_tolerance_config.gloo_comm_timeout)
    _set_pg_timeout(timeout=timeout, group=get_dp_group().cpu_group)


@contextmanager
def patch_get_all_weights(
    saved_expert_weights_dict: dict[str, torch.Tensor] | None = None,
    enable_fault_tolerance: bool = False,
    drafter_model: torch.nn.Module | None = None,
):
    if saved_expert_weights_dict is None or not enable_fault_tolerance:
        yield
        return

    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    loader = get_model_loader(LoadConfig())
    if not isinstance(loader, DefaultModelLoader):
        logger.warning(
            "Fault tolerance weight saving only supports DefaultModelLoader, "
            "Current loader type: %s. "
            "Scale-down weight reload will not available. ",
            type(loader).__name__,
        )
        yield
        return
    original_get_all_weights = DefaultModelLoader.get_all_weights

    def saving_get_all_weights(self_loader, model_config, model):
        for name, tensor in original_get_all_weights(self_loader, model_config, model):
            saved_expert_weights_dict[name] = tensor
            yield name, tensor

        if drafter_model is not None:
            for name, tensor in original_get_all_weights(self_loader, model_config, drafter_model):
                saved_expert_weights_dict[name] = tensor
                yield name, tensor

    DefaultModelLoader.get_all_weights = saving_get_all_weights
    try:
        yield
    finally:
        DefaultModelLoader.get_all_weights = original_get_all_weights


def reconfigure_moe(
    model_runner: NPUModelRunner,
    vllm_config: VllmConfig,
    num_global_logical_experts: int,
    num_global_new_phy_experts: int,
    log2phy: torch.Tensor,
):
    import vllm.envs as envs

    parallel_config = vllm_config.parallel_config
    new_ep_size = parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
    get_ascend_config().eplb_config.num_redundant_experts = num_global_new_phy_experts - num_global_logical_experts

    moe_modules = [module for module in model_runner.get_model().modules() if isinstance(module, FusedMoE)]
    draft_model = getattr(getattr(model_runner, "drafter", None), "model", None)
    if draft_model is not None:
        moe_modules.extend(module for module in draft_model.modules() if isinstance(module, FusedMoE))

    for cur_layer_id, module in enumerate(moe_modules):
        module.local_num_experts = num_global_new_phy_experts // new_ep_size
        module.global_num_experts = num_global_new_phy_experts
        module.global_redundant_expert_num = num_global_new_phy_experts - num_global_logical_experts
        sp_size = module.sp_size
        module.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=get_tp_group().world_size,
            pcp_size_=get_pcp_group().world_size,
            dp_size_=get_dp_group().world_size,
            vllm_parallel_config=parallel_config,
            sp_size_=sp_size,
        )
        module.moe_config = FusedMoEConfig(
            num_experts=module.global_num_experts,
            experts_per_token=module.top_k,
            hidden_dim=module.hidden_size,
            intermediate_size_per_partition=module.intermediate_size_per_partition,
            num_local_experts=module.local_num_experts,
            num_logical_experts=num_global_logical_experts,
            moe_parallel_config=module.moe_parallel_config,
            in_dtype=module.vllm_config.model_config.dtype,
            router_logits_dtype=None,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=False,
            is_act_and_mul=True,
            is_lora_enabled=module.vllm_config.lora_config is not None,
            activation=module.activation,
            device=module.vllm_config.device_config.device,
            routing_method=module.routing_method_type,
        )
        module.moe_config.num_experts = module.global_num_experts
        module.moe_config.num_local_experts = module.local_num_experts
        module.moe_config.global_redundant_expert_num = module.global_redundant_expert_num
        module.log2phy.copy_(log2phy[cur_layer_id].npu(), non_blocking=True)
