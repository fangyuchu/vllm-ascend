import copy
import gc
import math
from collections import defaultdict

import torch
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.model_loader import get_model_loader

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel, get_mc2_group, set_elastic_info
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.ops.fused_moe.fused_moe import setup_moe_comm_method
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


def gen_expert_backup_map(
    num_experts: int, ep_size: int, num_die_per_npu: int, global_expert_distribution: dict[int, list[int]]
) -> list[list[int]]:
    backup_experts = [[] for _ in range(ep_size)]
    if global_expert_distribution is None:
        global_expert_distribution = distribute_experts(num_experts, ep_size)

    def get_least_load_backup_rank(exclude_ranks: list[int]) -> int:
        assert len(exclude_ranks) != ep_size, "必须保留至少一个可用的备份rank"
        min_backup_count = float("inf")
        optimal_backup_rank = -1
        for rank in range(ep_size):
            if rank in exclude_ranks:
                continue
            current_backup_count = len(backup_experts[rank])
            if current_backup_count <= min_backup_count:
                min_backup_count = current_backup_count
                optimal_backup_rank = rank
        return optimal_backup_rank

    for rank_group_start in range(0, ep_size, num_die_per_npu):
        rank_group_end = min(ep_size, rank_group_start + num_die_per_npu)
        current_rank_group = list(range(rank_group_start, rank_group_end))

        current_group_experts = []
        for rank in current_rank_group:
            current_group_experts.extend(global_expert_distribution[rank])
        for expert_id in current_group_experts:
            backup_rank = get_least_load_backup_rank(current_rank_group)
            backup_experts[backup_rank].append(expert_id)
    return backup_experts


def distribute_experts(global_num_expert: int, ep_size: int) -> dict[int, list[int]]:
    init_global_expert_distribution = {}
    base = global_num_expert // ep_size
    remainder = global_num_expert % ep_size

    start_index = 0
    for rank in range(ep_size):
        num = base + (1 if rank < remainder else 0)
        expert_ids = list(range(start_index, start_index + num))
        init_global_expert_distribution[rank] = expert_ids
        start_index += num
    return init_global_expert_distribution


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

    # Mapping: logcial expert -> list of physical expert IDs assigned
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
                    "expert placement aborted. the distribution of redundant expert cannot"
                    "satisfy the requirement that physical replicas of each logical expert"
                )
    return global_log2phy_map


def init_global_expert_distribution(global_log2phy_map: dict[int, list[int]], ep_size: int) -> dict[int, list[int]]:
    num_phy_experts = sum(map(lambda x: len(x), global_log2phy_map.values()))
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


def get_expert_distribution_after_descale(
    excluded_dp_ranks: list[int],
    old_expert_distribution: dict[int, list[int]],
    global_log2phy_map: dict[int, list[int]],
    backup_expert_rank_mapping,
    use_mask_mc2,
) -> tuple[
    dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], dict[int, dict[int, tuple[int, int]]], bool
]:
    expert_distribution = copy.deepcopy(old_expert_distribution)
    num_experts_per_rank = len(expert_distribution[0])
    health_ranks = [r for r in range(len(expert_distribution)) if r not in excluded_dp_ranks]
    if not health_ranks:
        raise RuntimeError("All DP ranks are faulty")
    for rank in excluded_dp_ranks:
        expert_distribution.pop(rank)

    added_experts: dict[int, list[int]] = {r: [] for r in health_ranks}
    replaced_redundant_experts: dict[int : dict[int, tuple[int, int]]] = defaultdict(dict)
    rank_load: dict[int, float] = {r: 0.0 for r in health_ranks}

    # ======= 1: Remove failed physical experts & identify lost logical experts
    failed_logical_experts: set[int] = set()
    for rank in excluded_dp_ranks:
        global_offset = rank * num_experts_per_rank
        for local_idx, expert_id in enumerate(old_expert_distribution[rank]):
            golbal_pos = global_offset + local_idx
            assert golbal_pos in global_log2phy_map[expert_id], (
                f"physical position {golbal_pos} missing from global_log2phy_map[{expert_id}]"
            )
            global_log2phy_map[expert_id].remove(golbal_pos)
            if not global_log2phy_map[expert_id]:
                failed_logical_experts.add(expert_id)
    failed_logical_experts = sorted(failed_logical_experts)

    # ======= 2: collect all available redundant expert slots

    # {rank: [(redundant_expert_id, local_slot_idx)]}
    redundant_slots: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for expert_id, phy_positions in global_log2phy_map.items():
        if len(phy_positions) <= 1:
            continue
        for global_pos in phy_positions[1:]:
            rank = global_pos // num_experts_per_rank
            if rank in excluded_dp_ranks:
                continue
            local_idx = global_pos - rank * num_experts_per_rank
            redundant_slots[rank].append((expert_id, local_idx))

    total_redundant_slots = sum(len(v) for v in redundant_slots.values())

    # ======= 3: add extra redundant slots if necessary

    if total_redundant_slots < len(failed_logical_experts):
        use_mask_mc2 = False
        extra_needed = len(failed_logical_experts) - total_redundant_slots
        slots_per_rank = math.ceil(extra_needed / len(health_ranks))

        for rank in health_ranks:
            backup_experts = [eid for eid, r in backup_expert_rank_mapping.items() if r == rank]
            slot_gap = slots_per_rank
            while slot_gap > 0:
                for eid in backup_experts[:slot_gap]:
                    expert_distribution[rank].append(eid)
                    added_experts[rank].append(eid)
                    redundant_slots[rank].append((eid, len(expert_distribution[rank]) - 1))
                    slot_gap -= 1

    # ======== 4: restore lost logical experts

    remaining_lost = []

    # restore on backup rank(RAM load)
    for expert_id in failed_logical_experts:
        backup_rank = backup_expert_rank_mapping.get(expert_id)
        if backup_rank is None and backup_rank in health_ranks and redundant_slots[backup_rank]:
            redundant_id, slot_idx = redundant_slots[backup_rank].pop()
            expert_distribution[backup_rank][slot_idx] = expert_id
            replaced_redundant_experts[backup_rank][redundant_id] = (slot_idx, expert_id)
            rank_load[backup_rank] += 0.5
        else:
            remaining_lost.append(expert_id)

    # restore on least-loaded rank (disk load)
    for expert_id in remaining_lost:
        target_rank = min(rank_load, key=rank_load.get)
        while len(redundant_slots[target_rank]) == 0:
            rank_load[target_rank] = float("inf")
            target_rank = min(rank_load, key=rank_load.get)
        redundant_id, slot_idx = redundant_slots[target_rank].pop()
        expert_distribution[target_rank][slot_idx] = expert_id
        replaced_redundant_experts[target_rank][redundant_id] = (slot_idx, expert_id)
        rank_load[target_rank] += 1.0

    # ======== 5: rebuild global logical to physical mapping

    old_expert_distribution = defaultdict(list)
    flat_distribution = []
    for rank in sorted(expert_distribution):
        flat_distribution.extend(expert_distribution[rank])

    for phy_idx, expert_id in enumerate(flat_distribution):
        global_log2phy_map[expert_id].append(phy_idx)

    return global_log2phy_map, expert_distribution, added_experts, replaced_redundant_experts, use_mask_mc2


def destory_acl_graph(use_mask_mc2: bool, vllm_config: VllmConfig, model: NPUModelRunner) -> VllmConfig:
    if not use_mask_mc2:
        for entries in vllm_config.compilation_config.concrete_aclgraph_entry_list:
            for name, entry in entries.items():
                entry.aclgraph.reset()
                entry.aclgraph = None
        vllm_config.compilation_config.concrete_aclgraph_entry_list = []

        with set_current_vllm_config(vllm_config):
            reset_compile_wrapper(model.get_model())
        gc.collect()
        torch.npu.empty_cache()
        print(f"rank{vllm_config.parallel_config.data_parallel_rank} aclgraph reset success!")
        return vllm_config


def rebuild_acl_graph(use_mask_mc2: bool, worker: NPUWorker) -> None:
    if not use_mask_mc2:
        worker.determine_available_memory()
        worker.compile_or_warm_up_model()


def destory_comm_group(use_mask_mc2: bool) -> None:
    if not use_mask_mc2:
        get_dp_group().destory_cpu_group()
    else:
        destroy_ascend_model_parallel()
        cleanup_dist_env_and_memory()


def init_dp_device_group(vllm_config: VllmConfig) -> None:
    get_dp_group.cpu_group = stateless_init_torch_distributed_process_group(
        vllm_config.parallel_config.data_parallel_master_ip,
        vllm_config.parallel_config.data_parallel_master_port + 100,
        vllm_config.parallel_config.data_parallel_rank,
        vllm_config.parallel_config.data_parallel_size,
        backend="hccl",
        gloo_comm_timeout=60,
        enable_fault_tolerance=True,
    )


def reinit_comm_group(use_mask_mc2: bool, vllm_config: VllmConfig, worker: NPUWorker) -> None:
    if not use_mask_mc2:
        init_dp_device_group(vllm_config)
    else:
        worker._init_worker_distributed_environment()


def save_expert_weights_to_ram(
    experts_id_to_save: list[int],
    experts_saved_ids: list[int],
    experts_saved_weights: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    model_runner: NPUModelRunner,
    quant: bool | str = False,
) -> tuple[list[int], dict[str, torch.Tensor]]:
    """
    将指定未保存的专家权重加载并保存到内存中（RAM）

    Args:
        experts_id_to_save: 需要保存的专家ID列表
        experts_saved_ids: 已保存的专家ID列表（会在函数内更新）
        experts_saved_weights: 存储已保存专家权重的字典（会在函数内更新）
        vllm_config: VLLM配置对象
        model_runner: NPU模型运行器
        quant: 是否为量化模型（影响权重名称后缀）

    Returns:
        tuple: (更新后的已保存专家ID列表, 更新后的已保存权重字典)
    """
    # 转换为集合加速查找（O(1) vs 列表O(n)）
    saved_ids_set = set(experts_saved_ids)
    # 去重并筛选未保存的专家ID
    unsaved_expert_ids = list({eid for eid in experts_id_to_save if eid not in saved_ids_set})

    # 无需要保存的专家时直接返回原数据
    if not unsaved_expert_ids:
        return experts_saved_ids, experts_saved_weights

    # 定义权重名称后缀常量
    BASE_WEIGHT_SUFFIXES = {"down_proj.weight", "up_proj.weight", "gate_proj.weight"}
    QUANT_WEIGHT_SUFFIXES = {
        "down_proj.weight_offset",
        "up_proj.weight_offset",
        "gate_proj.weight_offset",
        "down_proj.weight_scale",
        "up_proj.weight_scale",
        "gate_proj.weight_scale",
    }

    # 根据量化状态确定需要保存的权重后缀
    weight_suffixes = BASE_WEIGHT_SUFFIXES.union(QUANT_WEIGHT_SUFFIXES) if quant else BASE_WEIGHT_SUFFIXES
    num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers

    # 生成需要保存的权重名称集合
    def _generate_expert_weight_name(layer_id: int, expert_id: int, suffix: str) -> str:
        """生成单个专家权重的完整名称"""
        return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"

    weights_to_save = set()
    for expert_id in unsaved_expert_ids:
        for layer_id in range(num_hidden_layers):
            for suffix in weight_suffixes:
                weights_to_save.add(_generate_expert_weight_name(layer_id, expert_id, suffix))

    # 加载并保存权重
    model_loader = get_model_loader(vllm_config.load_config)
    all_weight_iter = model_loader.get_all_weight(vllm_config.model_config, model_runner.model)

    for weight_name, weight_tensor in all_weight_iter:
        if weight_name in weights_to_save:
            if weight_name.rsplit(".", 1)[-1] in BASE_WEIGHT_SUFFIXES:
                weight_tensor = weight_tensor.transpose(0, 1).contiguous()
            experts_saved_weights[weight_name] = weight_tensor

    # 更新已保存的专家ID列表
    experts_saved_ids.extend(unsaved_expert_ids)

    # 返回更新后的两个核心数据
    return experts_saved_ids, experts_saved_weights


def expand_parameter(old_param, axis: int = 0, extra_lines: int = 1) -> torch.nn.Parameter:
    old_shape = old_param.shape
    new_shape = list(old_param.shape)
    new_shape[axis] += extra_lines
    new_tensor = torch.zeros(
        size=new_shape,
        device=old_param.device,
        dtype=old_param.dtype,
    )
    if axis == 0:
        new_tensor[: old_shape[0]] = old_param.data
    elif axis == 1:
        new_tensor[:, : old_shape[1]] = old_param.data
    else:
        raise NotImplementedError(f"axis {axis} is not supported")
    return torch.nn.Parameter(new_tensor, requires_grad=old_param.requires_grad)


def expand_expert_weights(model_runner: NPUModelRunner, added_experts: dict[int, list[int]], quant: bool | str) -> None:
    rank = model_runner.vllm_config.parallel_config.data_parallel_rank
    current_rank_expand_experts = added_experts[rank]
    expand_lines = len(current_rank_expand_experts)
    for module in model_runner.model.modules():
        if isinstance(module, FusedMoE) and expand_lines:
            if quant:
                # todo 待验证
                module.w2_weight_list = expand_parameter(module.w2_weight_list, 0, expand_lines)
                module.w13_weight_list = expand_parameter(module.w13_weight_list, 0, expand_lines)
                module.w2_weight_scale_list = expand_parameter(module.w2_weight_scale_list, 0, expand_lines)
                module.w13_weight_scale_fp32_list = expand_parameter(module.w13_weight_scale_fp32_list, 0, expand_lines)
            else:
                module.w2_weight = expand_parameter(module.w2_weight, 0, expand_lines)
                module.w13_weight = expand_parameter(module.w13_weight, 0, expand_lines)


def dynamic_merge_view(
    target_tensor: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    dim_size1 = tensor1.shape[dim]
    dim_size2 = tensor2.shape[dim]
    total_dim_size = dim_size1 + dim_size2

    non_dim_shapes = [s for i, s in enumerate(tensor1.shape) if i != dim]
    for i, s in enumerate(tensor2.shape):
        if i != dim and s != non_dim_shapes[i if i < dim else i - 1]:
            raise ValueError(f"非合并维度{i}大小不匹配：tensor1={s} vs tensor2 = {non_dim_shapes[i]}")
    if target_tensor.shape[dim] != total_dim_size:
        raise ValueError(f"目标张量在维度{dim}大小必须为{dim_size1}+{dim_size2}={total_dim_size}")

    top_view = target_tensor.narrow(dim, 0, dim_size1)
    bottom_view = target_tensor.narrow(dim, dim_size1, dim_size2)

    top_view.copy_(tensor1)
    bottom_view.copy_(tensor2)

    return target_tensor


def reload_fault_expert_weights(
    model_runner: NPUModelRunner,
    global_experts_distribution: dict[int, list[int]],
    experts_saved_weights: dict[str, torch.Tensor],
    vllm_config: VllmConfig,
    redistributed_experts: dict[int, list[int]],
    added_experts: dict[int, list[int]],
    replaced_redundant_experts: dict[int : dict[int, tuple[int, int]]],
    quant: bool | str = False,
) -> None:
    def _load_single_expert(expert_id: int, target_index: int, quant: bool | str = False):
        prefix = f"{module.layer_name}.{expert_id}"
        w1_weight = experts_saved_weights[f"{prefix}.gate_proj.weight"]
        w2_weight = experts_saved_weights[f"{prefix}.down_proj.weight"]
        w3_weight = experts_saved_weights[f"{prefix}.up_proj.weight"]
        if quant:
            device = module.w2_weight_list.device
            module._load_w2(
                expert_data=module.w2_weight_list[target_index],
                shard_dim=1,
                loaded_weight=w2_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight_list[target_index],
                shard_dim=1,
                shard_id="w1",
                loaded_weight=w1_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight_list[target_index],
                shard_id=1,
                shard_dim="w3",
                loaded_weight=w3_weight.to(device),
                tp_rank=module.tp_rank,
            )
            # todo 加载量化权重 module.w2_weight_scale_list module.w13_weight_scale_fp32_list
        else:
            device = module.w2_weight.device
            module._load_w2(
                expert_data=module.w2_weight[target_index],
                shard_dim=1,
                loaded_weight=w2_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight[target_index],
                shard_dim=1,
                shard_id="w1",
                loaded_weight=w1_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight[target_index],
                shard_id=1,
                shard_dim="w3",
                loaded_weight=w3_weight.to(device),
                tp_rank=module.tp_rank,
            )

    old_rank = vllm_config.parallel_config.data_parallel_rank
    experts_to_expand = added_experts[old_rank]
    init_local_experts_count = len(global_experts_distribution[old_rank])
    slot_to_routed_expert_map = {}
    for redundant_expert_id, (pos, routed_expert_id) in replaced_redundant_experts[old_rank].items():
        slot_to_routed_expert_map[pos] = routed_expert_id

    for module in model_runner.model.modules():
        if isinstance(module, FusedMoE):
            # 加载新增专家
            if experts_to_expand:
                for idx, eid in enumerate(experts_to_expand):
                    _load_single_expert(eid, init_local_experts_count + idx, quant=quant)
            # 故障专家替换冗余专家
            if slot_to_routed_expert_map:
                for slot_pos, expert_id in slot_to_routed_expert_map.items():
                    _load_single_expert(expert_id, slot_pos, quant=quant)

    global_experts_distribution = {}
    for i, rank in enumerate(list(sorted(redistributed_experts.keys()))):
        global_experts_distribution[i] = redistributed_experts[rank]


def update_parallel_config(original_config: VllmConfig, update_config: dict[str, int]) -> VllmConfig:  # , worker_guard)
    original_config.parallel_config.data_parallel_size = update_config["data_parallel_size"]
    original_config.parallel_config.data_parallel_size_local = update_config["data_parallel_size_local"]
    original_config.parallel_config.data_parallel_rank = update_config["data_parallel_rank"]
    original_config.parallel_config.data_parallel_rank_local = update_config["data_parallel_rank_local"]
    original_config.parallel_config.expert_parallel_size = update_config["expert_parallel_size"]
    original_config.parallel_config.data_parallel_master_port = update_config["data_parallel_master_port"]
    # if original_config.fault_tolerance.config.enabel_fault_tolerance:
    #     worker_guard.update(original_config)


def init_ep2dp_map(dp_size: int, tp_size: int) -> dict[int, int]:
    ep2dp_map = {}
    for dp_rank in range(dp_size):
        ep_start = dp_rank * tp_size
        ep_end = (dp_rank + 1) * tp_size
        for ep_rank in range(ep_start, ep_end):
            ep2dp_map[ep_rank] = dp_rank
    return ep2dp_map


def update_ep2dp_map(ep2dp_map: dict[int, int], exclude_dp_ranks: list[int], rank_mapping: dict[str, int]) -> None:
    for old_ep_rank, dp_rank in ep2dp_map.items():
        if dp_rank != -1:
            if dp_rank in exclude_dp_ranks:
                ep2dp_map[old_ep_rank] = -1
            else:
                ep2dp_map[old_ep_rank] = rank_mapping[str(dp_rank)]


def update_elastic_info(
    use_mask_mc2: bool,
    elastic_info: torch.Tensor,
    expert_num: int,
    raw_ep_size: int,
    ep2dp: dict[int, int],
    share_expert_num: int = 0,
) -> None:
    if use_mask_mc2:
        if elastic_info is None:
            elastic_info = torch.full((4 + 2 * raw_ep_size,), -1, dtype=torch.int32).npu().contiguous()
        raw_ep_ranks = sorted(ep2dp.keys())
        valid_ep_ranks = [ep for ep in raw_ep_ranks if ep2dp[ep] != -1]
        descale_ep_size = len(valid_ep_ranks)
        is_descale = 1 if descale_ep_size < raw_ep_size else 0

        # Table1: epRankID -> localEpRankId(-1表示失效）
        table1 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
        for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
            table1[ep_rank] = local_ep_rank

        # Table2: localEpRankId -> epRankID(-1表示填充）
        table2 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
        for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
            if local_ep_rank < descale_ep_size:
                table2[local_ep_rank] = ep_rank

        # update elastic_info
        elastic_info[0] = is_descale
        elastic_info[1] = descale_ep_size
        elastic_info[2] = share_expert_num
        elastic_info[3] = expert_num
        # update Tabel1
        table1_start = 4
        elastic_info[table1_start : table1_start + raw_ep_size] = table1
        # update Table2
        table2_start = table1_start + raw_ep_size
        elastic_info[table2_start : table2_start + raw_ep_size] = table2
        set_elastic_info(elastic_info)
    else:
        set_elastic_info(None)


def gen_local_log2phy_map(global_log2phy_map: dict[int, list[int]]) -> torch.Tensor:
    num_logical_exp = len(global_log2phy_map)
    log2phy = torch.zeros(num_logical_exp, dtype=torch.int32, device="cpu")
    for log_expert_id in sorted(global_log2phy_map.keys()):
        replica_list = global_log2phy_map[log_expert_id]
        # num_replicas = len(replica_list)
        # phy_id = replica_list[global_rank % num_replicas]
        # todo：暂时只能使用每个逻辑专家的0号物理专家，使用上两行的写法进行负载均衡会有精度问题
        phy_id = replica_list[0]
        log2phy[log_expert_id] = phy_id
    return log2phy.npu()


def reconfigure_moe(
    use_mask_mc2: bool,
    modelrunner: NPUModelRunner,
    vllm_config: VllmConfig,
    num_global_logical_experts: int,
    num_global_new_phy_experts: int,
    log2phy: torch.Tensor,
):
    import vllm.envs as envs

    parallel_config = vllm_config.parallel_config
    new_ep_size = parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
    get_ascend_config().eplb_config.num_redundant_experts = num_global_new_phy_experts - num_global_logical_experts

    moe_moules = [
        module
        for module in modelrunner.model.modules()
        if (module.__class__.__name__ == "AscendFusedMoE" or module.__class__.__name__ == "SharedFusedMoE")
    ]

    for module in moe_moules:
        module.local_num_experts = num_global_new_phy_experts // new_ep_size
        module.global_num_experts = num_global_logical_experts
        module.global_redundant_experts_num = num_global_new_phy_experts - num_global_logical_experts
        module.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size=get_tp_group.world_size,
            pcp_size=get_pcp_group.world_size,
            dp_size=get_dp_group.world_size,
            vllm_parallel_config=parallel_config,
        )
        module.moe_config = FusedMoEConfig(
            num_experts=module.global_num_experts,
            experts_per_token=module.topk,
            hidden_dim=module.hidden_dim,
            intermediate_size_per_partition=module.intermediate_size_per_partition,
            num_local_experts=module.local_num_experts,
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
        module.moe_config.global_redundant_expert_num = module.global_redundant_experts_num
        module.log2phy = log2phy
        if not use_mask_mc2:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
            module.moe_config.supports_eplb = module.quant_method.supports_eplb
            eplb_config = get_ascend_config().eplb_config
            module.global_expert_map, module._expert_map, _, _ = init_eplb_config(
                eplb_config, module.moe_counter, module.moe_config
            )
            with set_current_vllm_config(vllm_config):
                setup_moe_comm_method()
            changed_moe_load_shape = module.local_num_experts - module.moe_load.shape[0]
            if modelrunner.dynamic_eplb and changed_moe_load_shape:
                module.moe_load = expand_parameter(module.moe_load, 0, changed_moe_load_shape)
            if vllm_config.model_config.quantization is not None:
                from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

                module.quant_method.quant_method = AscendW8A8DynamicFusedMoEMethod()
                # todo support other quant like w4a4 w4a8 ...
