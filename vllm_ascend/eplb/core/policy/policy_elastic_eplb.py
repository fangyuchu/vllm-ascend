from typing import cast

import numpy as np

from .policy_abstract import DynamicConfig, EplbPolicy
from .policy_default_eplb import DefaultEplb


class DynamicTable:
    # workload_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: workload (heat) at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the workload (heat) of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
    workload_table = None

    # placement_table:
    # 3D matrix: [layer, gpus, experts_per_gpu_per_layer] -> value: physical expert ID at the corresponding position
    # Size: number of layers * number of GPUs * number of experts per GPU per layer
    # The element at (i, j, k) represents the physical expert ID of the k-th expert on the j-th GPU in the i-th layer
    # For experts that are not available or collected, the value is set to -1
    placement_table = None


class ElasticEPLB(EplbPolicy):
    def __init__(self, config: DynamicConfig, new_ep_size: int = None):
        super().__init__(config)
        self._new_ep_size = new_ep_size

    def set_new_ep_size(self, new_ep_size):
        self._new_ep_size = new_ep_size

    def rebalance_experts(self, current_expert_table, expert_workload):
        assert self._new_ep_size is not None
        new_ep_size = self._new_ep_size
        info = DynamicTable()
        info.workload_table = np.array(expert_workload)
        info.placement_table = np.array(current_expert_table)
        assert info.workload_table is not None
        layer_num, num_npus, experts_per_npu = info.workload_table.shape
        assert info.placement_table is not None
        assert num_npus != new_ep_size
        row = cast(np.ndarray, info.placement_table[0])
        expert_ids, counts = np.unique(row, return_counts=True)
        num_original_expert = len(expert_ids)
        layer_workloads = DefaultEplb.add_redundant(info.placement_table, info.workload_table, num_original_expert)
        max_heat_per_layer_before = DefaultEplb.calculate_max_heat_per_layer(info.workload_table, layer_num)
        npu_heat_all_origin = sum(max_heat_per_layer_before)
        num_redundancy_expert = new_ep_size * experts_per_npu - num_original_expert

        # Perform load balancing and deploy redundant experts
        layer_num = layer_workloads.shape[0]
        expert_num = layer_workloads.shape[1]
        # Validate that the number of experts, number of cards, and number of redundant experts
        # do not exceed the number of cards.
        if num_original_expert != expert_num:
            raise ValueError(
                f"the number of original experts {num_original_expert} must be equal to expert_num {expert_num}"
            )

        if num_npus <= 0:
            raise ValueError("the number of NPUs must be greater than 0")

        if experts_per_npu > expert_num:
            raise ValueError(
                f"the number of experts per NPU {experts_per_npu} must be less than the number of experts {expert_num}"
            )

        if new_ep_size * experts_per_npu < num_original_expert:
            raise ValueError(
                f"new_ep_size {new_ep_size} * experts_per_npu {experts_per_npu}  "
                f"must be grater than or equal to num_original_expert {num_original_expert}"
            )

        # Number of experts deployed on each card includes one redundant expert
        global_deployment: list[list[list[int]]] = [[[] for _ in range(new_ep_size)] for _ in range(layer_num)]
        # Iterate to obtain the placement strategy for each layer, taking computational balance into account
        max_heat_per_layer_after = np.zeros([layer_num])
        for layer in range(layer_num):
            # Get the expert IDs and their corresponding workloads for the current layer;
            # workloads need to be normalized, and one redundant expert is added per card
            weights = np.zeros((expert_num,), dtype="object")
            for expert_id, workload_weight in enumerate(layer_workloads[layer]):
                weights[expert_id] = (expert_id, workload_weight)

            # Obtain the globally balanced placement strategy for each layer
            result, layer_deployment = DefaultEplb.original_compute_balanced_pack_redundancy(
                weights, new_ep_size, num_redundancy_expert
            )

            global_deployment[layer] = layer_deployment
            max_heat_per_layer_after[layer] = max(result, key=lambda x: x["total_weight"])["total_weight"]

        new_global_deployment = global_deployment
        # Obtain the priority of each layer
        layer_changed_ratio = []
        for layer_idx in range(layer_num):
            layer_changed_ratio.append(max_heat_per_layer_after[layer_idx] / max_heat_per_layer_before[layer_idx])

        per_layer_priority = np.argsort(layer_changed_ratio)
        npu_heat_all_after = sum(max_heat_per_layer_after)

        change = 0
        if npu_heat_all_after < 0.95 * npu_heat_all_origin:
            change = 1

        return change, per_layer_priority, np.array(new_global_deployment).tolist()
