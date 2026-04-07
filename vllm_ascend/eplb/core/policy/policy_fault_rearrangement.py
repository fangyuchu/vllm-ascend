from collections import defaultdict

import numpy as np

from .policy_abstract import DynamicConfig, EplbPolicy


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


class FaultRearrangement(EplbPolicy):
    def __init__(self, config: DynamicConfig):
        super().__init__(config)

        self.n_experts = None
        self.k_replicas = None
        self.max_swap_times = 100
        self.num_max_com = 1
        self.swap_threshold = 0
        # A3(16), A2(8)
        self.n_cards_per_nodes = 16
        self.n_add_expert_per_card = 0
        self.failed_cards = []
        self.enable_d2d_after_failure = False

    def get_original_workload(self):
        workload_new = np.zeros((self.n_layer, self.n_experts))
        if self.enable_d2d_after_failure:
            for layer_idx in range(self.n_layer):
                workload_dict: dict[int, int] = defaultdict(int)

                placement_layer = self.org_deployment[layer_idx].copy()
                workload_layer = self.org_deployment[layer_idx].copy()
                for card_idx in range(self.n_org_cards):
                    for index in range(self.n_experts_per_card):
                        workload_dict[placement_layer[card_idx][index]] += workload_layer[card_idx][index]
                for expert_idx in range(self.n_experts):
                    workload_new[layer_idx][expert_idx] = workload_dict[expert_idx]

        return workload_new

    def constraint_expert_local_exchange(self, old_deployment, new_deployment):
        new_deployment_list = []
        old_deployment_list = []
        # todo 变量命名
        for card_id in range(self.n_remain_cards):
            current_list = [int(x) for x in old_deployment[card_id]]
            new_list = [int(x) for x in new_deployment[card_id]]
            num = len(new_list)

            new_index = [-1] * num
            new_result = [-1] * num
            remaining_elements = []

            for i in range(num):
                flag = True
                for j in range(num):
                    if new_list[i] == current_list[j] and new_index[j] == -1:
                        new_index[j] = 0
                        new_result[j] = current_list[j]
                        flag = False
                        break
                if flag:
                    remaining_elements.append(new_list[i])

            index = 0
            for k in range(num):
                if new_result[k] == -1:
                    new_result[k] = remaining_elements[index]
                    index += 1

            new_deployment_list.append(new_result.copy())
            old_deployment_list.append(current_list.copy())

        return new_deployment_list, old_deployment_list

    def get_cur_expert_maps_dict(self, old_placement):
        all_layer_cur_expert_maps = []
        n_layer, n_rank, _ = old_placement.shape
        for layer_id in range(n_layer):
            cur_expert_maps = defaultdict(list)
            for rank_id in range(n_rank):
                cur_expert_maps[rank_id] = old_placement[layer_id][rank_id].tolist()
            all_layer_cur_expert_maps.append(cur_expert_maps)

        return all_layer_cur_expert_maps

    def rebalance_experts(self, current_expert_table, expert_workload):
        if expert_workload is not None:
            self.org_workload = expert_workload.numpy()

        self.org_deployment = current_expert_table.numpy()

        self.n_layer, self.n_org_cards, self.n_experts_per_card = self.org_deployment.shape
        self.n_experts = self.org_deployment.max() + 1

        rank_indices = np.arange(self.n_org_cards)
        mask = ~np.isin(rank_indices, self.failed_cards)
        self.remain_deployment = self.org_deployment[:, mask, :]
        self.n_remain_cards = self.remain_deployment.shape[1]

        layer_workload = self.get_original_workload()

        if self.n_remain_cards == 0:
            raise ValueError("All cards are faulty, no available cards.")

        self.org_expert_per_card = self.n_experts // self.n_remain_cards + 1
        if self.n_experts_per_card < self.org_expert_per_card:
            self.n_add_expert_per_card = self.org_expert_per_card - self.n_experts_per_card

        redistributed_experts = []
        old_deployment_after_h2d = []
        all_layer_need_load_h2d = []

        for layer_id in range(self.n_layer):
            cur_layer_deployment = self.remain_deployment[layer_id]
            cur_workload = layer_workload[layer_id]

            new_deployment, old_deployment, need_load_h2d = self._execute_allocation(cur_layer_deployment, cur_workload)

            new_deployment_list, old_deployment_list = self.constraint_expert_local_exchange(
                old_deployment, new_deployment
            )

            redistributed_experts.append(new_deployment_list)
            old_deployment_after_h2d.append(old_deployment_list)
            all_layer_need_load_h2d.append(need_load_h2d)

        return (
            np.array(redistributed_experts).tolist(),
            np.array(old_deployment_after_h2d).tolist(),
            all_layer_need_load_h2d,
            self.n_add_expert_per_card,
        )

    def expert_exchange_between_ranks(
        self,
        rank_assignments: np.ndarray,
        rank_loads: np.ndarray,
        num_com_between_rank: np.ndarray,
        rev_expert_per_rank: defaultdict[int, set[int]],
        updated_weights: np.ndarray,
    ):
        rank_deploy_sets = []
        for rank_id in range(self.n_remain_cards):
            rank_deploy_sets.append(set(rank_assignments[rank_id]))

        max_swap_times = self.max_swap_times
        max_rank_load = 0
        exchange = True
        while max_swap_times > 0:
            max_swap_times -= 1
            sorted_rank_idx = np.argsort(rank_loads, kind="stable")
            max_load_rank_id = int(sorted_rank_idx[-1])
            max_rank_load = rank_loads[max_load_rank_id]

            if not exchange:
                break

            exchange = False
            for swap_rank_id in sorted_rank_idx[:-1]:
                if (
                    num_com_between_rank[swap_rank_id][max_load_rank_id] < self.num_max_com
                    and num_com_between_rank[max_load_rank_id][swap_rank_id] < self.num_max_com
                ):
                    swap_rank_load = rank_loads[swap_rank_id]

                    max_rank_expert, swap_rank_expert, max_weight = self.swap_experts_between_ranks(
                        rank_deploy_sets[max_load_rank_id],
                        rank_deploy_sets[swap_rank_id],
                        rev_expert_per_rank[max_load_rank_id],
                        rev_expert_per_rank[swap_rank_id],
                        updated_weights,
                        max_rank_load,
                        swap_rank_load,
                    )

                    if max_rank_load - max_weight < self.swap_threshold or max_rank_expert == -1:
                        continue

                    rank_deploy_sets[max_load_rank_id].remove(max_rank_expert)
                    rank_deploy_sets[swap_rank_id].remove(swap_rank_expert)
                    rank_deploy_sets[max_load_rank_id].add(swap_rank_expert)
                    rank_deploy_sets[swap_rank_id].add(max_rank_expert)

                    rank_loads[max_load_rank_id] += updated_weights[swap_rank_expert] - updated_weights[max_rank_expert]
                    rank_loads[swap_rank_id] += updated_weights[max_rank_expert] - updated_weights[swap_rank_expert]

                    rev_expert_per_rank[max_load_rank_id].add(swap_rank_expert)
                    rev_expert_per_rank[swap_rank_id].add(max_rank_expert)

                    num_com_between_rank[swap_rank_id][max_load_rank_id] += 1
                    num_com_between_rank[max_load_rank_id][swap_rank_id] += 1

                    exchange = True
                    break

        ranks_deployment_after_swap = [list(s) for s in rank_deploy_sets]

        return ranks_deployment_after_swap, max_rank_load

    def swap_experts_between_ranks(
        self,
        max_rank_deployment_set,
        swap_rank_deployment_set,
        max_rank_rev_expert,
        swap_rank_rev_expert,
        workload,
        max_rank_load,
        swap_rank_load,
    ):
        max_rank_expert = -1
        swap_rank_expert = -1
        max_weight = max_rank_load

        for cur_expert_id in max_rank_deployment_set:
            if cur_expert_id in swap_rank_deployment_set or cur_expert_id in max_rank_rev_expert:
                continue

            cur_weight = workload[cur_expert_id]

            for next_expert_id in swap_rank_deployment_set:
                if next_expert_id in max_rank_deployment_set or next_expert_id in swap_rank_rev_expert:
                    continue

                next_weight = workload[next_expert_id]

                cur_load_after_swap = max_rank_load - cur_weight + next_weight
                next_load_after_swap = swap_rank_load - next_weight + cur_weight
                max_load_after_swap = max(cur_load_after_swap, next_load_after_swap)
                if max_load_after_swap < max_weight:
                    max_weight = max_load_after_swap
                    max_rank_expert = cur_expert_id
                    swap_rank_expert = next_expert_id

        return max_rank_expert, swap_rank_expert, max_weight

    def sort_expert_ids_by_workload(self, expert_ids, workload):
        expert_ids_np = np.array(expert_ids)
        sorted_indices = np.argsort(workload[expert_ids_np])
        sorted_expert_id = expert_ids_np[sorted_indices]

        return sorted_expert_id.tolist()

    def _swap_two_cards(
        self,
        card_selected,
        card_selected_count,
        surplus_id,
        deficit_id,
        allocate_amount,
    ):
        for _ in range(allocate_amount):
            surplus_card_expert_id = min(card_selected[surplus_id])

            card_selected[surplus_id].remove(surplus_card_expert_id)
            card_selected[deficit_id].add(surplus_card_expert_id)

            card_selected_count[surplus_id] -= 1
            card_selected_count[deficit_id] += 1

    def _load_no_backup_experts(self, old_deployment, redundant_expert_pos, no_backup_experts, expert_from_rank):
        sorted_result = []
        for card_id in range(self.n_remain_cards):
            if len(redundant_expert_pos[card_id]) > 0:
                sorted_result.append((card_id, len(redundant_expert_pos[card_id])))

        sorted_result.sort(key=lambda x: x[1], reverse=True)
        sorted_card_id = [item[0] for item in sorted_result]

        need_load_h2d = defaultdict(list)
        index = 0

        while no_backup_experts:
            card_id = sorted_card_id[index]
            if redundant_expert_pos[card_id]:
                expert_id = no_backup_experts.pop()
                pos = redundant_expert_pos[card_id].pop()
                old_deployment[card_id][pos] = int(expert_id)
                expert_from_rank[expert_id] = card_id
                need_load_h2d[card_id].append((pos, int(expert_id)))
                index += 1
            else:
                sorted_card_id.pop(index)
            index = index % len(sorted_card_id)

        return need_load_h2d

    def recomputing_workload(self, rank_assignments, org_workload):
        num_per_existing_expert = np.zeros(self.n_experts, dtype=np.int64)

        for rank in rank_assignments:
            for expert_id in rank:
                if expert_id != -1:
                    num_per_existing_expert[expert_id] += 1

        for n in num_per_existing_expert:
            if n == 0:
                raise ValueError(f"Currently fewer than {self.n_experts} experts are deployed")

        update_workload = org_workload / num_per_existing_expert

        return update_workload, num_per_existing_expert

    def fill_in_undeployed_ranks(self, rank_assignments, org_workload, redundant_expert_pos):
        update_workload, num_per_existing_expert = self.recomputing_workload(rank_assignments, org_workload)

        for card_id in range(self.n_remain_cards):
            need_redundancy = len(redundant_expert_pos[card_id])

            if need_redundancy > 0:
                sorted_expert_ids = np.argsort(-update_workload, kind="stable")

                for i in range(len(sorted_expert_ids)):
                    expert_id = sorted_expert_ids[i]
                    if expert_id not in rank_assignments[card_id]:
                        index = redundant_expert_pos[card_id].pop()
                        rank_assignments[card_id][index] = expert_id
                        num_per_existing_expert[expert_id] += 1
                        update_workload[expert_id] = org_workload[expert_id] / num_per_existing_expert[expert_id]
                        need_redundancy -= 1
                        if need_redundancy <= 0:
                            break

        rank_loads = np.zeros(len(rank_assignments), dtype=np.float32)
        for rank_id in range(self.n_remain_cards):
            for expert_id in rank_assignments[rank_id]:
                rank_loads[rank_id] += update_workload[expert_id]

        return update_workload, rank_loads

    def find_min_rank(self, rank_ids, n_expert_per_rank):
        if len(rank_ids) == 1:
            return rank_ids[0]

        min_rank_id = rank_ids[0]
        min_val = n_expert_per_rank[min_rank_id]

        for cur_rank_id in rank_ids[1:]:
            current_val = n_expert_per_rank[cur_rank_id]

            if current_val < min_val:
                min_val = current_val
                min_rank_id = cur_rank_id

        return min_rank_id

    def statistics_expert_distribution(self, single_layer_deployment):
        expert_to_rank = defaultdict(list)
        n_exist_experts = np.zeros(self.n_experts, dtype=np.int64)
        for rank_id in range(self.n_remain_cards):
            for expert_id in single_layer_deployment[rank_id]:
                expert_to_rank[expert_id].append(rank_id)
                n_exist_experts[expert_id] += 1

        num_redundant_experts = int(np.sum(np.maximum(n_exist_experts - 1, 0)))
        sorted_expert_ids = np.argsort(n_exist_experts)

        rank_route_expert = defaultdict(set)
        n_expert_per_rank = np.zeros(self.n_experts)
        no_backup_experts = []
        expert_from_rank = [-1] * self.n_experts
        for expert_id in sorted_expert_ids:
            if expert_to_rank[expert_id]:
                min_rank = self.find_min_rank(expert_to_rank[expert_id], n_expert_per_rank)
                rank_route_expert[min_rank].add(expert_id)
                n_expert_per_rank[min_rank] += 1
                expert_from_rank[expert_id] = min_rank
            else:
                no_backup_experts.append(expert_id)

        redundant_expert_pos: list[list[int]] = [[] for _ in range(self.n_remain_cards)]

        for rank_id in range(self.n_remain_cards):
            for index, expert_id in enumerate(single_layer_deployment[rank_id]):
                if expert_id not in rank_route_expert[rank_id]:
                    redundant_expert_pos[rank_id].append(index)

        return redundant_expert_pos, expert_from_rank, no_backup_experts, num_redundant_experts

    def compute_redundant_assignments(
        self,
        org_workload: np.ndarray,
        num_redundant_experts: int,
    ) -> tuple[list[tuple[int, float]], np.ndarray]:
        current_weights = []
        for expert_id in range(self.n_experts):
            current_weights.append((expert_id, org_workload[expert_id]))

        redundant_assignments = [0] * self.n_experts

        for i in range(num_redundant_experts):
            sorted_indices = np.argsort([w for _, w in current_weights], kind="stable")[::-1]
            for index in sorted_indices:
                target_expert = current_weights[index]
                expert_id, original_weight = target_expert

                current_redundancy = redundant_assignments[expert_id] + 1
                if current_redundancy < self.n_remain_cards:
                    new_avg_weight = (original_weight * current_redundancy) / (current_redundancy + 1)
                    redundant_assignments[expert_id] += 1
                    current_weights[index] = (expert_id, new_avg_weight)
                    break

        update_weight = np.zeros(self.n_experts, dtype=np.float32)
        for expert_id, expert_weight in current_weights:
            update_weight[expert_id] = expert_weight

        redundant_expert_list = []
        if num_redundant_experts > 0:
            for expert_id in range(self.n_experts):
                for _ in range(redundant_assignments[expert_id]):
                    redundant_expert_list.append((expert_id, float(update_weight[expert_id])))

            redundant_expert_list.sort(key=lambda x: x[1], reverse=True)

        return redundant_expert_list, update_weight

    def non_redundant_expert_information(
        self,
        origin_deployment: np.ndarray,
        updated_weights: np.ndarray,
        redundant_expert_pos: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        rank_assignments = np.full((self.n_remain_cards, self.n_experts_per_card), fill_value=-1, dtype=np.int64)
        rank_loads = np.zeros(self.n_remain_cards, dtype=np.float32)

        for rank_id in range(self.n_remain_cards):
            for index, expert_id in enumerate(origin_deployment[rank_id]):
                if index in redundant_expert_pos[rank_id]:
                    continue
                rank_assignments[rank_id][index] = expert_id
                rank_loads[rank_id] += updated_weights[expert_id]

        return rank_assignments, rank_loads

    def expand_deployment_table(
        self,
        deployment,
        redundant_expert_pos,
        n_add_expert_per_card,
    ):
        for card_id in range(self.n_remain_cards):
            for _ in range(n_add_expert_per_card):
                deployment[card_id].append(-1)

        for rank_id in range(self.n_remain_cards):
            for i in range(n_add_expert_per_card):
                redundant_expert_pos[rank_id].append(i + self.n_experts_per_card)

        self.n_experts_per_card += self.n_add_expert_per_card

    def distribute_redundant_experts(
        self,
        rank_assignments: np.ndarray,
        rank_loads: np.ndarray,
        redundant_expert_list: list[tuple[int, float]],
        expert_from_rank: list[int],
        redundant_expert_pos: list[list[int]],
    ) -> tuple[np.ndarray, defaultdict[int, set[int]], list[int]]:
        rev_expert_per_rank = defaultdict(set)
        num_com_between_rank = np.zeros((self.n_remain_cards, self.n_remain_cards), dtype=np.int64)

        for expert_id, weight in redundant_expert_list:
            candidate = -1
            send_rank = expert_from_rank[expert_id]
            for rank_id in range(self.n_remain_cards):
                if len(redundant_expert_pos[rank_id]) == 0:
                    continue
                if expert_id in rank_assignments[rank_id]:
                    continue
                if num_com_between_rank[send_rank][rank_id] >= self.num_max_com:
                    continue
                if candidate == -1 or rank_loads[rank_id] < rank_loads[candidate]:
                    candidate = rank_id

            if candidate != -1:
                pos = redundant_expert_pos[candidate].pop()
                rank_assignments[candidate][pos] = expert_id
                rank_loads[candidate] += weight

                num_com_between_rank[send_rank][candidate] += 1
                rev_expert_per_rank[candidate].add(expert_id)

        undeployed_ranks = []
        for rank_id in range(self.n_remain_cards):
            if len(redundant_expert_pos[rank_id]) > 0:
                undeployed_ranks.append(rank_id)

        return num_com_between_rank, rev_expert_per_rank, undeployed_ranks

    def _execute_allocation(
        self,
        cur_layer_deployment: np.ndarray,
        org_workload: np.ndarray,
    ):
        redundant_expert_pos, expert_from_rank, no_backup_experts, num_redundant_experts = (
            self.statistics_expert_distribution(cur_layer_deployment)
        )

        if self.n_add_expert_per_card > 0:
            raise ValueError("Capacity expansion is not supported.")
        self.expand_deployment_table(cur_layer_deployment, redundant_expert_pos, self.n_add_expert_per_card)

        if no_backup_experts:
            need_load_h2d = self._load_no_backup_experts(
                cur_layer_deployment, redundant_expert_pos, no_backup_experts, expert_from_rank
            )
            num_redundant_experts -= len(no_backup_experts)
        else:
            need_load_h2d = defaultdict(list)

        if self.enable_d2d_after_failure:
            redundant_expert_list, update_weight = self.compute_redundant_assignments(
                org_workload, num_redundant_experts
            )

            rank_assignments, rank_loads = self.non_redundant_expert_information(
                cur_layer_deployment, update_weight, redundant_expert_pos
            )

            num_com_between_rank, rev_expert_per_rank, undeployed_ranks = self.distribute_redundant_experts(
                rank_assignments, rank_loads, redundant_expert_list, expert_from_rank, redundant_expert_pos
            )

            if len(undeployed_ranks) > 0:
                update_weight, rank_loads = self.fill_in_undeployed_ranks(
                    rank_assignments, org_workload, redundant_expert_pos
                )

            new_deployment, after_swap_max_load = self.expert_exchange_between_ranks(
                rank_assignments, rank_loads, num_com_between_rank, rev_expert_per_rank, update_weight
            )
        else:
            new_deployment = cur_layer_deployment

        return new_deployment, cur_layer_deployment, need_load_h2d
