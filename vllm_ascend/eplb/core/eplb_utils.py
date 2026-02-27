#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove eplb utils.
import json
import os.path

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_ep_group
from vllm.logger import logger

_GLOBAL_PLACEMENT = None


def expert_file_to_tensor(expert_map_path, layer_id):
    with open(expert_map_path) as f:
        data = json.load(f)
    physical_count = 0
    device_data = []
    if layer_id > data["moe_layer_count"]:
        raise ValueError("Invalid EPLB Table")
    if layer_id == data["moe_layer_count"]:
        logger.warning("Init expert map of mtp/eagle when using sample.")
        return None, None
    for device in data["layer_list"][layer_id]["device_list"]:
        physical_count += len(device["device_expert"])
        device_data.append(device["device_expert"])
    global_placement = torch.tensor(device_data, dtype=torch.int32)
    return global_placement, physical_count


def generate_global_placement(n_expert, ep_size, n_redundant):
    all_experts = np.arange(n_expert)
    groups = np.array_split(all_experts, ep_size)
    for i in range(n_redundant):
        j = i % ep_size + 1
        if len(groups[-j]) == 0:
            groups[-j] = np.append(groups[-j], j)
        else:
            groups[-j] = np.append(groups[-j], (groups[-j][-1] + 1) % n_expert)
    return torch.tensor(groups, dtype=torch.int32)


def init_eplb_config(eplb_config, layer_id, moe_config):
    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    global_placement = None
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    if expert_map_path:
        if not (os.path.exists(expert_map_path) and os.access(expert_map_path, os.R_OK)):
            raise ValueError("Invalid EPLB path")
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        if physical_count is not None:
            n_redundant = physical_count - n_experts
            if not moe_config.supports_eplb:
                raise ValueError("Eplb supports only w8a8_dynamic quantization.")
        else:
            eplb_enable = False

    if global_placement is None:
        global_placement = generate_global_placement(n_experts, ep_size, n_redundant)

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, None, n_redundant
    global_expert_map = []
    for rankid in range(ep_size):
        expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
        global_expert_map.append(expert_map)
        if rankid == moe_config.ep_rank:
            local_expert_map = expert_map.npu()
    global_expert_map = torch.stack(global_expert_map)
    log2phy = generate_log2phy_map(global_expert_map)[moe_config.ep_rank].npu() if eplb_enable else None

    return global_expert_map, local_expert_map, log2phy, n_redundant


def reinit_expert(local_expert_map, num_moe_layers, layer_id):
    global _GLOBAL_PLACEMENT
    cpu_group = get_ep_group().cpu_group
    rank = cpu_group.rank()
    world_size = cpu_group.size()
    num_local_experts = local_expert_map.max() + 1
    if _GLOBAL_PLACEMENT is None:
        flag = torch.tensor([False], dtype=torch.bool, device="cpu")
        dist.broadcast(flag, src=0, group=cpu_group)
        if flag:
            shape = (num_moe_layers, world_size, num_local_experts)
            _GLOBAL_PLACEMENT = torch.empty(shape, dtype=torch.int32, device="cpu")
            dist.broadcast(_GLOBAL_PLACEMENT, src=0, group=cpu_group)

    if _GLOBAL_PLACEMENT is not None:
        global_placement = _GLOBAL_PLACEMENT[layer_id]
        n_experts = len(torch.unique(global_placement))
        global_expert_map = []
        for rankid in range(world_size):
            local_expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
            local_placement = global_placement[rankid]
            local_expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
            global_expert_map.append(local_expert_map)
        global_expert_map = torch.stack(global_expert_map)
        log2phy = generate_log2phy_map(global_expert_map)[rank].npu()

        return global_expert_map, global_expert_map[rank].npu(), log2phy
    else:
        return None


def generate_log2phy_map(expert_map, sync=False):
    if sync:
        cpu_group = get_ep_group().cpu_group
        dice = torch.randint(low=0, high=256, size=(1,), device="cpu")
        dist.all_reduce(dice, op=dist.ReduceOp.SUM, group=cpu_group)
    else:
        dice = 0

    log2phy_map = expert_map.clone().cpu().to(torch.int32)
    device = "cpu"
    world_size, num_experts = log2phy_map.shape
    num_local_experts = log2phy_map.max() + 1
    row_indices = (
        torch.arange(world_size, device=device).view(-1, 1).expand(world_size, num_experts) * num_local_experts
    )
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]
    mask_valid = log2phy_map != -1
    valid_cnts = mask_valid.sum(dim=0)
    assert valid_cnts.all()
    sel_off = dice % valid_cnts
    sorted_vals, _ = log2phy_map.sort(dim=0, stable=True, descending=True)
    selected = sorted_vals[sel_off, torch.arange(num_experts, device=device)]
    log2phy_map = selected.unsqueeze(0).expand(world_size, -1)

    return log2phy_map
