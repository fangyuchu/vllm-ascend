# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Stable routing and context storage for MegaMoe restore-only expansion."""

import torch

from vllm_ascend.distributed.parallel_state import register_stateless_group_rank


def ep_to_mc2_ranks(ep_ranks: list[int], mc2_ranks: list[int]) -> list[int]:
    if len(set(ep_ranks)) != len(ep_ranks) or sorted(ep_ranks) != sorted(mc2_ranks):
        raise ValueError("MegaMoe requires identical EP and MC2 rank sets")
    return [mc2_ranks.index(rank) for rank in ep_ranks]


class MegaMoeCaptureState:
    def __init__(self, buffer, num_local_experts: int, rank_map: list[int]):
        if num_local_experts <= 0 or buffer.num_experts != num_local_experts * buffer.ep_world_size:
            raise ValueError("MegaMoe capture requires fixed local expert capacity")
        self.buffer = buffer
        self.num_local_experts = num_local_experts
        self.rank_map = torch.tensor(rank_map, dtype=torch.int64, device=buffer.context.device)
        self.set_rank_map(rank_map)
        self.capture_active = torch.zeros((), dtype=torch.bool, device=buffer.context.device)
        self.capture_experts: torch.Tensor | None = None

    @torch.inference_mode()
    def set_rank_map(self, rank_map: list[int]) -> None:
        if sorted(rank_map) != list(range(self.buffer.ep_world_size)):
            raise ValueError("MegaMoe rank map must be a full physical-rank permutation")
        self.rank_map.copy_(torch.tensor(rank_map, dtype=self.rank_map.dtype, device=self.rank_map.device))

    def route(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # EPLB counts are recorded before this conversion, in EP order.
        # Only the fused communication operator consumes MC2 physical IDs.
        ids = topk_ids.to(torch.int64)
        ranks = torch.div(ids.clamp_min(0), self.num_local_experts, rounding_mode="floor")
        physical = self.rank_map[ranks] * self.num_local_experts + ids.remainder(self.num_local_experts)
        physical = torch.where(ids >= 0, physical, ids).to(torch.int32)
        if self.capture_experts is not None:
            rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1)
            cols = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            dummy = self.capture_experts[(rows + cols) % self.capture_experts.numel()]
            physical = torch.where(self.capture_active, dummy, physical)
        return physical

    @torch.inference_mode()
    def prepare_capture(self, active_mc2_ranks: list[int], topk: int) -> None:
        if self.capture_experts is not None:
            raise RuntimeError("MegaMoe capture routing has already been allocated")
        if (
            not active_mc2_ranks
            or len(set(active_mc2_ranks)) != len(active_mc2_ranks)
            or any(rank < 0 or rank >= self.buffer.ep_world_size for rank in active_mc2_ranks)
            or len(active_mc2_ranks) * self.num_local_experts < topk
        ):
            raise ValueError("MegaMoe capture requires enough distinct experts on new ranks")
        self.capture_experts = torch.tensor(
            [
                rank * self.num_local_experts + slot
                for rank in active_mc2_ranks
                for slot in range(self.num_local_experts)
            ],
            dtype=torch.int32,
            device=self.buffer.context.device,
        )
        self.buffer.clean_mask_buffer()
        for rank in range(self.buffer.ep_world_size):
            if rank not in active_mc2_ranks:
                self.buffer.update_mask_buffer(rank, True)
        self.capture_active.fill_(True)

    @torch.inference_mode()
    def finish_capture(self) -> None:
        self.capture_active.zero_()
        self.buffer.clean_mask_buffer()

    @torch.inference_mode()
    def update_context(self, mc2_group) -> None:
        buffer = self.buffer
        if mc2_group.world_size != buffer.ep_world_size or mc2_group.rank_in_group != buffer.rank_id:
            raise ValueError("MegaMoe restore must preserve physical world size and local MC2 slot")
        if not callable(getattr(buffer, "update_group", None)) or buffer.mask_buffer is None:
            raise RuntimeError("MegaMoe restore requires PR #12345 and a mask allocated before capture")
        addresses = (buffer.context.data_ptr(), buffer.mask_buffer.data_ptr())
        scalars = (buffer.ccl_buffer_size, buffer.topo_type, buffer.rank_num_per_server)
        with register_stateless_group_rank(mc2_group.device_group, mc2_group.rank_in_group, mc2_group.world_size):
            buffer.update_group(mc2_group.device_group)
        if addresses != (buffer.context.data_ptr(), buffer.mask_buffer.data_ptr()):
            raise RuntimeError("MegaMoe update_group changed captured context or mask storage")
        if scalars != (buffer.ccl_buffer_size, buffer.topo_type, buffer.rank_num_per_server):
            raise RuntimeError("MegaMoe update_group changed captured communication parameters")
        # update_group invalidates old local-buffer views. Always acquire a new one.
        buffer.get_local_buffer_tensor(torch.uint8).zero_()
