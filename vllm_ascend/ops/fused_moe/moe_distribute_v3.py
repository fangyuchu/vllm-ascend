# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""MoeDistribute V3 adapter for the Ascend MC2 token dispatcher.

The optional CANN transformer-ops dependency is imported lazily so the
normal V2 path keeps the same import and startup behavior. CANN 9.1 renamed
``npu_ops_transformer`` to ``cann_ops_transformer`` and removed the ``npu_``
prefix from the buffer methods, so both interfaces are supported here.
"""

from __future__ import annotations

import importlib
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import (
    get_mc2_group,
    get_v3_elastic_info,
)
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


EXPERT_TOKEN_NUMS_TYPE_COUNT = 1


class MoeDistributeV3Adapter:
    """Wrap ``MoeDistributeBuffer`` behind the current MC2 interface.

    A buffer is keyed by its process group and static tensor dimensions. A
    graph-captured buffer can later switch to a same-sized MC2 group through
    ``update_ctx_to_mc2_group`` without rebuilding the graph.
    """

    _COMM_ALG = ""
    _INSTANCES: weakref.WeakSet[MoeDistributeV3Adapter] = weakref.WeakSet()

    def __init__(self, max_tokens_per_rank: int) -> None:
        self.max_tokens_per_rank = max_tokens_per_rank
        self._buffer = None
        self._buffer_key: tuple | None = None
        self._buffer_rank: int | None = None
        self._context_group = None
        self._capture_active: torch.Tensor | None = None
        self._ccl_buffer_size = None
        self._elastic_info_signature: tuple[int, ...] | None = None
        self._INSTANCES.add(self)

    @staticmethod
    def _current_elastic_info_signature() -> tuple[int, ...] | None:
        elastic_info = get_v3_elastic_info()
        if elastic_info is None:
            return None
        return tuple(int(value) for value in elastic_info.detach().cpu().tolist())

    @staticmethod
    @contextmanager
    def _register_stateless_group_rank(
        device_group,
        rank: int,
        world_size: int,
    ):
        """Temporarily expose a stateless PG to PyTorch rank lookup.

        ``MoeDistributeBuffer`` calls ``dist.get_rank(group)`` internally,
        while vLLM's stateless process groups deliberately are not registered
        in the default c10d world. Keep this compatibility shim tightly scoped
        to buffer construction/update until the operator accepts an explicit
        rank.
        """
        try:
            default_rank = dist.get_rank()
        except Exception:
            yield
            return

        had_mapping = device_group in _world.pg_group_ranks
        previous_mapping = _world.pg_group_ranks.get(device_group)
        try:
            group_ranks = {group_rank: group_rank for group_rank in range(world_size)}
            group_ranks[default_rank] = rank
            _world.pg_group_ranks[device_group] = group_ranks
            yield
        finally:
            if had_mapping:
                _world.pg_group_ranks[device_group] = previous_mapping
            else:
                _world.pg_group_ranks.pop(device_group, None)

    @staticmethod
    def _load_buffer_cls():
        import_errors = []
        for module_name in (
            "cann_ops_transformer.ops",
            "npu_ops_transformer.ops",
        ):
            try:
                module = importlib.import_module(module_name)
                return module.MoeDistributeBuffer
            except (ImportError, AttributeError) as exc:
                import_errors.append((module_name, exc))

        details = "; ".join(f"{module_name}: {error}" for module_name, error in import_errors)
        raise RuntimeError(
            "MoeDistribute V3 requires cann_ops_transformer (CANN 9.1+) "
            "or the legacy npu_ops_transformer package, with "
            f"MoeDistributeBuffer exported from its ops module. {details}"
        ) from import_errors[-1][1]

    @staticmethod
    def _get_buffer_method(buffer, method_name: str):
        method = getattr(buffer, method_name, None)
        if method is not None:
            return method

        legacy_method_name = f"npu_{method_name}"
        method = getattr(buffer, legacy_method_name, None)
        if method is not None:
            return method

        raise RuntimeError(
            f"{type(buffer).__name__} provides neither {method_name} nor the legacy {legacy_method_name} method."
        )

    def _restore_ccl_buffer_size(self) -> None:
        if self._buffer is None or self._ccl_buffer_size is None:
            return

        current_buffer_size = self._buffer.ccl_buffer_size
        if hasattr(current_buffer_size, "value"):
            # Legacy npu_ops_transformer stores the size in a ctypes scalar.
            current_buffer_size.value = self._ccl_buffer_size
        elif current_buffer_size != self._ccl_buffer_size:
            # cann_ops_transformer exposes the size as a plain integer. It is
            # normally unchanged after construction, so avoid assigning when
            # the extension already holds the desired value.
            self._buffer.ccl_buffer_size = self._ccl_buffer_size

    def _ensure_buffer(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        moe_expert_num: int,
    ):
        # update_ctx can precede the global group switch while old ranks serve.
        mc2_group = self._context_group or get_mc2_group()
        device_group = mc2_group.device_group
        hidden_size = hidden_states.shape[-1]
        topk = topk_ids.shape[-1]
        buffer_key = (
            id(device_group),
            mc2_group.world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            self._COMM_ALG,
        )
        if self._buffer is not None and self._buffer_key == buffer_key:
            return self._buffer

        buffer_cls = self._load_buffer_cls()
        ccl_buffer_size = buffer_cls.get_low_latency_ccl_buffer_size(
            mc2_group.world_size,
            self.max_tokens_per_rank,
            hidden_size,
            moe_expert_num,
            topk,
            comm_alg=self._COMM_ALG,
        )
        with self._register_stateless_group_rank(
            device_group,
            mc2_group.rank_in_group,
            mc2_group.world_size,
        ):
            self._buffer = buffer_cls(
                device_group,
                ccl_buffer_size=ccl_buffer_size,
                comm_alg=0,
            )
        self._buffer_key = buffer_key
        self._buffer_rank = mc2_group.rank_in_group
        self._ccl_buffer_size = ccl_buffer_size
        self._elastic_info_signature = self._current_elastic_info_signature()
        logger.info(
            "Initialized MoeDistribute V3 buffer: ep_rank=%s, ep_size=%s, hidden_size=%s, experts=%s, topk=%s",
            mc2_group.rank_in_group,
            mc2_group.world_size,
            hidden_size,
            moe_expert_num,
            topk,
        )
        return self._buffer

    def prepare_for_shape(
        self,
        *,
        hidden_size: int,
        topk: int,
        moe_expert_num: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> bool:
        hidden_states = torch.empty((1, hidden_size), dtype=dtype, device=device)
        topk_ids = torch.empty((1, topk), dtype=torch.int32, device=device)
        self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)
        return True

    def update_ctx_to_mc2_group(
        self,
        mc2_group: GroupCoordinator | None = None,
    ) -> bool:
        """Switch an existing buffer to a same-sized MC2 process group."""
        if self._buffer is None or self._buffer_key is None:
            return False

        mc2_group = mc2_group or get_mc2_group()
        if self._buffer_rank != mc2_group.rank_in_group:
            raise RuntimeError(
                "MoeDistribute V3 update_ctx cannot change captured ep_rank_id: "
                f"old={self._buffer_rank}, new={mc2_group.rank_in_group}"
            )
        if self._buffer_key[1] != mc2_group.world_size:
            raise RuntimeError(
                "MoeDistribute V3 update_ctx requires the captured EP size "
                f"to remain unchanged, but got old={self._buffer_key[1]}, "
                f"new={mc2_group.world_size}."
            )
        elastic_info_signature = self._current_elastic_info_signature()
        if self._buffer_key[0] == id(mc2_group.device_group) and self._elastic_info_signature == elastic_info_signature:
            return False

        self._restore_ccl_buffer_size()
        with (
            self._register_stateless_group_rank(
                mc2_group.device_group,
                mc2_group.rank_in_group,
                mc2_group.world_size,
            ),
            torch.inference_mode(),
        ):
            self._buffer.update_ctx(mc2_group.device_group)
        self._buffer_key = (id(mc2_group.device_group), *self._buffer_key[1:])
        self._context_group = mc2_group
        self._elastic_info_signature = elastic_info_signature
        return True

    @staticmethod
    def _check_quant_mode(quant_mode: int) -> None:
        if quant_mode not in (0, 2):
            raise RuntimeError(f"MoeDistribute V3 supports communication quant_mode 0 or 2, but got {quant_mode}.")

    @staticmethod
    def _normalize_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
        if topk_ids.dtype == torch.int32:
            return topk_ids
        return topk_ids.to(torch.int32)

    def _remap_topk_ids_for_capture(
        self,
        topk_ids: torch.Tensor,
        elastic_info: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm_ascend.distributed.elastic_ep.v3_capture import (
            get_v3_capture_session,
        )

        if elastic_info is None or get_v3_capture_session() is None:
            return topk_ids
        if self._capture_active is None:
            self._capture_active = torch.ones((), dtype=torch.bool, device=topk_ids.device)
        # Capture uses only the newly added ranks. Route deterministic dummy
        # tokens to their dense expert range and keep experts unique per row.
        active_experts = torch.clamp(
            elastic_info[3].to(topk_ids.device),
            min=1,
        )
        row = torch.arange(
            topk_ids.shape[0],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(1)
        column = torch.arange(
            topk_ids.shape[1],
            dtype=torch.int32,
            device=topk_ids.device,
        ).unsqueeze(0)
        remapped_topk_ids = torch.remainder(row + column, active_experts)
        # A committed topology may still need elastic rank translation. Keep
        # capture-only routing independent of that flag and of later FT masks.
        return torch.where(
            self._capture_active,
            remapped_topk_ids,
            topk_ids,
        )

    def finish_capture(self) -> None:
        if self._capture_active is not None:
            with torch.inference_mode():
                self._capture_active.zero_()

    @staticmethod
    def _is_graph_mode() -> bool:
        try:
            forward_context = get_forward_context()
        except Exception:
            return False
        additional_kwargs = getattr(forward_context, "additional_kwargs", None)
        if additional_kwargs is not None:
            mode = additional_kwargs.get("cudagraph_runtime_mode")
        else:
            mode = getattr(forward_context, "cudagraph_runtime_mode", None)
        if mode is None:
            return False
        if isinstance(mode, CUDAGraphMode):
            return mode != CUDAGraphMode.NONE
        try:
            return CUDAGraphMode(mode) != CUDAGraphMode.NONE
        except (TypeError, ValueError):
            return bool(mode)

    @classmethod
    def _active_mask_for_graph(
        cls,
        active_mask: torch.Tensor | None,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        if active_mask is None or not cls._is_graph_mode() or active_mask.shape[0] != topk_ids.shape[0]:
            return None
        return active_mask

    def dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        moe_expert_num: int,
        quant_mode: int,
    ) -> MoETokenDispatchOutput[MoEMC2CombineMetadata]:
        self._check_quant_mode(quant_mode)
        hidden_states = token_dispatch_input.hidden_states
        topk_ids = self._normalize_topk_ids(token_dispatch_input.topk_ids)
        elastic_info = get_v3_elastic_info()
        topk_ids = self._remap_topk_ids_for_capture(topk_ids, elastic_info)
        buffer = self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)

        kwargs = {
            "x": hidden_states,
            "topk_idx": topk_ids,
            "num_experts": moe_expert_num,
            "quant_mode": quant_mode,
            "comm_alg": self._COMM_ALG,
            "num_max_dispatch_tokens_per_rank": topk_ids.shape[0],
        }
        active_mask = self._active_mask_for_graph(
            token_dispatch_input.routing.mc2_mask,
            topk_ids,
        )
        if active_mask is not None:
            kwargs["x_active_mask"] = active_mask
        if elastic_info is not None:
            kwargs["elastic_info"] = elastic_info

        dispatch = self._get_buffer_method(buffer, "low_latency_dispatch")
        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            expand_scales,
        ) = dispatch(**kwargs)
        if not token_dispatch_input.quant.dispatch_with_quant:
            dynamic_scale = None

        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=EXPERT_TOKEN_NUMS_TYPE_COUNT,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=torch.empty(
                    0,
                    dtype=torch.int32,
                    device=hidden_states.device,
                ),
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask,
                ori_x=hidden_states,
            ),
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: MoEMC2CombineMetadata,
        moe_expert_num: int,
    ) -> torch.Tensor:
        if combine_metadata.ori_x is None:
            raise RuntimeError("MoeDistribute V3 combine requires the dispatch input ori_x.")
        topk_ids = self._normalize_topk_ids(combine_metadata.topk_ids)
        buffer = self._ensure_buffer(hidden_states, topk_ids, moe_expert_num)
        kwargs = {
            "x": hidden_states,
            "topk_idx": topk_ids,
            "topk_weights": combine_metadata.topk_weights.to(torch.float32),
            "assist_info_for_combine": combine_metadata.assist_info_for_combine,
            "ep_send_counts": combine_metadata.ep_recv_counts,
            "num_experts": moe_expert_num,
            "comm_alg": self._COMM_ALG,
            "num_max_dispatch_tokens_per_rank": topk_ids.shape[0],
            "ori_x": combine_metadata.ori_x,
            "expand_scales": combine_metadata.expand_scales,
        }
        active_mask = self._active_mask_for_graph(
            combine_metadata.mc2_mask,
            topk_ids,
        )
        if active_mask is not None:
            kwargs["x_active_mask"] = active_mask
        if (elastic_info := get_v3_elastic_info()) is not None:
            kwargs["elastic_info"] = elastic_info
        combine = self._get_buffer_method(buffer, "low_latency_combine")
        return combine(**kwargs)


def update_moe_distribute_v3_contexts(mc2_group=None) -> int:
    """Move the live V3 buffer to ``mc2_group`` on existing ranks.

    ``MoeDistributeBuffer.update_ctx`` is the existing-rank half of the same
    collective in which newly added ranks construct their buffer. Exactly one
    V3 dispatcher is expected to have been exercised by the model; entering
    the collective more than once would leave the new ranks unmatched.
    """
    initialized_adapters = [
        adapter
        for adapter in list(MoeDistributeV3Adapter._INSTANCES)
        if adapter._buffer is not None and adapter._buffer_key is not None
    ]
    if len(initialized_adapters) != 1:
        raise RuntimeError(
            "V3 scale-up requires exactly one initialized MoeDistribute "
            f"buffer on each existing rank, found {len(initialized_adapters)}."
        )

    return int(initialized_adapters[0].update_ctx_to_mc2_group(mc2_group))
