# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU expert redistribution and weight reload for fault-tolerance scale-down.

The reload deliberately does not degrade: a checkpoint, weight layout or quant
scheme it cannot handle raises. Every fallback that used to sit here read the
whole checkpoint, or wrote tensors that did not match the runtime layout, on
paths that could not succeed anyway.
"""

import glob
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
import torch_npu
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import (
    AscendW4A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    scale_from_float_to_int64,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, COMPRESSED_TENSORS_METHOD, maybe_trans_nz

__all__ = [
    "build_orig_to_dense_rank_table",
    "densify_routing_table_physical_ids",
    "reload_experts_from_disk",
]

# Checkpoint name suffixes (relative to "<layer_name>.<expert_id>.") of the
# tensors a reload may consume. weight_offset is loaded at startup but never
# consumed by the MoE apply path, so it is not reloaded here.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"
# scale_bias is only present in the msModelSlim (modelslim / "ascend") format;
# compressed_tensors computes it on the fly, so it simply never has these keys.
_W13_SCALE_BIAS_SUFFIXES = ("gate_up_proj.scale_bias", "gate_proj.scale_bias", "up_proj.scale_bias")
_W2_SCALE_BIAS_SUFFIX = "down_proj.scale_bias"

# Quant schemes the reload supports (gated in _reload_local_slots).
_SUPPORTED_QUANT_TYPES = {
    AscendUnquantizedFusedMoEMethod,
    AscendW8A8DynamicFusedMoEMethod,
    AscendW4A8DynamicFusedMoEMethod,
}

# (layer, expert) pairs per chunk: a larger chunk amortises the device rounds
# further but grows the stack's transient device memory.
_RELOAD_CHUNK_SIZE = 32

# Threads assembling one chunk; bounded because worker processes share host
# memory bandwidth.
_GATHER_MAX_WORKERS = 8


def _get_ckpt_name_mapper(model: torch.nn.Module) -> Callable[[str], str]:
    """Map raw checkpoint weight names into the model's runtime namespace."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        # No declared mapping means the checkpoint stores weights under
        # runtime names already (e.g. Qwen3-MoE), so identity is correct.
        return lambda name: name
    return lambda name: mapper._map_name(name) or name


def build_orig_to_dense_rank_table(ep_world_size: int, dead_ranks: set[int]) -> torch.Tensor:
    """Build the orig-rank -> densified-rank mapping table.

    Returns a ``[ep_world_size]`` int32 tensor where ``table[orig_rank]`` is
    the densified rank (-1 for dead ranks), i.e. the kernel's table1 view of
    the elastic_info layout. Densified ranks are assigned in ascending
    original-rank order over the survivors.
    """
    table = torch.full((ep_world_size,), -1, dtype=torch.int32)
    alive = sorted(set(range(ep_world_size)) - dead_ranks)
    for dense_rank, orig_rank in enumerate(alive):
        table[orig_rank] = dense_rank
    return table


def densify_routing_table_physical_ids(
    routing_table: torch.Tensor,
    orig_to_dense_rank: torch.Tensor,
    num_local_experts: int,
) -> None:
    """Renumber a routing table's physical ids into the densified id space.

    In scale-down mode the MC2 dispatch kernel computes a token's destination
    as ``table2[expert_id // num_local]`` and the combine kernel drops any id
    >= the shrunk physical expert count, so the ids produced by the EPLB
    mapping must be dense-rank-major: ``dense_rank * num_local + slot``.
    Keeping original ids only works when the dead ranks happen to be a suffix
    (then table2 is the identity on the alive prefix); a dead rank in the
    middle misroutes tokens or crashes the kernel on a -1 rank lookup.

    The update is an in-place ``copy_`` with an unchanged shape, so captured
    graphs keep pointing at valid storage.

    Args:
        routing_table: ``expert_replica_routing_table`` of one MoE layer
            (device, int32), holding original global physical ids.
        orig_to_dense_rank: ``[ep_world_size]`` original EP rank -> densified
            rank (-1 for dead ranks), i.e. elastic_info's table1.
        num_local_experts: physical slots per EP rank (unchanged by
            scale-down).
    """
    ids = routing_table.to(torch.int64)
    if bool((ids < 0).any()):
        raise RuntimeError(
            "[FT] expert replica routing table references empty slots after "
            "redistribution; every logical expert must have a live replica."
        )
    orig_rank = torch.div(ids, num_local_experts, rounding_mode="floor")
    dense_rank = orig_to_dense_rank.to(device=ids.device, dtype=torch.int64)[orig_rank]
    if bool((dense_rank < 0).any()):
        raise RuntimeError(
            "[FT] expert replica routing table references dead EP ranks after "
            "redistribution; the placement did not vacate the dead ranks."
        )
    dense_ids = dense_rank * num_local_experts + ids % num_local_experts
    routing_table.copy_(dense_ids.to(routing_table.dtype))


def _tp_shard_info(layer) -> tuple[int, int]:
    """(tp_rank, tp_size) of the MoE weights of a routed-experts module."""
    parallel_config = layer.moe_config.moe_parallel_config
    return parallel_config.tp_rank, parallel_config.tp_size


def _shard_row(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's row shard (dim 0) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[0] // tp_size
    return t.narrow(0, tp_rank * shard, shard)


def _shard_col(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's column shard (dim 1) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[1] // tp_size
    return t.narrow(1, tp_rank * shard, shard)


def _gather_w13(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Assemble one expert's w13 ([2I_local, H]) from checkpoint tensors."""
    fused = tensors.get("gate_up_proj.weight")
    if fused is not None:
        half = fused.shape[0] // 2
        gate, up = fused[:half], fused[half:]
    else:
        gate, up = tensors["gate_proj.weight"], tensors["up_proj.weight"]
    return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0)


def _gather_w13_scale(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """1-D w13 scale ([2I_local]) from checkpoint tensors."""
    fused = tensors.get("gate_up_proj.weight_scale")
    if fused is not None:
        half = fused.shape[0] // 2
        gate, up = fused[:half], fused[half:]
    else:
        gate = tensors["gate_proj.weight_scale"]
        up = tensors["up_proj.weight_scale"]
    return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0).view(-1)


def _gather_w13_scale_bias(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """2-D model-slim w13 scale_bias ([2I, 1]) from checkpoint tensors."""
    fused = tensors.get("gate_up_proj.scale_bias")
    if fused is not None:
        half = fused.shape[0] // 2
        gate, up = fused[:half], fused[half:]
    else:
        gate = tensors["gate_proj.scale_bias"]
        up = tensors["up_proj.scale_bias"]
    return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0)


def _reinterpret_fp32_to_int64(t: torch.Tensor) -> torch.Tensor:
    """Bit-preserving fp32 -> int64 cast (raw scale bits stored as int64).

    Mirrors ``AscendW4A8DynamicFusedMoEMethod._process_scale``'s
    reinterpretation; no transpose here (the caller supplies the flattened
    scale)."""
    arr = t.cpu().numpy()
    arr.dtype = np.uint32
    return torch.from_numpy(arr.astype(np.int64))


def _w4a8_scale_bias_modelslim(w2_scale_bias: torch.Tensor) -> torch.Tensor:
    """modelslim scale_bias: reduce the replica/group dim, [out, 16//tp] -> [out]."""
    return w2_scale_bias.transpose(0, 1).contiguous().sum(0)


def _assemble_expert(
    pair: tuple[int, int],
    routed_layers: list,
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    quant_type: type,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """CPU assembly of one expert's (w13, w2, w13_scale, w2_scale,
    w13_scale_bias, w2_scale_bias); thread-safe (read-only shared state).

    Scales/scale_bias are None for the unquantized scheme; the w2 scale is a
    view into its bucket (callers must not write it). Mirrors the relevant
    ``process_weights_after_loading`` per-expert transforms (TP sharding is a
    no-op under EP, which scale_down requires).
    """
    layer_idx, logical_id = pair
    routed = routed_layers[layer_idx]
    w8a8 = quant_type is AscendW8A8DynamicFusedMoEMethod
    w4a8 = quant_type is AscendW4A8DynamicFusedMoEMethod
    if getattr(routed, "w13_bias", None) is not None or getattr(routed, "w2_bias", None) is not None:
        raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
    tp_rank, tp_size = _tp_shard_info(routed)
    tensors = buckets[(layer_idx, logical_id)]

    w13_raw = _gather_w13(tensors, tp_rank, tp_size)
    w2_raw = _shard_col(tensors[_W2_WEIGHT_SUFFIX], tp_rank, tp_size)
    w13_scale_src = _gather_w13_scale(tensors, tp_rank, tp_size)
    w2_scale_src = tensors[_W2_SCALE_SUFFIX]

    if w4a8:
        method = getattr(routed.quant_method, "quant_method", routed.quant_method)
        if method.quant_method == COMPRESSED_TENSORS_METHOD:
            raise NotImplementedError(
                "[FT] scale_down W4A8 weight reload does not supportCOMPRESSED_TENSORS foemat yet."
            )
        w13_scale_bias_raw = _gather_w13_scale_bias(routed, tp_rank, tp_size)
        processed = method._process_modelslim_weights(
            w13_raw.unsqueeze(0),
            w2_raw.unsqueeze(0),
            w13_scale_src.unsqueeze(0),
            w2_scale_src.unsqueeze(0),
            w13_scale_bias_raw.unsqueeze(0),
            tensors[_W2_SCALE_BIAS_SUFFIX].unsqueeze(0),
        )
        return tuple(x[0] for x in processed)

    w13 = w13_raw.transpose(0, 1).contiguous()
    w2 = w2_raw.transpose(0, 1).contiguous()
    if not w8a8:
        return w13, w2, None, None, None, None

    method = getattr(routed.quant_method, "quant_method", routed.quant_method)
    _, _, w13_scale, w2_scale, _, _, _ = method._process_w8a8_weights(
        w13_raw.unsqueeze(0),
        w2_raw.unsqueeze(0),
        w13_scale_src.unsqueeze(0),
        w2_scale_src.unsqueeze(0),
    )
    return w13, w2, w13_scale[0], w2_scale[0], None, None


def _assemble_expert(
    pair: tuple[int, int],
    routed_layers: list,
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    quant_type: type,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """CPU assembly of one expert's (w13, w2, w13_scale, w2_scale,
    w13_scale_bias, w2_scale_bias); thread-safe (read-only shared state).

    Scales/scale_bias are None for the unquantized scheme; the w2 scale is a
    view into its bucket (callers must not write it). Mirrors the relevant
    ``process_weights_after_loading`` per-expert transforms (TP sharding is a
    no-op under EP, which scale_down requires).
    """
    layer_idx, logical_id = pair
    routed = routed_layers[layer_idx]
    w8a8 = quant_type is AscendW8A8DynamicFusedMoEMethod
    w4a8 = quant_type is AscendW4A8DynamicFusedMoEMethod
    if getattr(routed, "w13_bias", None) is not None or getattr(routed, "w2_bias", None) is not None:
        raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
    tp_rank, tp_size = _tp_shard_info(routed)
    tensors = buckets[(layer_idx, logical_id)]

    w13 = _gather_w13(tensors, tp_rank, tp_size).transpose(0, 1).contiguous()
    w2 = _shard_col(tensors[_W2_WEIGHT_SUFFIX], tp_rank, tp_size).transpose(0, 1).contiguous()

    if not w8a8 and not w4a8:
        return w13, w2, None, None, None, None

    w13_scale_src = _gather_w13_scale(tensors, tp_rank, tp_size)
    w2_scale_src = tensors[_W2_SCALE_SUFFIX]

    if w8a8:
        return w13, w2, w13_scale_src, w2_scale_src.view(-1), None, None

    # W4A8: the scheme instance carries the sub-format ("ascend" modelslim vs
    # "compressed-tensors"); both converge to identical runtime shapes below.
    method = getattr(routed.quant_method, "quant_method", routed.quant_method)
    modelslim = method.quant_method != COMPRESSED_TENSORS_METHOD

    # Scales: bit-reinterpret fp32 -> int64 (w13 already flattened to [2I];
    # w2 kept as [1, H] for npu_grouped_matmul).
    w13_scale = _reinterpret_fp32_to_int64(w13_scale_src.to(torch.float32))
    w2_scale = _reinterpret_fp32_to_int64(w2_scale_src.to(torch.float32).view(1, -1))

    if not modelslim:
        raise NotImplementedError("[FT] scale_down W4A8 weight reload does not supportCOMPRESSED_TENSORS foemat yet.")
    # weights already packed; scale_bias read from checkpoint. w13 is
    # already 1-D([2I]); w2's checkpoint [out, 16//tp] is reduced over the
    # replica/group dim down to [out]
    w13_scale_bias = _gather_w13_scale_bias(tensors, tp_rank, tp_size).view(-1)
    w2_scale_bias = _w4a8_scale_bias_modelslim(tensors[_W2_SCALE_BIAS_SUFFIX])

    return w13, w2, w13_scale, w2_scale, w13_scale_bias, w2_scale_bias


def _reload_local_slots(
    routed_layers: list,
    local_slots: dict[tuple[int, int], int],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Reload all local (layer, expert) pairs in quant-uniform chunks.

    Entries are grouped by (quant scheme, sub-format, weight shapes, dtypes,
    storage layout) so each stacked chunk is shape-uniform, e.g. DeepSeek-V4
    43-layers, 8 experts to reload, 344 pairs become ceil(344 / 32) = 11 chunks
    instead of 344 per-expert reloads. modelslim vs compressed_tensors produce
    different per-slot layouts and scale_bias sources, so W4A8 is split by
    sub-format to keep each chunk shape-uniform.
    """
    groups: dict[tuple, list[tuple[int, int]]] = {}
    for (layer_idx, logical_id), slot in sorted(local_slots.items()):
        routed = routed_layers[layer_idx]
        # Quantized layers wrap the scheme in AscendFusedMoEMethod; the
        # scheme class lives in its .quant_method attribute.
        quant_method = getattr(routed.quant_method, "quant_method", routed.quant_method)
        if type(quant_method) not in _SUPPORTED_QUANT_TYPES:
            raise NotImplementedError(
                f"[FT] scale_down weight reload is not implemented for quant method {type(quant_method).__name__}."
            )
        w13_weight_list = getattr(routed, "w13_weight_list", None)
        if w13_weight_list is not None:
            w13_shape, w13_dtype = w13_weight_list[slot].shape, w13_weight_list[slot].dtype
            w2_shape, w2_dtype = routed.w2_weight_list[slot].shape, routed.w2_weight_list[slot].dtype
        else:
            w13_shape, w13_dtype = routed.w13_weight.shape[1:], routed.w13_weight.dtype
            w2_shape, w2_dtype = routed.w2_weight.shape[1:], routed.w2_weight.dtype
        sub_format = quant_method.quant_method if type(quant_method) is AscendW4A8DynamicFusedMoEMethod else None
        key = (type(quant_method), sub_format, w13_shape, w13_dtype, w2_shape, w2_dtype, w13_weight_list is not None)
        groups.setdefault(key, []).append((layer_idx, logical_id))

    reloaded = 0
    for key, pairs in groups.items():
        quant_type = key[0]
        for start in range(0, len(pairs), _RELOAD_CHUNK_SIZE):
            chunk = pairs[start : start + _RELOAD_CHUNK_SIZE]
            reloaded += _reload_chunk(quant_type, routed_layers, chunk, local_slots, buckets)
    return reloaded


def _reload_chunk(
    quant_type: type,
    routed_layers: list,
    pairs: list[tuple[int, int]],
    local_slots: dict[tuple[int, int], int],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Apply one chunk of pairs sharing a quant scheme and layout.

    Assemble the chunk's (w13, w2, scales, scale_bias) pairs on the CPU in
    parallel, stack them, then pay one H2D + one format cast for the whole
    stack and scatter it with one non-blocking copy per slot, drained by a
    single sync. Shapes must match (see ``_reload_local_slots``); stacking and
    the device-side cast assume it.
    """
    w13s: list[torch.Tensor] = []
    w2s: list[torch.Tensor] = []
    w13_scales: list[torch.Tensor] = []
    w2_scales: list[torch.Tensor] = []
    w13_scale_biases: list[torch.Tensor] = []
    w2_scale_biases: list[torch.Tensor] = []
    w8a8 = quant_type is AscendW8A8DynamicFusedMoEMethod
    w4a8 = quant_type is AscendW4A8DynamicFusedMoEMethod
    has_scales = w8a8 or w4a8

    # CPU phase: assemble pairs in parallel (read-only shared state).
    assemble = partial(_assemble_expert, routed_layers=routed_layers, buckets=buckets, quant_type=quant_type)
    with ThreadPoolExecutor(max_workers=min(len(pairs), _GATHER_MAX_WORKERS)) as pool:
        for w13, w2, w13_scale, w2_scale, w13_scale_bias, w2_scale_bias in pool.map(assemble, pairs):
            w13s.append(w13)
            w2s.append(w2)
            if has_scales:
                w13_scales.append(w13_scale)
                w2_scales.append(w2_scale)
            if w4a8:
                w13_scale_biases.append(w13_scale_bias)
                w2_scale_biases.append(w2_scale_bias)

    w13_batch = torch.stack(w13s)
    w2_batch = torch.stack(w2s)
    if has_scales:
        w13_scale_batch = torch.stack(w13_scales)
        w2_scale_batch = torch.stack(w2_scales)
    if w4a8:
        w13_scale_bias_batch = torch.stack(w13_scale_biases)
        w2_scale_bias_batch = torch.stack(w2_scale_biases)

    # Device phase: one H2D + one format cast per weight class.
    first_layer = routed_layers[pairs[0][0]]
    first_slot = local_slots[pairs[0]]
    w13_weight_list = getattr(first_layer, "w13_weight_list", None)
    if w13_weight_list is not None:
        device = w13_weight_list[first_slot].device
        w13_dtype, w2_dtype = w13_weight_list[first_slot].dtype, first_layer.w2_weight_list[first_slot].dtype
    else:
        device = first_layer.w13_weight.device
        w13_dtype, w2_dtype = first_layer.w13_weight.dtype, first_layer.w2_weight.dtype

    if has_scales:
        w13_batch = w13_batch.to(device=device)
        w2_batch = w2_batch.to(device=device)
        w13_scale_batch = w13_scale_batch.to(device=device, dtype=torch.float32)
        w2_scale_batch = w2_scale_batch.to(device=device)
        if w4a8:
            # Weights int8, scales int64 (bit-reinterpreted), scale_bias fp32.
            w13_scale_bias_batch = w13_scale_bias_batch.to(device=device, dtype=torch.float32)
            w2_scale_bias_batch = w2_scale_bias_batch.to(device=device, dtype=torch.float32)
    else:
        # Whole-tensor layout policy; no-op when it does not force NZ.
        w13_batch = w13_batch.to(device=device, dtype=w13_dtype)
        w2_batch = w2_batch.to(device=device, dtype=w2_dtype)
    if w8a8:
        w13_batch = torch_npu.npu_format_cast(w13_batch, ACL_FORMAT_FRACTAL_NZ)
        w2_batch = torch_npu.npu_format_cast(w2_batch, ACL_FORMAT_FRACTAL_NZ)
    else:
        # W4A8 process_weights_after_loading also uses maybe_trans_nz.
        w13_batch = maybe_trans_nz(w13_batch)
        w2_batch = maybe_trans_nz(w2_batch)

    # Scatter phase: non-blocking copies drained by one sync.
    for i, (layer_idx, logical_id) in enumerate(pairs):
        slot = local_slots[(layer_idx, logical_id)]
        routed = routed_layers[layer_idx]
        w13_weight_list = getattr(routed, "w13_weight_list", None)
        if w13_weight_list is not None:
            w13_weight_list[slot].copy_(w13_batch[i], non_blocking=True)
            routed.w2_weight_list[slot].copy_(w2_batch[i], non_blocking=True)
        else:
            # Whole-tensor layout: the slot slice is one expert matrix.
            routed.w13_weight.data[slot].copy_(w13_batch[i], non_blocking=True)
            routed.w2_weight.data[slot].copy_(w2_batch[i], non_blocking=True)
        if w8a8:
            routed.w13_weight_scale_fp32_list[slot].copy_(w13_scale_batch[i], non_blocking=True)
            w2_scale_target = routed.w2_weight_scale_list[slot]
            routed.w2_weight_scale_list[slot].copy_(w2_scale_batch[i].to(w2_scale_target.dtype), non_blocking=True)
            # Optional fused scales (enable_fused_mc2, currently rejected
            # for scale_down).
            fused_w1_scale_list = getattr(routed, "fused_w1_scale_list", None)
            fused_w2_scale_list = getattr(routed, "fused_w2_scale_list", None)
            if fused_w1_scale_list is not None and fused_w2_scale_list is not None:
                fused_w1_scale_list[slot].copy_(scale_from_float_to_int64(w13_scales[i]))
                fused_w2_scale_list[slot].copy_(scale_from_float_to_int64(w2_scales[i]))
        if w4a8:
            # Scales are already int64 (bit-reinterpreted) and scale_bias fp32,
            # so a direct dtype-preserving copy into the int64/fp32 slot storage.
            routed.w13_weight_scale_list[slot].copy_(w13_scale_batch[i], non_blocking=True)
            routed.w2_weight_scale_list[slot].copy_(w2_scale_batch[i], non_blocking=True)
            routed.w13_scale_bias_list[slot].copy_(w13_scale_bias_batch[i], non_blocking=True)
            routed.w2_scale_bias_list[slot].copy_(w2_scale_bias_batch[i], non_blocking=True)
    torch.npu.synchronize()
    return len(pairs)


def _resolve_expert_tensor(
    name: str,
    wanted: dict[str, dict[int, tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
    """Split a normalized name into (key, suffix) if it names a wanted expert tensor.

    E.g. ``model.layers.3.ffn.experts.12.gate_proj.weight`` -> ``((3, 12), "gate_proj.weight")``.
    """
    head, _, suffix = name.rpartition(".")
    layer_name, _, expert_str = head.rpartition(".")
    if not expert_str.isdigit():
        # 2-segment suffix ("down_proj.weight"): peel one more segment.
        layer_name, _, last = layer_name.rpartition(".")
        suffix = f"{expert_str}.{suffix}"
        expert_str = last
    by_expert = wanted.get(layer_name)
    if by_expert is None or not expert_str.isdigit():
        return None
    key = by_expert.get(int(expert_str))
    return (key, suffix) if key is not None else None


def _collect_safetensors_shards(vllm_config: VllmConfig, model: torch.nn.Module) -> list[str]:
    """Collect the safetensors shards the model loads from the local checkpoint directory."""
    model_path = vllm_config.model_config.model
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"[FT] scale_down expert reload reads a local checkpoint only, but '{model_path}' is not a directory."
        )
    # A model may narrow the patterns it loads; honour that over our default.
    allow_patterns = getattr(model, "allow_patterns_overrides", None) or ["*.safetensors"]
    shards: list[str] = []
    for pattern in allow_patterns:
        shards += glob.glob(os.path.join(model_path, pattern))
        if shards:
            break
    # An override may select non-safetensors files; the header walk has
    # nothing to read there.
    shards = [shard for shard in shards if shard.endswith(".safetensors")]
    if len(shards) > 1:
        # Sharded and consolidated safetensors can coexist; the index file
        # records which set the model actually loads. It ships with the
        # checkpoint, and the filter no-ops without it.
        shards = filter_duplicate_safetensors_files(shards, model_path, SAFE_WEIGHTS_INDEX_NAME)
    return sorted(shards)


def _collect_matching_weights(
    shards: list[str],
    normalize: Callable[[str], str],
    wanted: dict[str, dict[int, tuple[int, int]]],
    wanted_suffixes: set[str],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    matched: set[tuple[int, int]],
) -> None:
    """Read only the wanted experts' tensors by walking safetensors shard headers.

    ``safe_open(...).keys()`` reads a shard's JSON header alone, so a name
    ``_resolve_expert_tensor`` rejects never reaches ``get_tensor``: reads are
    limited to the wanted experts plus one header per shard.
    """
    for st_file in shards:
        with safe_open(st_file, framework="pt") as f:
            for raw_name in f.keys():  # noqa: SIM118
                found = _resolve_expert_tensor(normalize(raw_name), wanted)
                if found is None:
                    continue
                key, suffix = found
                matched.add(key)
                if suffix in wanted_suffixes:
                    buckets.setdefault(key, {})[suffix] = f.get_tensor(raw_name)


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reassignments: set[tuple[int, int]],
) -> int:
    """Reload reassigned (moe_layer_idx, logical_expert_id) weights from disk.

    Ascend keeps expert weights in runtime layout (transposed, NZ-cast,
    per-slot lists, derived quant scales), so ``model.load_weights`` cannot
    write them back. Only reassigned experts whose replica lands in this
    rank's physical block are reloaded, into the slot given by
    ``eplb_state.logical_to_physical_map`` -- hence the call must follow the
    upstream ``rebuild_model_expert_maps``.
    """
    if not reassignments:
        return 0

    moe_layers = list(model.moe_layers)
    routed_layers = [getattr(layer, "routed_experts", layer) for layer in moe_layers]
    ep_rank = get_ep_group().rank_in_group

    local_slots: dict[tuple[int, int], int] = {}
    for layer_idx, logical_id in reassignments:
        layer_state = getattr(moe_layers[layer_idx], "eplb_state", None)
        l2p = getattr(layer_state, "logical_to_physical_map", None)
        if l2p is None:
            raise RuntimeError(
                f"[FT] MoE layer {layer_idx} has no EPLB placement map; the reassigned expert has nowhere to land."
            )
        num_local = routed_layers[layer_idx].moe_config.num_local_experts
        start = ep_rank * num_local
        for physical_id in l2p[logical_id].tolist():
            if start <= physical_id < start + num_local:
                local_slots[(layer_idx, logical_id)] = physical_id - start
                break

    if not local_slots:
        return 0

    normalize = _get_ckpt_name_mapper(model)

    # "<layer_name>.<expert_id>" -> key for O(1) matching.
    wanted: dict[str, dict[int, tuple[int, int]]] = {}
    for layer_idx, logical_id in local_slots:
        wanted.setdefault(routed_layers[layer_idx].layer_name, {})[logical_id] = (layer_idx, logical_id)

    wanted_suffixes = set(_W13_WEIGHT_SUFFIXES + _W13_SCALE_SUFFIXES)
    wanted_suffixes.add(_W2_WEIGHT_SUFFIX)
    wanted_suffixes.add(_W2_SCALE_SUFFIX)
    # modelslim W4A8 keeps scale_bias in the checkpoint; compressed_tensors
    # has no such keys (it derives scale_bias from weight*scale), so these
    # suffixes are simply never matched for CT.
    wanted_suffixes.update(_W13_SCALE_BIAS_SUFFIXES)
    wanted_suffixes.add(_W2_SCALE_BIAS_SUFFIX)

    buckets: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    matched: set[tuple[int, int]] = set()

    logger.info("[FT] Reloading %d reassigned (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    shards = _collect_safetensors_shards(vllm_config, model)
    if not shards:
        raise RuntimeError(
            "[FT] scale_down expert reload requires a safetensors checkpoint; "
            f"{vllm_config.model_config.model} has none."
        )
    _collect_matching_weights(shards, normalize, wanted, wanted_suffixes, buckets, matched)
    # Same reasoning: a pair with no checkpoint weight would leave its slot
    # holding the previous occupant's weights.
    unmatched = [pair for pair in local_slots if pair not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {unmatched[:5]}. The model's expert "
            "weights likely use a layout that does not follow "
            "'<layer_name>.<expert_id>.' (e.g. fused experts), or the "
            "checkpoint's naming differs from the runtime namespace without "
            "an hf_to_vllm_mapper declared on the model class."
        )

    reloaded = _reload_local_slots(routed_layers, local_slots, buckets)

    logger.info("[FT] Expert weight reload complete: %d (layer, expert) pair(s).", reloaded)
    return reloaded
