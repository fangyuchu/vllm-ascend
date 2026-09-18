# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU expert redistribution and weight reload for fault-tolerance scale-down.

Reuses the upstream placement math and adds the Ascend-specific pieces:
``reload_experts_from_disk`` (checkpoint reload that mirrors the runtime weight
layout) and ``densify_routing_table_physical_ids`` (kernel-facing routing id
renumbering).
"""

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import torch_npu
from safetensors import safe_open
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

# Re-exported upstream helpers, kept in one place so the sentinel has a
# single import site for the redistribution building blocks.
from vllm.v1.worker.sentinel.eplb_redistribute import (
    compute_dead_ep_ranks,
    mark_dead_expert_slots_inplace,
    redistribute_expert_placement,
)

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    scale_from_float_to_int64,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

__all__ = [
    "build_orig_to_dense_rank_table",
    "compute_dead_ep_ranks",
    "densify_routing_table_physical_ids",
    "mark_dead_expert_slots_inplace",
    "redistribute_expert_placement",
    "reload_experts_from_disk",
]

# Checkpoint name suffixes (relative to "<layer_name>.<expert_id>.") of the
# tensors a reload may consume. weight_offset is loaded at startup but never
# consumed by the MoE apply path, so it is not reloaded here.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"


def _get_ckpt_name_normalizer(model: torch.nn.Module) -> Callable[[str], str]:
    """Return the model's checkpoint -> runtime name normalizer, if any.

    Models whose raw checkpoint naming differs from the runtime module
    namespace (e.g. DeepSeek-V4's ``.ffn.``/``.w1./.w2./.w3.``/``.scale``)
    declare it as a ``ckpt_weight_name_normalizer`` attribute so the reload
    below can match raw checkpoint names against runtime-name prefixes. The
    identity mapping is used for models that load under the same names.
    """
    normalizer = getattr(model, "ckpt_weight_name_normalizer", None)
    return normalizer if normalizer is not None else (lambda name: name)


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


# Quant schemes supported by the scale_down reload (gate in _reload_batched).
_SUPPORTED_QUANT_TYPES = {
    AscendUnquantizedFusedMoEMethod,
    AscendW8A8DynamicFusedMoEMethod,
}

# Max (layer, slot) pairs per device batch; bounds _reload_chunk's
# transient device memory.
_RELOAD_CHUNK_SIZE = 32

# Threads assembling one chunk in parallel; bounded because multiple worker
# processes share the host's memory bandwidth.
_GATHER_MAX_WORKERS = 8


def _assemble_entry(
    entry: tuple[int, int, int],
    routed_layers: list,
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    w8a8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """CPU assembly of one entry's (w13, w2[, scales]); thread-safe
    (read-only shared state). Scales are None for the unquantized scheme."""
    layer_idx, logical_id, _slot = entry
    routed = routed_layers[layer_idx]
    if not w8a8 and (getattr(routed, "w13_bias", None) is not None or getattr(routed, "w2_bias", None) is not None):
        raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
    tp_rank, tp_size = _tp_shard_info(routed)
    tensors = buckets[(layer_idx, logical_id)]
    w13 = _gather_w13(tensors, tp_rank, tp_size).transpose(0, 1).contiguous()
    w2 = _shard_col(tensors[_W2_WEIGHT_SUFFIX], tp_rank, tp_size).transpose(0, 1).contiguous()
    if not w8a8:
        return w13, w2, None, None
    return w13, w2, _gather_w13_scale(tensors, tp_rank, tp_size), tensors[_W2_SCALE_SUFFIX].view(-1)


def _reload_batched(
    routed_layers: list,
    local_slots: dict[tuple[int, int], int],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Reload all local (layer, slot) pairs in quant-uniform batches.

    Entries are grouped by (quant scheme, weight shapes, dtypes, storage
    layout) so each stacked chunk is shape-uniform.
    """
    groups: dict[tuple, list[tuple[int, int, int]]] = {}
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
        key = (type(quant_method), w13_shape, w13_dtype, w2_shape, w2_dtype, w13_weight_list is not None)
        groups.setdefault(key, []).append((layer_idx, logical_id, slot))

    reloaded = 0
    for key, entries in groups.items():
        quant_type = key[0]
        for start in range(0, len(entries), _RELOAD_CHUNK_SIZE):
            reloaded += _reload_chunk(quant_type, routed_layers, entries[start : start + _RELOAD_CHUNK_SIZE], buckets)
    return reloaded


def _reload_chunk(
    quant_type: type,
    routed_layers: list,
    entries: list[tuple[int, int, int]],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Apply one chunk of same-quant entries.

    One stacked H2D + one format cast per weight class, then non-blocking
    scatter into the slot storage; falls back to one entry at a time if
    stacking fails.
    """
    w13s: list[torch.Tensor] = []
    w2s: list[torch.Tensor] = []
    w13_scales: list[torch.Tensor] = []
    w2_scales: list[torch.Tensor] = []
    w8a8 = quant_type is AscendW8A8DynamicFusedMoEMethod

    # CPU phase: assemble entries in parallel (read-only shared state).
    assemble = partial(_assemble_entry, routed_layers=routed_layers, buckets=buckets, w8a8=w8a8)
    with ThreadPoolExecutor(max_workers=min(len(entries), _GATHER_MAX_WORKERS)) as pool:
        for w13, w2, w13_scale, w2_scale in pool.map(assemble, entries):
            w13s.append(w13)
            w2s.append(w2)
            if w8a8:
                w13_scales.append(w13_scale)
                w2_scales.append(w2_scale)

    try:
        w13_batch = torch.stack(w13s)
        w2_batch = torch.stack(w2s)
        if w8a8:
            w13_scale_batch = torch.stack(w13_scales)
            w2_scale_batch = torch.stack(w2_scales)
    except RuntimeError:
        # Heterogeneous shapes that escaped grouping: reload one at a time
        # (a single entry always stacks).
        logger.warning(
            "[FT] reload chunk stack failed for %d entries; falling back to the per-entry path.",
            len(entries),
        )
        for entry in entries:
            _reload_chunk(quant_type, routed_layers, [entry], buckets)
        return len(entries)

    # Device phase: one H2D + one format cast per weight class.
    first_layer = routed_layers[entries[0][0]]
    first_slot = entries[0][2]
    w13_weight_list = getattr(first_layer, "w13_weight_list", None)
    if w13_weight_list is not None:
        device = w13_weight_list[first_slot].device
        w13_dtype, w2_dtype = w13_weight_list[first_slot].dtype, first_layer.w2_weight_list[first_slot].dtype
    else:
        device = first_layer.w13_weight.device
        w13_dtype, w2_dtype = first_layer.w13_weight.dtype, first_layer.w2_weight.dtype

    if w8a8:
        w13_batch = w13_batch.to(device=device)
        w2_batch = w2_batch.to(device=device)
        w13_scale_batch = w13_scale_batch.to(device=device, dtype=torch.float32)
        w2_scale_batch = w2_scale_batch.to(device=device)
    else:
        # Whole-tensor layout policy; no-op when it does not force NZ.
        w13_batch = w13_batch.to(device=device, dtype=w13_dtype)
        w2_batch = w2_batch.to(device=device, dtype=w2_dtype)
    if w8a8:
        w13_batch = torch_npu.npu_format_cast(w13_batch, ACL_FORMAT_FRACTAL_NZ)
        w2_batch = torch_npu.npu_format_cast(w2_batch, ACL_FORMAT_FRACTAL_NZ)
    else:
        w13_batch = maybe_trans_nz(w13_batch)
        w2_batch = maybe_trans_nz(w2_batch)

    # Scatter phase: non-blocking copies drained by one sync.
    for i, (layer_idx, _logical_id, slot) in enumerate(entries):
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
    torch.npu.synchronize()
    return len(entries)


def _match_weight_name(
    name: str,
    wanted: dict[str, dict[int, tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
    """Split a normalized name into (key, suffix) if it names a wanted expert tensor."""
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


def _collect_matching_weights(
    loader: DefaultModelLoader,
    vllm_config: VllmConfig,
    model: torch.nn.Module,
    normalize: Callable[[str], str],
    wanted: dict[str, dict[int, tuple[int, int]]],
    wanted_suffixes: set[str],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    matched: set[tuple[int, int]],
) -> bool:
    """Read only the wanted experts' tensors by walking safetensors shard
    headers (no full-checkpoint read); works for raw checkpoint namings the
    upstream local_expert_ids filter cannot parse (e.g. DeepSeek-V4
    ``.ffn.``). Returns False when there are no shards to walk.
    """
    primary = DefaultModelLoader.Source(
        vllm_config.model_config.model,
        vllm_config.model_config.revision,
        prefix="",
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    _, hf_weights_files, use_safetensors = loader._prepare_weights(
        primary.model_or_path,
        primary.subfolder,
        primary.revision,
        primary.fall_back_to_pt,
        primary.allow_patterns_overrides,
    )
    if not use_safetensors:
        return False

    for st_file in sorted(hf_weights_files):
        with safe_open(st_file, framework="pt") as f:
            for raw_name in f.keys():  # noqa: SIM118
                found = _match_weight_name(normalize(raw_name), wanted)
                if found is None:
                    continue
                key, suffix = found
                matched.add(key)
                if suffix in wanted_suffixes:
                    buckets.setdefault(key, {})[suffix] = f.get_tensor(raw_name)
    return True


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reassignments: set[tuple[int, int]],
) -> int:
    """Reload reassigned (moe_layer_idx, logical_expert_id) weights from disk.

    Ascend keeps expert weights in runtime layout (transposed, NZ-cast,
    per-slot lists, derived quant scales), so the standard
    ``model.load_weights`` path cannot write them back. Only reassigned
    experts whose replica lands in this rank's physical block are reloaded.
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
            continue
        num_local = routed_layers[layer_idx].moe_config.num_local_experts
        start = ep_rank * num_local
        for physical_id in l2p[logical_id].tolist():
            if start <= physical_id < start + num_local:
                local_slots[(layer_idx, logical_id)] = physical_id - start
                break

    if not local_slots:
        return 0

    normalize = _get_ckpt_name_normalizer(model)

    # "<layer_name>.<expert_id>" -> key for O(1) matching.
    wanted: dict[str, dict[int, tuple[int, int]]] = {}
    for layer_idx, logical_id in local_slots:
        wanted.setdefault(routed_layers[layer_idx].layer_name, {})[logical_id] = (layer_idx, logical_id)

    wanted_suffixes = set(_W13_WEIGHT_SUFFIXES + _W13_SCALE_SUFFIXES)
    wanted_suffixes.add(_W2_WEIGHT_SUFFIX)
    wanted_suffixes.add(_W2_SCALE_SUFFIX)

    buckets: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    matched: set[tuple[int, int]] = set()

    def full_scan(all_weights: Generator[tuple[str, torch.Tensor], None, None]) -> None:
        for raw_name, tensor in all_weights:
            found = _match_weight_name(normalize(raw_name), wanted)
            if found is not None:
                key, suffix = found
                matched.add(key)
                if suffix in wanted_suffixes:
                    buckets.setdefault(key, {})[suffix] = tensor

    logger.info("[FT] Reloading %d reassigned (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    loader = get_model_loader(vllm_config.load_config)
    if not (
        isinstance(loader, DefaultModelLoader)
        and _collect_matching_weights(loader, vllm_config, model, normalize, wanted, wanted_suffixes, buckets, matched)
    ):
        # Non-safetensors or custom loader: fall back to the full scan.
        full_scan(loader.get_all_weights(vllm_config.model_config, model))
    unmatched = [pair for pair in local_slots if pair not in matched]
    if unmatched and getattr(model, "secondary_weights", ()):
        # Models with secondary weight sources: let the full scan pick up
        # anything the shard walk missed.
        full_scan(loader.get_all_weights(vllm_config.model_config, model))
        unmatched = [pair for pair in local_slots if pair not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {unmatched[:5]}. The model's expert "
            "weights likely use a layout that does not follow "
            "'<layer_name>.<expert_id>.' (e.g. fused experts)."
        )

    reloaded = _reload_batched(routed_layers, local_slots, buckets)

    logger.info("[FT] Expert weight reload complete: %d (layer, slot) pair(s).", reloaded)
    return reloaded
