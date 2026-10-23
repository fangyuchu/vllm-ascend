# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU expert redistribution and weight reload for fault-tolerance scale-down.

The reload deliberately does not degrade: a checkpoint, weight layout or quant
scheme it cannot handle raises instead of falling back to a path that could not
succeed anyway.
"""

import glob
import os
from collections.abc import Callable, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

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
    process_scale_to_runtime,
    reduce_scale_bias_modelslim,
    w13_scale_to_runtime,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    flatten_expert_dims,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    ASCEND_QUANTIZATION_METHOD,
    maybe_trans_nz,
    transpose_weight_to_runtime,
)

__all__ = [
    "build_orig_to_dense_rank_table",
    "densify_routing_table_physical_ids",
    "reload_experts_from_disk",
]

# Checkpoint suffixes (relative to "<layer_name>.<expert_id>.") of the tensors
# a reload may consume. weight_offset is never read by the MoE apply path, so
# it is not reloaded.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"
# scale_bias exists only in the msModelSlim ("ascend") format; compressed_tensors
# derives it on the fly, so those checkpoints have no such keys.
_W13_SCALE_BIAS_SUFFIXES = ("gate_up_proj.scale_bias", "gate_proj.scale_bias", "up_proj.scale_bias")
_W2_SCALE_BIAS_SUFFIX = "down_proj.scale_bias"

# Every checkpoint key the reload may consume, across all supported schemes.
_WANTED_SUFFIXES = frozenset(
    _W13_WEIGHT_SUFFIXES
    + _W13_SCALE_SUFFIXES
    + _W13_SCALE_BIAS_SUFFIXES
    + (_W2_WEIGHT_SUFFIX, _W2_SCALE_SUFFIX, _W2_SCALE_BIAS_SUFFIX)
)

# (layer, expert) pairs per chunk: larger chunks amortise the device rounds
# further but grow the transient stacked buffer.
_RELOAD_CHUNK_SIZE = 32

# Threads assembling one chunk; bounded because worker processes share host
# memory bandwidth.
_GATHER_MAX_WORKERS = 8


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
    """Renumber one layer's ``expert_replica_routing_table`` into the densified id space.

    In scale-down mode the MC2 dispatch kernel indexes
    ``table2[expert_id // num_local]`` and the combine kernel drops ids beyond
    the shrunk expert count, so ids must be dense-rank-major:
    ``dense_rank * num_local + slot``. Original ids only work when the dead
    ranks form a suffix; a dead rank in the middle misroutes tokens or crashes
    on the -1 rank lookup.

    The update is an in-place ``copy_`` with an unchanged shape, so captured
    graphs keep pointing at valid storage.
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


class _WeightPipeline(NamedTuple):
    """One runtime attribute's reload: checkpoint gather + runtime transform.

    ``source`` gathers the checkpoint tensor (sharded for this TP rank).
    ``processor`` is the load path's ``process_weights_after_loading``
    transform, imported so the reload cannot drift from it.
    """

    source: Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]
    processor: Callable[[torch.Tensor], torch.Tensor]


# Weight sources: the checkpoint-side gather for one expert, sharded for this TP rank.
def _gate_up(suffix: str) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """w13 order out of the checkpoint: fused ``gate_up_proj``, or split gate+up."""

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        fused = tensors.get(f"gate_up_proj.{suffix}")
        if fused is not None:
            gate, up = fused.chunk(2)
        else:
            gate, up = tensors[f"gate_proj.{suffix}"], tensors[f"up_proj.{suffix}"]
        return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0)

    return weight_source


def _down_proj(suffix: str, *, shard: bool = False) -> Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]:
    """``down_proj.<suffix>`` as this rank holds it."""

    def weight_source(tensors: dict[str, torch.Tensor], tp_rank: int, tp_size: int) -> torch.Tensor:
        tensor = tensors[f"down_proj.{suffix}"]
        return _shard_col(tensor, tp_rank, tp_size) if shard else tensor

    return weight_source


def _single_expert(transform: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    The reload assembles experts one by one on the pool threads, so the shared
    transforms get a temporary leading expert dim that is squeezed back off.
    """

    def processor(expert: torch.Tensor) -> torch.Tensor:
        return transform(expert.unsqueeze(0)).squeeze(0)

    return processor


# The weight pipelines every scheme reloads; their runtime attributes are the
# ones that take the layout cast.
_WEIGHT_PIPELINES: dict[str, _WeightPipeline] = {
    "w13_weight": _WeightPipeline(_gate_up("weight"), transpose_weight_to_runtime),
    "w2_weight": _WeightPipeline(_down_proj("weight", shard=True), transpose_weight_to_runtime),
}
_WEIGHT_ATTRS = tuple(_WEIGHT_PIPELINES)


def _to_fractal_nz(weight: torch.Tensor) -> torch.Tensor:
    """Unconditional FRACTAL_NZ cast, as W8A8's weights are loaded with."""
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


@dataclass(eq=False)  # identity semantics: registry entries are singletons
class _ReloadScheme:
    """Everything the reload needs for one quant method."""

    quant_method: type
    pipelines: dict[str, _WeightPipeline]
    cast_layout: Callable[[torch.Tensor], torch.Tensor]

    def matches(self, method) -> bool:
        if type(method) is not self.quant_method:
            return False
        # W4A8 is the only class with several checkpoint sub-formats; only the
        # msModelSlim ("ascend") one maps to a reload scheme.
        if self.quant_method is AscendW4A8DynamicFusedMoEMethod:
            return method.quant_method == ASCEND_QUANTIZATION_METHOD
        return True


# One variable per scheme: identity, the scheme's pipeline table (the shared
# weight entries plus its own) and the layout cast. ``_SCHEMES`` is only the
# resolution list, so adding a scheme is one variable plus one tuple element.
_UNQUANTIZED_SCHEME = _ReloadScheme(
    AscendUnquantizedFusedMoEMethod,
    _WEIGHT_PIPELINES,
    maybe_trans_nz,
)
_W8A8_DYNAMIC_SCHEME = _ReloadScheme(
    AscendW8A8DynamicFusedMoEMethod,
    {
        **_WEIGHT_PIPELINES,
        "w13_weight_scale_fp32": _WeightPipeline(_gate_up("weight_scale"), _single_expert(flatten_expert_dims)),
        "w2_weight_scale": _WeightPipeline(_down_proj("weight_scale"), _single_expert(flatten_expert_dims)),
    },
    _to_fractal_nz,
)
_W4A8_ASCEND_SCHEME = _ReloadScheme(
    AscendW4A8DynamicFusedMoEMethod,
    {
        **_WEIGHT_PIPELINES,
        "w13_weight_scale": _WeightPipeline(_gate_up("weight_scale"), _single_expert(w13_scale_to_runtime)),
        "w2_weight_scale": _WeightPipeline(_down_proj("weight_scale"), process_scale_to_runtime),
        "w13_scale_bias": _WeightPipeline(_gate_up("scale_bias"), reduce_scale_bias_modelslim),
        "w2_scale_bias": _WeightPipeline(_down_proj("scale_bias"), reduce_scale_bias_modelslim),
    },
    maybe_trans_nz,
)

_SCHEMES: tuple[_ReloadScheme, ...] = (_UNQUANTIZED_SCHEME, _W8A8_DYNAMIC_SCHEME, _W4A8_ASCEND_SCHEME)


def _resolve_scheme(routed: torch.nn.Module) -> _ReloadScheme:
    """The reload rules a routed-experts layer uses."""

    method = getattr(routed.quant_method, "quant_method", routed.quant_method)
    for scheme in _SCHEMES:
        if scheme.matches(method):
            return scheme
    raise NotImplementedError(
        f"[FT] scale_down weight reload is not implemented for quant method {type(method).__name__}."
    )


def _slot_tensor(layer: torch.nn.Module, attr: str, slot: int) -> torch.Tensor:
    """The runtime storage of one per-slot tensor."""

    per_slot = getattr(layer, f"{attr}_list", None)
    return per_slot[slot] if per_slot is not None else getattr(layer, attr).data[slot]


def _reload_local_slots(
    routed_layers: list,
    local_slots: dict[tuple[int, int], int],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Reload all local (layer, expert) pairs in scheme-uniform chunks.

    Pairs are grouped by (scheme, per-slot layout) so every stacked chunk is
    shape- and dtype-uniform: e.g. 344 pairs become 11 chunks instead of 344
    per-expert reloads. The layout is read off the runtime slots, so a new
    scheme needs no change here and can never share a chunk with an
    incompatible layout.
    """
    groups: dict[tuple, list[tuple[int, int]]] = {}
    for (layer_idx, logical_id), slot in sorted(local_slots.items()):
        routed = routed_layers[layer_idx]
        if getattr(routed, "w13_bias", None) is not None or getattr(routed, "w2_bias", None) is not None:
            raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
        reload_scheme = _resolve_scheme(routed)
        layout = []
        for attr in reload_scheme.pipelines:
            tensor = _slot_tensor(routed, attr, slot)
            layout.append((attr, tensor.shape, tensor.dtype))
        groups.setdefault((reload_scheme, tuple(layout)), []).append((layer_idx, logical_id))

    reloaded = 0
    for (reload_scheme, _layout), pairs in groups.items():
        for start in range(0, len(pairs), _RELOAD_CHUNK_SIZE):
            chunk = pairs[start : start + _RELOAD_CHUNK_SIZE]
            reloaded += _reload_chunk(reload_scheme, routed_layers, chunk, local_slots, buckets)
    return reloaded


def _process_expert(
    pair: tuple[int, int],
    routed_layers: list,
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
    pipelines: dict[str, _WeightPipeline],
) -> dict[str, torch.Tensor]:
    """CPU-side gather + transform of one expert, keyed by runtime attribute."""

    # EP reports (0, 1) because each rank owns whole experts; the shard helpers
    # still split checkpoint tensors for TP>1 deployments.
    parallel_config = routed_layers[pair[0]].moe_config.moe_parallel_config
    tp_rank, tp_size = parallel_config.tp_rank, parallel_config.tp_size
    tensors = buckets[pair]
    return {
        attr: pipeline.processor(pipeline.source(tensors, tp_rank, tp_size)) for attr, pipeline in pipelines.items()
    }


def _reload_chunk(
    scheme: _ReloadScheme,
    routed_layers: list,
    pairs: list[tuple[int, int]],
    local_slots: dict[tuple[int, int], int],
    buckets: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Apply one chunk of pairs sharing a scheme and per-slot layout.

    Assemble the chunk in parallel on the CPU, then pay one H2D and one layout
    cast per attribute for the whole stack, and scatter with one non-blocking
    copy per slot drained by a single sync -- instead of a transpose-copy, H2D,
    cast and scatter per pair.
    """
    # CPU phase: assemble pairs in parallel (read-only shared state).
    assemble = partial(_process_expert, routed_layers=routed_layers, buckets=buckets, pipelines=scheme.pipelines)
    with ThreadPoolExecutor(max_workers=min(len(pairs), _GATHER_MAX_WORKERS)) as pool:
        per_expert = list(pool.map(assemble, pairs))

    # Device phase: one H2D and one layout cast per attribute for the whole
    # stack. Landing at the slot's dtype covers every scheme: int8 weights, the
    # fp32 W8A8 scale, W4A8's int64 bit scales and fp32 scale_bias.
    first_layer = routed_layers[pairs[0][0]]
    first_slot = local_slots[pairs[0]]
    landed: dict[str, torch.Tensor] = {}
    for attr in scheme.pipelines:
        ref = _slot_tensor(first_layer, attr, first_slot)
        stacked = torch.stack([expert[attr] for expert in per_expert])
        landed[attr] = stacked.to(device=ref.device, dtype=ref.dtype)
    for attr in _WEIGHT_ATTRS:
        landed[attr] = scheme.cast_layout(landed[attr])

    # Scatter phase: non-blocking copies drained by one sync.
    for i, pair in enumerate(pairs):
        layer, slot = routed_layers[pair[0]], local_slots[pair]
        for attr in landed:
            _slot_tensor(layer, attr, slot).copy_(landed[attr][i], non_blocking=True)
    torch.npu.synchronize()
    return len(pairs)


def _get_ckpt_name_mapper(model: torch.nn.Module) -> Callable[[str], str]:
    """Map raw checkpoint weight names into the model's runtime namespace."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        # No declared mapping means the checkpoint stores weights under
        # runtime names already (e.g. Qwen3-MoE), so identity is correct.
        return lambda name: name
    return lambda name: mapper._map_name(name) or name


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
        # selects the set the model loads (the filter no-ops without one).
        shards = filter_duplicate_safetensors_files(shards, model_path, SAFE_WEIGHTS_INDEX_NAME)
    return sorted(shards)


def _collect_matching_weights(
    shards: list[str],
    normalize: Callable[[str], str],
    wanted: dict[str, dict[int, tuple[int, int]]],
    wanted_suffixes: Set[str],
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
    write them back. Only reassigned experts landing in this rank's physical
    block are reloaded, into the slot given by
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

    buckets: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    matched: set[tuple[int, int]] = set()

    logger.info("[FT] Reloading %d reassigned (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    shards = _collect_safetensors_shards(vllm_config, model)
    if not shards:
        raise RuntimeError(
            "[FT] scale_down expert reload requires a safetensors checkpoint; "
            f"{vllm_config.model_config.model} has none."
        )
    _collect_matching_weights(shards, normalize, wanted, _WANTED_SUFFIXES, buckets, matched)
    # A pair with no checkpoint weight would silently keep the previous
    # occupant's weights in its slot, so fail instead.
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
