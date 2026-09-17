# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read-only FT snapshots; never read device memory before a device reset."""

import hashlib
import json

import torch
from vllm.logger import logger

from vllm_ascend import envs

_KFC_MAX_RANKS = 1024
_KFC_HEADER_WORDS = 4
_WORDS_PER_ADDRESS = 2


def _tensor_snapshot(tensor, *, read_device: bool) -> dict | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    result = {"ptr": hex(tensor.data_ptr()), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if read_device:
        cpu = tensor.detach().cpu().contiguous()
        result["sha256"] = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        # Context is opaque unless it matches the installed KFC ABI below.
        # Keep only its header and the small elastic metadata in the log.
        result["head"] = cpu.reshape(-1)[:16].tolist()
        if cpu.dtype == torch.int32 and cpu.numel() == _KFC_HEADER_WORDS + _WORDS_PER_ADDRESS * _KFC_MAX_RANKS:
            words = [int(value) & 0xFFFFFFFF for value in cpu.reshape(-1).tolist()]
            result["kfc_layout_candidate"] = {
                "ep_rank": words[0],
                "ranks_per_server": words[1],
                "kfc_context_addr": hex(words[2] | words[3] << 32),
                "nonzero_rank_buffers": {
                    str(rank): hex(words[4 + 2 * rank] | words[5 + 2 * rank] << 32)
                    for rank in range(_KFC_MAX_RANKS)
                    if words[4 + 2 * rank] or words[5 + 2 * rank]
                },
            }
    return result


def trace_adapter_context(adapter, stage: str, elastic_info, *, read_device: bool) -> None:
    if not envs.VLLM_ASCEND_FT_REPLAY_DEBUG:
        return
    # Baseline construction should happen outside capture. Never insert a
    # device-to-host read into a graph if a new buffer is created during capture.
    if read_device and torch.npu.is_current_stream_capturing():
        logger.info("[FT_V3_CONTEXT] stage=%s skipped=device_capture", stage)
        return
    logger.info("[FT_V3_CONTEXT] stage=%s begin read_device=%s adapter=%s", stage, read_device, hex(id(adapter)))
    try:
        buffer = adapter._buffer
        context = _tensor_snapshot(getattr(buffer, "context", None), read_device=read_device)
        elastic = _tensor_snapshot(elastic_info, read_device=read_device)
        previous = getattr(adapter, "_ft_context_last_snapshot", None)
        record = {
            "stage": stage,
            "adapter": hex(id(adapter)),
            "buffer": hex(id(buffer)),
            "buffer_type": f"{type(buffer).__module__}.{type(buffer).__name__}",
            "rank": adapter._buffer_rank,
            "buffer_key": adapter._buffer_key,
            "group_name": getattr(buffer, "group_name", None),
            "context": context,
            "elastic_info": elastic,
            "cached_elastic_signature": adapter._elastic_info_signature,
            "capture_active": _tensor_snapshot(adapter._capture_active, read_device=read_device),
            "read_device": read_device,
        }
        if previous:
            record["previous_stage"] = previous["stage"]
            for key in ("context", "elastic_info"):
                old, new = previous[key], record[key]
                if old is not None and new is not None:
                    record[f"{key}_ptr_changed"] = old["ptr"] != new["ptr"]
                    if read_device:
                        record[f"{key}_contents_changed"] = old.get("sha256") != new.get("sha256")
            if not read_device:
                record["cached_device_snapshot"] = previous
        logger.info("[FT_V3_CONTEXT] %s", json.dumps(record, sort_keys=True))
        if read_device:
            adapter._ft_context_last_snapshot = record
    except Exception:
        # A diagnostic exception must not replace the real recovery failure.
        logger.exception("[FT_V3_CONTEXT] stage=%s snapshot failed", stage)
