# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Helpers shared by Ascend workers for resolving MoE-owning modules."""

import torch.nn as nn
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
)


def unwrap_moe(model: nn.Module) -> nn.Module:
    """Resolve the module that owns ``moe_layers`` for the served model.

    A multimodal wrapper (e.g. ``AscendDeepseekV41ForCausalLM``) delegates to
    an inner language model holding the MoE mixin and ``moe_layers``; EPLB
    registers that inner model, so callers must operate on it too.
    """
    if not is_mixture_of_experts(model) and isinstance(model, SupportsMultiModal):
        return model.get_language_model()
    return model
