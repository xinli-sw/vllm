# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MiniMax shared NVFP4 expert with explicit MiniMax routing.

Allocation follows vLLM #58350. The modular TRTLLM call receives the existing
MiniMax top-k plus a unit-weight shared slot, preserving its routing semantics.
"""

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    _quantize_moe_weight_to_nvfp4,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def fusion_enabled(config):
    if os.environ.get("VLLM_MINIMAX_M3_FUSED_SHARED_EXPERTS", "0") != "1":
        return False
    parallel = get_current_vllm_config().parallel_config
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
        or parallel.enable_expert_parallel
        or parallel.enable_eplb
        or getattr(config, "n_shared_experts", None) != 1
        or not getattr(config, "use_routing_bias", False)
        or config.scoring_func != "sigmoid"
    ):
        raise ValueError(
            "MiniMax shared fusion requires SM100, one shared expert, "
            "sigmoid+bias routing and TP without EP/EPLB"
        )
    return True


def install_shared_weights(moe):
    shared = moe.shared_experts
    if shared is None:
        raise ValueError(
            "Shared weights have already been fused; reload is unsupported"
        )
    experts = moe.experts.routed_experts
    if not experts.quant_method.per_token_activation:
        raise ValueError("Shared fusion requires per-token NVFP4 activation scaling")
    if experts.quant_method.experts_cls.is_monolithic():
        raise ValueError("Shared fusion requires the modular TRTLLM kernel")
    slot = experts.global_num_experts
    if experts.w13_weight.shape[0] != slot + 1:
        raise ValueError("Expected exactly one appended shared-expert row")
    with torch.no_grad():
        for proj, stem in ((shared.gate_up_proj, "w13"), (shared.down_proj, "w2")):
            method = proj.quant_method
            # Use #55684's serialized-source dequantization path. Quantization
            # happens before any per-linear GEMM repacking.
            weight = method.get_weight_for_quantization(proj).contiguous()
            packed, blocks, scale = _quantize_moe_weight_to_nvfp4(
                weight.unsqueeze(0), moe_tp_size=moe.tp_size
            )
            getattr(experts, stem + "_weight")[slot].copy_(packed[0])
            getattr(experts, stem + "_weight_scale")[slot].copy_(blocks[0])
            getattr(experts, stem + "_weight_scale_2")[slot].fill_(scale.item())
            getattr(experts, stem + "_input_scale")[slot].fill_(1.0)
    # The MoE runner received shared_experts=None. Drop the load-only module so
    # neither its old weights nor its separate GEMMs survive into serving.
    moe.shared_experts = None
    logger.info_once(
        "MiniMax shared-expert fusion: explicit routing, NVFP4 W4A4, "
        "per-token activations, one shared slot"
    )
