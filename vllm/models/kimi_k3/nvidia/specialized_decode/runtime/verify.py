# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3 target's vocabulary-sharded logits rows for spec-decode verification, with
the specialized decode on Rubin-class devices: this rank's LM-head shard of the logits
(no full-vocabulary all-gather), which ``runtime.sampled_verify`` verifies. Up to 8 rows
(B1) take the LM-head GEMV with BF16 logits, 10-64 rows the shard GEMM with evict-first
weight loads (cutedsl.verify.lmhead; FP32 accumulation orders stated there), more rows
the stock shard GEMM.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Tensor-parallel sizes served by the vocabulary-sharded verification on Rubin (compute
# capability 10.7). GB300 keeps the gathered logits.
SHARDED_VERIFY_TP = (4, 8, 16)
# Vocabulary rows per rank: a multiple of this (256 threads x 8 BF16 logits per 16-byte
# load), the shard layouts the verification is validated on.
VOCAB_ALIGN = 2048


class VerifyLogits:
    """Per-rank prepared LM-head shard projections of the target's logits rows; built
    with the specialized decode (after the weights load, before CUDA graph capture)."""

    def __init__(self, wide=None, logits=None):
        # 10-64 rows: the evict-first shard GEMM (PreparedWideLmHead).
        self.wide = wide
        # Up to 8 rows: the LM-head GEMV with BF16 logits (PreparedLmHeadLogits).
        self.logits = logits

    @classmethod
    def build(cls, vllm_config, model):
        """None unless this deployment speculates and this device and TP size are
        covered. ``model``: the KimiLinearForCausalLM (weights loaded)."""
        if vllm_config.speculative_config is None:
            return None
        device = torch.device(torch.accelerator.current_device_index())
        if torch.cuda.get_device_capability(device) != (10, 7):
            return None
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        lm_head, processor = model.lm_head, model.logits_processor
        shard = getattr(lm_head, "shard_indices", None)
        if (
            tp not in SHARDED_VERIFY_TP
            or shard is None
            or shard.num_org_vocab_padding != 0
            or processor.scale != 1.0
            or processor.soft_cap is not None
            or processor.logits_as_input
            or processor.head_dtype not in (None, torch.bfloat16)
            or vllm_config.model_config.dtype != torch.bfloat16
            or shard.num_added_elements != 0
        ):
            return None
        vocab_rank = shard.org_vocab_end_index - shard.org_vocab_start_index
        if (
            vocab_rank % VOCAB_ALIGN
            or shard.org_vocab_start_index != rank * vocab_rank
            or processor.org_vocab_size != tp * vocab_rank
        ):
            return None
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        from ..cutedsl.verify import lmhead

        wide = logits = None
        weight = lm_head.weight[:vocab_rank]
        if (
            isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod)
            and getattr(lm_head, "bias", None) is None
            and lmhead.supports(weight)
        ):
            wide = lmhead.PreparedWideLmHead(weight, device)
            logits = lmhead.PreparedLmHeadLogits(weight, device)
        logger.info_once(
            "Kimi K3: vocabulary-sharded verification logits (LM-head GEMV up to %d "
            "rows, evict-first LM head GEMM above)",
            lmhead.TOKENS if logits is not None else 0,
        )
        return cls(wide, logits)

    def shard_logits(self, processor, lm_head, hidden_states):
        """This rank's vocabulary shard of the logits rows [rows, vocab_rank] BF16: up
        to 8 rows the LM-head GEMV (its FP32 accumulation order), 10-64 rows the
        evict-first shard GEMM, else the stock LM-head GEMM without its all-gather
        (bitwise compute_logits')."""
        if self.logits is not None and self.logits.takes(hidden_states):
            return self.logits.run(hidden_states)
        if self.wide is not None and self.wide.takes(hidden_states):
            return self.wide.run(hidden_states)
        return processor._get_logits(hidden_states, lm_head, None, skip_gather=True)
