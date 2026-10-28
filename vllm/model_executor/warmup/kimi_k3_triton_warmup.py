# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_kda_layer(worker: Worker) -> KimiK3DeltaAttention | None:
    # Kimi model construction already imports kda. Avoid importing it here.
    kda = sys.modules.get("vllm.models.kimi_k3.nvidia.kda")
    if kda is None:
        return None

    compilation_config = getattr(
        worker.model_runner,
        "compilation_config",
        None,
    )
    static_context = getattr(compilation_config, "static_forward_context", None)
    if not isinstance(static_context, dict):
        return None
    return next(
        (
            layer
            for layer in static_context.values()
            if isinstance(layer, kda.KimiK3DeltaAttention)
        ),
        None,
    )


def _warm_attn_res(worker: Worker) -> None:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import (
        attn_res,
        get_attn_res_triton_warmup_profiles,
    )

    config = worker.model_config.hf_text_config
    block_size = getattr(config, "attn_res_block_size", None)
    if block_size is None:
        return

    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 2:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros(
        (1, max_blocks, hidden_size),
        dtype=dtype,
        device=device,
    )
    norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)
    qk_weight = torch.zeros_like(norm_weight)
    output_norm_weight = torch.zeros_like(norm_weight)

    for (
        num_blocks,
        has_delta,
        block_write_idx,
        apply_output_norm,
    ) in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta if has_delta else None,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight if apply_output_norm else None,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=eps if apply_output_norm else 0.0,
        )


def _warm_recurrent_kda(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda,
        get_fused_recurrent_kda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_kda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 speculative KDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_kda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


def _warm_prefill_glue(worker: Worker, layer: KimiK3DeltaAttention) -> None:
    """Prefill-only kernels that dummy runs never reach (KDA's _forward returns
    without attention metadata), once each at the serving strides, so no
    Triton kernel JIT-compiles during inference:
    - KDA: short conv (generic and tile), output gate norm (every TILES
      variant, prefill and projection-overlap gate strides), initial-state
      gather, checkpoint export, segment-scan chain;
    - DCP context: merge, fp8 dequant, fp8 K / V pack;
    - MLA: q_a / kv_a norm (every TOK variant) and the fp8 epilogue extension.
    Cache line 0 is the null block: the state kernels skip it (no writes)."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        gather_initial_states,
    )
    from vllm.models.kimi_k3.nvidia import dcp_context
    from vllm.models.kimi_k3.nvidia.ops import (
        kda_conv,
        kda_norm,
        kda_segscan,
        mla_qkv_norm,
        mla_qkv_quant,
    )
    from vllm.v1.attention.backends.utils import (
        NULL_BLOCK_ID,
        compute_causal_conv1d_metadata,
    )

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    conv_state, rec_state = kv_cache[0], kv_cache[1]
    if not isinstance(conv_state, torch.Tensor) or not conv_state.numel():
        return
    device = conv_state.device
    dtype = worker.model_config.dtype
    P = int(layer.local_projection_size)
    H = int(layer.local_num_heads)
    D = int(layer.head_dim)
    # Row stride of the prefill projection output (KDA forward's split sizes).
    width = 4 * P + D + H + int(getattr(layer, "in_proj_padding", 0) or 0)
    logger.info("Warming up Kimi-K3 prefill glue kernels.")

    # KDA short conv: q / k / v column slices of the projection (row stride
    # width), the layer's conv cache views, one 16-token sequence.
    T = 16
    proj = torch.zeros((T, width), dtype=dtype, device=device)
    qsl_cpu = torch.tensor([0, T], dtype=torch.int32)
    nums_dict, batch_ptr, offs = compute_causal_conv1d_metadata(qsl_cpu, device=device)
    md = SimpleNamespace(
        nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=offs
    )
    qsl = qsl_cpu.to(device)
    null = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    no_init = torch.zeros(1, dtype=torch.bool, device=device)
    w = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    for x, wp, st in zip(
        proj[:, : 3 * P].split(P, dim=-1),
        w.split(P, dim=0),
        conv_state.split(P, dim=-2),
    ):
        xt = x.transpose(0, 1)
        causal_conv1d_fn(
            xt,
            wp,
            None,
            activation="silu",
            conv_states=st,
            has_initial_state=no_init,
            cache_indices=null,
            query_start_loc=qsl,
            metadata=md,
        )
        if kda_conv.supported(xt, wp, st, md):
            kda_conv.kda_prefill_conv(xt, wp, st, qsl, null, no_init, md)

    # Output gate norm: the gate is a view of the prefill projection (row
    # stride width) or of the decode projection-overlap output (4 P).
    for stride in sorted({width, 4 * P}):
        kda_norm.warmup(H, stride, layer.o_norm.weight, float(layer.o_norm.eps))

    gather_initial_states(rec_state, null, no_init)
    exporter = getattr(layer, "_checkpoint_exporter", None)
    if exporter is not None:
        ck = torch.empty((1, H, D, D), dtype=rec_state.dtype, device=device)
        exporter.export(
            SimpleNamespace(
                checkpoint_offsets=torch.zeros(1, dtype=torch.int32, device=device),
                state_indices=null,
            ),
            raw_qkv=proj[:, : 3 * P],
            conv_state=conv_state,
            recurrent_checkpoint=ck,
            recurrent_state=rec_state,
            cu_seqlens=qsl,
        )
    bufs = getattr(layer, "_segscan_bufs", None)
    if bufs is not None:
        init = torch.zeros((1, H, D, D), dtype=rec_state.dtype, device=device)
        plan = SimpleNamespace(first=null * 0, count=null * 0 + 1, num_seqs=1)
        kda_segscan.chain_scan(init, bufs.M, bufs.C, bufs.X, plan)  # type: ignore[arg-type]

    # DCP context loop and MLA prefill (same local head count as KDA).
    config = worker.model_config.hf_text_config
    dcp_context.warmup_triton(H, dtype, device)
    q_lora = getattr(config, "q_lora_rank", None)
    if q_lora:
        kv_lora = int(config.kv_lora_rank)
        rope = int(config.qk_rope_head_dim)
        mla_qkv_norm.warmup(
            int(q_lora),
            kv_lora,
            int(q_lora) + kv_lora + rope,
            dtype,
            float(config.rms_norm_eps),
            device,
        )
    cache_dtype = str(worker.cache_config.cache_dtype)
    if cache_dtype.startswith("fp8") and mla_qkv_quant.eligible(proj):
        from vllm.models.kimi_k3.nvidia import mla as k3_mla

        if k3_mla._QKV_QUANT_JIT:
            mla_qkv_quant.load_extension()
    torch.accelerator.synchronize()


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if not current_platform.is_cuda():
        return

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    _warm_attn_res(worker)
    _warm_recurrent_kda(layer, worker.model_config.dtype)
    try:
        _warm_prefill_glue(worker, layer)
    except Exception:  # never block startup; the JIT monitor will flag misses
        logger.warning("Kimi-K3 prefill warmup failed.", exc_info=True)
