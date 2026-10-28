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


class _KeyOnly(Exception):
    pass


def _warm_stock_mla_decode(worker: Worker) -> None:
    """Compile tokenspeed's generic MLA decode kernel (TOKENSPEED_MLA ``forward_mqa``)
    for every compile key an eager step can reach. The specialized decode covers the
    captured steady steps; the stock path still serves the decode rows of eager and
    mixed steps (target), the drafter's uncaptured steps, and prompts or chunk tails
    of at most q tokens (MLA treats them as decodes, at their own q_len). Its key
    (``mla_decode._get_compiled_mla_kernel``) depends on the batch and max_seq_len
    through the split count (workspace or not, the reducer's D tiles and max splits),
    so each layer shape walks batch x split-defining lengths with the compile
    stubbed to record the key, then compiles each distinct key once."""
    from tokenspeed_mla import mla_decode as md
    from tokenspeed_mla import tokenspeed_mla_decode

    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLAImpl,
        _get_workspace,
    )

    config = worker.vllm_config
    context = config.compilation_config.static_forward_context
    spec = config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    # past ~SMs / 2 requests tokenspeed's split is 1 and the reducer's tiles settle
    max_reqs = min(config.scheduler_config.max_num_seqs, 128)
    speculator = getattr(worker.model_runner, "speculator", None)
    draft_names = set(getattr(speculator, "draft_attn_layer_names", ()) or ())
    draft_q = getattr(speculator, "num_query_per_req", num_spec + 1)
    draft_causal = not getattr(speculator, "requires_non_causal", False)
    interleave = config.parallel_config.cp_kv_cache_interleave_size
    device = torch.device("cuda", torch.accelerator.current_device_index())

    # layer shape -> (q_len -> request counts), the impl and its cache
    shapes: dict = {}
    for name, layer in context.items():
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, TokenspeedMLAImpl):
            continue
        read = getattr(layer, "_attn_read_kv_cache", None)
        cache = read() if callable(read) else getattr(layer, "kv_cache", None)
        if isinstance(cache, (list, tuple)):
            cache = cache[0] if cache else None
        if cache is None or cache.numel() == 0:
            continue
        if cache.dtype == torch.uint8:
            cache = cache.view(torch.float8_e4m3fn)
        heads = impl.num_heads * max(impl.dcp_world_size, 1)  # DCP gathers q heads
        if name in draft_names:
            causal = (draft_causal,)
            qlens = {q: range(1, max_reqs + 1) for q in {draft_q - 1, draft_q} if q > 0}
        else:
            causal = (True,)
            # steady width at every batch; shorter "decodes" (short prompts, chunk
            # tails) at a few requests
            qlens = {q: range(1, 9) for q in range(1, num_spec + 1)}
            qlens[num_spec + 1] = range(1, max_reqs + 1)
        key: tuple = (heads, cache.shape[1], causal, impl.need_to_return_lse_for_decode)
        key += (impl.dcp_world_size, impl.dcp_rank, impl._min_split_kv)
        shapes.setdefault(key, (qlens, impl, cache))

    keys: dict = {}

    def record(**kwargs):
        keys.setdefault(tuple(kwargs.items()), kwargs)
        raise _KeyOnly

    compile_fn = md._get_compiled_mla_kernel
    md._get_compiled_mla_kernel = record
    try:
        for (heads, _, causal, lse, cp, _, min_split), (
            qlens,
            impl,
            cache,
        ) in shapes.items():
            workspace = _get_workspace(
                device, heads, impl.kv_lora_rank, min_split, impl._max_decode_tokens
            )
            i32 = dict(dtype=torch.int32, device=device)
            for q_len, batches in qlens.items():
                top = max(batches)
                query = torch.zeros(
                    (top, q_len, heads, cache.shape[-1]),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                tables = torch.zeros((top, 1), **i32)
                lens = torch.ones(top, **i32)
                for b in batches:
                    # max_seq_len sets the split through its 128-key tiles: 1..64
                    # tiles reach every split count tokenspeed can pick
                    for tiles in range(1, 65):
                        for c in causal:
                            try:
                                tokenspeed_mla_decode(
                                    query=query[:b],
                                    kv_cache=cache,
                                    workspace_buffer=workspace,
                                    kv_lora_rank=impl.kv_lora_rank,
                                    qk_rope_head_dim=impl.qk_rope_head_dim,
                                    block_tables=tables[:b],
                                    seq_lens=lens[:b],
                                    max_seq_len=128 * tiles,
                                    softmax_scale=1.0,
                                    output_scale=1.0,
                                    enable_pdl=False,
                                    return_lse=lse,
                                    causal_mask=c,
                                    causal_seqs=lens[:b] if cp > 1 else None,
                                    cp_world=cp,
                                    cp_rank=impl.dcp_rank,
                                    cp_interleave_size=interleave,
                                    min_split_kv=min_split,
                                )
                            except _KeyOnly:
                                pass
                            except (AssertionError, ValueError):
                                pass  # beyond the serving workspace: unreachable
    finally:
        md._get_compiled_mla_kernel = compile_fn
    for kwargs in keys.values():
        compile_fn(**kwargs)
    if keys:
        logger.info(
            "Kimi-K3: compiled %d tokenspeed MLA decode keys for %d layer shapes.",
            len(keys),
            len(shapes),
        )


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
        _warm_stock_mla_decode(worker)
    except Exception:  # never block startup; the JIT monitor will flag misses
        logger.warning("Kimi-K3 stock MLA decode warmup failed.", exc_info=True)
