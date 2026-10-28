# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 KDA prefill short convolution (width 4, SiLU), bitwise equal to
``causal_conv1d_fn`` (vllm/model_executor/layers/mamba/ops/causal_conv1d.py).

The generic kernel walks each program's 8 tokens one after another, one
dependent [BLOCK_N] global load per iteration, with 256 channels per program:
at the 16k prefill shape it runs 37.5 us per q / k / v call (1 GPU; 42 us
in-model) against a ~12-14 us strided-copy floor. Here the program keeps the
generic kernel's mapping (the metadata's BLOCK_M = 8 (sequence, chunk) lists),
1D [BLOCK_N] layout, taps in the same order (acc = 0; acc += x[t-3] * w0; ...;
acc += x[t] * w3, bf16 products into an fp32 accumulator), SiLU expression and
conv-state update (by each sequence's chunk-0 program, after it has read the
old state). What changes: the 8 row loads are issued up front and the 8 token
bodies are unrolled, and a program covers 512 channels (4 warps): 18.6 us.
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

_BLOCK_M = 8  # must match the causal-conv1d metadata (compute_causal_conv1d_metadata)
_BLOCK_N = int(os.environ.get("VLLM_K3_KDA_CONV_BN", "512"))
_WARPS = int(os.environ.get("VLLM_K3_KDA_CONV_WARPS", "4"))


@triton.jit
def _tap4(c0, c1, c2, xt, w0, w1, w2, w3, BLOCK_N: tl.constexpr):
    # The generic kernel's loop body for one token: bf16 products, fp32
    # accumulator, taps oldest first, then SiLU.
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc += c0 * w0
    acc += c1 * w1
    acc += c2 * w2
    acc += xt * w3
    return acc / (1 + tl.exp(-acc))


@triton.jit(do_not_specialize=["num_cache_lines"])
def _kda_conv_fwd_kernel(
    x_ptr,  # (dim, cu_seqlen) view, channel stride 1
    w_ptr,  # (dim, 4)
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,
    dim: tl.constexpr,
    num_cache_lines,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    state_len = 3  # KERNEL_WIDTH - 1
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    conv_states_input_coord = tl.load(
        cache_indices_ptr + idx_seq * stride_cache_indices
    ).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            col2 = tl.load(prior_tokens, mask_w, 0.0)
            col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        tl.debug_barrier()

        # Conv-state update: the generic kernel's STEP 2, unchanged.
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        col1 = tl.load(
            prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
        )
        col0 = tl.load(
            prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
        )

    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # The generic kernel's token loop (BLOCK_M = 8 tokens), with the 8 row
    # loads issued up front instead of one per iteration.
    xb = x_base + token_offset * stride_x_token
    ob = (
        o_ptr
        + (sequence_start_index + token_offset) * stride_o_token
        + (idx_feats * stride_o_dim)
    )
    x0 = tl.load(xb + 0 * stride_x_token, mask=mask_w & (segment_len > 0))
    x1 = tl.load(xb + 1 * stride_x_token, mask=mask_w & (segment_len > 1))
    x2 = tl.load(xb + 2 * stride_x_token, mask=mask_w & (segment_len > 2))
    x3 = tl.load(xb + 3 * stride_x_token, mask=mask_w & (segment_len > 3))
    x4 = tl.load(xb + 4 * stride_x_token, mask=mask_w & (segment_len > 4))
    x5 = tl.load(xb + 5 * stride_x_token, mask=mask_w & (segment_len > 5))
    x6 = tl.load(xb + 6 * stride_x_token, mask=mask_w & (segment_len > 6))
    x7 = tl.load(xb + 7 * stride_x_token, mask=mask_w & (segment_len > 7))
    o = _tap4(col0, col1, col2, x0, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 0 * stride_o_token, o, mask=mask_w & (segment_len > 0))
    o = _tap4(col1, col2, x0, x1, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 1 * stride_o_token, o, mask=mask_w & (segment_len > 1))
    o = _tap4(col2, x0, x1, x2, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 2 * stride_o_token, o, mask=mask_w & (segment_len > 2))
    o = _tap4(x0, x1, x2, x3, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 3 * stride_o_token, o, mask=mask_w & (segment_len > 3))
    o = _tap4(x1, x2, x3, x4, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 4 * stride_o_token, o, mask=mask_w & (segment_len > 4))
    o = _tap4(x2, x3, x4, x5, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 5 * stride_o_token, o, mask=mask_w & (segment_len > 5))
    o = _tap4(x3, x4, x5, x6, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 6 * stride_o_token, o, mask=mask_w & (segment_len > 6))
    o = _tap4(x4, x5, x6, x7, w_col0, w_col1, w_col2, w_col3, BLOCK_N)
    tl.store(ob + 7 * stride_o_token, o, mask=mask_w & (segment_len > 7))


def supported(x: torch.Tensor, weight: torch.Tensor, conv_states, metadata) -> bool:
    return (
        weight.shape[1] == 4
        and x.dim() == 2
        and x.stride(0) == 1
        and conv_states is not None
        and conv_states.dtype == x.dtype
        and metadata is not None
        and metadata.nums_dict is not None
        and _BLOCK_M in metadata.nums_dict
        and metadata.nums_dict[_BLOCK_M]["batch_ptr"] is not None
    )


def kda_prefill_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    metadata,
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """causal_conv1d_fn(x, weight, None, activation="silu", conv_states=...,
    has_initial_state=..., cache_indices=..., query_start_loc=...,
    metadata=metadata) for width 4, channel-last x (dim, tokens)."""
    out = torch.empty_like(x)
    dim, _ = x.shape
    nums = metadata.nums_dict[_BLOCK_M]
    tot = nums["tot"]
    if tot == 0:
        return out
    grid = (tot, triton.cdiv(dim, _BLOCK_N))
    _kda_conv_fwd_kernel[grid](
        x,
        weight,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        nums["batch_ptr"],
        nums["token_chunk_offset_ptr"],
        out,
        dim,
        conv_states.size(0),
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        out.stride(0),
        out.stride(1),
        pad_slot_id,
        null_block_id,
        HAS_NULL_BLOCK=null_block_id is not None,
        NP2_STATELEN=4,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_WARPS,
        num_stages=2,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out
