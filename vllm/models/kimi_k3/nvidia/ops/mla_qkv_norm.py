# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA q_a / kv_a RMSNorm, bitwise equal to
``vllm.models.common.ops.fused_qk_rmsnorm.fused_q_kv_rmsnorm``.

The common kernel runs one program per (token, q-or-kv row): at a 16k prefill
that is 32k programs of one 3 KB / 1 KB row each, latency bound (44.8 us per
call in-model for ~132 MB). Here a program walks TOK tokens of its task in a
software-pipelined loop; each row is the common kernel's body unchanged (same
BLOCK_SIZE, warps and expressions, so the same layout and reduction order).
"""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

_TOK = int(
    os.environ.get("VLLM_K3_MLA_QKV_NORM_TOK_MAX", "8")
)  # tokens per program, at most
_STAGES = int(os.environ.get("VLLM_K3_MLA_QKV_NORM_STAGES", "3"))
_MIN_PROGRAMS = 848  # keep >= 4 programs per SM (212 SMs) before taking more rows each


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_q_kv_rmsnorm_rows_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    num_tokens,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOK: tl.constexpr,
    STAGES: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid_task = tl.program_id(1)
    if pid_task == 0:
        SIZE = Q_SIZE
        in_ptr = q_ptr
        in_stride = q_in_stride
        weight_ptr = q_weight_ptr
        out_ptr = q_out_ptr
        out_stride = q_out_stride
    else:
        SIZE = KV_SIZE
        in_ptr = kv_ptr
        in_stride = kv_in_stride
        weight_ptr = kv_weight_ptr
        out_ptr = kv_out_ptr
        out_stride = kv_out_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    for k in tl.range(0, TOK, num_stages=STAGES):
        token_idx = tl.program_id(0).to(tl.int64) * TOK + k
        ok = token_idx < num_tokens
        row_in = in_ptr + token_idx * in_stride
        row_out = out_ptr + token_idx * out_stride
        # The common kernel's row body (RMSNorm in fp32, one cast at store).
        x = tl.load(row_in + block, mask=mask & ok, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / SIZE
        rrms = tl.rsqrt(variance + eps)
        y = x * rrms * w
        tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask & ok)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for the common fused_q_kv_rmsnorm (same outputs, bitwise)."""
    assert qr.ndim == 2 and kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    qr_out = torch.empty(qr.shape, dtype=qr.dtype, device=qr.device)
    kv_out = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    n = qr.shape[0]
    if n == 0:
        return qr_out, kv_out
    tok = 1
    while tok < _TOK and triton.cdiv(n, tok * 2) * 2 >= _MIN_PROGRAMS:
        tok *= 2
    _launch(qr, kv, qr_out, kv_out, q_weight, kv_weight, eps, n, tok)
    return qr_out, kv_out


def _launch(qr, kv, qr_out, kv_out, q_weight, kv_weight, eps, n, tok):
    block_size = next_power_of_2(max(qr.shape[1], kv.shape[1]))
    _fused_q_kv_rmsnorm_rows_kernel[(triton.cdiv(n, tok), 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        n,
        Q_SIZE=qr.shape[1],
        KV_SIZE=kv.shape[1],
        BLOCK_SIZE=block_size,
        TOK=tok,
        STAGES=_STAGES,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=8 if block_size >= 2048 else 4,
    )


def warmup(
    q_size: int, kv_size: int, in_row_stride: int, weight_dtype, eps: float, device
) -> None:
    """Compile every TOK variant for q / kv column slices of a [n, in_row_stride]
    projection output (one row of dummy data)."""
    a = torch.zeros((1, in_row_stride), dtype=weight_dtype, device=device)
    qr, kv = a[:, :q_size], a[:, q_size : q_size + kv_size]
    qw = torch.zeros(q_size, dtype=weight_dtype, device=device)
    kw = torch.zeros(kv_size, dtype=weight_dtype, device=device)
    qo = torch.empty(qr.shape, dtype=qr.dtype, device=device)
    ko = torch.empty(kv.shape, dtype=kv.dtype, device=device)
    tok = 1
    while tok <= _TOK:
        _launch(qr, kv, qo, ko, qw, kw, eps, 1, tok)
        tok *= 2
