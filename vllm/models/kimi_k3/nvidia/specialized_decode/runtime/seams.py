# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue of the stock seams the specialized decode runs at wide decode shapes.

``gate_sigmoid_mul``: the MLA output gate of the stock attention seam (S2) as one
kernel: ``core * sigmoid(gate)`` computed in FP32 and rounded once, the arithmetic of
the stock layer's compiled gate, read from the strided gate columns of the
in-projection rows and written into the rows the o-projection reads.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gate_kernel(
    core, gate, out, core_stride, gate_stride, out_stride,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    c = tl.load(core + row * core_stride + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate + row * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = c * tl.sigmoid(g)
    tl.store(out + row * out_stride + cols, y.to(out.dtype.element_ty), mask=mask)


def gate_sigmoid_mul(core, gate, out):
    """``out = core * sigmoid(gate)`` (FP32, one rounding); rows may be strided."""
    rows, width = core.shape
    block = triton.next_power_of_2(width)
    _gate_kernel[(rows,)](
        core, gate, out, core.stride(0), gate.stride(0), out.stride(0),
        WIDTH=width, BLOCK=block,
    )  # fmt: skip
    return out


def warm_gate(pitch, width, device):
    """Compile the gate kernel for in-projection rows of ``pitch`` columns outside any
    CUDA graph capture (the seam only runs inside captures)."""
    rows = torch.zeros((2, pitch), dtype=torch.bfloat16, device=device)
    core = torch.zeros((2, width), dtype=torch.bfloat16, device=device)
    gate_sigmoid_mul(core, rows[:, pitch - width :], torch.empty_like(core))
