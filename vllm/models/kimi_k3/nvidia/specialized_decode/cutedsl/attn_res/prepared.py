# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ...shapes import Shape, rows_policy
from ..common.host import (
    architecture,
    check_registers,
    dynamic_fake,
    dynamic_view,
    fake,
    kernel_registers,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..common.policy import batch_table
from .kernel import ClusterAttnRes

# Devices with more SMs than GB300's 152 (Rubin): the AttnRes stores the updated prefix
# row (and its aux copies and saved block) before its second cluster exchange instead
# of after it, so fewer of its stores are outstanding when its grid completes and the
# PDL successor is released. Same values and order per address. They also spread each
# cluster exchange's DSMEM pushes over one thread per (value, peer) (``spread_push``;
# same values; 4 real ranks, us/layer: KdaHead next -0.01..-0.02, in-proj -0.04).
STORES_FIRST_SMS = 152
# Rubin: register budgets (per thread) of the vote-polled (``poll_warp``) variants by
# block count: the LSU-poll variants' allocation, where the vote alone takes more (6
# blocks without a write: 88). The in-projection after an AttnRes loads its CTAs
# beside the AttnRes CTAs, and a larger AttnRes allocation delays them (16 ranks, B8
# serving, 88 instead of 80 registers: MLA in-projection 59% early instead of 100%,
# KDA 88%; +14.8 / +6.1 us/step). A variant over its budget is compiled again with a
# launch bound (``ClusterAttnRes.min_blocks``; same code, no spills) and checked.
RUBIN_VOTE_REGISTERS = {6: 80}
ROWS_TOKENS: dict = {"*": "193-256:2"}


def rows_tokens(rows):
    return rows_policy(ROWS_TOKENS, Shape(rows), 1)


@cache
def compile_consumer(
    nb, delta, write, final, slots, cap, arch, aux=False, sanitize=False, carveout=None,
    stores_first=False, spread_push=False, poll_warp=False, min_blocks=0, tokens=1,
):  # fmt: skip
    """``carveout`` only keys the cache: consumers with a preferred carve-out get their
    own function (the attribute is per function). ``tokens`` > 1: the token-batched
    ``ClusterAttnResRows`` (``ROWS_TOKENS``)."""
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    options = dict(
        slots=slots, has_delta=delta, write=write, final=final, sanitize=sanitize,
        stores_first=stores_first, spread_push=spread_push, poll_warp=poll_warp,
        min_blocks=min_blocks,
    )  # fmt: skip
    kernel: ClusterAttnRes
    if tokens > 1:
        from .rows import ClusterAttnResRows

        kernel = ClusterAttnResRows(nb, tokens=tokens, **options)
    else:
        kernel = ClusterAttnRes(nb, **options)
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    args += (row,) * aux  # the updated prefix row, also stored to an aux stream
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring.
    ``carveout``: a preferred shared-memory carve-out (percent) for its launches; 100
    lets a big-smem PDL successor become resident on this kernel's SMs while it runs
    (the default carve-out of this small-smem kernel blocks it)."""

    def __init__(self, weights, key, eps, carveout=None):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.sanitize = False
        self.carveout = carveout
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.stores_first = self.spread_push = sms > STORES_FIRST_SMS
        # Rubin: the Lamport delta poll exits by a warp vote (``ClusterAttnRes``).
        self.poll_warp = sms > STORES_FIRST_SMS
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.ring_rows: Any = None  # wide batches' ring (``bind_ring_rows``)
        self.rows_codes: dict = {}  # ROWS_TOKENS: tokens -> (code, aux code or None)
        self.aux: Any = None

    def _compile(self, slots, cap, aux=False, tokens=1):
        arch = architecture(self.device)
        sanitize = self.sanitize and slots > 1
        first, spread = self.stores_first, self.spread_push
        vote = self.poll_warp and slots > 0
        shape = (*self.key, slots, cap, arch, aux, sanitize, self.carveout)
        shape += (first, spread)
        code = compile_consumer(*shape, vote, 0, tokens)
        # The register budget keeps one-token CTAs resident beside the in-projection;
        # the token-batched variant has a fraction of their CTAs.
        budget = RUBIN_VOTE_REGISTERS.get(self.key[0]) if vote and tokens == 1 else None
        if budget and kernel_registers(code, self.device.index) > budget:
            code = compile_consumer(*shape, vote, 64 * 1024 // (128 * budget))
            what = f"AttnRes ({self.key[0]} blocks, vote poll)"
            check_registers(code, self.device.index, budget, what)
        if self.carveout is None:
            return code
        return prepare_shared_carveout(code, self.device.index, self.carveout)

    @classmethod
    def for_target(cls, target, *, final=False, has_delta=True, carveout=None):
        if final:
            norm, query = target.output_attn_res_norm, target.output_attn_res_proj
            output, nb, write = target.norm, target.num_attn_res_blocks, -1
        else:
            norm = target.self_attention_res_norm
            query, output = target.self_attention_res_proj, target.input_layernorm
            nb = target.prev_valid_blocks
            write = target.block_write_idx if target.is_block_write_layer else -1
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        return cls(weights, (nb, has_delta, write, final), eps, carveout)

    @classmethod
    def for_post(cls, decoder):
        """The AttnRes after the attention's o-projection (the stock decoder layer's
        ``_post_attn_norm``), for steps whose o-projection runs unfused: the delta is
        the reduced o-projection output; a block-write layer starts a new block: the
        kernel (no delta) takes that output as the new prefix and stores it."""
        write = decoder.is_block_write_layer
        norm, query = decoder.mlp_res_norm, decoder.mlp_res_proj
        output = decoder.post_attention_layernorm
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        key = (decoder.prev_valid_blocks + write, not write, -1, False)
        return cls(weights, key, eps)

    def bind_ring(self, mailbox, *, sanitize=False):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm.
        ``sanitize``: the sum maps -0 to +0 (``ClusterAttnRes``)."""
        shape = tuple(mailbox.shape[:2])
        self.sanitize = sanitize
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_ring_rows(self, mailbox):
        """The wide batches' Lamport input ring [slots, cap > 64, 7168] (the wide MoE
        tail's output); ``run_ring`` reads it for batches above the bound ring's cap."""
        slots, cap = shape = tuple(mailbox.shape[:2])
        self.ring_rows = self._compile(slots, cap), to_cute(mailbox), shape
        tokens = {t for t in batch_table(ROWS_TOKENS.get("*", "")).values() if t > 1}
        self.rows_codes = {
            t: (self._compile(slots, cap, tokens=t), None) for t in tokens
        }

    def bind_aux(self):
        """Variants that also store the updated prefix rows (a stock aux stream)."""
        ring = rows = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, True)
        if self.ring_rows is not None:
            slots, cap = self.ring_rows[2]
            rows = self._compile(slots, cap, True)
            self.rows_codes = {
                t: (code, self._compile(slots, cap, True, tokens=t))
                for t, (code, _) in self.rows_codes.items()
            }
        self.aux = self._compile(0, 0, True), ring, rows

    def run_ring(self, prefix, blocks, out, aux=None):
        rows = prefix.shape[0]
        if rows > self.ring[2][1]:  # the rings re-arm what they read: no row past cap
            if self.ring_rows is None or rows > self.ring_rows[2][1]:
                raise ValueError(f"AttnRes ring input of {rows} rows: no ring holds it")
            code = self.ring_rows[0] if aux is None else self.aux[2]
            batched = self.rows_codes.get(rows_tokens(rows))
            if batched is not None:
                code = batched[0] if aux is None else batched[1]
            return self._launch(code, self.ring_rows[1], prefix, blocks, out, aux)
        code = self.ring[0] if aux is None else self.aux[1]
        return self._launch(code, self.ring[1], prefix, blocks, out, aux)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux)

    def _launch(self, code, source, prefix, blocks, out, aux=None):
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *aux
        code(source, prefix, blocks, *args, stream(self.device))
        return out
