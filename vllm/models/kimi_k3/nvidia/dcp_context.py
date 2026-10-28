# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 chunked-context MLA prefill under decode context parallelism.

The impl's DCP loop all-gathers each chunk's latent context, then reorders it
(``reorg_kvcache``: a Python loop of slices and two ``torch.cat`` copies) into
one contiguous key run per request, so that the ragged context FMHA sees one
sequence per request. Each request's FMHA then launches ``ceil(q / 256) x
heads`` CTAs that each walk every key of the chunk: a 128-token query fills 12
of the GPU's SMs.

Here the all-gathered buffer is used in place. Context attention does not
depend on key order, so each (rank, request) block of the gathered buffer --
its valid prefix -- is its own segment, and the FMHA runs over all segments of
the chunk at once with the request's query rows replicated per segment
(``cu_seqlens``). Long segments are split further until the launch has enough
CTAs to fill the GPU. Padding rows between segments are covered by one-row
dummy entries whose output is dropped, so every entry's keys are exactly
``cum_seq_lens_kv[b] .. cum_seq_lens_kv[b + 1]``.

The per-segment partials (the kernel writes bf16 outputs and fp32 lse) are
merged in fp32 into an fp32 running context partial over every chunk; the bf16
context partial is rounded once from it. Today's loop rounds each chunk's
partial and every running merge to bf16, so this adds no bf16 rounding of a
merged value.
"""

from __future__ import annotations

import bisect
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

# Split segments for about this many query-tile CTAs, never past _MAX_CTAS
# (one wave on the 212 SMs of a Rubin GPU).
_TARGET_CTAS = 192
_MAX_CTAS = 212
# Never split a segment below this many keys.
_MIN_SPLIT_KEYS = 1024
# Query tile of the TRTLLM context kernel.
_Q_TILE = 256
_LN2 = math.log(2.0)
# VLLM_K3_DCP_CTX_SEGMENTS=0 restores the impl's loop (A/B and fallback).
_ENABLED = os.environ.get("VLLM_K3_DCP_CTX_SEGMENTS", "1") == "1"
# VLLM_K3_DCP_CTX_FP8_GATHER=0: all-gather the dequantized bf16 latent as before.
_FP8_GATHER = os.environ.get("VLLM_K3_DCP_CTX_FP8_GATHER", "1") == "1"
# VLLM_K3_DCP_CTX_TRITON_PACK=0: the fp8 K/V pack on the _C kernel as before.
_TRITON_PACK = os.environ.get("VLLM_K3_DCP_CTX_TRITON_PACK", "1") == "1"


def use_triton_pack() -> bool:
    return _TRITON_PACK


def use_segments(prefill) -> bool:
    backend = prefill.prefill_backend
    return (
        _ENABLED
        and backend is not None
        and backend.get_name() == "TRTLLM_RAGGED"
        and prefill.query_lens_cpu is not None
    )


# VLLM_K3_DCP_CTX_DIRECT_FMHA=0: go through flashinfer's python wrapper. On (the
# default), the trtllm-gen ragged runner is called directly: the wrapper's
# per-call checks and allocations cost ~90 us of host time, and every argument
# they derive is fixed per plan (all rows active, fp32 bmm scale, no sinks).
_DIRECT_FMHA = os.environ.get("VLLM_K3_DCP_CTX_DIRECT_FMHA", "1") == "1"
_RUNNER = None


def _runner():
    global _RUNNER
    if _RUNNER is None:
        from flashinfer.prefill import get_trtllm_gen_fmha_module
        from flashinfer.utils import get_device_sm_count

        _RUNNER = (
            get_trtllm_gen_fmha_module().trtllm_ragged_attention,
            get_device_sm_count(
                torch.device("cuda", torch.accelerator.current_device_index())
            ),
        )
    return _RUNNER


def run_segments_fmha(
    backend, plan: SegmentPlan, q_rep, k, v, out_dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ragged context FMHA over every entry of the chunk; returns the
    entries' bf16 outputs [rows, H, Dv] and fp32 log2 lse [rows, H]."""
    rows, heads = q_rep.shape[0], q_rep.shape[1]
    out = torch.empty((rows, heads, v.shape[2]), dtype=out_dtype, device=q_rep.device)
    if _DIRECT_FMHA:
        run, sm_count = _runner()
        lse = torch.empty((rows, heads), dtype=torch.float32, device=q_rep.device)
        ws = backend._workspace_buffer
        # trtllm_ragged_attention_deepseek's launch, all rows active (every
        # entry has >= 1 query row and >= 1 key), float scales, no sinks.
        run(
            out,
            q_rep,
            k,
            v,
            ws,
            plan.kv_lens,
            plan.max_q_len,
            plan.max_kv_len,
            backend.scale,
            1.0,
            1.0,
            plan.num_entries,
            -1,
            plan.cum_q,
            plan.cum_kv,
            sm_count,
            False,
            False,
            ws.numel() * ws.element_size(),
            None,
            None,
            lse,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            0,
        )
        return out, lse
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

    o, lse = trtllm_ragged_attention_deepseek(
        query=q_rep,
        key=k,
        value=v,
        workspace_buffer=backend._workspace_buffer,
        seq_lens=plan.kv_lens,
        max_q_len=plan.max_q_len,
        max_kv_len=plan.max_kv_len,
        bmm1_scale=backend.scale,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=plan.num_entries,
        window_left=-1,
        cum_seq_lens_q=plan.cum_q,
        cum_seq_lens_kv=plan.cum_kv,
        enable_pdl=False,
        is_causal=False,
        return_lse=True,
        out=out,
        skip_all_rows_active_check=True,
    )
    return o, lse


_WAVE_SPLIT = os.environ.get("VLLM_K3_DCP_CTX_WAVE_SPLIT", "1") == "1"
_WAVE_MAX_SPLITS = 8
_WAVE_MIN_GAIN = 0.05
_WAVE_ROW_KEYS = 0.00301 / 0.00539  # one replicated query row, in key-units of one wave
_SM_COUNT: dict[str, int] = {}


def _sm_count(device) -> int:
    key = str(device)
    n = _SM_COUNT.get(key)
    if n is None:
        n = _SM_COUNT[key] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return n


def _wave_split_piece(
    runs, query_lens, q_tiles, num_heads: int, longest: int, device
) -> int:
    """Key-piece length (multiple of 128) for launches of >= one wave."""
    sms = _sm_count(device)
    # dummy one-row entries cover the gaps between runs (buffer order)
    spans = sorted((s, s + n) for rr in runs for s, n in rr)
    gaps = sum(1 for k, (s, _) in enumerate(spans) if s > (spans[k - 1][1] if k else 0))
    best_cost = best_piece = None
    base = None
    for s in range(1, _WAVE_MAX_SPLITS + 1):
        piece = longest if s == 1 else math.ceil(longest / s / 128) * 128
        if s > 1 and piece >= longest:
            continue
        ctas = num_heads * gaps
        rows = gaps
        for i in range(len(runs)):
            for _, n in runs[i]:
                parts = math.ceil(n / piece)
                ctas += parts * q_tiles[i] * num_heads
                rows += parts * query_lens[i]
        cost = math.ceil(ctas / sms) * piece + _WAVE_ROW_KEYS * rows
        if s == 1:
            base = cost
        if best_cost is None or cost < best_cost:
            best_cost, best_piece = cost, piece
    # s == 1 always runs: base, best_cost and best_piece are set here
    if best_piece != longest and best_cost > (1.0 - _WAVE_MIN_GAIN) * base:  # type: ignore[operator]
        return longest
    return best_piece  # type: ignore[return-value]


_TAIL_SPLIT = os.environ.get("VLLM_K3_DCP_CTX_TAIL_SPLIT", "1") == "1"
_CTA_KEYS = 1024
_TAIL_STEP = 3072  # short-part sizes tried: multiples of this, up to 1/4 of the run
_TAIL_MIN_GAIN = 0.03  # model gain vs the unsplit launch


def _schedule_keys(groups: list[tuple[int, int]], sms: int) -> float:
    """Makespan (keys) of [(ctas, keys)] dispatched in order to the first free SM."""
    free: list[tuple[float, int]] = [(0.0, sms)]  # (free at, SMs), sorted
    end = 0.0
    for n, keys in groups:
        d = keys + _CTA_KEYS
        while n > 0:
            t, c = free.pop(0)
            a = min(c, n)
            n -= a
            if c > a:
                free.insert(0, (t, c - a))
            done = t + d
            end = max(end, done)
            k = bisect.bisect_left(free, (done, -1))
            if k < len(free) and free[k][0] == done:
                free[k] = (done, free[k][1] + a)
            else:
                free.insert(k, (done, a))
    return end


def _pieces_cost(pieces: list[int], ctas: int, rows: int, sms: int) -> float:
    return _schedule_keys(
        [(ctas, p) for p in pieces], sms
    ) + _WAVE_ROW_KEYS * rows * len(pieces)


def _equal_pieces(n: int, piece: int) -> list[int]:
    parts = math.ceil(n / piece)
    step = math.ceil(n / parts / 128) * 128 if parts > 1 else n
    out = []
    off = 0
    while off < n:
        out.append(min(step, n - off))
        off += out[-1]
    return out


def _tail_split_pieces(
    n: int, q_tiles: int, query_len: int, num_heads: int, device
) -> list[int] | None:
    """Unequal key pieces (long first) for one unsplit run, or None."""
    sms = _sm_count(device)
    ctas = q_tiles * num_heads
    if ctas < sms:
        return None
    unsplit = _pieces_cost([n], ctas, query_len, sms)
    best = None
    for m in (1, 2):
        sizes = range(_TAIL_STEP, n // 4 + 2 * _TAIL_STEP, _TAIL_STEP)
        cost = {
            S: _pieces_cost([n - S] + [S // m] * m, ctas, query_len, sms) for S in sizes
        }
        for S in range(2 * _TAIL_STEP, n // 4 + 1, _TAIL_STEP):
            worst = max(cost[S - _TAIL_STEP], cost[S], cost[S + _TAIL_STEP])
            if best is None or worst < best[0]:
                best = (worst, [n - S] + [S // m] * m)
    if best is None or best[0] > (1.0 - _TAIL_MIN_GAIN) * unsplit:
        return None
    return best[1]


_PLANS: dict[tuple, SegmentPlan] = {}


@dataclass
class SegmentPlan:
    """One chunk's FMHA entries over the all-gathered (rank-major) buffer."""

    num_entries: int
    num_rows: int  # replicated query rows (incl. one row per dummy entry)
    replicate: bool  # False: the entries are exactly the chunk's requests
    max_q_len: int
    max_kv_len: int
    q_index: torch.Tensor | None  # [num_rows] int32 chunk-local query row
    cum_q: torch.Tensor  # [E + 1] int32
    cum_kv: torch.Tensor  # [E + 1] int32
    kv_lens: torch.Tensor  # [E] int32
    # merge metadata
    req_qstart: torch.Tensor  # [R] int32 chunk-local first query row
    req_ent_ptr: torch.Tensor  # [R + 1] int32 into ent_rowbase
    ent_rowbase: torch.Tensor  # [real entries] int32 first replicated row
    token_req: torch.Tensor  # [chunk tokens] int32
    num_tokens: int
    # one-request chunks: copy all-gathered rows fill_src -> fill_dst first
    fill_src: torch.Tensor | None = None
    fill_dst: torch.Tensor | None = None

    def fill_holes(self, gathered: torch.Tensor) -> None:
        if self.fill_dst is not None:
            g = gathered.view(torch.uint8) if gathered.element_size() == 1 else gathered
            g.index_copy_(0, self.fill_dst, g.index_select(0, self.fill_src))


def plan_chunk_segments(
    chunk, query_lens: list[int], num_heads: int, device: torch.device
) -> SegmentPlan:
    """Build (and cache on the chunk) the segment plan for one DCP chunk."""
    plan = getattr(chunk, "_k3_segments", None)
    if plan is not None:
        return plan
    # Chunks with the same layout (the full chunks of one long request) share
    # one plan object, so the layer can also reuse its replicated query.
    sig = (
        chunk.num_local_context_tokens,
        tuple(chunk.padded_local_seq_lens),
        tuple(map(tuple, chunk.local_context_lens_allranks)),
        tuple(chunk.local_starts),
        tuple(query_lens),
        num_heads,
        str(device),
    )
    plan = _PLANS.get(sig)
    if plan is not None:
        chunk._k3_segments = plan
        return plan
    toks = chunk.num_local_context_tokens
    pads = chunk.padded_local_seq_lens
    lens_allranks = chunk.local_context_lens_allranks
    local_starts = chunk.local_starts
    num_reqs = len(pads)
    world = len(lens_allranks[0])
    req_off = np.concatenate([[0], np.cumsum(pads)]).astype(np.int64)

    # Valid (start, len) runs per request, in buffer order; merge adjacent runs.
    runs: list[list[list[int]]] = [[] for _ in range(num_reqs)]
    for r in range(world):
        for i in range(num_reqs):
            n = min(max(0, lens_allranks[i][r] - local_starts[i]), pads[i])
            if n == 0:
                continue
            s = r * toks + int(req_off[i])
            if runs[i] and runs[i][-1][0] + runs[i][-1][1] == s:
                runs[i][-1][1] += n
            else:
                runs[i].append([s, n])

    # One request: move the valid rows past the end of [0, total) into the
    # padding holes inside it (at most world - 1 latent rows, a copy of the same
    # values), so the chunk is one contiguous run: no dummy entries and no
    # per-rank query replication.
    fill = None
    if num_reqs == 1 and len(runs[0]) > 1:
        total = sum(n for _, n in runs[0])
        valid = np.zeros(toks * world, dtype=bool)
        for s, n in runs[0]:
            valid[s : s + n] = True
        holes = np.flatnonzero(~valid[:total])
        srcs = np.flatnonzero(valid[total:]) + total
        assert len(holes) == len(srcs)
        fill = (srcs, holes)
        runs = [[[0, total]]]

    # Cut runs into pieces of at most `piece` keys (a multiple of 128). The
    # TRTLLM context kernel is fastest with the launch in one wave: measured on
    # 61k keys (kpf2-fp2), 12 x 16 = 192 CTAs at N = 128 runs 30 us vs 301 us
    # unsplit, while a second partial wave costs more than it saves. So pieces
    # are sized for ~_TARGET_CTAS CTAs and never push the launch past _MAX_CTAS
    # (dummy entries count: 1 query tile each).
    q_tiles = [math.ceil(q / _Q_TILE) for q in query_lens]
    longest = max(n for rr in runs for _, n in rr)
    num_runs = sum(len(rr) for rr in runs)

    def launch_ctas(piece: int) -> int:
        c = sum(
            math.ceil(n / piece) * q_tiles[i] * num_heads
            for i in range(num_reqs)
            for _, n in runs[i]
        )
        return c + num_heads * num_runs  # upper bound on dummy entries (gaps)

    piece = longest
    if launch_ctas(longest) < _TARGET_CTAS:
        work = sum(
            n * q_tiles[i] * num_heads for i in range(num_reqs) for _, n in runs[i]
        )
        piece = max(_MIN_SPLIT_KEYS, math.ceil(work / _TARGET_CTAS / 128) * 128)
        while piece < longest and launch_ctas(piece) > _MAX_CTAS:
            piece += 128 * max(1, piece // 1024)
        piece = min(piece, longest)
    tail = None
    if piece == longest and _WAVE_SPLIT:
        piece = _wave_split_piece(runs, query_lens, q_tiles, num_heads, longest, device)
        if _TAIL_SPLIT and piece == longest and num_reqs == 1 and len(runs[0]) == 1:
            tail = _tail_split_pieces(
                longest, q_tiles[0], query_lens[0], num_heads, device
            )
    real: list[tuple[int, int, int]] = []
    for i in range(num_reqs):
        for s, n in runs[i]:
            off = 0
            for m in tail if tail is not None else _equal_pieces(n, piece):
                real.append((i, s + off, m))
                off += m
    real.sort(key=lambda e: e[1])

    replicate = not (
        len(real) == num_reqs
        and all(e[0] == k for k, e in enumerate(real))
        and all(real[k][1] + real[k][2] == real[k + 1][1] for k in range(len(real) - 1))
        and real[0][1] == 0
    )

    qstart = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int64)
    q_index: list[np.ndarray] = []
    cum_q = [0]
    cum_kv = [0]
    kv_lens: list[int] = []
    ent_rows: list[list[int]] = [[] for _ in range(num_reqs)]
    cursor = 0
    rows = 0
    max_kv = 0
    for i, s, n in real:
        if s > cursor:  # padding rows: a one-row dummy entry, output dropped
            q_index.append(np.zeros(1, dtype=np.int64))
            rows += 1
            cum_q.append(rows)
            kv_lens.append(s - cursor)
            cum_kv.append(s)
            max_kv = max(max_kv, s - cursor)
        ent_rows[i].append(rows)
        q_index.append(np.arange(qstart[i], qstart[i + 1], dtype=np.int64))
        rows += query_lens[i]
        cum_q.append(rows)
        kv_lens.append(n)
        cum_kv.append(s + n)
        max_kv = max(max_kv, n)
        cursor = s + n
    num_entries = len(kv_lens)

    req_ent_ptr = np.concatenate([[0], np.cumsum([len(e) for e in ent_rows])])
    ent_rowbase = np.concatenate([np.asarray(e, dtype=np.int64) for e in ent_rows])
    token_req = np.repeat(np.arange(num_reqs), query_lens)
    num_tokens = int(qstart[-1])

    parts = [
        np.asarray(cum_q),
        np.asarray(cum_kv),
        np.asarray(kv_lens),
        qstart[:-1],
        req_ent_ptr,
        ent_rowbase,
        token_req,
    ]
    if replicate:
        parts.append(np.concatenate(q_index))
    if fill is not None:
        parts += [fill[0], fill[1]]
    sizes = [len(p) for p in parts]
    flat = async_tensor_h2d(
        np.concatenate(parts).astype(np.int32), device=device, dtype=torch.int32
    )
    views = list(torch.split(flat, sizes))
    fill_src = fill_dst = None
    if fill is not None:
        fill_src, fill_dst = views[-2].long(), views[-1].long()
    plan = SegmentPlan(
        num_entries=num_entries,
        num_rows=rows,
        replicate=replicate,
        max_q_len=max(query_lens) if not replicate else max(max(query_lens), 1),
        max_kv_len=max_kv,
        q_index=views[7] if replicate else None,
        cum_q=views[0],
        cum_kv=views[1],
        kv_lens=views[2],
        req_qstart=views[3],
        req_ent_ptr=views[4],
        ent_rowbase=views[5],
        token_req=views[6],
        num_tokens=num_tokens,
        fill_src=fill_src,
        fill_dst=fill_dst,
    )
    chunk._k3_segments = plan
    if len(_PLANS) > 256:
        _PLANS.clear()
    _PLANS[sig] = plan
    return plan


# The plan's index arrays are views into one int32 buffer (arbitrary 4-byte
# offsets), so their pointers are not specialized on alignment either.
@triton.jit(
    do_not_specialize=[
        "token_req_ptr",
        "req_qstart_ptr",
        "req_ent_ptr_ptr",
        "ent_rowbase_ptr",
        "token_base",
        "num_total_tokens",
    ]
)
def _merge_segments_kernel(
    o_ptr,  # [rows, H, D] bf16 entry partials
    lse_ptr,  # [rows, H] fp32, log2
    acc_o_ptr,  # [T, H, D] fp32 running partial (whole prefill)
    acc_lse_ptr,  # [T, H] fp32 log2, -inf = empty
    out_ptr,  # [T, H, D] bf16 context partial
    out_lse_ptr,  # [H, T] fp32 natural log
    token_req_ptr,
    req_qstart_ptr,
    req_ent_ptr_ptr,
    ent_rowbase_ptr,
    token_base,
    num_total_tokens,
    H: tl.constexpr,
    BH: tl.constexpr,
    D: tl.constexpr,
    LN2: tl.constexpr,
):
    # One program per query token, all heads at once (the per-token entry
    # lookups are paid once).
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    hm = h < H
    d = tl.arange(0, D)
    hd = h[:, None] * D + d[None, :]
    req = tl.load(token_req_ptr + t)
    local = t - tl.load(req_qstart_ptr + req)
    e0 = tl.load(req_ent_ptr_ptr + req)
    e1 = tl.load(req_ent_ptr_ptr + req + 1)
    g = (token_base + t).to(tl.int64)
    m = tl.load(acc_lse_ptr + g * H + h, mask=hm, other=float("-inf"))
    has = m != float("-inf")
    acc = tl.load(
        acc_o_ptr + g * H * D + hd, mask=has[:, None] & hm[:, None], other=0.0
    )
    s = tl.where(has, 1.0, 0.0)
    for e in range(e0, e1):
        row = (tl.load(ent_rowbase_ptr + e) + local).to(tl.int64)
        lse = tl.load(lse_ptr + row * H + h, mask=hm, other=float("-inf"))
        o = tl.load(o_ptr + row * H * D + hd, mask=hm[:, None], other=0.0).to(
            tl.float32
        )
        m_new = tl.maximum(m, lse)
        live = m_new != float("-inf")
        a = tl.where(live, tl.exp2(m - m_new), 0.0)
        b = tl.where(live, tl.exp2(lse - m_new), 0.0)
        acc = acc * a[:, None] + o * b[:, None]
        s = s * a + b
        m = tl.where(live, m_new, m)
    live = m != float("-inf")
    acc = acc / tl.where(live, s, 1.0)[:, None]
    m = tl.where(live, m + tl.log2(tl.where(live, s, 1.0)), m)
    tl.store(acc_o_ptr + g * H * D + hd, acc, mask=hm[:, None])
    tl.store(acc_lse_ptr + g * H + h, m, mask=hm)
    tl.store(
        out_ptr + g * H * D + hd, acc.to(out_ptr.dtype.element_ty), mask=hm[:, None]
    )
    tl.store(out_lse_ptr + h * num_total_tokens + g, m * LN2, mask=hm)


def merge_segments(
    plan: SegmentPlan,
    o: torch.Tensor,
    lse: torch.Tensor,
    acc_o: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    out_lse: torch.Tensor,
    token_base: int,
) -> None:
    """Fold one chunk's entry partials into the fp32 running partial and write
    the bf16 / natural-log context partial for the chunk's tokens."""
    num_heads, head_dim = o.shape[1], o.shape[2]
    assert o.is_contiguous() and lse.is_contiguous()
    _merge_segments_kernel[(plan.num_tokens,)](
        o,
        lse,
        acc_o,
        acc_lse,
        out,
        out_lse,
        plan.token_req,
        plan.req_qstart,
        plan.req_ent_ptr,
        plan.ent_rowbase,
        token_base,
        out.shape[0],
        H=num_heads,
        BH=triton.next_power_of_2(num_heads),
        D=head_dim,
        LN2=_LN2,
        num_warps=4,
    )


# fp8 context gather ---------------------------------------------------------
#
# A plain fp8 cache holds the latent as fp8 bytes (scaled by the layer's
# _k_scale). The impl dequantizes them to bf16 in the local gather and then
# all-gathers bf16. Here the local gather copies the fp8 bytes, the DCP
# all-gather moves fp8 (half the bytes), and one elementwise pass dequantizes
# the all-gathered buffer with gather_and_maybe_dequant_cache's exact formula,
# bf16(float(x) * scale), into the bf16 workspace.


class Fp8ContextGather:
    capacity: int  # set by init_fp8_gather

    def __init__(self, manager, max_gathered: int, dim: int, device: torch.device):
        from functools import partial

        from vllm.v1.attention.ops.dcp import get_direct_dcp_kv_gather_workspace

        world = manager.group.world_size
        fp8 = torch.float8_e4m3fn
        self.local = torch.empty((max_gathered // world, dim), dtype=fp8, device=device)
        self.gathered = torch.empty((max_gathered, dim), dtype=fp8, device=device)
        direct = get_direct_dcp_kv_gather_workspace(
            manager.group, device, max_gathered, dim, fp8, manager.num_ubatches
        )
        if direct is not None:
            self._gather = direct.gather
        else:
            self._gather = partial(
                torch.distributed.all_gather_into_tensor,
                group=manager.group.device_group,
            )

    def gather(self, toks: int, world: int) -> torch.Tensor:
        out = self.gathered[: toks * world]
        self._gather(out, self.local[:toks])
        return out


# One set of buffers per DCP group, shared by every MLA layer (layers run one
# after another). Allocated from the layers' __init__ (init_fp8_gather), i.e.
# before the memory profile, so KV cache sizing accounts for them. The chunked
# workspace size is only final after KV planning (it is aligned to the KV block
# size, which the hybrid page unification sets later: 65,536 -> 66,304 rows at
# block 896), so init sizes for 65,536 + _GATHER_HEADROOM rows; a larger
# runtime workspace falls back to the bf16 all-gather.
_FP8_GATHERS: dict[tuple, Fp8ContextGather] = {}
_GATHER_HEADROOM = 8192


def use_fp8_gather(kv_cache_dtype: str) -> bool:
    return _FP8_GATHER and kv_cache_dtype in ("fp8", "fp8_e4m3")


def _gather_key(manager, dim: int, device) -> tuple:
    return (id(manager.group), dim, str(torch.device(device)), manager.num_ubatches)


def init_fp8_gather(manager, min_gathered: int, dim: int, device) -> Fp8ContextGather:
    key = _gather_key(manager, dim, device)
    g = _FP8_GATHERS.get(key)
    if g is None:
        world = manager.group.world_size
        cap = -(-(min_gathered + _GATHER_HEADROOM) // world) * world
        g = _FP8_GATHERS[key] = Fp8ContextGather(manager, cap, dim, device)
        g.capacity = cap
    return g


_warned = False


def get_fp8_gather(manager, max_gathered: int, dim: int, device):
    """The init-allocated buffers if they hold max_gathered rows, else None."""
    global _warned
    g = _FP8_GATHERS.get(_gather_key(manager, dim, device))
    if g is not None and g.capacity >= max_gathered:
        return g
    if not _warned:
        _warned = True
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "k3 fp8 DCP context gather: no init-allocated buffers for %d rows "
            "(have %s); using the bf16 all-gather",
            max_gathered,
            None if g is None else g.capacity,
        )
    return None


@triton.jit(do_not_specialize=["n"])
def _dequant_fp8_kernel(src_ptr, dst_ptr, scale_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    scale = tl.load(scale_ptr)
    x = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    tl.store(dst_ptr + offs, (x * scale).to(dst_ptr.dtype.element_ty), mask=mask)


def dequant_fp8(src: torch.Tensor, dst: torch.Tensor, scale: torch.Tensor) -> None:
    """Dst = bf16(float(src) * scale), elementwise (gather_and_maybe_dequant_cache)."""
    assert src.is_contiguous() and dst.is_contiguous() and src.shape == dst.shape
    n = src.numel()
    block = 4096
    _dequant_fp8_kernel[(triton.cdiv(n, block),)](
        src, dst, scale, n, BLOCK=block, num_warps=8
    )


@triton.jit(do_not_specialize=["M"])
def _pack_fp8_kernel(
    kv_ptr,
    kv_stride,  # [M, H, NOPE + DV] bf16 (row stride kv_stride, head stride NOPE + DV)
    pe_ptr,
    pe_stride,  # [M, ROPE] bf16
    k_ptr,  # [M, H, NOPE + ROPE] fp8
    v_ptr,  # [M, H, DV] fp8
    M,
    H: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    DV: tl.constexpr,
    BM: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.program_id(1)
    mask = (rm < M)[:, None]
    r64 = rm.to(tl.int64)[:, None]
    cn = tl.arange(0, NOPE)[None, :]
    cv = tl.arange(0, DV)[None, :]
    ce = tl.arange(0, ROPE)[None, :]
    src = kv_ptr + r64 * kv_stride + h * (NOPE + DV)
    kn = tl.load(src + cn, mask=mask)
    vv = tl.load(src + NOPE + cv, mask=mask)
    pe = tl.load(pe_ptr + r64 * pe_stride + ce, mask=mask)
    kdst = k_ptr + r64 * (H * (NOPE + ROPE)) + h * (NOPE + ROPE)
    tl.store(kdst + cn, kn.to(tl.float8e4nv), mask=mask)
    tl.store(kdst + NOPE + ce, pe.to(tl.float8e4nv), mask=mask)
    tl.store(v_ptr + r64 * (H * DV) + h * DV + cv, vv.to(tl.float8e4nv), mask=mask)


def pack_fp8(
    kv_nope: torch.Tensor,  # [M, H, NOPE + DV] bf16, rows may be strided
    k_pe: torch.Tensor,  # [M, ROPE] or [M, 1, ROPE] bf16, rows may be strided
    nope: int,
    bm: int = 64,
    num_warps: int = 4,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 K = [k_nope | k_pe] and fp8 V from one kv_b_proj output (the same
    casts as fused_mla_kv_concat_quant_fp8)."""
    m, heads, width = kv_nope.shape
    dv = width - nope
    k_pe = k_pe.reshape(m, k_pe.shape[-1])
    rope = k_pe.shape[-1]
    assert kv_nope.stride(2) == 1 and kv_nope.stride(1) == width and k_pe.stride(1) == 1
    if out is None:
        k = torch.empty(
            (m, heads, nope + rope), dtype=torch.float8_e4m3fn, device=kv_nope.device
        )
        v = torch.empty(
            (m, heads, dv), dtype=torch.float8_e4m3fn, device=kv_nope.device
        )
    else:
        k, v = out
        assert k.is_contiguous() and v.is_contiguous() and k.shape[0] == v.shape[0] == m
    if m:
        _pack_fp8_kernel[(triton.cdiv(m, bm), heads)](
            kv_nope,
            kv_nope.stride(0),
            k_pe,
            k_pe.stride(0),
            k,
            v,
            m,
            H=heads,
            NOPE=nope,
            ROPE=rope,
            DV=dv,
            BM=bm,
            num_warps=num_warps,
        )
    return k, v


# Persistent K / V buffers and the latent sub-block scratch -----------------
#
# Allocated at layer init (with the fp8 gather), shared by every MLA layer, so
# KV cache sizing accounts for them and large context chunks
# (VLLM_MLA_DCP_CONTEXT_CHUNK_TOKENS) cost no unaccounted transient. kv_b and
# the pack run in sub-blocks of _SUB_ROWS rows, so their bf16 output stays at
# today's per-chunk size whatever the chunk.
_SUB_ROWS = 65536


class ContextKVBuffers:
    def __init__(self, capacity: int, heads: int, qk: int, dv: int, lat: int, device):
        fp8 = torch.float8_e4m3fn
        self.capacity = capacity
        self.k = torch.empty((capacity, heads, qk), dtype=fp8, device=device)
        self.v = torch.empty((capacity, heads, dv), dtype=fp8, device=device)
        self.latent = torch.empty(
            (min(capacity, _SUB_ROWS), lat), dtype=torch.bfloat16, device=device
        )


_KV_BUFFERS: dict[tuple, ContextKVBuffers] = {}


def init_kv_buffers(
    capacity: int, heads: int, qk: int, dv: int, lat: int, device
) -> ContextKVBuffers:
    key = (heads, qk, dv, lat, str(torch.device(device)))
    b = _KV_BUFFERS.get(key)
    if b is None or b.capacity < capacity:
        b = _KV_BUFFERS[key] = ContextKVBuffers(capacity, heads, qk, dv, lat, device)
    return b


def get_kv_buffers(rows: int, heads: int, qk: int, dv: int, lat: int, device):
    b = _KV_BUFFERS.get((heads, qk, dv, lat, str(torch.device(device))))
    return b if b is not None and b.capacity >= rows else None


def produce_kv_fp8(
    g8: torch.Tensor,  # [R, lat] fp8 all-gathered latent (holes filled)
    kv_b,  # callable: bf16 [n, kv_lora_rank] -> bf16 [n, H * (nope + dv)]
    k_scale: torch.Tensor,
    bufs: ContextKVBuffers,
    kv_lora_rank: int,
    nope: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 K / V for R all-gathered rows, in sub-blocks: dequant (the cache's
    formula) -> kv_b -> pack, into the persistent buffers."""
    rows = g8.shape[0]
    k, v = bufs.k[:rows], bufs.v[:rows]
    sub = bufs.latent.shape[0]
    for a in range(0, rows, sub):
        b = min(rows, a + sub)
        lat = bufs.latent[: b - a]
        dequant_fp8(g8[a:b], lat, k_scale)
        kv_nope = kv_b(lat[:, :kv_lora_rank]).view(b - a, heads, -1)
        pack_fp8(kv_nope, lat[:, kv_lora_rank:], nope, out=(k[a:b], v[a:b]))
    return k, v


# VLLM_K3_CTXREC_FUSED=1 (dev): the fp8-gather context path runs dequant + kv_b + pack
# as one CuTe DSL kernel (ops/ctxrec_cute.py; same dequant and casts, fp32 GEMM
# accumulation in tcgen05 order). Only for a BF16 kv_b weight [heads * 256, 512] fed
# the bf16 latent (its input cast, if any, is to bf16: a no-op), and the K3 head
# geometry.
_CTXREC_FUSED = os.environ.get("VLLM_K3_CTXREC_FUSED", "0") == "1"


def use_ctxrec_fused(kv_b_proj, kv_b_proj_input_dtype) -> bool:
    if not _CTXREC_FUSED or kv_b_proj_input_dtype not in (None, torch.bfloat16):
        return False
    w = getattr(kv_b_proj, "weight", None)
    return (
        w is not None
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[1] == 512
        and w.shape[0] % 256 == 0
        and w.is_contiguous()
        and getattr(kv_b_proj, "bias", None) is None
    )


def ctxrec_fused(
    g8: torch.Tensor,
    k_scale: torch.Tensor,
    w: torch.Tensor,
    bufs: ContextKVBuffers,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 K / V for R all-gathered rows in one launch (replaces produce_kv_fp8)."""
    from vllm.models.kimi_k3.nvidia.ops.ctxrec_cute import ctxrec_cute

    rows = g8.shape[0]
    k, v = bufs.k[:rows], bufs.v[:rows]
    ctxrec_cute(g8.contiguous(), k_scale, w, k, v, heads)
    return k, v


def warmup_triton(heads: int, out_dtype: torch.dtype, device: torch.device) -> None:
    """Compile the context loop's Triton kernels once at the serving shapes and
    strides (their size arguments are not specialized): the fp32 merge, the fp8
    latent dequant and the fp8 K / V pack (and the fused-kv_b dequant when on)."""
    from types import SimpleNamespace

    i32 = torch.int32
    lora, rope, nope, dv = 512, 64, 128, 128
    one = torch.zeros(1, dtype=i32, device=device)
    plan = SimpleNamespace(
        num_tokens=1,
        token_req=one,
        req_qstart=one,
        req_ent_ptr=torch.tensor([0, 1], dtype=i32, device=device),
        ent_rowbase=one,
    )
    o = torch.zeros((1, heads, dv), dtype=out_dtype, device=device)
    lse = torch.zeros((1, heads), dtype=torch.float32, device=device)
    acc_o = torch.zeros((1, heads, dv), dtype=torch.float32, device=device)
    acc_lse = torch.full((1, heads), float("-inf"), dtype=torch.float32, device=device)
    out = torch.empty((1, heads, dv), dtype=out_dtype, device=device)
    out_lse = torch.empty((heads, 1), dtype=torch.float32, device=device)
    merge_segments(plan, o, lse, acc_o, acc_lse, out, out_lse, 0)  # type: ignore[arg-type]
    scale = torch.ones(1, dtype=torch.float32, device=device)
    src = torch.zeros((16, lora + rope), dtype=torch.float8_e4m3fn, device=device)
    gathered = torch.empty((16, lora + rope), dtype=torch.bfloat16, device=device)
    dequant_fp8(src, gathered, scale)
    kvb = torch.zeros((16, heads * (nope + dv)), dtype=torch.bfloat16, device=device)
    pack_fp8(kvb.view(16, heads, nope + dv), gathered[:, lora:], nope)
    if _CTXREC_FUSED:
        # one compile covers every row count (symbolic R)
        from vllm.models.kimi_k3.nvidia.ops.ctxrec_cute import ctxrec_cute

        k8 = torch.empty(
            (16, heads, nope + rope), dtype=torch.float8_e4m3fn, device=device
        )
        v8 = torch.empty((16, heads, dv), dtype=torch.float8_e4m3fn, device=device)
        w = torch.zeros(
            (heads * (nope + dv), lora), dtype=torch.bfloat16, device=device
        )
        ctxrec_cute(src, scale, w, k8, v8, heads)
    kpe = globals().get("dequant_fp8_kpe")
    if globals().get("_FUSED_KVB", False) and kpe is not None:
        k = torch.empty(
            (16, heads, nope + rope), dtype=torch.float8_e4m3fn, device=device
        )
        kpe(src, gathered, k, scale, lora, nope)
