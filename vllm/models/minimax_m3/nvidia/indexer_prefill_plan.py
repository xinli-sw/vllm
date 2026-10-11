# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host metadata helpers for MiniMax indexer prefill."""

import math
from collections.abc import Callable

import numpy as np
import torch

PAGE_SIZE = 128


def _segment_page_index(
    seg_req: list[int], seg_kv: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(row, col)`` into the prefill block table for every segment's pages
    ``0 .. cdiv(kv, PAGE_SIZE) - 1``, concatenated in segment order (the fmha
    plan's ``kv_page_indptr`` order)."""
    pages = np.asarray(
        [(kv + PAGE_SIZE - 1) // PAGE_SIZE for kv in seg_kv], dtype=np.int64
    )
    rows = np.repeat(np.asarray(seg_req, dtype=np.int64), pages)
    starts = np.cumsum(pages) - pages
    cols = np.arange(int(pages.sum()), dtype=np.int64) - np.repeat(starts, pages)
    return rows, cols


def _host_prefill_kv_lens(
    seq_lens_cpu_upper_bound: torch.Tensor | None,
    qo_lens: torch.Tensor,
    lo: int,
    hi: int,
    decode_threshold: int,
) -> torch.Tensor | None:
    """KV lengths of prefill rows ``[lo, hi)`` from host data, or None if the
    host values cannot be trusted (the caller then reads them from the GPU).

    ``seq_lens_cpu_upper_bound`` is exact on prefill rows and only optimistic
    on async spec-decode rows, whose query length is <= ``decode_threshold``;
    any such row on the prefill side falls back to the device lengths."""
    if seq_lens_cpu_upper_bound is None or seq_lens_cpu_upper_bound.is_cuda:
        return None
    if qo_lens.numel() == 0 or seq_lens_cpu_upper_bound.shape[0] < hi:
        return None
    if int(qo_lens.min()) <= decode_threshold:
        return None
    kv_lens = seq_lens_cpu_upper_bound[lo:hi].to(torch.int32)
    if bool((kv_lens < qo_lens).any()):
        return None
    return kv_lens



# Prefill split-KV score (attention_config.minimax_m3_indexer_prefill_kv_split).
# fmha_sm100 splits the KV walk only on its 128-query tile, so a split plan cuts
# each request into segments of at most this many queries.
_PREFILL_SEG_LEN = 128
_PREFILL_MAX_KV_SPLITS = 32
# Below this context the serial KV walk is short; keep the unsplit plan.
_PREFILL_SPLIT_MIN_KV = 2048
# Bound on total_q * num_kv_splits * num_heads (rows of the split workspace).
_PREFILL_SPLIT_MAX_WS_ROWS = 1 << 18


def _plan_prefill_segments(
    qo_lens: list[int],
    kv_lens: list[int],
    num_heads: int,
    num_sms: int,
    max_splits: int,
) -> tuple[list[int], list[int], list[int], int]:
    """Prefill score plan layout ``(seg_qo, seg_kv, seg_req, num_kv_splits)``.

    Returns the unsplit layout (one segment per request, ``num_kv_splits=1``)
    unless the unsplit plan under-fills the GPU on a long context. Then each
    request is cut into <=128-query segments and the KV walk is split to fill
    about one wave of CTAs.

    This is exact for the score-only output: query rows are independent, and
    segment ``[s, e)`` of a request with context ``c`` gets ``kv_len = c + e``
    with bottom-right causal masking, i.e. the same keys its queries see in the
    unsplit plan. Each 128-key score tile is written by exactly one split, so
    there is no cross-split combine.
    """
    unsplit = (list(qo_lens), list(kv_lens), list(range(len(qo_lens))), 1)
    if max_splits <= 1 or not qo_lens or max(kv_lens) < _PREFILL_SPLIT_MIN_KV:
        return unsplit
    unsplit_tile = 128 if max(qo_lens) <= 128 else 256
    unsplit_ctas = sum(math.ceil(q / unsplit_tile) for q in qo_lens) * num_heads
    if unsplit_ctas >= num_sms:
        return unsplit
    seg_qo: list[int] = []
    seg_kv: list[int] = []
    seg_req: list[int] = []
    for req, (q, kv) in enumerate(zip(qo_lens, kv_lens)):
        context = kv - q
        for start in range(0, q, _PREFILL_SEG_LEN):
            end = min(start + _PREFILL_SEG_LEN, q)
            seg_qo.append(end - start)
            seg_kv.append(context + end)
            seg_req.append(req)
    # Several index heads with <=32-query segments would select a pack-GQA
    # split variant; keep those shapes on the unsplit plan.
    if not seg_qo or (num_heads > 1 and max(seg_qo) <= 32):
        return unsplit
    splits = min(max_splits, math.ceil(num_sms / (len(seg_qo) * num_heads)))
    # The fmha planner keeps >= 4 KV iterations (256-key tiles) per split.
    splits = min(splits, max(1, math.ceil(max(seg_kv) / 256) // 4))
    splits = min(
        splits, max(1, _PREFILL_SPLIT_MAX_WS_ROWS // (sum(seg_qo) * num_heads))
    )
    # The planner cuts each segment's KV iterations into min(splits, iters // 4)
    # pieces of ceil(iters / pieces); keep every piece non-empty.
    iters = (np.asarray(seg_kv, dtype=np.int64) + 255) // 256
    while splits > 1:
        pieces = np.minimum(splits, np.maximum(1, iters // 4))
        step = (iters + pieces - 1) // pieces
        if np.array_equal((iters + step - 1) // step, pieces):
            break
        splits -= 1
    if splits <= 1:
        return unsplit
    return seg_qo, seg_kv, seg_req, splits



def _prefill_score_variants(
    num_heads: int, pack_factor_fn: Callable[[int, int, int], int]
) -> list[tuple]:
    """fmha_sm100 OnlyScore variants ``(qo_tile, single_wg, sparse_mode,
    page_size, split_kv, pack_factor)`` the prefill score can select when the
    split is enabled (unsplit and split plans).

    Mirrors fmha_sm100's dispatch: qo_tile 256 above 128 (packed) queries,
    else 128; single_wg at <= 64; pack-GQA (``pack_factor_fn`` is fmha_sm100's
    ``_compute_pack_factor``) only at <= 32 queries, on the 128 tile. Split
    plans use 128-query segments with pack factor 1 (see
    ``_plan_prefill_segments``)."""
    variants = {
        (256, False, 2, PAGE_SIZE, False, 1),
        (128, False, 2, PAGE_SIZE, False, 1),
        (128, True, 2, PAGE_SIZE, False, 1),
        (128, False, 2, PAGE_SIZE, True, 1),
        (128, True, 2, PAGE_SIZE, True, 1),
    }
    for q in range(1, 33):
        pack = pack_factor_fn(q, num_heads, 1)
        if pack > 1:
            variants.add((128, q * pack <= 64, 2, PAGE_SIZE, False, pack))
    return sorted(variants)

