# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host metadata helpers for MiniMax indexer prefill."""

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


