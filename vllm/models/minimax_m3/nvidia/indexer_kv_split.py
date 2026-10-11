# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-split planning for the MiniMax-M3 MSA prefill index score.

The stock builder plans the prefill ``fmha_sm100`` OnlyScore pass with
``num_kv_splits=1``. With one index head per rank (e.g. TP4), a prefill chunk
then runs one work item per 256-row query tile, each scanning the request's
whole cached index-K serially: a 512-token chunk keeps 2 of 148 SMs busy however
long the context is.

fmha_sm100 honours split-KV only for 128-row query tiles (``max_qo_len <= 128``),
so this slices every prefill request into pseudo-requests of at most 128 query
tokens (each with its own bottom-right causal offset and page list) and plans
them with ``num_kv_splits > 1``. Each work item writes the max logit of every
128-key block it visits at the block's absolute index; a block belongs to
exactly one split and a max needs no cross-split merge, so the scores (and
hence the top-k) are bit-identical to the stock plan.

On by default; ``VLLM_MINIMAX_M3_INDEXER_PREFILL_KV_SPLIT=0`` keeps the stock
plan.
"""

import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# fmha_sm100 runs split-KV only with 128-row query tiles.
SLICE_LEN = 128
MAX_SPLITS = 64
# Work items per SM the split count aims for.
CTAS_PER_SM = 2
# Steps with more prefill tokens than this keep the stock plan: they already
# fill the GPU with query tiles.
MAX_PREFILL_TOKENS = 16384
# The plan caches a bf16 split-KV O workspace of q * splits * heads * 128
# elements even in OnlyScore mode; bound it.
WORKSPACE_BYTES = 64 << 20


def slice_prefill(
    qo_lens: torch.Tensor, kv_lens: torch.Tensor, slice_len: int = SLICE_LEN
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut each request's queries into consecutive slices of <= slice_len.

    ``qo_lens``/``kv_lens`` are CPU int tensors per request (kv includes the
    request's own queries). Returns per-slice ``(request, qo_len, kv_len)``
    where a slice's kv_len stops at its last query, so its bottom-right causal
    offset ``kv_len - qo_len`` is the absolute position of its first query.
    Slices keep request order, so their query rows are the request's rows.
    """
    qo = qo_lens.to(torch.int64)
    kv = kv_lens.to(torch.int64)
    num_slices = (qo + slice_len - 1) // slice_len
    req = torch.repeat_interleave(torch.arange(qo.numel()), num_slices)
    first = torch.cumsum(num_slices, 0) - num_slices
    idx = torch.arange(req.numel()) - torch.repeat_interleave(first, num_slices)
    start = idx * slice_len
    slice_qo = torch.clamp(qo[req] - start, max=slice_len)
    slice_kv = kv[req] - qo[req] + start + slice_qo
    return req, slice_qo.to(torch.int32), slice_kv.to(torch.int32)


def page_gather_index(
    slice_req: torch.Tensor, slice_kv: torch.Tensor, page_size: int, row_stride: int
) -> torch.Tensor:
    """Flat block-table index of every slice's pages, concatenated in slice
    order: slice i lists its request's first cdiv(kv_len_i, page_size) pages
    (the plan's kv_page_indptr is built the same way)."""
    num_pages = (slice_kv.to(torch.int64) + page_size - 1) // page_size
    rows = torch.repeat_interleave(slice_req.to(torch.int64), num_pages)
    first = torch.cumsum(num_pages, 0) - num_pages
    cols = torch.arange(rows.numel()) - torch.repeat_interleave(first, num_pages)
    return rows * row_stride + cols


def choose_num_splits(
    num_slices: int, total_q: int, num_heads: int, num_sms: int
) -> int:
    """Splits so that slices x splits give ``CTAS_PER_SM`` work items per SM,
    capped by ``MAX_SPLITS`` and the plan's O workspace bound."""
    workspace_cap = WORKSPACE_BYTES // max(1, total_q * num_heads * 128 * 2)
    splits = math.ceil(CTAS_PER_SM * num_sms / max(1, num_slices))
    return max(1, min(splits, MAX_SPLITS, workspace_cap))


class PrefillKVSplitPlanner:
    """Builds the split-KV prefill score plan and its page table."""

    def __init__(
        self,
        num_heads: int,
        page_size: int,
        max_k_tiles: int,
        device: torch.device,
    ) -> None:
        self.num_heads = num_heads
        self.page_size = page_size
        self.max_k_tiles = max_k_tiles
        self.device = device
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        logger.info_once(
            "MiniMax-M3 MSA indexer: split-KV prefill score plan on "
            "(<= %d prefill tokens per step)",
            MAX_PREFILL_TOKENS,
        )

    def plan(
        self,
        side_qo: torch.Tensor,
        side_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[dict, torch.Tensor] | None:
        """Split plan + page table for the prefill side, or None for stock.

        ``side_qo``/``side_kv`` are CPU per-request lengths; ``block_table`` is
        the prefill side's [num_prefills, max_pages] device block table.
        """
        total_q = int(side_qo.sum())
        if total_q > MAX_PREFILL_TOKENS:
            return None
        # A slice with an odd page count also writes -inf at block index
        # cdiv(kv_len, page_size); keep that inside the score row.
        if int(side_kv.max()) > (self.max_k_tiles - 1) * self.page_size:
            return None
        from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

        slice_req, slice_qo, slice_kv = slice_prefill(side_qo, side_kv)
        splits = choose_num_splits(
            slice_req.numel(), total_q, self.num_heads, self.num_sms
        )
        if splits == 1:
            return None
        plan = _fmha_sm100_plan(
            slice_qo,
            slice_kv,
            self.num_heads,
            num_kv_heads=1,
            qo_offset=slice_kv - slice_qo,  # absolute position of first query
            page_size=self.page_size,
            output_maxscore=True,
            causal=True,
            num_kv_splits=splits,
        )
        plan["max_k_tiles"] = self.max_k_tiles
        table = block_table if block_table.is_contiguous() else block_table.contiguous()
        flat = page_gather_index(slice_req, slice_kv, self.page_size, table.shape[1])
        page_table = (
            table.view(-1)
            .index_select(0, flat.to(table.device, non_blocking=True))
            .to(torch.int32)
        )
        return plan, page_table
