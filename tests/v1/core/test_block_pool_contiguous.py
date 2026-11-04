# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool.get_new_blocks_contiguous / select_contiguous_blocks."""

import pytest

from vllm.v1.core.block_pool import BlockPool, select_contiguous_blocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def _blocks(ids):
    return [KVCacheBlock(block_id=i) for i in ids]


def test_select_takes_whole_runs_oldest_first():
    # Eviction order: 7 is oldest; runs {7,8,9}, {2,3}, {20}, {11,12,13,14}.
    cands = _blocks([7, 20, 2, 12, 8, 3, 13, 9, 11, 14])
    chosen = [b.block_id for b in select_contiguous_blocks(cands, 5)]
    # Runs rank by their oldest member: {7,8,9}=0, {20}=1, {2,3}=2, {11..14}=3,
    # so {7,8,9} and {20} are taken whole and {2,3} is cut to one block.
    assert chosen == [2, 7, 8, 9, 20]


def test_select_cuts_last_run_around_its_oldest_member():
    cands = _blocks([0, 15, 10, 11, 12, 13, 14, 16])
    # Runs: {0}=0, {10..16}=1 (oldest member 15 at position 1). The 3-block
    # slice of {10..16} must contain 15 and stay inside the run.
    chosen = [b.block_id for b in select_contiguous_blocks(cands, 4)]
    assert chosen == [0, 14, 15, 16]
    assert [b.block_id for b in select_contiguous_blocks(cands, 1)] == [0]


def test_select_all_candidates():
    cands = _blocks([5, 1, 3])
    assert [b.block_id for b in select_contiguous_blocks(cands, 3)] == [1, 3, 5]


def _pool(n):
    return BlockPool(num_gpu_blocks=n, enable_caching=True, hash_block_size=16)


def test_get_new_blocks_contiguous_bookkeeping():
    pool = _pool(33)  # block 0 is the null block
    blocks = pool.get_new_blocks(32)
    by_id = {b.block_id: b for b in blocks}
    ids = sorted(by_id)
    pool.free_blocks([by_id[i] for i in ids[0::2] + ids[1::2]])
    free_before = pool.get_num_free_blocks()
    oldest = next(pool.free_block_queue.iter_blocks_after(None))

    got = pool.get_new_blocks_contiguous(6, window=16)
    got_ids = [b.block_id for b in got]
    assert got_ids == list(range(got_ids[0], got_ids[0] + 6))
    assert oldest.block_id in got_ids
    assert all(b.ref_cnt == 1 for b in got)
    assert pool.get_num_free_blocks() == free_before - 6
    remaining = list(pool.free_block_queue.iter_blocks_after(None))
    assert len(remaining) == free_before - 6
    assert not set(got_ids) & {b.block_id for b in remaining}

    # Window 0 is plain LRU allocation.
    nxt = list(pool.free_block_queue.iter_blocks_after(None))[:3]
    assert pool.get_new_blocks_contiguous(3, window=0) == nxt

    with pytest.raises(ValueError):
        pool.get_new_blocks_contiguous(pool.get_num_free_blocks() + 1, window=4)
