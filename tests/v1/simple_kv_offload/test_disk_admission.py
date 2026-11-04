# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-load admission control for SimpleCPUOffloadConnector."""

from __future__ import annotations

import pytest

from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    _alloc_and_register,
    _make_kv_cache_config,
    _make_vllm_config,
    make_request,
    make_scheduler_output,
    simulate_load_completion,
    simulate_store_completion,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.admission import Admission, DiskLoadAdmission
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

pytestmark = pytest.mark.skip_global_cleanup

BPB = 1000  # bytes per block for the pure admission tests


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _adm(clock: FakeClock, **kw) -> DiskLoadAdmission:
    kw.setdefault("min_inflight_blocks", 10)
    return DiskLoadAdmission(
        bytes_per_block=BPB, tokens_per_block=16, clock=clock, **kw
    )


# ---------------------------------------------------------------------------
# DiskLoadAdmission
# ---------------------------------------------------------------------------
def test_fixed_cap_defers_and_readmits_fifo() -> None:
    clock = FakeClock()
    adm = _adm(clock, max_inflight_bytes=10 * BPB)
    assert adm.cap_blocks() == 10

    assert adm.check("a", 6, 96) is Admission.ADMIT
    adm.on_admitted("a", 6)
    # b does not fit -> deferred; c fits but must not jump the queue.
    assert adm.check("b", 6, 96) is Admission.DEFER
    assert adm.check("c", 2, 32) is Admission.DEFER
    adm.end_step()

    adm.on_released(6, completed=True)
    assert adm.inflight_blocks == 0
    assert adm.check("b", 6, 96) is Admission.ADMIT
    adm.on_admitted("b", 6)
    assert adm.check("c", 2, 32) is Admission.ADMIT
    adm.on_admitted("c", 2)
    assert adm.inflight_blocks == 8
    assert not adm._deferred_since


def test_oversized_request_admitted_when_idle() -> None:
    clock = FakeClock()
    adm = _adm(clock, max_inflight_bytes=4 * BPB)
    assert adm.check("big", 50, 800) is Admission.ADMIT
    adm.on_admitted("big", 50)
    assert adm.check("small", 1, 16) is Admission.DEFER
    adm.end_step()
    adm.on_released(50, completed=True)
    assert adm.check("small", 1, 16) is Admission.ADMIT


def test_disabled_always_admits() -> None:
    adm = _adm(FakeClock(), max_inflight_bytes=-1)
    adm.on_admitted("a", 10**6)
    assert adm.check("b", 10**6, 1) is Admission.ADMIT


def test_adaptive_cap_tracks_measured_rate() -> None:
    clock = FakeClock()
    adm = _adm(clock, admission_ms=1000)
    # Before any sample: 4x the floor.
    assert adm.cap_blocks() == 40
    # Drain 100 blocks/s for a few seconds while busy.
    adm.on_admitted("x", 1000)
    for _ in range(20):
        clock.t += 0.5
        adm.on_released(50, completed=True)
    assert adm.rate_blocks_per_s == pytest.approx(100.0)
    assert adm.cap_blocks() == 100  # 100 blocks/s * 1 s
    # Never below the floor.
    adm.peak_rate_blocks_per_s = 1.0
    assert adm.cap_blocks() == 10


def test_adaptive_cap_holds_peak_while_throttled() -> None:
    """While the cap binds, the observed rate is cap / latency; the budget
    must not shrink with it."""
    clock = FakeClock()
    adm = _adm(clock, admission_ms=1000)
    adm.on_admitted("x", 10**6)
    for _ in range(10):
        clock.t += 1.0
        adm.on_released(100, completed=True)
    assert adm.cap_blocks() == 100
    for _ in range(5):
        clock.t += 1.0
        adm.on_released(20, completed=True)
    assert adm.rate_blocks_per_s < 50
    assert adm.cap_blocks() >= 95


def test_idle_time_does_not_dilute_rate() -> None:
    clock = FakeClock()
    adm = _adm(clock)
    adm.on_admitted("a", 100)
    clock.t += 1.0
    adm.on_released(100, completed=True)
    assert adm.rate_blocks_per_s == pytest.approx(100.0)
    clock.t += 50.0  # idle
    adm.on_admitted("b", 100)
    clock.t += 1.0
    adm.on_released(100, completed=True)
    assert adm.rate_blocks_per_s == pytest.approx(100.0)


def test_dropped_loads_do_not_count_toward_rate() -> None:
    clock = FakeClock()
    adm = _adm(clock)
    adm.on_admitted("a", 100)
    clock.t += 1.0
    adm.on_released(100, completed=False)
    assert adm.rate_blocks_per_s is None
    assert adm.inflight_blocks == 0


def test_defer_timeout_falls_back_to_recompute() -> None:
    clock = FakeClock()
    adm = _adm(clock, max_inflight_bytes=10 * BPB, defer_timeout_ms=500)
    adm.on_admitted("a", 10)
    assert adm.check("b", 5, 80) is Admission.DEFER
    adm.end_step()
    clock.t += 0.4
    assert adm.check("b", 5, 80) is Admission.DEFER
    adm.end_step()
    clock.t += 0.2
    assert adm.check("b", 5, 80) is Admission.RECOMPUTE
    assert "b" not in adm._deferred_since


def test_recompute_when_load_slower_than_compute() -> None:
    clock = FakeClock()
    adm = _adm(clock, max_inflight_bytes=1000 * BPB, recompute_tokens_per_s=1600)
    adm.rate_blocks_per_s = 10.0
    adm.on_admitted("a", 90)
    # (90 + 10) blocks at 10 blocks/s = 10 s > 160 tokens / 1600 tok/s.
    assert adm.check("b", 10, 160) is Admission.RECOMPUTE
    # A long hit is still cheaper to load: 10 s < 32000 / 1600 = 20 s.
    assert adm.check("c", 10, 32000) is Admission.ADMIT


# ---------------------------------------------------------------------------
# SimpleCPUOffloadScheduler (disk mode) integration
# ---------------------------------------------------------------------------
def _disk_scheduler(max_inflight_blocks: int, num_gpu_blocks: int = 64, **kw):
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    sched = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=0,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        lazy_offload=False,
        disk_capacity_bytes=_BYTES_PER_BLOCK * 32,
        disk_max_inflight_load_bytes=max_inflight_blocks * _BYTES_PER_BLOCK,
        **kw,
    )
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        num_hash_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
    )
    sched.bind_gpu_block_pool(pool)

    class _Fix:
        pass

    fix = _Fix()
    fix.scheduler, fix.gpu_block_pool = sched, pool
    return fix


def _store(fix, num_blocks: int) -> Request:
    sched = fix.scheduler
    req = make_request(num_blocks=num_blocks)
    kv_blocks = _alloc_and_register(fix, req, num_blocks)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    meta = sched.build_connector_meta(
        make_scheduler_output(
            {req.request_id: num_blocks * BLOCK_SIZE},
            new_reqs={req.request_id: kv_blocks.get_block_ids()},
        )
    )
    simulate_store_completion(sched, meta.store_event)
    sched.request_finished(req, block_ids=[])
    return req


def _clone(req: Request, rid: str) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )


def _accept(fix, req: Request, tokens: int) -> None:
    n = tokens // BLOCK_SIZE
    kv = KVCacheBlocks(blocks=(fix.gpu_block_pool.get_new_blocks(n),))
    fix.scheduler.update_state_after_alloc(req, kv, num_external_tokens=tokens)


def test_scheduler_defers_disk_hits_over_budget() -> None:
    fix = _disk_scheduler(max_inflight_blocks=4)
    sched = fix.scheduler
    src = _store(fix, 3)

    r1, r2 = _clone(src, "r1"), _clone(src, "r2")
    hit, is_async = sched.get_num_new_matched_tokens(r1, 0)
    assert (hit, is_async) == (3 * BLOCK_SIZE, True)
    _accept(fix, r1, hit)
    assert sched._load_admission.inflight_blocks == 3

    # 3 + 3 > 4: not ready, and nothing pinned for r2.
    assert sched.get_num_new_matched_tokens(r2, 0) == (None, False)
    assert "r2" not in sched._pending_cpu_hits

    # Load is issued, still deferred while it runs.
    meta = sched.build_connector_meta(make_scheduler_output({"r1": 1}))
    assert meta.load_event >= 0
    assert sched.get_num_new_matched_tokens(r2, 0) == (None, False)
    sched.build_connector_meta(make_scheduler_output({}))

    simulate_load_completion(sched, {"r1"})
    assert sched._load_admission.inflight_blocks == 0
    hit2, is_async2 = sched.get_num_new_matched_tokens(r2, 0)
    assert (hit2, is_async2) == (3 * BLOCK_SIZE, True)
    _accept(fix, r2, hit2)
    assert sched._load_admission.inflight_blocks == 3


def test_scheduler_releases_budget_for_unsent_finished_load() -> None:
    fix = _disk_scheduler(max_inflight_blocks=4)
    sched = fix.scheduler
    src = _store(fix, 3)
    r1 = _clone(src, "r1")
    hit, _ = sched.get_num_new_matched_tokens(r1, 0)
    _accept(fix, r1, hit)
    # Aborted before build_connector_meta sent the load.
    sched.request_finished(r1, block_ids=[])
    assert sched._load_admission.inflight_blocks == 0
    assert sched._load_admission.rate_blocks_per_s is None


def test_scheduler_defer_timeout_recomputes() -> None:
    fix = _disk_scheduler(max_inflight_blocks=4, disk_load_defer_timeout_ms=1)
    sched = fix.scheduler
    clock = FakeClock()
    sched._load_admission._clock = clock
    src = _store(fix, 3)
    r1, r2 = _clone(src, "r1"), _clone(src, "r2")
    hit, _ = sched.get_num_new_matched_tokens(r1, 0)
    _accept(fix, r1, hit)
    assert sched.get_num_new_matched_tokens(r2, 0) == (None, False)
    sched.build_connector_meta(make_scheduler_output({"r1": 1}))
    clock.t += 1.0
    assert sched.get_num_new_matched_tokens(r2, 0) == (0, False)


def test_cpu_backend_has_no_admission() -> None:
    vllm_config = _make_vllm_config()
    sched = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=_make_kv_cache_config(16),
        cpu_capacity_bytes=_BYTES_PER_BLOCK * 8,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        disk_max_inflight_load_bytes=1,
    )
    assert sched._load_admission is None


def test_auto_floor_is_fraction_of_gpu_pool() -> None:
    fix = _disk_scheduler(max_inflight_blocks=0, num_gpu_blocks=200)
    adm = fix.scheduler._load_admission
    assert adm.fixed_cap_blocks is None
    assert adm.min_cap_blocks == 10  # 5% of 200 GPU blocks
    assert adm.cap_blocks() == 40  # before any rate sample


def test_negative_cap_disables_admission() -> None:
    fix = _disk_scheduler(max_inflight_blocks=-1)
    assert fix.scheduler._load_admission is None
