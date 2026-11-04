# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission control for disk-tier loads of SimpleCPUOffloadConnector.

A disk hit accepted by the scheduler allocates its GPU blocks right away and
then waits (WAITING_FOR_REMOTE_KVS) until the disk read lands. When hits
arrive faster than the drives can serve them, accepted loads pile up: they
hold GPU KV while waiting, the running batch starves for blocks, and
throughput collapses even though the disk is fully busy.

``DiskLoadAdmission`` bounds the load backlog. Every accepted load counts
against an in-flight budget (blocks accepted but not yet completed, whether
or not they were sent to the worker). A new disk hit that would exceed the
budget is deferred: the connector reports "not ready" so the request takes
no GPU blocks and is asked again on a later step. Deferral is FIFO: once a
request is deferred in a step, later disk hits in that step are deferred too,
so a large request cannot be starved by smaller ones behind it. When nothing
is in flight, the head request is always admitted, however large.

The budget is fixed (``disk_max_inflight_load_bytes`` > 0) or adapts to the
measured load rate: ``peak_rate * disk_load_admission_ms``, floored at
``disk_min_inflight_load_bytes`` (default: a small fraction of the GPU KV
pool). The rate is sampled over busy windows and held at its recent peak:
while the budget binds, the observed rate is at most budget / latency, so a
plain average would shrink the budget step after step whenever per-load
latency (a couple of engine steps) exceeds the horizon. Optional fallbacks
turn a deferred hit into
recompute after ``disk_load_defer_timeout_ms``, or when the estimated load
time exceeds the estimated recompute time at
``disk_recompute_tokens_per_s``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum, auto

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_DISK_LOAD_ADMISSION_MS = 2000.0
# Auto floor of the budget, as a fraction of the GPU KV pool.
DEFAULT_DISK_MIN_INFLIGHT_GPU_FRACTION = 0.05
# Cap before the first bandwidth sample, as a multiple of the floor.
_INITIAL_CAP_FLOOR_MULT = 4
# Minimum busy-window length for one bandwidth sample.
_BW_SAMPLE_S = 0.25
_BW_EMA_ALPHA = 0.3
# Half-life of the peak-held rate once samples fall below it.
_PEAK_HALF_LIFE_S = 120.0
_LOG_INTERVAL_S = 30.0


class Admission(Enum):
    ADMIT = auto()
    DEFER = auto()
    RECOMPUTE = auto()


class DiskLoadAdmission:
    """Scheduler-side budget for disk loads accepted but not yet completed.

    All sizes are in offload blocks (one block = all layers of one scheduler
    block on this rank).
    """

    def __init__(
        self,
        bytes_per_block: int,
        tokens_per_block: int,
        max_inflight_bytes: int = 0,
        admission_ms: float = DEFAULT_DISK_LOAD_ADMISSION_MS,
        min_inflight_blocks: int = 1,
        defer_timeout_ms: float = 0.0,
        recompute_tokens_per_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.bytes_per_block = max(1, int(bytes_per_block))
        self.tokens_per_block = max(1, int(tokens_per_block))
        # <0 disables admission control, 0 adapts, >0 is a fixed budget.
        self.enabled = max_inflight_bytes >= 0
        self.fixed_cap_blocks = (
            max(1, max_inflight_bytes // self.bytes_per_block)
            if max_inflight_bytes > 0
            else None
        )
        self.admission_s = max(0.0, admission_ms) / 1000.0
        self.min_cap_blocks = max(1, int(min_inflight_blocks))
        self.defer_timeout_s = max(0.0, defer_timeout_ms) / 1000.0
        self.recompute_tokens_per_s = max(0.0, recompute_tokens_per_s)
        self._clock = clock

        self.inflight_blocks = 0
        self._inflight_reqs = 0
        # Completed blocks per second: EMA over busy windows, and its peak
        # (slowly decaying) which sizes the adaptive budget.
        self.rate_blocks_per_s: float | None = None
        self.peak_rate_blocks_per_s: float | None = None
        self._peak_t = 0.0
        self._win_start: float | None = None
        self._win_blocks = 0
        # req_id -> first time it was deferred.
        self._deferred_since: dict[str, float] = {}
        # Set once a request is deferred in the current step (FIFO gate).
        self._gate_closed = False

        # Cumulative counters (logged periodically).
        self.num_deferred = 0
        self.num_admitted = 0
        self.num_recompute = 0
        self._last_log = clock()

    # ------------------------------------------------------------------
    def cap_blocks(self) -> int:
        if self.fixed_cap_blocks is not None:
            return self.fixed_cap_blocks
        if self.peak_rate_blocks_per_s is None:
            return self.min_cap_blocks * _INITIAL_CAP_FLOOR_MULT
        return max(
            self.min_cap_blocks, int(self.peak_rate_blocks_per_s * self.admission_s)
        )

    def check(self, req_id: str, num_blocks: int, num_tokens: int) -> Admission:
        """Decide whether a disk hit of ``num_blocks`` may take GPU blocks now.

        Does not reserve anything: the budget is charged by ``on_admitted``
        once the scheduler actually allocates for the load.
        """
        if not self.enabled or num_blocks <= 0:
            return Admission.ADMIT
        now = self._clock()
        since = self._deferred_since.get(req_id)

        if self._should_recompute(since, now, num_blocks, num_tokens):
            self._deferred_since.pop(req_id, None)
            self.num_recompute += 1
            return Admission.RECOMPUTE

        fits = self.inflight_blocks + num_blocks <= self.cap_blocks()
        if self.inflight_blocks == 0 or (fits and not self._gate_closed):
            return Admission.ADMIT

        if since is None:
            self._deferred_since[req_id] = now
        self._gate_closed = True
        self.num_deferred += 1
        return Admission.DEFER

    def _should_recompute(
        self, since: float | None, now: float, num_blocks: int, num_tokens: int
    ) -> bool:
        if (
            since is not None
            and self.defer_timeout_s > 0
            and now - since >= self.defer_timeout_s
        ):
            return True
        if self.recompute_tokens_per_s > 0 and self.rate_blocks_per_s:
            # Everything already accepted drains first.
            est_load_s = (self.inflight_blocks + num_blocks) / self.rate_blocks_per_s
            est_recompute_s = num_tokens / self.recompute_tokens_per_s
            if est_load_s > est_recompute_s:
                return True
        return False

    # ------------------------------------------------------------------
    def on_admitted(self, req_id: str, num_blocks: int) -> None:
        self._deferred_since.pop(req_id, None)
        if num_blocks <= 0:
            return
        if self.inflight_blocks == 0:
            self._win_start = self._clock()
            self._win_blocks = 0
        self.inflight_blocks += num_blocks
        self._inflight_reqs += 1
        self.num_admitted += 1

    def on_released(self, num_blocks: int, completed: bool) -> None:
        """A load accepted via ``on_admitted`` left the backlog.

        ``completed`` is True when the worker finished the read (counts toward
        the bandwidth estimate), False when it was dropped before being sent.
        """
        if num_blocks <= 0:
            return
        self.inflight_blocks = max(0, self.inflight_blocks - num_blocks)
        self._inflight_reqs = max(0, self._inflight_reqs - 1)
        now = self._clock()
        if completed and self._win_start is not None:
            self._win_blocks += num_blocks
            elapsed = now - self._win_start
            if elapsed >= _BW_SAMPLE_S:
                self._add_rate_sample(self._win_blocks / elapsed, now)
                self._win_start = now
                self._win_blocks = 0
        if self.inflight_blocks == 0:
            # Idle time must not dilute the next sample.
            self._win_start = None
            self._win_blocks = 0

    def _add_rate_sample(self, sample: float, now: float) -> None:
        rate = self.rate_blocks_per_s
        rate = sample if rate is None else rate + _BW_EMA_ALPHA * (sample - rate)
        self.rate_blocks_per_s = rate
        peak = self.peak_rate_blocks_per_s
        if peak is not None:
            peak *= 0.5 ** ((now - self._peak_t) / _PEAK_HALF_LIFE_S)
        self.peak_rate_blocks_per_s = rate if peak is None else max(peak, rate)
        self._peak_t = now

    def on_request_gone(self, req_id: str) -> None:
        self._deferred_since.pop(req_id, None)

    def end_step(self) -> None:
        self._gate_closed = False
        if not self.enabled:
            return
        now = self._clock()
        if now - self._last_log >= _LOG_INTERVAL_S:
            self._last_log = now
            rate = self.rate_blocks_per_s
            logger.info(
                "SimpleCPUOffload disk admission: cap=%d blocks (%.2f GB), "
                "inflight=%d blocks/%d reqs, waiting=%d, rate=%s GB/s "
                "(peak %s), "
                "admitted=%d deferred_steps=%d recompute=%d",
                self.cap_blocks(),
                self.cap_blocks() * self.bytes_per_block / 1e9,
                self.inflight_blocks,
                self._inflight_reqs,
                len(self._deferred_since),
                self._gbps(rate),
                self._gbps(self.peak_rate_blocks_per_s),
                self.num_admitted,
                self.num_deferred,
                self.num_recompute,
            )

    def _gbps(self, rate: float | None) -> str:
        return "n/a" if rate is None else f"{rate * self.bytes_per_block / 1e9:.2f}"

    def reset_deferrals(self) -> None:
        self._deferred_since.clear()
        self._gate_closed = False
