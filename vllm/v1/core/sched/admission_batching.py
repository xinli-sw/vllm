# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouping of WAITING admissions under the active-request cap.

With a saturated ``max_num_active_seqs`` cap, each completion frees one slot
and the next step admits one waiting request. That request's prefill turns a
decode step into a mixed step that stalls every running decode. Admitting
waiting requests in groups lets one mixed step carry several prefills.

Opt-in via ``SchedulerConfig.admission_batch_size`` (``>= 2``), tuned by
``admission_batch_max_wait_steps`` and ``admission_batch_overshoot``.

Grouping engages only when 0 < free slots < ``batch_size``, at least
``batch_size`` requests are waiting, and the step would otherwise be a pure
decode step. In every other state admission is unchanged, and
``max_wait_steps=0`` without ``overshoot`` is stock admission.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig


class AdmissionBatcher:
    """Decides how far a scheduler step may admit WAITING requests.

    Stateful across steps: it remembers when the current hold began so a free
    slot is held empty for at most ``max_wait_steps`` steps.
    """

    def __init__(
        self, batch_size: int = 2, max_wait_steps: int = 32, overshoot: bool = True
    ) -> None:
        if batch_size < 2:
            raise ValueError(f"admission batch_size must be >= 2: {batch_size}")
        if max_wait_steps < 0:
            raise ValueError(f"admission max_wait_steps must be >= 0: {max_wait_steps}")
        self.batch_size = batch_size
        self.max_wait_steps = max_wait_steps
        self.overshoot = overshoot
        # Scheduler step at which the pending hold began; None when none pends.
        self._hold_start_step: int | None = None

    @classmethod
    def from_scheduler_config(
        cls, scheduler_config: "SchedulerConfig"
    ) -> "AdmissionBatcher | None":
        """Build a batcher if admission batching is enabled, else None."""
        if (
            scheduler_config.admission_batch_size <= 1
            or scheduler_config.max_num_active_seqs is None
        ):
            return None
        return cls(
            batch_size=scheduler_config.admission_batch_size,
            max_wait_steps=scheduler_config.admission_batch_max_wait_steps,
            overshoot=scheduler_config.admission_batch_overshoot,
        )

    def __repr__(self) -> str:
        return (
            f"AdmissionBatcher(batch_size={self.batch_size}, "
            f"max_wait_steps={self.max_wait_steps}, overshoot={self.overshoot})"
        )

    def admission_limit(
        self,
        *,
        step: int,
        num_active: int,
        max_active: int,
        max_running: int,
        num_waiting: int,
        can_group: bool,
        stalls_decodes: bool,
    ) -> int:
        """Active-request count up to which this step may admit waiting requests.

        Returns ``max_active`` for stock admission, ``num_active`` to admit
        nothing this step, or a value above ``max_active`` to overshoot.

        Args:
            step: Monotonic scheduler step counter.
            num_active: Requests currently occupying an active slot.
            max_active: The active-request cap (``max_num_active_seqs``).
            max_running: Model-runner slot count (``max_num_seqs``); overshoot
                never exceeds it.
            num_waiting: Requests in the waiting queue.
            can_group: Whether the waiting queue holds only fresh requests
                (none holding KV blocks or deferred), so grouping is safe.
            stalls_decodes: Whether this step schedules decodes and no prefill
                chunk, i.e. whether an admission would turn it into a mixed
                step.

        """
        num_free = max_active - num_active
        if (
            not can_group
            or num_free <= 0
            or num_free >= self.batch_size
            or num_waiting < self.batch_size
        ):
            self._hold_start_step = None
            return max_active
        if not stalls_decodes:
            # Nothing to stall, or a prefill already rides this step.
            return max_active
        if self._hold_start_step is None:
            self._hold_start_step = step
        if step - self._hold_start_step < self.max_wait_steps:
            return num_active
        # Expired: stays released until the slot or queue state changes.
        if self.overshoot:
            return min(num_active + self.batch_size, max_running)
        return max_active
