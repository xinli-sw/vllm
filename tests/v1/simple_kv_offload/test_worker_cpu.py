# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only worker tests: CUDA events/backends are replaced by fakes."""

from __future__ import annotations

from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


class _Ev:
    def __init__(self, done: bool = True) -> None:
        self.done = done
        self.synced = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.synced = True
        self.done = True


class _FlushingBackend:
    """Holds launched batches until flush(), like DiskBackend with slow IO."""

    def __init__(self) -> None:
        self.held: list[tuple[int, list]] = []
        self.flushed = 0
        self.launches: list[tuple[list[int], list[int], bool, int]] = []

    def launch_copy(
        self, src, dst, is_store, event_idx, events_list, wait_event=None
    ) -> None:
        self.launches.append((list(src), list(dst), is_store, event_idx))
        self.held.append((event_idx, events_list))

    def flush(self, timeout=None) -> bool:
        self.flushed += 1
        for idx, lst in self.held:
            lst.append((idx, _Ev(done=False)))
        self.held.clear()
        return True


def _worker(backend, disk_mode: bool = True) -> SimpleCPUOffloadWorker:
    w = SimpleCPUOffloadWorker(None, None, 0)
    w._backend = backend
    w.disk_mode = disk_mode
    return w


def test_preemption_flush_drains_backend_before_sync():
    backend = _FlushingBackend()
    w = _worker(backend)
    meta = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[1, 2],
        load_cpu_blocks=[5, 6],
        load_event_to_reqs={0: ["r0"]},
    )
    w.bind_connector_metadata(meta)
    w.start_load_kv()
    assert w._load_events == []  # still queued inside the backend

    w.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
    assert backend.flushed == 1
    assert w._load_events == []

    _, recving = w.get_finished(set())
    assert recving == {"r0"}


def test_no_flush_without_preemption():
    backend = _FlushingBackend()
    w = _worker(backend)
    w.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=False))
    assert backend.flushed == 0
