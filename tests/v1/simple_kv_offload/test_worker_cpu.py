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


class _ManualBackend:
    """Records launches; the test decides when (and in which order) each
    launched copy completes."""

    def __init__(self) -> None:
        self.launches: list[tuple[list[int], list[int], int, list]] = []

    def launch_copy(
        self, src, dst, is_store, event_idx, events_list, wait_event=None
    ) -> None:
        self.launches.append((list(src), list(dst), event_idx, events_list))

    def complete(self, event_idx: int, done: bool = True) -> _Ev:
        for _, _, idx, lst in self.launches:
            if idx == event_idx:
                ev = _Ev(done=done)
                lst.append((idx, ev))
                return ev
        raise KeyError(event_idx)


def _split_meta() -> SimpleCPUOffloadMetadata:
    return SimpleCPUOffloadMetadata(
        load_event=3,
        load_cpu_blocks=[10, 11, 12, 20, 30, 31],
        load_gpu_blocks=[1, 2, 3, 4, 5, 6],
        load_event_splits=[(3, 3), (4, 1), (5, 2)],
        load_event_to_reqs={3: ["a"], 4: ["b"], 5: ["c"]},
    )


def test_disk_mode_launches_per_request_and_reports_out_of_order():
    backend = _ManualBackend()
    w = _worker(backend, disk_mode=True)
    w.bind_connector_metadata(_split_meta())
    w.start_load_kv()
    assert [(s, d, i) for s, d, i, _ in backend.launches] == [
        ([10, 11, 12], [1, 2, 3], 3),
        ([20], [4], 4),
        ([30, 31], [5, 6], 5),
    ]
    assert w.get_finished(set()) == (None, None)

    # Request "c" lands first, then "a"; "b"'s copy is still running.
    backend.complete(5)
    assert w.get_finished(set())[1] == {"c"}
    ev_b = backend.complete(4, done=False)
    backend.complete(3)
    assert w.get_finished(set())[1] == {"a"}
    assert w.get_finished(set())[1] is None
    ev_b.done = True
    assert w.get_finished(set())[1] == {"b"}
    assert not w._pending_load_event_indices
    assert not w._done_load_events
    assert not w._load_events


def test_cpu_mode_single_copy_covers_all_requests():
    backend = _ManualBackend()
    w = _worker(backend, disk_mode=False)
    w.bind_connector_metadata(_split_meta())
    w.start_load_kv()
    assert len(backend.launches) == 1
    src, dst, idx, _ = backend.launches[0]
    assert src == [10, 11, 12, 20, 30, 31] and dst == [1, 2, 3, 4, 5, 6]
    assert idx == 5
    backend.complete(5)
    assert w.get_finished(set())[1] == {"a", "b", "c"}
    assert not w._pending_load_event_indices and not w._done_load_events


def test_legacy_metadata_without_splits():
    backend = _ManualBackend()
    w = _worker(backend, disk_mode=True)
    meta = SimpleCPUOffloadMetadata(
        load_event=7,
        load_cpu_blocks=[1, 2],
        load_gpu_blocks=[3, 4],
        load_event_to_reqs={7: ["x", "y"]},
    )
    w.bind_connector_metadata(meta)
    w.start_load_kv()
    assert len(backend.launches) == 1
    backend.complete(7)
    assert w.get_finished(set())[1] == {"x", "y"}


def test_poll_keeps_events_appended_concurrently():
    """Entries appended after the poll snapshot must not be dropped."""
    backend = _ManualBackend()
    w = _worker(backend, disk_mode=True)
    w.bind_connector_metadata(_split_meta())
    w.start_load_kv()

    class _AppendingEv(_Ev):
        def query(self):
            # Simulate a backend thread appending mid-poll.
            backend.complete(4)
            return True

    w._load_events.append((3, _AppendingEv()))
    w._poll_load_events()
    assert [i for i, _ in w._load_events] == [4]
    assert w._done_load_events == {3}
