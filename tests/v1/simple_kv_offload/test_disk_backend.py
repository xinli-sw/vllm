# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the DiskBackend data path.

CUDA pieces (pinning, batch memcpy, streams, events, set_device) are replaced
by synchronous host fakes, so these run without a GPU and exercise the real
threading, coalescing, slot management, file IO and in-order completion.
"""

from __future__ import annotations

import os
import random
import threading
import time

import pytest
import torch

from vllm.v1.simple_kv_offload import disk_backend as db


class _FakeEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class _FakeStream:
    def wait_event(self, event) -> None:
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(db, "pin_tensor", lambda t: None)
    monkeypatch.setattr(db, "build_params", lambda src, dst, stream, **kw: (src, dst))

    def _copy(src_ids, dst_ids, params):
        src, dst = params
        for name in src:
            for s, d in zip(src_ids, dst_ids):
                dst[name][d].copy_(src[name][s])

    monkeypatch.setattr(db, "copy_blocks", _copy)
    monkeypatch.setattr(db, "_record_event", lambda stream: _FakeEvent())
    monkeypatch.setattr(db.current_platform, "set_device", lambda d: None)


def _make_backend(
    tmp_path,
    *,
    num_gpu_blocks=64,
    num_disk_slots=128,
    tensor_bytes=(4096, 8192, 4096),
    **kw,
):
    gpu = {
        f"t{i}": torch.zeros((num_gpu_blocks, b), dtype=torch.int8)
        for i, b in enumerate(tensor_bytes)
    }
    backend = db.DiskBackend()
    backend.init(
        gpu,
        torch.device("cpu"),
        _FakeStream(),
        _FakeStream(),
        str(tmp_path / "kv.bin"),
        num_disk_slots,
        sum(tensor_bytes),
        use_page_cache=True,
        **kw,
    )
    return backend, gpu


def _fill(gpu, blocks, seed):
    g = torch.Generator().manual_seed(seed)
    for t in gpu.values():
        for b in blocks:
            t[b] = torch.randint(
                -128, 127, (t.shape[1],), generator=g, dtype=torch.int8
            )


def _wait(events, n, timeout=20.0):
    deadline = time.monotonic() + timeout
    while len(events) < n:
        assert time.monotonic() < deadline, f"timed out: {len(events)}/{n} events"
        time.sleep(0.005)


@pytest.mark.parametrize("coalesce", [1, 4, 16])
@pytest.mark.parametrize("threads", [1, 8])
def test_round_trip(tmp_path, fake_cuda, coalesce, threads):
    backend, gpu = _make_backend(
        tmp_path,
        max_coalesce_blocks=coalesce,
        num_io_threads=threads,
        num_buffer_slots=8,
    )
    try:
        src = list(range(0, 32))
        # Mix contiguous and scattered disk slots.
        slots = list(range(10, 20)) + [3, 50, 51, 52, 7, 90] + list(range(60, 76))
        _fill(gpu, src, seed=1)
        expected = {n: t[src].clone() for n, t in gpu.items()}

        store_events: list = []
        backend.launch_copy(src, slots, True, 0, store_events)
        _wait(store_events, 1)

        dst = list(range(32, 64))
        load_events: list = []
        backend.launch_copy(slots, dst, False, 0, load_events)
        _wait(load_events, 1)
        for n, t in gpu.items():
            assert torch.equal(t[dst], expected[n]), n
    finally:
        backend.shutdown()


def test_events_in_order_under_random_latency(tmp_path, fake_cuda, monkeypatch):
    """With ordered_load_completion, later batches may finish IO first but are
    reported after earlier ones."""
    real_preadv = os.preadv
    rng = random.Random(0)
    lock = threading.Lock()

    def slow_preadv(fd, bufs, off):
        with lock:
            delay = rng.random() * 0.01
        time.sleep(delay)
        return real_preadv(fd, bufs, off)

    backend, gpu = _make_backend(
        tmp_path,
        num_io_threads=8,
        num_buffer_slots=16,
        max_coalesce_blocks=2,
        ordered_load_completion=True,
    )
    try:
        src = list(range(64))
        _fill(gpu, src, seed=2)
        expected = {n: t.clone() for n, t in gpu.items()}
        st: list = []
        backend.launch_copy(src, list(range(64)), True, 0, st)
        _wait(st, 1)
        for t in gpu.values():
            t.zero_()

        monkeypatch.setattr(os, "preadv", slow_preadv)
        load_events: list = []
        # 20 batches of varying size; first ones are largest.
        sizes = [12, 1, 6, 1, 3, 9, 2, 1, 4, 1, 5, 2, 3, 1, 1, 2, 3, 4, 2, 1]
        start = 0
        for idx, n in enumerate(sizes):
            blocks = list(range(start, start + n))
            backend.launch_copy(blocks, blocks, False, idx, load_events)
            start += n
        _wait(load_events, len(sizes))
        assert [i for i, _ in load_events] == list(range(len(sizes)))
        for n, t in gpu.items():
            assert torch.equal(t[:start], expected[n][:start])
    finally:
        backend.shutdown()


def test_store_events_in_order(tmp_path, fake_cuda, monkeypatch):
    real_pwritev = os.pwritev
    rng = random.Random(1)
    lock = threading.Lock()

    def slow_pwritev(fd, bufs, off):
        with lock:
            delay = rng.random() * 0.01
        time.sleep(delay)
        return real_pwritev(fd, bufs, off)

    monkeypatch.setattr(os, "pwritev", slow_pwritev)
    backend, gpu = _make_backend(tmp_path, num_io_threads=8, num_buffer_slots=4)
    try:
        events: list = []
        for idx in range(15):
            n = (idx % 4) + 1
            backend.launch_copy(
                list(range(n)), list(range(idx * 4, idx * 4 + n)), True, idx, events
            )
        _wait(events, 15)
        assert [i for i, _ in events] == list(range(15))
    finally:
        backend.shutdown()


def test_coalescing(tmp_path, fake_cuda, monkeypatch):
    calls: list[tuple[int, int, int]] = []
    real_preadv = os.preadv

    def spy(fd, bufs, off):
        calls.append((off, len(bufs), sum(memoryview(b).nbytes for b in bufs)))
        return real_preadv(fd, bufs, off)

    tensor_bytes = (4096, 8192)
    bpb = sum(tensor_bytes)
    backend, gpu = _make_backend(
        tmp_path,
        tensor_bytes=tensor_bytes,
        max_coalesce_blocks=4,
        num_buffer_slots=8,
    )
    monkeypatch.setattr(os, "preadv", spy)
    try:
        # Unsorted input: 0..9 contiguous -> runs of 4,4,2; 20 alone; 30,31.
        slots = [5, 1, 0, 2, 3, 4, 31, 6, 7, 8, 9, 20, 30]
        ev: list = []
        backend.launch_copy(slots, list(range(len(slots))), False, 0, ev)
        _wait(ev, 1)
        got = sorted((off // bpb, nbytes // bpb) for off, _, nbytes in calls)
        assert got == [(0, 4), (4, 4), (8, 2), (20, 1), (30, 2)]
        # One iovec per tensor per block.
        assert all(n_iov == (nb // bpb) * 2 for _, n_iov, nb in calls)
    finally:
        backend.shutdown()


def test_coalesce_capped_by_buffer_slots(tmp_path, fake_cuda):
    backend, _ = _make_backend(tmp_path, max_coalesce_blocks=64, num_buffer_slots=6)
    try:
        assert backend._max_run_blocks == 3
    finally:
        backend.shutdown()


def test_split_runs():
    pairs = [(5, 50), (1, 10), (2, 20), (3, 30), (9, 90), (3, 31)]
    runs = db._split_runs(pairs, max_run=2)
    flat = [[s for s, _ in r] for r in runs]
    assert flat == [[1, 2], [3], [3], [5], [9]]


def test_rw_full_handles_short_transfers_and_iov_max(monkeypatch):
    monkeypatch.setattr(db, "_IOV_MAX", 3)
    data = bytes(range(256)) * 4  # 1024 bytes
    bufs = [bytearray(100) for _ in range(10)] + [bytearray(24)]
    views = [memoryview(b) for b in bufs]
    calls = []

    def fake_preadv(fd, chunk, off):
        assert len(chunk) <= 3
        calls.append(off)
        # Transfer at most 70 bytes per call to force partial iovecs.
        budget = 70
        done = 0
        for v in chunk:
            n = min(len(v), budget - done)
            v[:n] = data[off + done : off + done + n]
            done += n
            if done == budget:
                break
        return done

    db._rw_full(fake_preadv, -1, views, 0, "read")
    assert b"".join(bytes(b) for b in bufs) == data
    assert len(calls) > 11


def test_rw_full_raises_on_eof():
    with pytest.raises(OSError, match="Short read"):
        db._rw_full(lambda fd, c, o: 0, -1, [memoryview(bytearray(8))], 0, "read")


def test_io_pool_load_priority_and_store_cap():
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []
    lock = threading.Lock()
    store_running = 0
    max_store_running = 0

    def blocker():
        started.set()
        release.wait()

    def load_job(i):
        def f():
            with lock:
                order.append(f"L{i}")

        return f

    def store_job(i):
        def f():
            nonlocal store_running, max_store_running
            with lock:
                store_running += 1
                max_store_running = max(max_store_running, store_running)
                order.append(f"S{i}")
            time.sleep(0.01)
            with lock:
                store_running -= 1

        return f

    pool = db._IOPool(1, max_store_io=1, device=None)
    pool.submit(blocker, is_store=False)
    started.wait()
    for i in range(3):
        pool.submit(store_job(i), is_store=True)
    for i in range(3):
        pool.submit(load_job(i), is_store=False)
    release.set()
    assert pool.shutdown(timeout=5.0)
    assert order == ["L0", "L1", "L2", "S0", "S1", "S2"]

    # Store cap: 4 threads, at most 2 stores concurrently.
    order.clear()
    max_store_running = 0
    pool = db._IOPool(4, max_store_io=2, device=None)
    for i in range(12):
        pool.submit(store_job(i), is_store=True)
    assert pool.shutdown(timeout=5.0)
    assert len(order) == 12
    assert max_store_running <= 2


def test_slot_pool_deferred_release():
    class _Ev:
        def __init__(self):
            self.done = False
            self.synced = False

        def query(self):
            return self.done

        def synchronize(self):
            self.synced = True
            self.done = True

    pool = db._SlotPool(4)
    a = pool.acquire(3)
    ev = _Ev()
    pool.release_after(ev, a)
    assert pool.try_acquire(2) is None
    b = pool.acquire(1)
    assert len(b) == 1
    # Blocking acquire syncs the oldest pending DMA to free slots.
    c = pool.acquire(3)
    assert ev.synced and sorted(c) == sorted(a)


def test_shutdown_drains_and_unlinks(tmp_path, fake_cuda):
    backend, gpu = _make_backend(tmp_path, num_io_threads=4, num_buffer_slots=4)
    path = tmp_path / "kv.bin"
    assert path.exists()
    events: list = []
    for idx in range(5):
        backend.launch_copy(
            list(range(8)), list(range(idx * 8, idx * 8 + 8)), True, idx, events
        )
    assert backend.flush(timeout=20.0)
    backend.shutdown()
    assert [i for i, _ in events] == list(range(5))
    assert not path.exists()
    assert backend._fd == -1
    backend.shutdown()  # idempotent


def test_io_error_fails_batch_and_surfaces(tmp_path, fake_cuda, monkeypatch):
    def broken(fd, bufs, off):
        raise OSError(5, "EIO")

    backend, gpu = _make_backend(tmp_path, num_io_threads=2, num_buffer_slots=4)
    try:
        monkeypatch.setattr(os, "preadv", broken)
        events: list = []
        backend.launch_copy([0, 1], [0, 1], False, 0, events)
        assert not backend.flush(timeout=5.0)
        assert events == []
        with pytest.raises(RuntimeError, match="IO failed"):
            backend.launch_copy([0], [0], False, 1, events)
    finally:
        backend.shutdown()


def test_empty_batch_completes(tmp_path, fake_cuda):
    backend, _ = _make_backend(tmp_path)
    try:
        events: list = []
        backend.launch_copy([], [], False, 0, events)
        backend.launch_copy([], [], True, 0, events)
        _wait(events, 2)
    finally:
        backend.shutdown()


def test_loads_complete_out_of_order(tmp_path, fake_cuda, monkeypatch):
    """Default: a small load is reported while an earlier big one is stuck."""
    real_preadv = os.preadv
    gate = threading.Event()
    slow_off = 0  # disk slot 0 belongs to the first (big) batch

    def gated_preadv(fd, bufs, off):
        if off == slow_off:
            gate.wait(10)
        return real_preadv(fd, bufs, off)

    backend, gpu = _make_backend(
        tmp_path, num_io_threads=4, num_buffer_slots=8, max_coalesce_blocks=1
    )
    try:
        _fill(gpu, range(8), seed=3)
        expected = {n: t[:8].clone() for n, t in gpu.items()}
        st: list = []
        backend.launch_copy(list(range(8)), list(range(8)), True, 0, st)
        _wait(st, 1)
        for t in gpu.values():
            t.zero_()
        monkeypatch.setattr(os, "preadv", gated_preadv)
        events: list = []
        backend.launch_copy([0, 1, 2], [0, 1, 2], False, 0, events)
        backend.launch_copy([5], [5], False, 1, events)
        _wait(events, 1)
        assert [i for i, _ in events] == [1]
        gate.set()
        _wait(events, 2)
        assert [i for i, _ in events] == [1, 0]
        for n, t in gpu.items():
            for b in (0, 1, 2, 5):
                assert torch.equal(t[b], expected[n][b])
    finally:
        gate.set()
        backend.shutdown()
