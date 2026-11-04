# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Data path (high queue depth):

- Staging is *block-major*, matching the file: slot ``s`` holds a whole block
  contiguously (layer ``i`` at ``s * stride + offset[i]``), so a block is one
  contiguous buffer and the per-layer scatter/gather happens in the batched
  GPU DMA. ``stride`` is the block size rounded up to 4 KiB (O_DIRECT).
- Each ``launch_copy`` call is one *batch*. A per-direction coordinator thread
  splits a batch into *runs* of blocks whose disk slots are contiguous (up to
  ``max_coalesce_blocks`` and ``max_io_bytes``). A run gets contiguous staging
  slots, so it is a single one-buffer ``preadv``/``pwritev`` of up to
  ``max_io_bytes``.
- Runs are executed by a shared pool of IO threads (``preadv``/``pwritev``
  release the GIL), so many reads/writes are in flight at once.
- Each direction owns a ring of pinned, O_DIRECT-aligned staging slots. A load
  run reads into its slots and immediately issues the H2D DMA, so DMA of
  finished runs overlaps with further reads. Slots return to the pool once
  their DMA completes. A store run is DMA'd into staging by the coordinator and
  handed to an IO thread once that DMA has completed.
- Loads have priority over stores: IO threads always prefer queued load runs,
  and at most ``max_store_io`` threads may run stores at the same time, so
  stores can never occupy every thread. Stores use their own staging ring.
- Batches are issued in parallel. Store batches are *completed in order*: a
  store's event is appended to ``events_list`` only once it and every earlier
  store are done, preserving the worker's store high-water mark. Load batches
  (one per request) are reported as soon as each is done; the worker tracks
  loads as a set of completed events. ``ordered_load_completion=True``
  restores strict ordering for loads.

Same ``launch_copy`` interface as DmaCopyBackend so the worker can swap
backends without changing calling code.
"""

from __future__ import annotations

import contextlib
import functools
import mmap
import os
import queue
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_strided_params,
    copy_blocks,
    pin_tensor,
    unpin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096

# Defaults, overridable through kv_connector_extra_config.
# Measured on GB300 (4x NVMe RAID0): loads reach the fio ceiling with fewer
# than 4 IO threads per rank, so keep pinned staging modest.
DEFAULT_DISK_IO_THREADS = 8
# Staging per direction is sized in bytes unless disk_buffer_slots is given.
DEFAULT_DISK_STAGING_BYTES = 128 << 20
DEFAULT_DISK_MAX_COALESCE_BLOCKS = 32
DEFAULT_DISK_MAX_IO_BYTES = 16 << 20
# Scheduler side: LRU window searched for consecutive free disk slots.
DEFAULT_DISK_ALLOC_WINDOW_BLOCKS = 256


def _iov_max() -> int:
    try:
        v = os.sysconf("SC_IOV_MAX")
    except (AttributeError, ValueError, OSError):
        v = -1
    return v if v > 0 else 1024


_IOV_MAX = _iov_max()


def _page_size() -> int:
    """Host page size, at least the O_DIRECT alignment (64 KiB on some arm64)."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        page = mmap.PAGESIZE
    return max(_ALIGNMENT, int(page))


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


class StagingRegion:
    """Pinned host staging: one page-aligned, page-padded anonymous mmap.

    Registered with cudaHostRegister exactly once. cudaHostRegister works on
    whole pages, so buffers carved from the malloc heap can share a page with
    another registration (or with a still-registered buffer from an earlier
    init) and fail with cudaErrorHostMemoryAlreadyRegistered on 64 KiB-page
    kernels.
    """

    def __init__(self, nbytes: int) -> None:
        page = _page_size()
        self.nbytes = _round_up(max(nbytes, 1), page)
        # Over-allocate one page so the region can start on a page boundary
        # even if the configured page size exceeds mmap's own alignment.
        self._mmap = mmap.mmap(-1, self.nbytes + page)
        full = torch.frombuffer(self._mmap, dtype=torch.int8)
        shift = -full.data_ptr() % page
        self.base = full[shift : shift + self.nbytes]
        self.view = memoryview(self.base.numpy())
        self.registered = False

    def data_ptr(self) -> int:
        return self.base.data_ptr()

    def register(self) -> None:
        pin_tensor(self.base)
        self.registered = True

    def unregister(self) -> None:
        if self.registered:
            unpin_tensor(self.base)
            self.registered = False


def _record_event(stream: torch.cuda.Stream) -> torch.Event:
    event = torch.Event()
    event.record(stream)
    return event


def _rw_full(
    fn: Callable[[int, Sequence[memoryview], int], int],
    fd: int,
    views: list[memoryview],
    offset: int,
    what: str,
) -> None:
    """Run preadv/pwritev over ``views`` until every byte is transferred.

    Splits at IOV_MAX and resumes after short transfers. A zero-length
    transfer (EOF / no progress) is an error.
    """
    i = 0
    while i < len(views):
        chunk = views[i : i + _IOV_MAX]
        want = sum(v.nbytes for v in chunk)
        done = fn(fd, chunk, offset)
        if done <= 0:
            raise OSError(
                f"Short {what}: expected {want} bytes at {offset}, got {done}"
            )
        offset += done
        if done == want:
            i += len(chunk)
            continue
        # Partial transfer: skip fully covered iovecs, trim the next one.
        for v in chunk:
            if done >= v.nbytes:
                done -= v.nbytes
                i += 1
            else:
                views = views[:i] + [v[done:]] + views[i + 1 :]
                break


def _split_runs(
    pairs: list[tuple[int, int]], max_run: int
) -> list[list[tuple[int, int]]]:
    """Group (disk_slot, other) pairs into runs of consecutive disk slots.

    ``pairs`` are sorted by disk slot first; each run has strictly consecutive
    disk slots and at most ``max_run`` entries.
    """
    pairs = sorted(pairs, key=lambda p: p[0])
    runs: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for p in pairs:
        if cur and (p[0] != cur[-1][0] + 1 or len(cur) >= max_run):
            runs.append(cur)
            cur = []
        cur.append(p)
    if cur:
        runs.append(cur)
    return runs


class _BackendShutdown(Exception):
    pass


class _SlotPool:
    """Staging slots handed out as contiguous ranges (one IO buffer per run),
    optionally released only after a DMA event."""

    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self._is_free = [True] * num_slots
        self._pending: deque[tuple[torch.Event, list[int]]] = deque()
        self._cond = threading.Condition()
        self._closed = False

    def _reclaim_locked(self) -> None:
        while self._pending and self._pending[0][0].query():
            for s in self._pending.popleft()[1]:
                self._is_free[s] = True

    def _take_locked(self, k: int) -> list[int] | None:
        """First-fit range of ``k`` contiguous free slots."""
        run = 0
        for i, free in enumerate(self._is_free):
            run = run + 1 if free else 0
            if run == k:
                start = i - k + 1
                for j in range(start, i + 1):
                    self._is_free[j] = False
                return list(range(start, i + 1))
        return None

    def try_acquire(self, k: int) -> list[int] | None:
        with self._cond:
            self._reclaim_locked()
            return self._take_locked(k)

    def acquire(self, k: int) -> list[int]:
        assert 0 < k <= self.num_slots
        while True:
            to_sync: tuple[torch.Event, list[int]] | None = None
            with self._cond:
                self._reclaim_locked()
                slots = self._take_locked(k)
                if slots is not None:
                    return slots
                if self._pending:
                    to_sync = self._pending.popleft()
                elif self._closed:
                    raise _BackendShutdown
                else:
                    self._cond.wait(timeout=1.0)
                    continue
            # Block on the oldest outstanding DMA without holding the lock.
            to_sync[0].synchronize()
            self.release(to_sync[1])

    def release(self, slots: list[int]) -> None:
        with self._cond:
            for s in slots:
                self._is_free[s] = True
            self._cond.notify_all()

    def release_after(self, event: torch.Event, slots: list[int]) -> None:
        with self._cond:
            self._pending.append((event, slots))
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def drain(self) -> None:
        """Wait for every deferred-release DMA event."""
        while True:
            with self._cond:
                if not self._pending:
                    return
                ev, slots = self._pending.popleft()
            ev.synchronize()
            self.release(slots)


class _IOPool:
    """IO thread pool with two priority classes (loads before stores)."""

    def __init__(
        self, num_threads: int, max_store_io: int, device: torch.device | None
    ) -> None:
        self._loads: deque[Callable[[], None]] = deque()
        self._stores: deque[Callable[[], None]] = deque()
        self._cond = threading.Condition()
        self._max_store_io = max(1, max_store_io)
        self._store_running = 0
        self._stop = False
        self._device = device
        self._threads = [
            threading.Thread(target=self._run, name=f"kv-disk-io-{i}", daemon=True)
            for i in range(num_threads)
        ]
        for t in self._threads:
            t.start()

    def submit(self, job: Callable[[], None], is_store: bool) -> None:
        with self._cond:
            (self._stores if is_store else self._loads).append(job)
            self._cond.notify()

    def _next_job(self) -> tuple[Callable[[], None], bool] | None:
        with self._cond:
            while True:
                if self._loads:
                    return self._loads.popleft(), False
                if self._stores and self._store_running < self._max_store_io:
                    self._store_running += 1
                    return self._stores.popleft(), True
                if self._stop and not self._loads and not self._stores:
                    return None
                self._cond.wait()

    def _run(self) -> None:
        if self._device is not None:
            current_platform.set_device(self._device)
        while True:
            nxt = self._next_job()
            if nxt is None:
                return
            job, is_store = nxt
            try:
                job()
            finally:
                if is_store:
                    with self._cond:
                        self._store_running -= 1
                        self._cond.notify_all()

    def shutdown(self, timeout: float) -> bool:
        """Drain queued jobs, stop threads. Returns True if all threads exited."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=timeout)
        return not any(t.is_alive() for t in self._threads)


@dataclass
class _Batch:
    """One launch_copy call; completes once all its runs are done."""

    event_idx: int
    events_list: list[tuple[int, torch.Event]]
    remaining: int = 0
    sealed: bool = False  # all runs dispatched; remaining is final
    failed: bool = False


class DiskBackend:
    """Async disk offload backend: parallel coalesced IO + pipelined DMA."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._io_pool: _IOPool | None = None
        self._load_slots: _SlotPool | None = None
        self._store_slots: _SlotPool | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_region: StagingRegion | None = None
        self._load_region: StagingRegion | None = None
        self._stride: int = 0  # bytes per block on disk and in staging
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        self._max_run_blocks: int = 1
        # In-order completion bookkeeping.
        self._order_lock = threading.Condition()
        self._load_batches: deque[_Batch] = deque()
        self._store_batches: deque[_Batch] = deque()
        self._num_outstanding = 0  # launched, event not yet appended
        self._ordered_load_completion = False
        self._error: BaseException | None = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int | None = None,
        use_page_cache: bool = False,
        num_io_threads: int = DEFAULT_DISK_IO_THREADS,
        max_coalesce_blocks: int = DEFAULT_DISK_MAX_COALESCE_BLOCKS,
        max_store_io_threads: int | None = None,
        ordered_load_completion: bool = False,
        max_io_bytes: int = DEFAULT_DISK_MAX_IO_BYTES,
        staging_bytes: int = DEFAULT_DISK_STAGING_BYTES,
    ) -> None:
        self._ordered_load_completion = ordered_load_completion
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]
        assert sum(self._per_tensor_bpb) == total_block_bytes
        # Pad each block to 4 KiB so every slot (in the file and in staging)
        # starts O_DIRECT aligned, whatever the per-layer sizes are.
        stride = _round_up(total_block_bytes, _ALIGNMENT)
        self._stride = stride
        num_io_threads = max(1, int(num_io_threads))

        run_blocks = max(1, min(int(max_coalesce_blocks), int(max_io_bytes) // stride))
        if num_buffer_slots is None:
            # Room for several full-size runs in flight per direction.
            num_buffer_slots = max(2 * run_blocks, int(staging_bytes) // stride)
        num_buffer_slots = max(1, int(num_buffer_slots))
        self._num_buffer_slots = num_buffer_slots
        # Keep at least two runs' worth of slots so IO and DMA overlap.
        self._max_run_blocks = max(1, min(run_blocks, num_buffer_slots // 2))

        # Block-major staging per direction, one registration each.
        self._store_region = StagingRegion(num_buffer_slots * stride)
        self._load_region = StagingRegion(num_buffer_slots * stride)
        self._store_region.register()
        try:
            self._load_region.register()
        except BaseException:
            self._store_region.unregister()
            raise

        gpu_bases = [t.data_ptr() for t in gpu_caches.values()]
        offsets = [0]
        for bpb in self._per_tensor_bpb[:-1]:
            offsets.append(offsets[-1] + bpb)
        n = len(gpu_bases)

        def staging_bases(region: StagingRegion) -> list[int]:
            return [region.data_ptr() + o for o in offsets]

        self._store_params = build_strided_params(
            gpu_bases,
            staging_bases(self._store_region),
            self._per_tensor_bpb,
            self._per_tensor_bpb,
            [stride] * n,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_strided_params(
            staging_bases(self._load_region),
            gpu_bases,
            self._per_tensor_bpb,
            [stride] * n,
            self._per_tensor_bpb,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_disk_slots * stride)

        if max_store_io_threads is None:
            max_store_io_threads = max(1, num_io_threads // 2)
        max_store_io_threads = max(1, min(int(max_store_io_threads), num_io_threads))

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, block=%d bytes "
            "(%d tensors, stride %d), staging=%d slots per direction "
            "(%.2f GB pinned), io_threads=%d (store<=%d), max IO=%d blocks "
            "(%.1f MB), page_cache=%s",
            disk_path,
            num_disk_slots,
            (num_disk_slots * stride) / (1024**3),
            total_block_bytes,
            n,
            stride,
            num_buffer_slots,
            2 * num_buffer_slots * stride / (1024**3),
            num_io_threads,
            max_store_io_threads,
            self._max_run_blocks,
            self._max_run_blocks * stride / 1e6,
            use_page_cache,
        )

        self._load_slots = _SlotPool(num_buffer_slots)
        self._store_slots = _SlotPool(num_buffer_slots)
        self._io_pool = _IOPool(num_io_threads, max_store_io_threads, device)

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            name="kv-disk-store",
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            name="kv-disk-load",
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    # ------------------------------------------------------------------ API

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        if self._error is not None:
            raise RuntimeError("DiskBackend IO failed earlier") from self._error
        assert len(src_blocks) == len(dst_blocks)
        with self._order_lock:
            self._num_outstanding += 1
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event))

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every launched batch has appended its event.

        Returns False on timeout or if a batch failed.
        """
        with self._order_lock:
            ok = self._order_lock.wait_for(
                lambda: self._num_outstanding == 0 or self._error is not None,
                timeout=timeout,
            )
            return ok and self._error is None

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        io_done = True
        if self._io_pool is not None:
            io_done = self._io_pool.shutdown(timeout=10.0)
        for pool in (self._load_slots, self._store_slots):
            if pool is not None:
                pool.close()
        threads_done = io_done and not any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        )
        if threads_done:
            # Wait for in-flight H2D copies out of load staging, then release
            # the registrations so a later init in this process can register.
            if self._load_slots is not None:
                self._load_slots.drain()
            for region in (self._store_region, self._load_region):
                if region is not None:
                    try:
                        region.unregister()
                    except RuntimeError:
                        logger.exception("Failed to unregister disk staging")
        else:
            logger.warning("IO thread still running; leaving staging registered")
        if self._fd < 0:
            return
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if not threads_done:
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        os.close(self._fd)
        self._fd = -1

    # ------------------------------------------------------- completion order

    def _register_batch(
        self, is_store: bool, event_idx: int, events_list: list
    ) -> _Batch:
        batch = _Batch(event_idx=event_idx, events_list=events_list)
        with self._order_lock:
            (self._store_batches if is_store else self._load_batches).append(batch)
        return batch

    def _run_done(self, batch: _Batch, is_store: bool, ok: bool) -> None:
        with self._order_lock:
            batch.remaining -= 1
            if not ok:
                batch.failed = True
            self._drain_locked(is_store)

    def _seal(self, batch: _Batch, is_store: bool) -> None:
        # ``failed`` may have been set by this coordinator without the lock;
        # it is published here together with ``sealed``.
        with self._order_lock:
            batch.sealed = True
            self._drain_locked(is_store)

    def _drain_locked(self, is_store: bool) -> None:
        """Append events for finished batches.

        Stores are reported strictly in launch order (the worker tracks them
        with a high-water mark). Loads are reported as soon as each batch is
        done unless ``ordered_load_completion`` was requested.
        """
        batches = self._store_batches if is_store else self._load_batches
        stream = self._store_stream if is_store else self._load_stream
        in_order = is_store or self._ordered_load_completion
        appended = False
        i = 0
        while i < len(batches):
            b = batches[i]
            # A failed batch never completes: reporting it would publish
            # garbage (store) or hand garbage KV to a request (load).
            if not b.sealed or b.remaining > 0 or b.failed:
                if in_order:
                    break
                i += 1
                continue
            del batches[i]
            # Loads: recorded after every DMA of this batch was issued on the
            # load stream, so the event fires once the KV is on the GPU.
            # Stores: the data is already on disk; the event is just a marker.
            b.events_list.append((b.event_idx, _record_event(stream)))
            self._num_outstanding -= 1
            appended = True
        if appended:
            self._order_lock.notify_all()

    def _fail(self, exc: BaseException) -> None:
        logger.exception("DiskBackend IO failure", exc_info=exc)
        with self._order_lock:
            if self._error is None:
                self._error = exc
            self._order_lock.notify_all()

    def _slot_span(self, region: StagingRegion, slots: list[int]) -> memoryview:
        """One buffer covering contiguous staging slots."""
        first = slots[0]
        assert slots[-1] == first + len(slots) - 1
        return region.view[first * self._stride : (first + len(slots)) * self._stride]

    # ------------------------------------------------------------- loads

    def _load_loop(self, device: torch.device, stream: torch.cuda.Stream) -> None:
        current_platform.set_device(device)
        assert self._load_slots is not None and self._io_pool is not None
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            disk_slots, gpu_blocks, event_idx, events_list, wait_event = item
            batch = self._register_batch(False, event_idx, events_list)
            if wait_event is not None:
                stream.wait_event(wait_event)
            try:
                for run in _split_runs(
                    list(zip(disk_slots, gpu_blocks)), self._max_run_blocks
                ):
                    slots = self._load_slots.acquire(len(run))
                    with self._order_lock:
                        batch.remaining += 1
                    self._io_pool.submit(
                        functools.partial(self._load_run, batch, run, slots, stream),
                        is_store=False,
                    )
            except _BackendShutdown:
                return
            except Exception as e:  # noqa: BLE001
                self._fail(e)
                batch.failed = True
            self._seal(batch, False)

    def _load_run(
        self,
        batch: _Batch,
        run: list[tuple[int, int]],
        slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        assert self._load_slots is not None and self._load_params is not None
        ok = False
        dma_issued = False
        try:
            assert self._load_region is not None
            _rw_full(
                os.preadv,
                self._fd,
                [self._slot_span(self._load_region, slots)],
                run[0][0] * self._stride,
                "read",
            )
            copy_blocks(slots, [g for _, g in run], self._load_params)
            dma_issued = True
            self._load_slots.release_after(_record_event(stream), slots)
            ok = True
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            if not dma_issued:
                self._load_slots.release(slots)
        finally:
            self._run_done(batch, False, ok)

    # ------------------------------------------------------------- stores

    def _store_loop(self, device: torch.device, stream: torch.cuda.Stream) -> None:
        current_platform.set_device(device)
        assert self._store_slots is not None and self._io_pool is not None
        assert self._store_params is not None
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            gpu_blocks, disk_slots, event_idx, events_list, wait_event = item
            batch = self._register_batch(True, event_idx, events_list)
            if wait_event is not None:
                stream.wait_event(wait_event)
            # (dma_event, run, slots) staged on GPU->host but not yet written.
            staged: deque[tuple[torch.Event, list[tuple[int, int]], list[int]]] = (
                deque()
            )
            try:
                for run in _split_runs(
                    list(zip(disk_slots, gpu_blocks)), self._max_run_blocks
                ):
                    slots = self._acquire_store_slots(len(run), staged, batch)
                    copy_blocks([g for _, g in run], slots, self._store_params)
                    staged.append((_record_event(stream), run, slots))
                    with self._order_lock:
                        batch.remaining += 1
                    # Hand off whatever DMA already finished, without blocking.
                    while staged and staged[0][0].query():
                        self._submit_store(batch, *staged.popleft()[1:])
                while staged:
                    ev, run, slots = staged.popleft()
                    ev.synchronize()
                    self._submit_store(batch, run, slots)
            except _BackendShutdown:
                return
            except Exception as e:  # noqa: BLE001
                self._fail(e)
                batch.failed = True
            self._seal(batch, True)

    def _acquire_store_slots(
        self,
        k: int,
        staged: deque[tuple[torch.Event, list[tuple[int, int]], list[int]]],
        batch: _Batch,
    ) -> list[int]:
        assert self._store_slots is not None
        while True:
            slots = self._store_slots.try_acquire(k)
            if slots is not None:
                return slots
            if not staged:
                # Everything we own is being written; wait for a write.
                return self._store_slots.acquire(k)
            # Free up slots by pushing the oldest staged run to disk.
            ev, run, s = staged.popleft()
            ev.synchronize()
            self._submit_store(batch, run, s)

    def _submit_store(
        self, batch: _Batch, run: list[tuple[int, int]], slots: list[int]
    ) -> None:
        assert self._io_pool is not None
        self._io_pool.submit(
            functools.partial(self._store_run, batch, run, slots), is_store=True
        )

    def _store_run(
        self, batch: _Batch, run: list[tuple[int, int]], slots: list[int]
    ) -> None:
        assert self._store_slots is not None
        ok = False
        try:
            assert self._store_region is not None
            _rw_full(
                os.pwritev,
                self._fd,
                [self._slot_span(self._store_region, slots)],
                run[0][0] * self._stride,
                "write",
            )
            ok = True
        except Exception as e:  # noqa: BLE001
            self._fail(e)
        finally:
            self._store_slots.release(slots)
            self._run_done(batch, True, ok)
