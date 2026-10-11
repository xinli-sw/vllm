# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)

# Bounds of the host-side poll for a gated copy's wait_event. The copy thread
# backs off from the minimum to the maximum while a copy is gated and wakes at
# once for newly queued copies.
_GATE_POLL_MIN_S = 5e-5
_GATE_POLL_MAX_S = 5e-4

# Paced batch copies. One cuMemcpyBatchAsync call holds a driver lock that every
# other CUDA call in the process waits on, for ~1.6 us per descriptor (one per
# block per KV region): a CPU-tier load of a few thousand blocks issued as one
# call blocks the worker's main thread for ~1 s, and with it the next step. The
# copy thread instead issues at most _COPY_CHUNK_DESCRIPTORS descriptors per
# call (0 disables chunking), and starts a call only while the worker's main
# thread is between steps (see DmaCopyBackend.set_host_busy), waiting at most
# _COPY_MAX_PAUSE_S per call for it.
_COPY_CHUNK_DESCRIPTORS = int(
    os.environ.get("VLLM_SIMPLE_KV_COPY_CHUNK_DESCRIPTORS", "768")
)
_COPY_MAX_PAUSE_S = (
    float(os.environ.get("VLLM_SIMPLE_KV_COPY_MAX_PAUSE_MS", "50")) / 1000.0
)
_COPY_PAUSE_POLL_S = 1e-4


class _Copy(NamedTuple):
    src_blocks: list[int]
    dst_blocks: list[int]
    params: BatchMemcpyParams
    is_store: bool
    event_idx: int
    events_list: list[tuple[int, torch.Event]]
    wait_event: torch.Event | None
    pace: Callable[[], None] | None


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread).

    Args:
        host_gated_stores: Gate each GPU->CPU store on its compute-done event
            on the host (the copy thread polls ``event.query()`` and issues the
            batch memcpy once it has completed) instead of enqueuing a
            device-side ``stream.wait_event()`` ahead of the memcpy. A batch
            memcpy queued behind a device-side wait can block inside the
            driver, holding a lock every other CUDA call in the process needs,
            until compute reaches the event.

    """

    def __init__(self, host_gated_stores: bool = False) -> None:
        self._host_gated_stores = host_gated_stores
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        # Copies queued but not yet submitted to their stream (see wait_idle).
        self._num_queued = 0
        self._idle = threading.Condition()
        # Set while the worker's main thread issues a step's CUDA work (copy
        # calls then wait); cleared between steps, and ignored while draining.
        self._host_busy = False
        self._draining = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in _copy_loop); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(
                self._queue,
                device,
                load_stream,
                store_stream,
                self._submitted,
                self._host_gated_stores,
            ),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        with self._idle:
            self._num_queued += 1
        self._queue.put(
            _Copy(
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
                self._pace if _COPY_CHUNK_DESCRIPTORS > 0 else None,
            )
        )

    def set_host_busy(self, busy: bool) -> None:
        """Mark whether the worker's main thread is issuing a step's CUDA work;
        while it is, the copy thread holds back its next batch-copy call."""
        self._host_busy = busy

    def _pace(self) -> None:
        """Copy thread, before each batch-copy call: wait (bounded) until the
        main thread is between steps."""
        if not self._host_busy or self._draining:
            return
        deadline = time.monotonic() + _COPY_MAX_PAUSE_S
        while self._host_busy and not self._draining and time.monotonic() < deadline:
            time.sleep(_COPY_PAUSE_POLL_S)

    def wait_idle(self) -> None:
        """Block until every queued copy is submitted to its stream, i.e. its
        completion event is in its ``events_list``. Returns early if the copy
        thread has died. Queued copies are issued without pacing meanwhile."""
        thread = self._thread
        self._draining = True
        try:
            with self._idle:
                while self._num_queued and thread is not None and thread.is_alive():
                    self._idle.wait(timeout=1.0)
        finally:
            self._draining = False

    def _submitted(self) -> None:
        with self._idle:
            self._num_queued -= 1
            if not self._num_queued:
                self._idle.notify_all()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        submitted: Callable[[], None],
        host_gated: bool = False,
    ) -> None:
        current_platform.set_device(device)
        streams = (load_stream, store_stream)
        if not host_gated:
            while (item := q.get()) is not None:
                stream = streams[item.is_store]
                if item.wait_event is not None:
                    stream.wait_event(item.wait_event)
                DmaCopyBackend._submit(item, stream)
                submitted()
            return
        # Per stream (indexed by is_store), copies waiting on the host for their
        # wait_event, oldest first; later copies to that stream queue behind
        # them. Gating on the host replaces stream.wait_event(): a batch memcpy
        # queued behind a device-side wait on compute can block in the driver,
        # holding a lock every other CUDA call in the process needs, until
        # compute reaches the event. Loads carry no wait_event, so they are not
        # held behind a gated store.
        gated: tuple[deque[_Copy], deque[_Copy]] = (deque(), deque())
        poll_s = _GATE_POLL_MIN_S
        stopping = False
        while True:
            for stream, waiting in zip(streams, gated):
                while waiting and DmaCopyBackend._ready(waiting[0]):
                    DmaCopyBackend._submit(waiting.popleft(), stream)
                    submitted()
                    poll_s = _GATE_POLL_MIN_S
            any_gated = bool(gated[0] or gated[1])
            if stopping and not any_gated:
                return
            try:
                item = q.get(timeout=poll_s) if any_gated else q.get()
            except queue.Empty:
                poll_s = min(2 * poll_s, _GATE_POLL_MAX_S)
                continue
            if item is None:
                stopping = True
                continue
            waiting = gated[item.is_store]
            if waiting or not DmaCopyBackend._ready(item):
                waiting.append(item)
            else:
                DmaCopyBackend._submit(item, streams[item.is_store])
                submitted()

    @staticmethod
    def _ready(copy: _Copy) -> bool:
        return copy.wait_event is None or copy.wait_event.query()

    @staticmethod
    def _submit(copy: _Copy, stream: torch.cuda.Stream) -> None:
        copy_blocks(
            copy.src_blocks,
            copy.dst_blocks,
            copy.params,
            max_descriptors=_COPY_CHUNK_DESCRIPTORS,
            before_chunk=copy.pace,
        )
        event = torch.Event()
        event.record(stream)
        copy.events_list.append((copy.event_idx, event))
