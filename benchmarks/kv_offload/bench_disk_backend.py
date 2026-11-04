# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the SimpleCPUOffloadConnector disk tier.

Measures GPU<->NVMe KV block load/store throughput and per-batch latency for

  * ``legacy``: the pre-rewrite DiskBackend (one IO thread per direction,
    one block per preadv/pwritev, 2 staging slots), embedded below, and
  * ``new``: vllm.v1.simple_kv_offload.disk_backend.DiskBackend,

over a sweep of in-flight batch counts, IO threads, staging slots and
coalescing, plus a pure-disk ceiling (raw O_DIRECT preadv/pwritev from many
threads, no GPU).

Examples (GB200 tray, md0 mounted at /mnt/nvme)::

    # Default: MiniMax-M3.1 TP4 rank layout (1.8227 MB block over 125 tensors
    # of ~14.6 KB). Legacy issues 125-iovec preadv/pwritev per block, which
    # NVMe splits into ~10-15 KB requests; new issues one buffer per run.
    python benchmarks/kv_offload/bench_disk_backend.py --disk-dir /mnt/nvme/kvbench

    # DEP prefill layout used earlier (~5.63 MB over 8 tensors)
    python benchmarks/kv_offload/bench_disk_backend.py --disk-dir /mnt/nvme/kvbench \
        --block-bytes 5632000 --num-tensors 8

    # Whole-tray picture: 4 GPUs hammering the same RAID at once
    python benchmarks/kv_offload/bench_disk_backend.py --disk-dir /mnt/nvme/kvbench \\
        --num-procs 4 --json out.json

    # Only the raw disk ceiling
    python benchmarks/kv_offload/bench_disk_backend.py --disk-dir /mnt/nvme/kvbench \\
        --backends none --raw-threads 1,4,16,32,64 --raw-io-mb 1,4,16

Results print as a table and, with ``--json``, as a JSON list of rows.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import mmap
import os
import queue
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass

O_DIRECT = getattr(os, "O_DIRECT", 0)
ALIGN = 4096


# --------------------------------------------------------------------------
# Raw disk ceiling (no torch / GPU)
# --------------------------------------------------------------------------


def _aligned_buf(nbytes: int) -> mmap.mmap:
    # Anonymous mmap is page aligned, which satisfies O_DIRECT.
    return mmap.mmap(-1, nbytes)


def raw_disk_bench(
    path: str,
    file_bytes: int,
    io_bytes: int,
    threads: int,
    seconds: float,
    write: bool,
    direct: bool,
    pattern: str,
) -> float:
    """Return GB/s for ``threads`` threads doing ``io_bytes`` IOs."""
    flags = os.O_RDWR | (O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    n_ios = file_bytes // io_bytes
    assert n_ios > 0, "file too small for io size"
    stop = time.monotonic() + seconds
    totals = [0] * threads
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        buf = _aligned_buf(io_bytes)
        mv = memoryview(buf)
        rng = random.Random(tid)
        idx = tid * (n_ios // threads)
        try:
            while time.monotonic() < stop:
                if pattern == "random":
                    off = rng.randrange(n_ios) * io_bytes
                else:
                    off = (idx % n_ios) * io_bytes
                    idx += 1
                n = os.pwritev(fd, [mv], off) if write else os.preadv(fd, [mv], off)
                totals[tid] += n
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            mv.release()
            buf.close()

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.monotonic() - t0
    os.close(fd)
    if errors:
        raise errors[0]
    return sum(totals) / dt / 1e9


def prepare_file(path: str, nbytes: int) -> None:
    """Create and fully write ``path`` so reads hit real extents."""
    if os.path.exists(path) and os.path.getsize(path) >= nbytes:
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT | O_DIRECT, 0o600)
    chunk = 64 << 20
    buf = _aligned_buf(chunk)
    mv = memoryview(buf)
    try:
        off = 0
        while off < nbytes:
            n = min(chunk, nbytes - off)
            os.pwritev(fd, [mv[:n]], off)
            off += n
    finally:
        mv.release()
        buf.close()
        os.close(fd)


# --------------------------------------------------------------------------
# Legacy backend (copied from vLLM dd4e40142a, behavior preserved)
# --------------------------------------------------------------------------


def _make_legacy_cls():
    import torch

    from vllm.platforms import current_platform
    from vllm.v1.simple_kv_offload.cuda_mem_ops import (
        CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        build_params,
        copy_blocks,
    )
    from vllm.v1.simple_kv_offload.disk_backend import StagingRegion

    class LegacyDiskBackend:
        def init(
            self,
            gpu_caches,
            device,
            load_stream,
            store_stream,
            disk_path,
            num_disk_slots,
            total_block_bytes,
            num_buffer_slots=2,
            use_page_cache=False,
            **_ignored,
        ):
            self._T = total_block_bytes
            self._n = num_buffer_slots
            names = list(gpu_caches)
            # Same page-aligned, unregistered-on-shutdown staging as the new
            # backend; "legacy" here means the old IO pattern only.
            # Old per-tensor layout ([tensor][slot][bpb]), each tensor's
            # buffer 4 KiB aligned, so a block is one iovec per tensor.
            bpbs = [g.stride(0) * g.element_size() for g in gpu_caches.values()]
            offs, off = [], 0
            for bpb in bpbs:
                offs.append(off)
                off += (num_buffer_slots * bpb + ALIGN - 1) // ALIGN * ALIGN
            self._regions = [StagingRegion(off), StagingRegion(off)]
            for r in self._regions:
                r.register()

            def per_tensor(r):
                return {
                    n: r.base[o : o + num_buffer_slots * b].view(num_buffer_slots, b)
                    for n, o, b in zip(names, offs, bpbs)
                }

            self._sbuf = per_tensor(self._regions[0])
            self._lbuf = per_tensor(self._regions[1])
            self._sviews = [
                [memoryview(self._sbuf[n][s].numpy()) for n in names]
                for s in range(num_buffer_slots)
            ]
            self._lviews = [
                [memoryview(self._lbuf[n][s].numpy()) for n in names]
                for s in range(num_buffer_slots)
            ]
            self._sp = build_params(
                gpu_caches,
                self._sbuf,
                store_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            )
            self._lp = build_params(
                self._lbuf,
                gpu_caches,
                load_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            )
            with contextlib.suppress(FileNotFoundError):
                os.unlink(disk_path)
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if not use_page_cache:
                flags |= O_DIRECT
            self._fd = os.open(disk_path, flags, 0o600)
            self._path = disk_path
            os.ftruncate(self._fd, num_disk_slots * total_block_bytes)
            self._sq: queue.SimpleQueue = queue.SimpleQueue()
            self._lq: queue.SimpleQueue = queue.SimpleQueue()
            self._threads = [
                threading.Thread(
                    target=self._loop, args=(self._sq, True, device, store_stream)
                ),
                threading.Thread(
                    target=self._loop, args=(self._lq, False, device, load_stream)
                ),
            ]
            for t in self._threads:
                t.daemon = True
                t.start()

        def launch_copy(
            self, src, dst, is_store, event_idx, events_list, wait_event=None
        ):
            (self._sq if is_store else self._lq).put(
                (src, dst, event_idx, events_list, wait_event)
            )

        def flush(self, timeout=None):
            return True

        def shutdown(self):
            self._sq.put(None)
            self._lq.put(None)
            for t in self._threads:
                t.join(timeout=10)
            torch.cuda.synchronize()
            for r in self._regions:
                r.unregister()
            with contextlib.suppress(OSError):
                os.unlink(self._path)
            os.close(self._fd)

        def _loop(self, q, is_store, device, stream):
            current_platform.set_device(device)
            while True:
                item = q.get()
                if item is None:
                    return
                src, dst, idx, evs, wait_event = item
                if wait_event is not None:
                    stream.wait_event(wait_event)
                if is_store:
                    self._do_store(src, dst, stream)
                else:
                    self._do_load(src, dst, stream)
                ev = torch.Event()
                ev.record(stream)
                evs.append((idx, ev))

        def _do_store(self, gpu_blocks, disk_slots, stream):
            n = self._n
            pending = [None] * n
            for i, (g, d) in enumerate(zip(gpu_blocks, disk_slots)):
                b = i % n
                if pending[b] is not None:
                    pending[b][0].synchronize()
                    os.pwritev(self._fd, self._sviews[b], pending[b][1])
                copy_blocks([g], [b], self._sp)
                ev = torch.Event()
                ev.record(stream)
                pending[b] = (ev, d * self._T)
            for b, p in enumerate(pending):
                if p is not None:
                    p[0].synchronize()
                    os.pwritev(self._fd, self._sviews[b], p[1])

        def _do_load(self, disk_slots, gpu_blocks, stream):
            n = self._n
            prev = [None] * n
            for i, (d, g) in enumerate(zip(disk_slots, gpu_blocks)):
                b = i % n
                if prev[b] is not None:
                    prev[b].synchronize()
                os.preadv(self._fd, self._lviews[b], d * self._T)
                copy_blocks([b], [g], self._lp)
                ev = torch.Event()
                ev.record(stream)
                prev[b] = ev
            for ev in prev:
                if ev is not None:
                    ev.synchronize()

    return LegacyDiskBackend


# --------------------------------------------------------------------------
# Backend benchmark
# --------------------------------------------------------------------------


@dataclass
class Row:
    kind: str  # "backend" | "raw"
    backend: str
    op: str  # "store" | "load" | "read" | "write"
    procs: int
    inflight: int
    batch_blocks: int
    io_threads: int
    buffer_slots: int
    coalesce: int
    layout: str
    io_mb: float
    gbps: float  # aggregate across procs
    lat_p50_ms: float = 0.0
    lat_p99_ms: float = 0.0
    lat_max_ms: float = 0.0
    verified: bool | None = None


def _tensor_sizes(block_bytes: int, num_tensors: int, align: int) -> list[int]:
    """Split ``block_bytes`` over tensors in ``align``-byte units (sum exact)."""
    units, rest = divmod(block_bytes, align)
    base, extra = divmod(units, num_tensors)
    sizes = [(base + (1 if i < extra else 0)) * align for i in range(num_tensors)]
    sizes[-1] += rest
    assert all(x > 0 for x in sizes), "too many tensors for block size"
    return sizes


def _run_phase(backend, batches, is_store, inflight, stream):
    """Launch ``batches`` keeping at most ``inflight`` outstanding.

    Returns (seconds, per-batch latencies in seconds).
    """
    import torch

    events: list = []
    launch_t: list[float] = []
    lat: list[float] = []
    done = 0  # completed batches (loads may complete out of order)
    wait_event = None
    if is_store:
        wait_event = torch.Event()
        wait_event.record(torch.cuda.current_stream())
    t0 = time.perf_counter()
    nxt = 0
    while done < len(batches):
        while nxt < len(batches) and nxt - done < inflight:
            src, dst = batches[nxt]
            launch_t.append(time.perf_counter())
            backend.launch_copy(src, dst, is_store, nxt, events, wait_event)
            nxt += 1
        if len(events) > done:
            idx, ev = events[done]
            if is_store:
                assert idx == done, f"out-of-order store {idx} != {done}"
            ev.synchronize()
            lat.append(time.perf_counter() - launch_t[idx])
            done += 1
        else:
            time.sleep(0.0002)
    return time.perf_counter() - t0, lat


def _make_batches(n_batches, batch_blocks, num_gpu_blocks, num_slots, layout, seed):
    rng = random.Random(seed)
    total = n_batches * batch_blocks
    assert total <= num_slots, "increase --disk-gb or lower batch count"
    if layout == "contiguous":
        slots = list(range(total))
    else:
        slots = rng.sample(range(num_slots), total)
    batches = []
    for b in range(n_batches):
        disk = slots[b * batch_blocks : (b + 1) * batch_blocks]
        gpu = [(b * batch_blocks + i) % num_gpu_blocks for i in range(batch_blocks)]
        batches.append((gpu, disk))
    return batches


def backend_sweep(args, rank: int, barrier) -> list[Row]:
    import torch

    from vllm.v1.simple_kv_offload.disk_backend import DiskBackend

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    sizes = _tensor_sizes(args.block_bytes, args.num_tensors, args.tensor_align)
    block_bytes = sum(sizes)
    # New backend pads each block to 4 KiB on disk; size the slot count so
    # both backends fit in --disk-gb.
    num_slots = int(args.disk_gb * (1 << 30)) // (
        (block_bytes + ALIGN - 1) // ALIGN * ALIGN
    )
    gpu = {
        f"layer{i}": torch.randint(
            -128, 127, (args.num_gpu_blocks, b), dtype=torch.int8, device=device
        )
        for i, b in enumerate(sizes)
    }
    ref = {n: t.clone() for n, t in gpu.items()} if args.verify else None
    legacy_cls = _make_legacy_cls()
    low_pri, _ = torch.cuda.Stream.priority_range()
    rows: list[Row] = []

    configs = []
    for be in args.backends:
        if be == "legacy":
            for inflight, bb, layout in itertools.product(
                args.inflight, args.batch_blocks, args.layouts
            ):
                configs.append((be, inflight, bb, 1, 2, 1, 0.0, layout))
        elif be == "new":
            for inflight, bb, th, sl, co, mio, layout in itertools.product(
                args.inflight,
                args.batch_blocks,
                args.io_threads,
                args.buffer_slots,
                args.coalesce,
                args.max_io_mb,
                args.layouts,
            ):
                configs.append((be, inflight, bb, th, sl, co, mio, layout))

    for be, inflight, bb, th, sl, co, mio, layout in configs:
        n_batches = max(inflight * 2, args.total_blocks // bb)
        batches = _make_batches(
            n_batches, bb, args.num_gpu_blocks, num_slots, layout, seed=rank
        )
        backend = legacy_cls() if be == "legacy" else DiskBackend()
        path = os.path.join(args.disk_dir, f"kvbench.rank{rank}.bin")
        backend.init(
            gpu,
            device,
            torch.cuda.Stream(priority=low_pri),
            torch.cuda.Stream(priority=low_pri),
            path,
            num_slots,
            block_bytes,
            num_buffer_slots=sl or None,
            use_page_cache=args.use_page_cache,
            num_io_threads=th,
            max_coalesce_blocks=co,
            max_io_bytes=int(mio * (1 << 20)) if mio else 1 << 62,
        )
        eff_slots = getattr(backend, "_num_buffer_slots", sl)
        eff_run = getattr(backend, "_max_run_blocks", co)
        try:
            n_bytes = n_batches * bb * block_bytes
            for op, is_store in (("store", True), ("load", False)):
                if op == "load" and ref is not None:
                    for t in gpu.values():
                        t.zero_()
                torch.cuda.synchronize()
                if barrier is not None:
                    barrier.wait()
                seq = (
                    batches
                    if is_store
                    else [(disk, gpu_ids) for gpu_ids, disk in batches]
                )
                secs, lat = _run_phase(
                    backend, seq, is_store, inflight, torch.cuda.current_stream()
                )
                verified = None
                if op == "load" and ref is not None:
                    torch.cuda.synchronize()
                    # GPU ids repeat modulo the pool size, but every copy of a
                    # block id was stored from the same (unchanged) data.
                    ids = torch.tensor(
                        sorted({b for gpu_ids, _ in batches for b in gpu_ids}),
                        device=device,
                    )
                    verified = all(torch.equal(gpu[n][ids], ref[n][ids]) for n in gpu)
                    for n in gpu:  # restore for the next config's store
                        gpu[n].copy_(ref[n])
                lat_ms = sorted(x * 1e3 for x in lat)
                p99 = lat_ms[min(len(lat_ms) - 1, int(len(lat_ms) * 0.99))]
                rows.append(
                    Row(
                        kind="backend",
                        backend=be,
                        op=op,
                        procs=1,
                        inflight=inflight,
                        batch_blocks=bb,
                        io_threads=th,
                        buffer_slots=eff_slots,
                        coalesce=eff_run,
                        layout=layout,
                        io_mb=round(block_bytes / 1e6, 3),
                        gbps=n_bytes / secs / 1e9,
                        lat_p50_ms=statistics.median(lat_ms),
                        lat_p99_ms=p99,
                        lat_max_ms=lat_ms[-1],
                        verified=verified,
                    )
                )
                if rank == 0 and args.progress:
                    r = rows[-1]
                    print(
                        f"  [{be} {op} inflight={inflight} bb={bb} th={th} "
                        f"slots={sl} co={co} {layout}] {r.gbps:.2f} GB/s/rank "
                        f"p50={r.lat_p50_ms:.1f}ms",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            backend.shutdown()
    return rows


def raw_sweep(args, rank: int, barrier) -> list[Row]:
    rows: list[Row] = []
    path = os.path.join(args.disk_dir, f"kvbench.raw.rank{rank}.bin")
    file_bytes = int(args.raw_file_gb * (1 << 30)) // ALIGN * ALIGN
    prepare_file(path, file_bytes)
    try:
        for write, mb, th in itertools.product(
            (False, True), args.raw_io_mb, args.raw_threads
        ):
            io_bytes = int(mb * (1 << 20)) // ALIGN * ALIGN
            if barrier is not None:
                barrier.wait()
            gbps = raw_disk_bench(
                path,
                file_bytes,
                io_bytes,
                th,
                args.raw_seconds,
                write,
                direct=not args.use_page_cache,
                pattern=args.raw_pattern,
            )
            rows.append(
                Row(
                    kind="raw",
                    backend="raw-odirect" if not args.use_page_cache else "raw-buf",
                    op="write" if write else "read",
                    procs=1,
                    inflight=th,
                    batch_blocks=0,
                    io_threads=th,
                    buffer_slots=0,
                    coalesce=0,
                    layout=args.raw_pattern,
                    io_mb=mb,
                    gbps=gbps,
                )
            )
            if rank == 0 and args.progress:
                print(
                    f"  [raw {'write' if write else 'read'} {mb}MB x{th}] "
                    f"{gbps:.2f} GB/s/proc",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    return rows


def _proc_main(args, rank, barrier, out_q):
    try:
        rows = []
        if args.raw_threads:
            rows += raw_sweep(args, rank, barrier)
        if args.backends:
            rows += backend_sweep(args, rank, barrier)
        out_q.put((rank, [asdict(r) for r in rows], None))
    except BaseException as e:  # noqa: BLE001
        import traceback

        out_q.put((rank, [], traceback.format_exc()))
        if barrier is not None:
            barrier.abort()
        raise SystemExit(1) from e


def _aggregate(per_rank: list[list[dict]]) -> list[dict]:
    """Sum GB/s across procs row-by-row (all procs run the same sweep)."""
    if len(per_rank) == 1:
        return per_rank[0]
    out = []
    for rows in zip(*per_rank):
        r = dict(rows[0])
        r["procs"] = len(rows)
        r["gbps"] = sum(x["gbps"] for x in rows)
        r["lat_p50_ms"] = statistics.median(x["lat_p50_ms"] for x in rows)
        r["lat_p99_ms"] = max(x["lat_p99_ms"] for x in rows)
        r["lat_max_ms"] = max(x["lat_max_ms"] for x in rows)
        v = [x["verified"] for x in rows]
        r["verified"] = None if v[0] is None else all(v)
        out.append(r)
    return out


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("backend", "backend", "{}"),
        ("op", "op", "{}"),
        ("procs", "P", "{}"),
        ("inflight", "infl", "{}"),
        ("batch_blocks", "blk/b", "{}"),
        ("io_threads", "thr", "{}"),
        ("buffer_slots", "slots", "{}"),
        ("coalesce", "coal", "{}"),
        ("layout", "layout", "{}"),
        ("io_mb", "MB/io|blk", "{}"),
        ("gbps", "GB/s", "{:.2f}"),
        ("lat_p50_ms", "p50ms", "{:.1f}"),
        ("lat_p99_ms", "p99ms", "{:.1f}"),
        ("verified", "ok", "{}"),
    ]
    cells = [[h for _, h, _ in cols]] + [
        [f.format(r[k]) if r[k] is not None else "-" for k, _, f in cols] for r in rows
    ]
    widths = [max(len(c[i]) for c in cells) for i in range(len(cols))]
    for i, c in enumerate(cells):
        print("  ".join(x.rjust(w) for x, w in zip(c, widths)))
        if i == 0:
            print("  ".join("-" * w for w in widths))


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--disk-dir", required=True, help="directory on the target disk")
    p.add_argument(
        "--block-bytes",
        type=int,
        default=1_822_720,
        help="bytes per KV block per rank (sum over all tensors)",
    )
    p.add_argument("--num-tensors", type=int, default=125)
    p.add_argument(
        "--tensor-align",
        type=int,
        default=512,
        help="per-tensor byte granularity (512 keeps legacy O_DIRECT iovecs "
        "valid; real layers are not 4 KiB multiples)",
    )
    p.add_argument("--num-gpu-blocks", type=int, default=256)
    p.add_argument("--disk-gb", type=float, default=32.0, help="backend file size")
    p.add_argument(
        "--total-blocks",
        type=int,
        default=1024,
        help="blocks moved per measurement (per direction, per proc)",
    )
    p.add_argument("--backends", default="legacy,new", help="legacy,new or none")
    p.add_argument("--inflight", type=_ints, default=[1, 4, 16])
    p.add_argument("--batch-blocks", type=_ints, default=[8, 64])
    p.add_argument("--io-threads", type=_ints, default=[4, 8])
    p.add_argument(
        "--buffer-slots",
        type=_ints,
        default=[0],
        help="staging slots per direction; 0 = backend default (sized in bytes)",
    )
    p.add_argument("--coalesce", type=_ints, default=[32], help="max blocks per IO")
    p.add_argument(
        "--max-io-mb", type=_floats, default=[16.0], help="max bytes per IO (MiB)"
    )
    p.add_argument("--layouts", default="random,contiguous")
    p.add_argument("--use-page-cache", action="store_true")
    p.add_argument("--verify", action="store_true", help="check load == store data")
    p.add_argument("--raw-threads", type=_ints, default=[1, 8, 32])
    p.add_argument(
        "--raw-io-mb",
        type=_floats,
        default=[0.015, 1.82, 16],
        help="raw IO sizes (MiB); 0.015 ~ one layer of the default layout",
    )
    p.add_argument("--raw-seconds", type=float, default=5.0)
    p.add_argument("--raw-file-gb", type=float, default=16.0)
    p.add_argument("--raw-pattern", choices=["random", "sequential"], default="random")
    p.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="processes run in lock-step, proc i on GPU i (whole-tray view)",
    )
    p.add_argument("--json", default=None, help="write rows to this JSON file")
    p.add_argument("--quiet", dest="progress", action="store_false")
    args = p.parse_args()
    args.backends = [b for b in args.backends.split(",") if b and b != "none"]
    args.layouts = [x for x in args.layouts.split(",") if x]
    os.makedirs(args.disk_dir, exist_ok=True)

    if args.num_procs == 1:
        import queue as _q

        out: _q.Queue = _q.Queue()
        _proc_main(args, 0, None, out)
        _, rows, err = out.get()
        per_rank = [rows]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(args.num_procs)
        out = ctx.Queue()
        procs = [
            ctx.Process(target=_proc_main, args=(args, r, barrier, out))
            for r in range(args.num_procs)
        ]
        for pr in procs:
            pr.start()
        results = {}
        for _ in procs:
            rank, rows, err = out.get()
            if err:
                print(f"rank {rank} failed:\n{err}", file=sys.stderr)
            results[rank] = rows
        for pr in procs:
            pr.join()
        per_rank = [results[r] for r in sorted(results)]
        if any(not r for r in per_rank):
            sys.exit(1)

    rows = _aggregate(per_rank)
    print()
    print(
        f"block={args.block_bytes / 1e6:.4f} MB"
        f" ({args.num_tensors} tensors), page_cache={args.use_page_cache}, "
        f"procs={args.num_procs}; GB/s is aggregate across procs"
    )
    _print_table(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
