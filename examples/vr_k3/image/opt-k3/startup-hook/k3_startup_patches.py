# Kimi K3 serving patches applied by sitecustomize.py (this directory), each behind its own env knob.
#
# K3_FST_BULK=1: fastsafetensors' multi-rank consumer (PipelineParallel._consume_single_batch) broadcasts each
#   file's bytes in <= 1 GiB chunks (one NCCL broadcast per chunk) instead of one broadcast + device sync per
#   tensor (497,220 tensors for Kimi K3). Every rank still gets a fresh tensor per key, allocated exactly as the
#   stock path allocates it (framework.get_empty_tensor), holding the file's bytes for that key, in the stock
#   yield order (file by file, tensors in data-offset order). So the yielded (name, tensor) sequence is the same
#   bytes by construction; K3_WEIGHT_DIGEST checks it.
# K3_WEIGHT_DIGEST=params|all: logs per-rank GPU digests (position-sensitive 64-bit polynomial hashes over raw bytes) of
#   (a) the weight iterator's yielded sequence (names, order, bytes) and (b) every parameter and buffer of the
#   loaded model (by name) after load_model, as "[k3-weights]" lines on stderr. "params": (b) only (cheap, after
#   the "Loading weights took" stamp); "all": both ((a) adds a few GPU ops per tensor to the load).
import os
import sys
import time
import zlib

CHUNK = 1 << 30
_P = 0x100000001B3  # FNV-64 prime: order mixing of per-item hashes (int64 wrap-around)


def log(msg):
    print(f"[k3-weights] {os.uname().nodename.split('.')[0]} pid={os.getpid()} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# K3_FST_BULK
# ---------------------------------------------------------------------------------------------------------------
def patch_fastsafetensors(parallel_loader):
    import torch
    import torch.distributed as dist

    PipelineParallel = parallel_loader.PipelineParallel
    TimingContext = parallel_loader.TimingContext
    stock = PipelineParallel._consume_single_batch
    scratch = {}

    def buffer(nbytes, device):
        key = str(device)
        buf = scratch.get(key)
        if buf is None or buf.numel() < nbytes:
            buf = torch.empty(max(nbytes, CHUNK), dtype=torch.uint8, device=device)
            scratch[key] = buf
        return buf[:nbytes]

    from fastsafetensors.frameworks._torch import dtype_convert

    native = {}

    def native_shape(framework, dtype, shape):
        key = (dtype, tuple(shape))
        if key not in native:
            native[key] = tuple(framework.get_native_shape(dtype, list(shape)))
        return native[key]

    def bulk_tensors(fb):
        """Per chunk: one broadcast; per tensor: a fresh torch.empty (exactly framework.get_empty_tensor's
        allocation) filled by one batched copy (torch._foreach_copy_) per chunk."""
        pg = fb.pg
        real = getattr(pg, "real_pg", None)
        framework = fb.framework
        for rank in sorted(fb.rank_loaders):
            src = dist.get_global_rank(real, rank) if real is not dist.group.WORLD else rank
            for lidx, factory in enumerate(fb.rank_loaders[rank]):
                frames = [
                    (name, frame)
                    for name, frame in factory.metadata.tensors.items()  # data-offset order
                    if fb.key_to_rank_lidx.get(name) == (rank, lidx)
                ]
                owner = pg.rank() == rank
                device = factory.device.as_str()
                i = 0
                while i < len(frames):
                    group, size = [], 0  # consecutive tensors, 256-byte aligned slots, <= CHUNK (or one big tensor)
                    while i < len(frames):
                        name, frame = frames[i]
                        s, e = frame.data_offsets
                        n = e - s
                        slot = (size + 255) // 256 * 256
                        if group and slot + n > CHUNK:
                            break
                        group.append((name, frame, slot, n))
                        size = slot + n
                        i += 1
                    buf = buffer(size, device)
                    views = [buf[slot : slot + n] for _, _, slot, n in group if n]
                    if owner and views:
                        srcs = [
                            factory.tensors[name].get_raw().reshape(-1).view(torch.uint8)
                            for name, _, _, n in group
                            if n
                        ]
                        torch._foreach_copy_(views, srcs)
                    dist.broadcast(buf, src, group=real)
                    out, dsts, outs = [], [], []
                    for name, frame, slot, n in group:
                        dst = torch.empty(
                            native_shape(framework, frame.dtype, frame.shape),
                            dtype=dtype_convert[frame.dtype],
                            device=device,
                        )
                        out.append((name, dst))
                        if n:
                            b = dst.reshape(-1).view(torch.uint8)
                            if b.numel() != n:
                                raise RuntimeError(f"K3_FST_BULK: {name}: {b.numel()} B vs file {n} B")
                            outs.append(b)
                    if outs:
                        torch._foreach_copy_(outs, views)
                    yield from out
                # The stock path frees the owner's file buffer once every tensor was instantiated.
                if owner:
                    torch.cuda.current_stream().synchronize()
                    factory.free_dev_ptrs()

    def _consume_single_batch(self):
        fb_pg = None
        with TimingContext("wait_queue", self._log_message) as timer:
            batch_item = self.batch_queue.get()
            if self.queue_size == 0 and self.consumer_processed is not None:
                self.consumer_processed.set()
            if batch_item is None:
                self._log_error("get batch_item is None, will break")
                return
            if isinstance(batch_item, Exception):
                self._log_error("get batch_item is Exception, will raise")
                raise batch_item
            batch = batch_item
            timer.batch_id = batch.batch_id
            fb_pg = batch.fb.pg
        if fb_pg.size() <= 1 or getattr(fb_pg, "real_pg", None) is None:
            raise RuntimeError("K3_FST_BULK needs a multi-rank torch process group")
        try:
            yield from bulk_tensors(batch.fb)
        finally:
            batch.fb.close()
        if self.queue_size < 0 and self.consumer_processed is not None:
            self.consumer_processed.set()

    PipelineParallel._consume_single_batch = _consume_single_batch
    PipelineParallel._k3_stock_consume = stock
    # The 1 GiB broadcast buffer must not outlive the load: vLLM counts live allocations after load_model as model
    # memory (startup-g4: 131.5 vs 129.31 GiB), which shrinks the KV cache. Free it when the loader closes.
    Loader = parallel_loader.ParallelLoader
    stock_close = Loader.close

    def close(self, *args, **kwargs):
        try:
            return stock_close(self, *args, **kwargs)
        finally:
            scratch.clear()
            native.clear()

    Loader.close = close
    log("K3_FST_BULK: fastsafetensors consumer broadcasts files in 1 GiB chunks")


# ---------------------------------------------------------------------------------------------------------------
# K3_WEIGHT_DIGEST
# ---------------------------------------------------------------------------------------------------------------
class Digest:
    """Order-sensitive digest of (name, bytes) items, accumulated on the GPU (int64 wrap-around)."""

    def __init__(self, device):
        import torch

        self.torch = torch
        g = torch.Generator(device="cpu").manual_seed(0x4B33)
        w = torch.randint(-(2**62), 2**62, (1 << 22,), generator=g, dtype=torch.int64) | 1
        self.w = w.to(device)
        self.h = torch.zeros((), dtype=torch.int64, device=device)
        self.items = 0
        self.nbytes = 0

    def tensor_hash(self, t):
        torch = self.torch
        b = t.detach().reshape(-1).view(torch.uint8)
        n = b.numel()
        if n % 8:
            b = torch.cat([b, b.new_zeros(8 - n % 8)])
        x = b.view(torch.int64)
        h = torch.zeros((), dtype=torch.int64, device=x.device)
        L = self.w.numel()
        for j in range(0, x.numel(), L):
            part = x[j : j + L]
            h = h * _P + (part * self.w[: part.numel()]).sum()
        return h + n

    def add(self, name, t):
        self.h = self.h * _P + self.tensor_hash(t) + zlib.crc32(name.encode())
        self.items += 1
        self.nbytes += t.numel() * t.element_size()

    def value(self):
        return f"{int(self.h.item()) & 0xFFFFFFFFFFFFFFFF:016x}"


def _rank():
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else -1
    except Exception:
        return -1


def patch_default_loader(default_loader):
    stock = default_loader.fastsafetensors_weights_iterator

    def fastsafetensors_weights_iterator(*args, **kwargs):
        import torch

        digest, start = None, time.perf_counter()
        for name, t in stock(*args, **kwargs):
            if digest is None:
                digest = Digest(t.device)
            digest.add(name, t)
            yield name, t
        if digest is not None:
            log(f"rank={_rank()} iterator tensors={digest.items} bytes={digest.nbytes} digest={digest.value()} "
                f"({time.perf_counter() - start:.1f}s incl. load)")

    default_loader.fastsafetensors_weights_iterator = fastsafetensors_weights_iterator


def patch_model_runner(model_runner):
    Runner = model_runner.GPUModelRunner
    stock = Runner.load_model

    def load_model(self, *args, **kwargs):
        out = stock(self, *args, **kwargs)
        start = time.perf_counter()
        items = list(self.model.named_parameters(remove_duplicate=False)) + list(
            self.model.named_buffers(remove_duplicate=False)
        )
        items.sort(key=lambda kv: kv[0])
        digest = Digest(next(t.device for _, t in items if t.is_cuda))
        for name, t in items:
            if t.is_cuda:
                digest.add(name, t.contiguous())
        log(f"rank={_rank()} params tensors={digest.items} bytes={digest.nbytes} digest={digest.value()} "
            f"({time.perf_counter() - start:.1f}s)")
        return out

    Runner.load_model = load_model


# ---------------------------------------------------------------------------------------------------------------
# K3_WSNAP_DIR: per-hold host-RAM weight snapshot
# ---------------------------------------------------------------------------------------------------------------
# K3_WSNAP_DIR=<node-local dir, e.g. /dev/shm/$USER-k3-wsnap/<alloc>> with K3_WSNAP_MODEL=<target model path>:
# DefaultModelLoader.load_weights of the target either
#   (restore) copies every tensor that a stock load_weights wrote from this rank's snapshot file (~33 GB/s per GPU
#             from /dev/shm), when ALL ranks hold a valid snapshot for the same key (one all-reduce MIN), or
#   (dump)    runs the stock load_weights and then writes the tensors it changed (per-tensor GPU digests before and
#             after) to <dir>/rank<R>.<key>.{bin,json}.
# process_weights_after_loading and everything after it run unchanged in both cases: only the bytes load_weights
# would have written come from the snapshot, into the same tensors model init allocated.
# The dump is refused (stock load, nothing written, a "[k3-wsnap] refuse" line) when
#   - free space in the snapshot filesystem < 1.1 x (local ranks x this rank's bytes), or MemAvailable is short;
#   - load_weights replaced or added a parameter / buffer / plain tensor attribute, or changed a simple module
#     attribute (int/float/bool/str/None/tuple): a byte snapshot could not reproduce that.
# Key: model path, rank, world size, the param/buffer manifest (names, shapes, dtypes, strides), K3_WSNAP_KEY
# (serve.sh: weight-relevant server args and K3_EXTRA_ENV without per-run output knobs) and the content of every
# vllm/models/kimi_k3 .py file outside nvidia/specialized_decode/ (built after load_model; it never touches
# load_weights) plus vllm/torch versions.
# Dumping a new key first deletes this rank's other snapshots and every other hold's directory next to <dir>.
def patch_wsnap(default_loader):
    import hashlib
    import json
    import logging
    import mmap

    import torch
    import torch.distributed as dist

    Loader = default_loader.DefaultModelLoader
    stock = Loader.load_weights
    root = os.environ["K3_WSNAP_DIR"].rstrip("/")
    target = os.environ.get("K3_WSNAP_MODEL", "")
    vlog = logging.getLogger(default_loader.__name__)
    ALIGN = 4096

    def wlog(msg):
        print(f"[k3-wsnap] {os.uname().nodename.split('.')[0]} pid={os.getpid()} rank={_rank()} {msg}",
              file=sys.stderr, flush=True)

    def tensors(model):
        out = {}
        for n, t in model.named_parameters(remove_duplicate=True):
            out[n] = t
        for n, t in model.named_buffers(remove_duplicate=True):
            out.setdefault(n, t)
        return out

    def manifest(ts):
        h = hashlib.sha256()
        for n, t in ts.items():
            h.update(f"{n}|{tuple(t.shape)}|{t.dtype}|{tuple(t.stride())}|{t.device.type};".encode())
        return h.hexdigest()

    SIMPLE = (int, float, bool, str, type(None))

    def module_state(model):
        """Simple attributes, and plain tensor attributes (not parameters / buffers) by identity, per module."""
        simple, plain = {}, {}
        for mn, m in model.named_modules():
            for k, v in vars(m).items():
                if isinstance(v, SIMPLE) or (isinstance(v, tuple) and all(isinstance(x, SIMPLE) for x in v)):
                    simple[f"{mn}.{k}"] = repr(v)
                elif isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter):
                    plain[f"{mn}.{k}"] = (id(v), v.data_ptr(), tuple(v.shape), str(v.dtype))
        return simple, plain

    code = {}

    def code_hash():
        if "h" not in code:
            import importlib.util

            h = hashlib.sha256()
            spec = importlib.util.find_spec("vllm.models.kimi_k3")
            base = list(spec.submodule_search_locations)[0]
            for dirpath, dirnames, files in os.walk(base):
                dirnames.sort()
                rel = os.path.relpath(dirpath, base)
                if rel.replace(os.sep, "/").startswith("nvidia/specialized_decode"):
                    continue
                for f in sorted(files):
                    if f.endswith(".py"):
                        with open(os.path.join(dirpath, f), "rb") as fh:
                            h.update(f"{rel}/{f}:".encode() + hashlib.sha256(fh.read()).digest())
            import vllm

            h.update(f"vllm={getattr(vllm, '__version__', '?')} torch={torch.__version__}".encode())
            code["h"] = h.hexdigest()
        return code["h"]

    def key_of(model_config, ts):
        h = hashlib.sha256()
        world = dist.get_world_size() if dist.is_initialized() else 1
        for part in (str(model_config.model), str(_rank()), str(world), manifest(ts),
                     os.environ.get("K3_WSNAP_KEY", ""), code_hash()):
            h.update(part.encode() + b"\0")
        return h.hexdigest()[:20]

    def paths(key):
        r = _rank()
        return os.path.join(root, f"rank{r}.{key}.bin"), os.path.join(root, f"rank{r}.{key}.json")

    def valid(key):
        b, j = paths(key)
        try:
            idx = json.load(open(j))
            return idx.get("key") == key and os.path.getsize(b) == idx["size"]
        except (OSError, ValueError, KeyError):
            return False

    def all_ranks(flag):
        if not dist.is_initialized():
            return flag
        t = torch.tensor([1 if flag else 0], dtype=torch.int32, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def u8(t):
        return t.detach().reshape(-1).view(torch.uint8)

    def restore(ts, key):
        b, j = paths(key)
        idx = json.load(open(j))
        start = time.perf_counter()
        fd = os.open(b, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, idx["size"], prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer: we only read from it
            host = torch.frombuffer(mm, dtype=torch.uint8)
        nbytes = 0
        for name, off, n in idx["items"]:
            t = ts[name].data
            src = host[off: off + n]
            if t.is_contiguous():
                u8(t).copy_(src)
            else:
                tmp = torch.empty(t.shape, dtype=t.dtype, device=t.device)
                u8(tmp).copy_(src)
                t.copy_(tmp)
            nbytes += n
        torch.cuda.synchronize()
        del host
        mm.close()
        return len(idx["items"]), nbytes, time.perf_counter() - start

    def changed_names(ts, before, cpu_before):
        after = {n: Digest.tensor_hash(DIG[0], t) for n, t in ts.items() if t.is_cuda}
        names = [n for n in after if n not in before or bool((after[n] != before[n]).item())]
        names += [n for n, old in cpu_before.items() if not torch.equal(old, ts[n].detach())]
        return names

    DIG = [None]

    def space_ok(total):
        need = 1.1 * total * max(1, torch.cuda.device_count())
        st = os.statvfs(root)
        free = st.f_bavail * st.f_frsize
        avail = 0
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        except OSError:
            pass
        ok = free >= need and avail >= need + (64 << 30)
        return ok, f"need {need / 2**30:.0f} GiB (x{torch.cuda.device_count()} ranks +10%), free {free / 2**30:.0f} GiB, " \
                   f"MemAvailable {avail / 2**30:.0f} GiB"

    def cleanup_other(key):
        parent = os.path.dirname(root)
        mine = os.path.basename(root)
        import shutil

        try:
            for d in os.listdir(parent):  # other holds' snapshots on this node (ours are exclusive per node)
                if d != mine and os.path.isdir(os.path.join(parent, d)):
                    shutil.rmtree(os.path.join(parent, d), ignore_errors=True)
            r = _rank()
            for f in os.listdir(root):  # this rank's other keys
                if f.startswith(f"rank{r}.") and f".{key}." not in f:
                    os.unlink(os.path.join(root, f))
        except OSError:
            pass

    def dump(ts, key, names):
        b, j = paths(key)
        items, off = [], 0
        for n in names:
            t = ts[n]
            nb = t.numel() * t.element_size()
            off = (off + ALIGN - 1) // ALIGN * ALIGN
            items.append((n, off, nb))
            off += nb
        total = max(off, 1)
        cleanup_other(key)
        ok, why = space_ok(total)
        if not ok:
            wlog(f"refuse dump: {why}")
            return
        start = time.perf_counter()
        tmp = f"{b}.tmp.{os.getpid()}"
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            try:  # reserve the pages now: a full tmpfs or user quota must fail here, not SIGBUS on an mmap store
                os.posix_fallocate(fd, 0, total)
            except OSError as e:
                os.close(fd)
                fd = -1
                os.unlink(tmp)
                wlog(f"refuse dump: cannot reserve {total / 2**30:.1f} GiB in {root}: {e!r}")
                return
            mm = mmap.mmap(fd, total)
        finally:
            if fd >= 0:
                os.close(fd)
        host = torch.frombuffer(mm, dtype=torch.uint8)
        for n, o, nb in items:
            t = ts[n].data
            host[o: o + nb].copy_(u8(t if t.is_contiguous() else t.contiguous()))
        torch.cuda.synchronize()
        del host
        mm.close()
        os.replace(tmp, b)
        with open(f"{j}.tmp.{os.getpid()}", "w") as f:
            json.dump(dict(key=key, size=total, items=items, rank=_rank(), host=os.uname().nodename,
                           created=time.time(), model=target), f)
        os.replace(f"{j}.tmp.{os.getpid()}", j)  # the index last: it marks the snapshot complete
        wlog(f"dumped {len(items)} tensors {total / 2**30:.1f} GiB in {time.perf_counter() - start:.1f}s -> {b}")

    last = os.environ.get("K3_WSNAP_LAST", "0") == "1"

    def drop_mine():
        r = _rank()
        try:
            for f in os.listdir(root):
                if f.startswith(f"rank{r}."):
                    os.unlink(os.path.join(root, f))
        except OSError:
            pass

    def load_weights(self, model, model_config):
        if not target or os.path.realpath(str(model_config.model)) != os.path.realpath(target):
            return stock(self, model, model_config)
        os.makedirs(root, exist_ok=True)
        ts = tensors(model)
        key = key_of(model_config, ts)
        if _rank() == 0:  # which part of the key differs between servers that do not share a snapshot
            params = {n for n, _ in model.named_parameters(remove_duplicate=True)}
            wlog(f"key {key}: params {manifest({n: t for n, t in ts.items() if n in params})[:8]} buffers "
                 f"{manifest({n: t for n, t in ts.items() if n not in params})[:8]} ({len(ts) - len(params)}) env "
                 f"{hashlib.sha256(os.environ.get('K3_WSNAP_KEY', '').encode()).hexdigest()[:8]} code {code_hash()[:8]}")
        if all_ranks(valid(key)):
            self._init_ep_weight_filter(model_config)
            self._init_mm_encoder_only_weight_filter(model, model_config)
            self.counter_before_loading_weights = time.perf_counter()
            n, nbytes, sec = restore(ts, key)
            self.counter_after_loading_weights = time.perf_counter()
            wlog(f"restored {n} tensors {nbytes / 2**30:.1f} GiB in {sec:.1f}s (key {key})")
            vlog.info("Loading weights took %.2f seconds (k3 snapshot restore)",
                      self.counter_after_loading_weights - self.counter_before_loading_weights)
            if last:  # the hold's last server: free the node's RAM now
                drop_mine()
                wlog("K3_WSNAP_LAST: deleted this rank's snapshot after the restore")
            return
        if last:  # no later server on this hold could use a new snapshot
            drop_mine()
            return stock(self, model, model_config)
        # Dump launch: stock load, bracketed by per-tensor digests and the module-state guard.
        DIG[0] = Digest(torch.device("cuda", torch.cuda.current_device()))
        before = {n: Digest.tensor_hash(DIG[0], t) for n, t in ts.items() if t.is_cuda}
        cpu = {n: t for n, t in ts.items() if t.device.type == "cpu"}
        big_cpu = [n for n, t in cpu.items() if t.numel() * t.element_size() > (1 << 30)]
        other = [n for n, t in ts.items() if t.device.type not in ("cuda", "cpu", "meta")]  # meta holds no data
        cpu_before = {n: t.detach().clone() for n, t in cpu.items()} if not big_cpu else {}
        simple0, plain0 = module_state(model)
        out = stock(self, model, model_config)
        ts1 = tensors(model)
        simple1, plain1 = module_state(model)
        why = None
        if list(ts1) != list(ts) or any(ts1[n] is not ts[n] or ts1[n].data_ptr() != ts[n].data_ptr() for n in ts):
            why = "load_weights replaced or added parameters / buffers"
        elif plain1 != plain0:
            why = f"load_weights changed plain tensor attributes: {sorted(set(plain1) ^ set(plain0))[:5]}"
        elif simple1 != simple0:
            diff = [k for k in set(simple0) | set(simple1) if simple0.get(k) != simple1.get(k)]
            why = f"load_weights changed module attributes: {sorted(diff)[:5]}"
        elif big_cpu or other:
            why = f"tensors the snapshot does not handle: CPU > 1 GiB {big_cpu[:3]}, other devices {other[:3]}"
        if why:
            wlog(f"refuse dump: {why}")
            return out
        names = changed_names(ts, before, cpu_before)
        DIG[0] = None
        dump(ts, key, names)
        return out

    Loader.load_weights = load_weights
    wlog(f"K3_WSNAP: target {target} snapshots under {root}")


# ---------------------------------------------------------------------------------------------------------------
# K3_KVPLAN_KEY: vLLM's startup plan (VLLM_ENABLE_STARTUP_PLAN=1), keyed by our code too
# ---------------------------------------------------------------------------------------------------------------
def _kvplan_disabled(root):
    return os.path.exists(os.path.join(root, "DISABLE"))


def patch_startup_plan(startup_plan):
    root = os.environ.get("K3_KVPLAN_DIR", "")
    key = os.environ.get("K3_KVPLAN_KEY", "")
    if not root or not key:
        return
    stock = startup_plan._plan_path
    host = os.uname().nodename.split('.')[0]

    def _plan_path(fingerprint):
        sub = os.path.join("disabled", f"{host}.{os.getpid()}") if _kvplan_disabled(root) else "profiled"
        return os.path.join(root, key, sub, os.path.basename(stock(fingerprint)))

    startup_plan._plan_path = _plan_path


def _kvlog(rank, msg):
    print(f"[k3-kvplan] {os.uname().nodename.split('.')[0]} pid={os.getpid()} rank={rank} {msg}", file=sys.stderr,
          flush=True)


def _world_sum(vals):
    """(element-wise sum of vals over the vLLM world group, world size); no collective for a single rank."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return list(vals), 1
    from vllm.distributed.parallel_state import get_world_group

    g = get_world_group()
    if g.world_size == 1:
        return list(vals), 1
    t = torch.tensor(list(vals), dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=g.cpu_group)
    return [int(x) for x in t.tolist()], g.world_size


def patch_gpu_worker_plan(gpu_worker):
    stock = gpu_worker.maybe_save_startup_plan
    stock_apply = gpu_worker.maybe_apply_startup_plan

    def maybe_save_startup_plan(worker, kv_cache_memory_bytes):
        profiled = getattr(worker, "available_kv_cache_memory_bytes", None)
        if isinstance(profiled, (int, float)) and profiled > 0:
            kv_cache_memory_bytes = int(profiled)
        return stock(worker, kv_cache_memory_bytes)

    def maybe_apply_startup_plan(worker):
        import vllm.envs as envs

        cc = worker.cache_config
        explicit = cc.kv_cache_memory_bytes is not None
        stock_apply(worker)
        if not envs.VLLM_ENABLE_STARTUP_PLAN:
            return
        skip = cc.kv_cache_memory_bytes is not None
        (n_skip, n_explicit), world = _world_sum((int(skip), int(explicit)))
        rank = getattr(worker, "rank", -1)
        if n_skip in (0, world):
            if rank == 0:
                _kvlog(rank, f"ranks agree: {n_skip}/{world} skip memory profiling ({n_explicit} explicit)"
                             f" -> {'every rank skips it' if n_skip else 'every rank profiles'}")
            return
        if n_explicit:
            raise RuntimeError(f"[k3-kvplan] {n_skip}/{world} ranks skip memory profiling ({n_explicit} with an explicit "
                               "--kv-cache-memory-bytes), the rest profile: the ranks would hang; refusing to start")
        if skip:
            cc.kv_cache_memory_bytes = None
        _kvlog(rank, f"ranks disagree: {n_skip}/{world} have an applicable plan"
                     f"{' (this one, dropped)' if skip else ''} -> every rank profiles")

    gpu_worker.maybe_save_startup_plan = maybe_save_startup_plan
    if os.environ.get("K3_KVPLAN_AGREE", "1") != "0":
        gpu_worker.maybe_apply_startup_plan = maybe_apply_startup_plan


def targets():
    """{module name: patch function} for the knobs set in this process's environment."""
    t = {}
    if os.environ.get("K3_KVPLAN_KEY"):
        t["vllm.v1.worker.startup_plan"] = patch_startup_plan
        t["vllm.v1.worker.gpu_worker"] = patch_gpu_worker_plan
    if os.environ.get("K3_FST_BULK", "0") == "1":
        t["fastsafetensors.parallel_loader"] = patch_fastsafetensors
    digest = os.environ.get("K3_WEIGHT_DIGEST", "")
    loader_patches = []
    if os.environ.get("K3_WSNAP_DIR"):
        loader_patches.append(patch_wsnap)
    if digest in ("all", "1"):
        loader_patches.append(patch_default_loader)
    if loader_patches:
        def patch_loader(module):
            for p in loader_patches:
                p(module)

        t["vllm.model_executor.model_loader.default_loader"] = patch_loader
    if digest in ("all", "1", "params"):
        t["vllm.v1.worker.gpu.model_runner"] = patch_model_runner
    return t
