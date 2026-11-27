# Kimi K3 serving hook, active only when scripts/serve.sh puts this directory on
# PYTHONPATH (K3_CUTE_CACHE=<dir>): a persistent CuTe DSL compile cache for
# cute.compile().
#
# Why: cute.compile() always passes no_cache=True / compile_only=True, so the DSL's
# own module-hash file cache is never used for our kernels and every server start
# re-lowers (MLIR pipeline, PTX, ptxas) every kernel in every worker.
#
# What: for cute.compile() calls only, this re-enables that cache under the DSL's own
# key, BaseDSL.get_module_hash(): sha256 over the traced module's bytecode, the
# compile-affecting CUTE_DSL_* settings, the compile options and
# CutlassBaseDSL.get_version() (the DSL's python sources and IR library). The entries
# live in $K3_CUTE_CACHE/cute/<toolchain salt>/. Tracing still runs on every start (the
# key is the traced IR); a hit skips lowering and builds a fresh executor from the
# cached lowered module, exactly the DSL's file-cache hit path. Every hit reloads the
# file, so no compiled object (or CUfunction, e.g. a carveout attribute) is shared
# between cute.compile() calls. KEEP=ptx/cubin/sass and CUTE_DSL_NO_CACHE /
# DISABLE_FILE_CACHING leave the compile uncached, as the DSL does.
#
# Determinism check: a fresh compile whose key already has an entry (written by another
# rank compiling the same IR at the same time) is byte-compared with that entry, and
# the result is counted; a mismatch is logged with the function name and key.
#
# K3_CUTE_CACHE_VERIFY=1 (tests): every hit also compiles fresh, uncached, and
# byte-compares that module with the cached entry (counted as vsame / vdiff); the hit is
# still what the caller gets.
#
#
# Stats: one "[k3-cute-cache]" line per process on stderr, 20 s after its last compile
# (and every 200 compiles): calls, hits, misses, entries written, byte-identical /
# different, and the seconds spent tracing vs in the backend (lowering or loading).
import os
import sys


_HOOK_MARK = "# Kimi K3 serving hook, active only when scripts/serve.sh puts this directory on"


def _chain_next_sitecustomize():
    """Run the next sitecustomize on sys.path that has not run yet (the distribution's, or another hook's), tracked in
    sys._k3_sitecustomize_origins as scripts/infra3/hook and scripts/rfvr/hook do (so none runs twice). Other copies of
    this hook on the path (e.g. the live one behind a staged copy put first on PYTHONPATH) count as run: only the first
    copy patches."""
    import importlib.machinery
    import importlib.util

    seen = sys.__dict__.setdefault("_k3_sitecustomize_origins", set())
    seen.add(os.path.abspath(__file__))
    first = getattr(sys.modules.get("sitecustomize"), "__file__", None)
    if first:
        seen.add(os.path.abspath(first))
    specs = []
    for entry in list(sys.path):
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry or os.getcwd()])
        if spec is not None and spec.origin is not None:
            specs.append(spec)
    for spec in specs:  # mark every copy of this hook first, so a hook chained below never runs one
        try:
            with open(spec.origin) as f:
                if f.readline().rstrip("\n") == _HOOK_MARK:
                    seen.add(os.path.abspath(spec.origin))
        except OSError:
            pass
    for spec in specs:
        origin = os.path.abspath(spec.origin)
        if origin in seen:
            continue
        seen.add(origin)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


_ROOT = os.environ.get("K3_CUTE_CACHE", "")


def _install(targets):
    """Run ``targets[name](module)`` right after ``name`` is first imported (a patch that fails is logged and
    skipped: the module stays stock)."""
    import importlib.abc

    pending = dict(targets)

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner, patch):
            self._inner, self._patch = inner, patch

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                self._patch(module)
            except Exception as e:  # never break the import; the module stays stock
                print(f"[k3-hook] patch of {module.__name__} failed, left stock: {e!r}", file=sys.stderr, flush=True)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in pending:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None:
                        spec.loader = _Loader(spec.loader, pending.pop(fullname))
                    if not pending:
                        sys.meta_path.remove(self)
                    return spec
            return None

    if pending:
        sys.meta_path.insert(0, _Finder())


def _salt():
    """Toolchain identity on top of the DSL's key: the DSL version and its libraries."""
    import hashlib

    import cutlass

    h = hashlib.sha256()
    h.update(str(getattr(cutlass, "__version__", "?")).encode())
    base = os.path.dirname(os.path.dirname(os.path.abspath(cutlass.__file__)))
    for dirpath, _, files in sorted(os.walk(base)):
        for f in sorted(files):
            if ".so" in f:
                st = os.stat(os.path.join(dirpath, f))
                h.update(f"{dirpath}/{f}:{st.st_size}:{int(st.st_mtime)};".encode())
    return h.hexdigest()[:16]


def _patch(dsl_module):
    import io
    import json
    import threading
    import time
    import uuid

    from cutlass.base_dsl import compile_backend, compiler
    from cutlass.base_dsl.cache_helpers import write_bytecode_with_crc32

    BaseDSL = dsl_module.BaseDSL
    state = threading.local()
    stats = dict(calls=0, hits=0, misses=0, written=0, same=0, diff=0, uncached=0,
                 fallback=0, vsame=0, vdiff=0, vequiv=0, sequiv=0, t_norm=0.0, t_total=0.0, t_backend=0.0,
                 t_hit=0.0, t_miss=0.0, t_verify=0.0)
    verify = os.environ.get("K3_CUTE_CACHE_VERIFY", "0") == "1"
    tcache = {}
    ocache = {}
    lock = threading.Lock()
    timer = [None]
    host = os.uname().nodename
    cache_dir = [None]

    def directory():
        if cache_dir[0] is None:
            d = os.path.join(_ROOT, "cute", _salt())
            os.makedirs(d, exist_ok=True)
            cache_dir[0] = d
        return cache_dir[0]

    def report():
        s = stats
        veq = f" vequiv={s['vequiv']}" if s.get("vequiv") else ""
        print(
            f"[k3-cute-cache] {host} pid={os.getpid()} calls={s['calls']} hits={s['hits']} "
            f"misses={s['misses']} written={s['written']} same={s['same']} diff={s['diff']} "
            f"uncached={s['uncached']} fallback={s['fallback']} vsame={s['vsame']} "
            f"vdiff={s['vdiff']}{veq} total={s['t_total'] - s['t_verify']:.1f}s "
            f"trace={s['t_total'] - s['t_backend']:.1f}s backend={s['t_backend']:.1f}s "
            f"(hit {s['t_hit']:.1f}s, miss {s['t_miss']:.1f}s) dir={cache_dir[0]}"
            + (f" {tcache['report']()}" if tcache else "")
            + (f" {ocache['report']()}" if ocache else "")
            + (f" pathnorm[{pathnorm}] {s.get('t_norm', 0.0):.1f}s same-but-locations={s.get('sequiv', 0)}"
               if pathnorm in ("1", "text") else ""),
            file=sys.stderr,
            flush=True,
        )

    def schedule_report():
        if timer[0] is not None:
            timer[0].cancel()
        t = threading.Timer(20.0, report)
        t.daemon = True
        timer[0] = t
        t.start()

    def active(dsl):
        env = dsl.envar
        return not (env.keep_ptx or env.keep_cubin or env.keep_sass or env.no_cache
                    or env.disable_file_caching)

    # K3_CUTE_CACHE_LOG=<dir>: one JSON line per cute.compile() call in <dir>/<host>-<pid>.jsonl (family, call site,
    # serving call chain, trace / backend seconds, hit or miss, module key). Off when unset.
    log_dir = os.environ.get("K3_CUTE_CACHE_LOG", "")
    log_file = [None]

    def family(func):
        import inspect

        try:
            if inspect.ismethod(func):
                c = type(func.__self__)
                return f"{c.__module__}.{c.__qualname__}.{func.__name__}"
            if inspect.isfunction(func):
                return f"{func.__module__}.{func.__qualname__}"
            c = type(func)
            return f"{c.__module__}.{c.__qualname__}"
        except Exception:
            return "?"

    def short(path):
        for mark in ("vllm/models/", "vllm/", "scripts/"):
            i = path.rfind(mark)
            if i >= 0:
                return path[i + len(mark):]
        return path

    def call_site():
        f = sys._getframe(2)
        site, chain = None, []
        while f is not None:
            name = f.f_code.co_filename
            if "/cutlass/" not in name and "sitecustomize" not in name:
                if site is None:
                    site = f"{short(name)}:{f.f_lineno}"
                elif ("kimi_k3" in name or "/v1/" in name) and len(chain) < 8:
                    chain.append(f.f_code.co_name)
            f = f.f_back
        return site, chain

    def call_log(rec):
        try:
            if log_file[0] is None:
                os.makedirs(log_dir, exist_ok=True)
                log_file[0] = open(os.path.join(log_dir, f"{host.split('.')[0]}-{os.getpid()}.jsonl"), "a",
                                   buffering=1)
            log_file[0].write(json.dumps(rec) + "\n")
        except OSError:
            pass

    # 1. cute.compile(): time the call; mark it as ours for the backend hooks below.
    orig_compile = compiler.CompileCallable._compile

    def _compile(self, *args, **kwargs):
        start = time.perf_counter()
        state.ours = True
        state.backend = 0.0
        state.key = None
        state.hit = None
        wall = time.time()
        state.skip_hash = None
        state.tc = None
        ok = False
        try:
            out = orig_compile(self, *args, **kwargs)
            ok = True
            return out
        finally:
            state.ours = False
            if tcache:
                if ok:
                    tcache["finish"](state.key)
                else:
                    state.tc = state.skip_hash = None
            elapsed = time.perf_counter() - start
            with lock:
                stats["calls"] += 1
                stats["t_total"] += elapsed
                stats["t_backend"] += state.backend
                if stats["calls"] % 200 == 0:
                    report()
            schedule_report()
            if log_dir:
                site, chain = call_site()
                call_log(dict(t=round(wall, 3), fn=family(args[0]) if args else "?", site=site, chain=chain,
                              trace=round(elapsed - state.backend, 4), backend=round(state.backend, 4),
                              hit=state.hit, key=(state.key or "")[:16], rank=os.environ.get("RANK")))

    compiler.CompileCallable._compile = _compile
    import atexit

    atexit.register(lambda: stats["calls"] and report())

    # 2. compile_and_cache(): key the compile and route it through the DSL's file cache.
    orig_cac = BaseDSL.compile_and_cache

    def compile_and_cache(self, module, module_hash, function_name, pipeline, sig,
                          no_cache, no_jit_engine, *args, **kwargs):
        ours = getattr(state, "ours", False) and no_cache and module_hash is None
        if not ours or not active(self):
            if ours:
                stats["uncached"] += 1
            return orig_cac(self, module, module_hash, function_name, pipeline, sig,
                            no_cache, no_jit_engine, *args, **kwargs)
        start = time.perf_counter()
        skip = getattr(state, "skip_hash", None)
        key = skip or module_key(self, module, function_name)
        state.key = key
        state.cached = True
        state.function = function_name
        state.hit = False
        # Only file entries serve hits: never the in-memory cache (shared objects).
        self.jit_cache.delete(key)
        try:
            fn = orig_cac(self, module, key, function_name, pipeline, sig, False,
                          no_jit_engine, *args, **kwargs)
        except Exception as e:
            if not state.hit or skip:
                raise
            # A cached entry failed to build: compile it fresh, uncached (today's path).
            print(f"[k3-cute-cache] hit failed for {function_name} {key}: {e!r}; "
                  "compiling fresh", file=sys.stderr, flush=True)
            stats["fallback"] += 1
            state.cached = False
            fn = None
        finally:
            self.jit_cache.delete(key)
        if skip and not state.hit:  # the stand-in module of a skipped trace must never be lowered
            raise RuntimeError(f"[k3-trace] skipped trace of {function_name}: lowered module {key} not loadable")
        if fn is None:
            fn = orig_cac(self, module, None, function_name, pipeline, sig, True,
                          no_jit_engine, *args, **kwargs)
        state.cached = False
        elapsed = time.perf_counter() - start
        if verify and state.hit and not skip:  # (a skipped trace has no traced module to lower)
            # The traced module is untouched by the hit: lower it fresh and compare. Bytes that differ but print
            # the same without debug info (another overlay's source locations: K3_CUTE_PATHNORM) count as vequiv.
            vstart = time.perf_counter()
            fresh = orig_cac(self, module, None, function_name, pipeline, sig, True,
                             no_jit_engine, *args, **kwargs)
            ok = module_bytes(fresh) == entry_bytes(self.name, key)
            equiv = not ok and nodebug_text(fresh.ir_module) == entry_text(self.name, key)
            with lock:
                stats["vsame" if ok else "vequiv" if equiv else "vdiff"] += 1
                stats["t_verify"] += time.perf_counter() - vstart
            state.backend += time.perf_counter() - vstart
            if not ok and not equiv:
                print(f"[k3-cute-cache] VERIFY DIFF {function_name} key={key}",
                      file=sys.stderr, flush=True)
        state.backend += elapsed
        with lock:
            stats["hits" if state.hit else "misses"] += 1
            stats["t_hit" if state.hit else "t_miss"] += elapsed
        return fn

    BaseDSL.compile_and_cache = compile_and_cache

    pathnorm = os.environ.get("K3_CUTE_PATHNORM", "0")

    def module_key(dsl, module, function_name):
        """K3_CUTE_PATHNORM: the DSL's key over the module without source locations (=1: a clone stripped by
        strip-debuginfo, written as bytecode; =text: printed without debug info). Anything else: the DSL's key."""
        if pathnorm not in ("1", "text"):
            return dsl.get_module_hash(module, function_name)
        start = time.perf_counter()
        if pathnorm == "text":
            data = module.operation.get_asm(enable_debug_info=False).encode()
        else:
            from cutlass._mlir import passmanager

            clone = module.operation.clone()
            try:
                passmanager.PassManager.parse("builtin.module(strip-debuginfo)", context=module.context).run(clone)
                buf = io.BytesIO()
                clone.write_bytecode(buf)
                data = buf.getvalue()
            finally:
                clone.erase()
        h = dsl.get_version().copy()
        h.update(b"k3-pathnorm-v1" + data + dsl.envar.cache_key_str().encode() + dsl.compile_options.to_str().encode())
        stats["t_norm"] = stats.get("t_norm", 0.0) + time.perf_counter() - start
        return h.hexdigest()

    def module_bytes(fn):
        buffer = io.BytesIO()
        write_bytecode_with_crc32(buffer, fn.ir_module)
        return buffer.getvalue()

    def entry_bytes(dsl_name, key):
        with open(os.path.join(directory(), f"{dsl_name.lower()}_{key}.mlir"), "rb") as f:
            return f.read()

    def nodebug_text(module):
        return module.operation.get_asm(enable_debug_info=False)

    def entry_text(dsl_name, key):
        from cutlass._mlir import ir
        from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32

        try:
            with ir.Context() as ctx:
                ctx.allow_unregistered_dialects = True
                with open(os.path.join(directory(), f"{dsl_name.lower()}_{key}.mlir"), "rb") as f:
                    return nodebug_text(read_bytecode_and_check_crc32(f))
        except Exception as e:  # noqa: BLE001
            return f"<unreadable {e!r}>"

    # 3. The backend's file load / dump, redirected to our directory for our compiles.
    orig_load = compile_backend.load_cache_from_path
    orig_dump = compile_backend.dump_cache_to_path

    def load_cache_from_path(dsl_name, file, path=None, bytecode_reader=None):
        if not getattr(state, "cached", False):
            return orig_load(dsl_name, file, path, bytecode_reader)
        fn = orig_load(dsl_name, file, directory(), bytecode_reader)
        state.hit = fn is not None
        return fn

    def dump_cache_to_path(dsl_name, jit_function, file, path=None, bytecode_writer=None):
        if not getattr(state, "cached", False):
            return orig_dump(dsl_name, jit_function, file, path, bytecode_writer)
        target = os.path.join(directory(), f"{dsl_name.lower()}_{file}.mlir")
        buffer = io.BytesIO()
        if bytecode_writer is not None:
            bytecode_writer(buffer)
        else:
            jit_function.ir_module.operation.write_bytecode(buffer)
        data = buffer.getvalue()
        try:
            with open(target, "rb") as f:
                existing = f.read()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            same = existing == data
            # K3_CUTE_PATHNORM: an entry another overlay wrote differs in its source locations only.
            equiv = not same and pathnorm in ("1", "text") and \
                nodebug_text(jit_function.ir_module) == entry_text(dsl_name, file)
            with lock:
                stats["same" if same or equiv else "diff"] += 1
                if equiv:
                    stats["sequiv"] = stats.get("sequiv", 0) + 1
            if not same and not equiv:
                print(f"[k3-cute-cache] DIFF {state.function} key={file}: fresh "
                      f"{len(data)} B vs cached {len(existing)} B", file=sys.stderr,
                      flush=True)
            return
        tmp = f"{target}.tmp.{host}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)  # atomic: readers never see a partial entry
            with lock:
                stats["written"] += 1
        except OSError as e:
            print(f"[k3-cute-cache] write failed {target}: {e!r}", file=sys.stderr, flush=True)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    compile_backend.load_cache_from_path = load_cache_from_path
    compile_backend.dump_cache_to_path = dump_cache_to_path

    if os.environ.get("K3_TRACE_CACHE", "") in ("shadow", "1"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import k3_trace_cache

            salt_once = []

            def salt():
                if not salt_once:
                    salt_once.append(_salt())
                return salt_once[0]

            tcache.update(k3_trace_cache.install(dsl_module, dict(
                state=state, lock=lock, stats=stats, directory=directory, salt=salt, root=_ROOT)))
        finally:
            sys.path.pop(0)

    if os.environ.get("K3_OBJ_CACHE", "0") == "1":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import k3_obj_cache

            ocache.update(k3_obj_cache.install(dict(state=state, stats=stats, salt=_salt, root=_ROOT,
                                                    on_import=_install)))
        finally:
            sys.path.pop(0)


_targets = {"cutlass.base_dsl.dsl": _patch} if _ROOT else {}
if any(os.environ.get(k) for k in ("K3_FST_BULK", "K3_WEIGHT_DIGEST", "K3_WSNAP_DIR", "K3_KVPLAN_KEY")):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import k3_startup_patches

        _targets.update(k3_startup_patches.targets())
    finally:
        sys.path.pop(0)
_install(_targets)
_chain_next_sitecustomize()
