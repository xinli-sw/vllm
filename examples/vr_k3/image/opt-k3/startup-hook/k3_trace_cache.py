import enum
import hashlib
import io
import json
import os
import pickle
import re
import sys
import time
import types
import uuid
ADDR = re.compile('0x[0-9a-fA-F]{5,}| object at ')
MAX_NODES = 50000

class Unkeyable(Exception):
    pass

def install(dsl_module, hook):
    import torch
    BaseDSL = dsl_module.BaseDSL
    state, lock, stats, directory, salt = (hook['state'], hook['lock'], hook['stats'], hook['directory'], hook['salt'])
    mode = os.environ.get('K3_TRACE_CACHE', '')
    shadow = mode == 'shadow'
    verify_mode = os.environ.get('K3_TRACE_VERIFY', 'rot16')
    label = os.environ.get('K3_RUN_LABEL', '') or f'pid{os.getpid()}'
    host = os.uname().nodename.split('.')[0]
    project = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'k3-trace')
    mismatch_log = os.environ.get('K3_TRACE_MISMATCH_LOG', os.path.join(project, 'trace-mismatch.jsonl'))
    keep_ir = os.environ.get('K3_TRACE_KEEP_IR', '')
    for k in ('skips', 'tkeyed', 'tunkeyed', 'timpure', 'tverify', 'tshadow_ok', 'tmismatch', 'tsaved', 'twritten', 'tequiv', 'dstale', 'dxov', 'dnodeps', 't_deps', 't_dpick'):
        stats.setdefault(k, 0)
    cache = {}
    why_counts = {}
    stale_counts = {}
    deps = None
    if os.environ.get('K3_TRACE_DEPS', '0') == '1':
        try:
            import k3_trace_deps
            deps = k3_trace_deps.Deps(lambda m: tlog(m), use_globals=os.environ.get('K3_TRACE_DEPS_GLOBALS', '1') == '1')
            if not deps.ok:
                deps = None
        except Exception as e:
            print(f'[k3-trace] K3_TRACE_DEPS unavailable ({e!r}): whole-tree keys', file=sys.stderr, flush=True)
            deps = None
    MAX_VARIANTS = 6
    share = os.environ.get('K3_TRACE_SHARE', '') == 'shadow'
    share_deps = None
    if share:
        try:
            import k3_trace_deps
            share_deps = deps or k3_trace_deps.Deps(lambda m: tlog(m))
            if not share_deps.ok:
                share, share_deps = (False, None)
        except Exception as e:
            print(f'[k3-trace] K3_TRACE_SHARE unavailable ({e!r})', file=sys.stderr, flush=True)
            share, share_deps = (False, None)
    rec_deps = deps or share_deps
    for k in ('shits', 'sok', 'sequiv', 'smismatch', 'sstale', 'scfg', 'swritten', 's_would_save'):
        stats.setdefault(k, 0)

    def tlog(msg):
        print(f'[k3-trace] {host} pid={os.getpid()} {msg}', file=sys.stderr, flush=True)

    def code_hash():
        if 'code' not in cache:
            cache['code'] = _code_hash(tree=deps is None)
        return cache['code']

    def tree_hash():
        if 'tree' not in cache:
            cache['tree'] = code_hash() if deps is None else _code_hash(tree=True)
        return cache['tree']

    def _code_hash(tree):
        import importlib.util
        h = hashlib.sha256()
        if not tree:
            h.update(b'deps-v1')
        spec = importlib.util.find_spec('vllm')
        pkg = list(spec.submodule_search_locations)[0]
        k3 = os.path.join(pkg, 'models', 'kimi_k3')
        for dirpath, dirnames, files in os.walk(k3 + '/') if tree else ():
            dirnames.sort()
            for f in sorted(files):
                if f.endswith('.py'):
                    p = os.path.join(dirpath, f)
                    with open(p, 'rb') as fh:
                        h.update(os.path.relpath(p, k3).encode() + b':' + hashlib.sha256(fh.read()).digest())
        if tree and 'dist-packages' not in pkg and ('site-packages' not in pkg):
            for dirpath, dirnames, files in os.walk(pkg):
                dirnames[:] = sorted((d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))))
                for f in sorted(files):
                    p = os.path.join(dirpath, f)
                    if f.endswith('.py') and (not os.path.islink(p)) and ('/models/kimi_k3/' not in p):
                        with open(p, 'rb') as fh:
                            h.update(os.path.relpath(p, pkg).encode() + b':' + hashlib.sha256(fh.read()).digest())
        here = os.path.dirname(os.path.abspath(__file__))
        for f in sorted(os.listdir(here)):
            if f.endswith('.py'):
                with open(os.path.join(here, f), 'rb') as fh:
                    h.update(f.encode() + hashlib.sha256(fh.read()).digest())
        vers = []
        for mod in ('vllm', 'torch', 'flashinfer', 'tokenspeed_mla', 'cutlass', 'cuda'):
            m = sys.modules.get(mod)
            vers.append(f'{mod}={(getattr(m, '__version__', '?') if m else '-')}')
        h.update(';'.join(vers).encode())
        if tree:
            h.update(os.environ.get('PYTHONPATH', '').encode())
        return h.hexdigest()[:16]

    def process_key():
        if 'proc' in cache:
            return cache['proc']
        h = hashlib.sha256()
        dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
        if dev >= 0:
            p = torch.cuda.get_device_properties(dev)
            h.update(f'{p.name}|{p.major}.{p.minor}|{p.multi_processor_count}|{p.total_memory}|dev{dev}'.encode())
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else -1
            world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            rank, world = (-1, 1)
        h.update(f'rank={rank}/{world}|{os.environ.get('K3_TRACE_KEY', '')}|{salt()}|{code_hash()}'.encode())
        for k in sorted(os.environ):
            if k.startswith(('CUTE_DSL_', 'CUTLASS_')) and k != 'CUTE_DSL_DUMP_DIR':
                h.update(f'{k}={os.environ[k]};'.encode())
        if rank >= 0:
            cache['proc'] = h.hexdigest()
        return h.hexdigest()

    def share_process_key():
        if 'sproc' in cache:
            return cache['sproc']
        h = hashlib.sha256()
        dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
        if dev >= 0:
            p = torch.cuda.get_device_properties(dev)
            h.update(f'{p.name}|{p.major}.{p.minor}|{p.multi_processor_count}|{p.total_memory}|dev{dev}'.encode())
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else -1
            world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            rank, world = (-1, 1)
        h.update(f'share|rank={rank}/{world}|{salt()}|{share_code()}'.encode())
        for k in sorted(os.environ):
            if k.startswith(('CUTE_DSL_', 'CUTLASS_')) and k != 'CUTE_DSL_DUMP_DIR':
                h.update(f'{k}={os.environ[k]};'.encode())
        if rank >= 0:
            cache['sproc'] = h.hexdigest()
        return h.hexdigest()

    def share_code():
        if 'scode' not in cache:
            cache['scode'] = code_hash() if deps is not None else _code_hash(tree=False)
        return cache['scode']

    def share_path(key):
        d = os.path.join(hook['root'], 'trace-share', salt(), share_code())
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{key}.pkl')

    def share_read(key):
        try:
            with open(share_path(key), 'rb') as f:
                data = pickle.load(f)
            return list(data.get('variants') or ())
        except Exception:
            return []

    def share_pick(key):
        variants = share_read(key)
        if not variants:
            return None
        start, memo = (time.perf_counter(), {})
        best = None
        for e in sorted(variants, key=lambda e: -e.get('created', 0)):
            if share_deps.valid(e.get('deps'), memo):
                best = e
                break
        stats['t_deps'] += time.perf_counter() - start
        if best is None:
            stats['sstale'] += 1
        return best

    def share_write(key, new):
        variants = share_read(key)
        for e in variants:
            if e.get('depsig') == new['depsig'] and e.get('module_hash') == new['module_hash'] and (e.get('pure') == new['pure']):
                return
        variants = [e for e in variants if e.get('depsig') != new['depsig']] + [new]
        variants.sort(key=lambda e: e.get('created', 0))
        p = share_path(key)
        tmp = f'{p}.tmp.{host}.{os.getpid()}.{uuid.uuid4().hex[:6]}'
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(dict(variants=variants[-MAX_VARIANTS:]), f, protocol=4)
            os.replace(tmp, p)
            stats['swritten'] += 1
        except Exception as e:
            tlog(f'share write failed {p}: {e!r}')
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def entry_dir():
        d = os.path.join(hook['root'], 'trace' if deps is None else 'trace-deps', salt(), code_hash())
        os.makedirs(d, exist_ok=True)
        return d
    arg_mods = [None]

    def canon(x, budget, seen, volatile):
        budget[0] -= 1
        if budget[0] < 0:
            raise Unkeyable('too large')
        if x is None or isinstance(x, (bool, int, str, bytes)):
            return (type(x).__name__, x)
        if isinstance(x, float):
            return ('float', repr(x))
        if isinstance(x, enum.Enum):
            return ('E', type(x).__module__, type(x).__qualname__, x.name)
        t = type(x)
        mod = getattr(t, '__module__', '') or ''
        if isinstance(x, torch.dtype):
            return ('dtype', str(x))
        if isinstance(x, torch.device):
            return ('device', str(x))
        if isinstance(x, torch.Size):
            return ('size', tuple(x))
        if isinstance(x, torch.Tensor):
            if x.is_meta or x.device.type == 'meta':
                return ('Tm', str(x.dtype), tuple(x.shape), tuple(x.stride()))
            ptr = x.data_ptr()
            volatile.append(ptr)
            return ('T', str(x.dtype), tuple(x.shape), tuple(x.stride()), x.device.type, ptr % 1024)
        if isinstance(x, (torch.cuda.Stream, torch.cuda.Event)):
            return ('tstream', t.__name__)
        if isinstance(x, (list, tuple)):
            return (t.__name__, tuple((canon(v, budget, seen, volatile) for v in x)))
        if isinstance(x, dict):
            return ('dict', tuple(sorted(((repr(canon(k, budget, seen, volatile)), canon(v, budget, seen, volatile)) for k, v in x.items()))))
        if isinstance(x, (set, frozenset)):
            return ('set', tuple(sorted((repr(canon(v, budget, seen, volatile)) for v in x))))
        if isinstance(x, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            f = getattr(x, '__func__', x)
            if arg_mods[0] is not None and getattr(f, '__module__', None):
                arg_mods[0].add(f.__module__)
            out = ('F', getattr(f, '__module__', ''), getattr(f, '__qualname__', repr(f)))
            if isinstance(x, types.MethodType) and (not isinstance(x.__self__, type)):
                out += (canon(x.__self__, budget, seen, volatile),)
            if getattr(f, '__closure__', None):
                out += (tuple((canon(c.cell_contents, budget, seen, volatile) for c in f.__closure__ if _cell_ok(c))),)
            return out
        if isinstance(x, type):
            if arg_mods[0] is not None:
                arg_mods[0].add(x.__module__)
            return ('C', x.__module__, x.__qualname__)
        if isinstance(x, types.ModuleType):
            return ('M', x.__name__)
        if 'stream' in t.__name__.lower() and mod.startswith('cuda'):
            return ('cstream', t.__name__)
        if id(x) in seen:
            return ('cycle', t.__qualname__)
        seen.add(id(x))
        items = {}
        d = getattr(x, '__dict__', None)
        if isinstance(d, dict):
            items.update(d)
        slots = [s for s in _slots(t) if s not in ('__dict__', '__weakref__')]
        for s in slots:
            if hasattr(x, s):
                items[s] = getattr(x, s)
        if items or isinstance(d, dict) or slots:
            if arg_mods[0] is not None and mod:
                arg_mods[0].add(mod)
            return ('O', mod, t.__qualname__, tuple(((k, canon(v, budget, seen, volatile)) for k, v in sorted(items.items(), key=lambda kv: str(kv[0])))))
        r = repr(x)
        if ADDR.search(r):
            raise Unkeyable(f'{mod}.{t.__qualname__}')
        return ('R', mod, t.__qualname__, r)

    def _cell_ok(c):
        try:
            c.cell_contents
            return True
        except ValueError:
            return False

    def _slots(t):
        out = []
        for c in t.__mro__:
            s = c.__dict__.get('__slots__', ())
            out += [s] if isinstance(s, str) else list(s)
        return out
    SIMPLE = (bool, int, float, str, type(None), enum.Enum)

    def globals_part(fn, budget, seen, volatile):
        g = getattr(fn, '__globals__', {})
        out = []
        for k in sorted(g):
            v = g[k]
            if k.startswith('__'):
                continue
            if isinstance(v, SIMPLE) or (isinstance(v, tuple) and all((isinstance(e, SIMPLE) for e in v))):
                out.append((k, canon(v, budget, seen, volatile)))
        return tuple(out)

    def call_key(dsl, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig):
        budget, seen, volatile = ([MAX_NODES], set(), [])
        arg_mods[0] = set() if rec_deps is not None else None
        parts = (process_key(), function_name, getattr(funcBody, '__module__', ''), getattr(funcBody, '__qualname__', ''), canon(getattr(funcBody, '__defaults__', None), budget, seen, volatile), canon(getattr(funcBody, '__kwdefaults__', None), budget, seen, volatile), tuple((canon(c.cell_contents, budget, seen, volatile) for c in funcBody.__closure__ or () if _cell_ok(c))), globals_part(funcBody, budget, seen, volatile), canon(list(args), budget, seen, volatile), canon(dict(kwonlyargs or {}), budget, seen, volatile), str(sig), tuple((str(t) for t in func_types)), repr(sorted(((str(k), str(v)) for k, v in (gpu_module_attrs or {}).items()))), dsl.compile_options.to_str(), dsl.envar.cache_key_str())
        last_parts[0] = parts
        h = hashlib.sha256(repr(parts).encode()).hexdigest()[:32]
        v = hashlib.sha256(repr(volatile).encode()).hexdigest()[:16]
        return (h, v)
    last_parts = [None]
    PART_NAMES = ('process', 'name', 'module', 'qualname', 'defaults', 'kwdefaults', 'closure', 'globals', 'args', 'kwargs', 'sig', 'types', 'gpu_attrs', 'options', 'env')
    impure_seen = set()

    def diff_paths(a, b, path, out, limit=4):
        if len(out) >= limit or a == b:
            return
        if isinstance(a, tuple) and isinstance(b, tuple) and (len(a) == 4 == len(b)) and (a[0] == b[0] == 'O'):
            da, db = (dict(a[3]), dict(b[3]))
            for k in sorted(set(da) | set(db), key=str):
                if da.get(k) != db.get(k):
                    if k in da and k in db:
                        diff_paths(da[k], db[k], f'{path}.{k}', out, limit)
                    else:
                        out.append(f'{path}.{k} {('added' if k in db else 'removed')}')
            return
        if isinstance(a, tuple) and isinstance(b, tuple) and (len(a) == len(b)):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    diff_paths(x, y, f'{path}[{i}]', out, limit)
            return
        out.append(path)

    def take_arg_mods(funcBody):
        mods, arg_mods[0] = (arg_mods[0] or set(), None)
        if getattr(funcBody, '__module__', None):
            mods.add(funcBody.__module__)
        return mods

    def entry_path(key):
        return os.path.join(entry_dir(), f'{key}.pkl')

    def read_entry(key):
        try:
            with open(entry_path(key), 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def write_entry(key, entry):
        p = entry_path(key)
        tmp = f'{p}.tmp.{host}.{os.getpid()}.{uuid.uuid4().hex[:6]}'
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(entry, f, protocol=4)
            os.replace(tmp, p)
            stats['twritten'] += 1
        except Exception as e:
            tlog(f'entry write failed {p}: {e!r}')
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def pick(key):
        data = read_entry(key)
        variants = data.get('variants') if isinstance(data, dict) else None
        if not variants:
            return None
        start, memo = (time.perf_counter(), {})
        best = None
        for e in sorted(variants, key=lambda e: -e.get('created', 0)):
            if deps.valid(e.get('deps'), memo):
                best = e
                break
        stats['t_deps'] += time.perf_counter() - start
        stats['t_dpick'] += time.perf_counter() - start
        if best is None:
            stats['dstale'] += 1
            if deps.last_stale:
                k = f'{deps.last_stale[1]}:{deps.last_stale[0].rpartition('|')[2]}'
                stale_counts[k] = stale_counts.get(k, 0) + 1
        return best

    def write_variant(key, new):
        data = read_entry(key)
        variants = list(data.get('variants') or ()) if isinstance(data, dict) else []
        for e in variants:
            if e.get('depsig') == new['depsig'] and e.get('module_hash') == new['module_hash'] and (e.get('pure') == new['pure']):
                return
        variants = [e for e in variants if e.get('depsig') != new['depsig']] + [new]
        variants.sort(key=lambda e: e.get('created', 0))
        write_entry(key, dict(variants=variants[-MAX_VARIANTS:]))

    def sampled(key):
        if verify_mode == '1':
            return True
        if verify_mode == '0':
            return False
        return int(hashlib.sha256(f'{label}|{key}'.encode()).hexdigest()[:8], 16) % 16 == 0

    def dump_ir(key, entry, new):
        if not (entry.get('ir') or new.get('ir')):
            return
        d = os.path.join(os.path.dirname(mismatch_log), 'trace-mismatch-ir')
        try:
            os.makedirs(d, exist_ok=True)
            for side, e in (('cached', entry), ('fresh', new)):
                if e.get('ir'):
                    with open(os.path.join(d, f'{key}.{side}.{e.get('label', '')}.{host}.{os.getpid()}.mlirbc'), 'wb') as f:
                        f.write(e['ir'])
        except OSError as e:
            tlog(f'dump-ir failed: {e!r}')
    share_log = os.environ.get('K3_TRACE_SHARE_LOG', os.path.join(os.path.dirname(mismatch_log), 'trace-share-mismatch.jsonl'))

    def mismatch(kind, fn, key, entry, module_hash, extra, log=None):
        rec = dict(t=time.time(), kind=kind, host=host, pid=os.getpid(), label=label, fn=fn, key=key, cached=entry.get('module_hash'), traced=module_hash, code=code_hash(), **extra)
        tlog(f'{kind} {fn} key={key} cached={entry.get('module_hash', '')[:16]} traced={(module_hash or '')[:16]} {extra}')
        try:
            os.makedirs(os.path.dirname(log or mismatch_log), exist_ok=True)
            with open(log or mismatch_log, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError:
            pass

    def lowered_exists(module_hash):
        return os.path.exists(os.path.join(directory(), f'cute_dsl_{module_hash}.mlir'))

    def lowered_text(module_hash):
        from cutlass._mlir import ir
        from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            with open(os.path.join(directory(), f'cute_dsl_{module_hash}.mlir'), 'rb') as f:
                m = read_bytecode_and_check_crc32(f)
            buf = io.StringIO()
            m.operation.print(file=buf)
            return buf.getvalue()

    def lowered_equal(a, b):
        try:
            return lowered_text(a) == lowered_text(b)
        except Exception as e:
            tlog(f'lowered compare failed {a[:12]} {b[:12]}: {e!r}')
            return False

    def peek_spec():
        try:
            from cutlass.base_dsl import pyir_core
            return pyir_core._PYIR_SPEC_SEALED[0]
        except Exception:
            return None
    orig_goi = BaseDSL.generate_original_ir

    def generate_original_ir(self, ir, func, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig, location=None, *, no_cache=False):
        state.tc = None
        if not getattr(state, 'ours', False) or not getattr(state, 'active', True):
            return orig_goi(self, ir, func, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig, location, no_cache=no_cache)
        fname = f'{getattr(funcBody, '__module__', '')}.{getattr(funcBody, '__qualname__', '')}'
        try:
            key, vol = call_key(self, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig)
        except Exception as e:
            reason = str(e)[:80] if isinstance(e, Unkeyable) else f'key error {e!r}'[:80]
            stats['tunkeyed'] += 1
            why_counts[reason] = why_counts.get(reason, 0) + 1
            if (fname, 'unkeyable') not in impure_seen and len(impure_seen) < 64:
                impure_seen.add((fname, 'unkeyable'))
                tlog(f'unkeyable {fname}: {reason}')
            state.tc = dict(unkeyable=reason)
            return orig_goi(self, ir, func, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig, location, no_cache=no_cache)
        stats['tkeyed'] += 1
        mods = take_arg_mods(funcBody) if rec_deps is not None else None
        try:
            entry = read_entry(key) if deps is None else pick(key)
            usable = entry is not None and entry.get('pure') and lowered_exists(entry['module_hash'])
        except Exception as e:
            tlog(f'entry read failed {key}: {e!r}')
            entry, usable = (None, False)
        if usable and (not shadow) and (not sampled(key)):
            self.kernel_info = pickle.loads(entry['kernel_info'])
            self.num_kernels = entry['num_kernels']
            self.launch_inner_count = entry.get('launch_inner_count', 0)
            state.skip_hash = entry['module_hash']
            state.tc = dict(key=key, skip=True, fn=fname)
            stats['skips'] += 1
            stats['tsaved'] += entry.get('trace_s', 0.0)
            if deps is not None and entry.get('tree') != tree_hash():
                stats['dxov'] += 1
            with ir.Location.unknown():
                module = ir.Module.create()
            module.operation.attributes['gpu.container_module'] = ir.UnitAttr.get()
            return (module, None, None)
        sentry, skey = (None, None)
        if share:
            try:
                skey = hashlib.sha256(repr((share_process_key(),) + tuple(last_parts[0][1:])).encode()).hexdigest()[:32]
                sentry = share_pick(skey)
            except Exception as e:
                tlog(f'share lookup failed {fname}: {e!r}')
                skey = None
        files = rec_deps.begin() if rec_deps is not None else None
        start = time.perf_counter()
        try:
            out = orig_goi(self, ir, func, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig, location, no_cache=no_cache)
        finally:
            if files is not None:
                rec_deps.end(files)
        trace_s = time.perf_counter() - start
        dep_map = None
        if rec_deps is not None and files is not None:
            dstart = time.perf_counter()
            try:
                dep_map = rec_deps.collect(files, mods)
            except Exception as e:
                tlog(f'deps collect failed {fname}: {e!r}')
            stats['t_deps'] += time.perf_counter() - dstart
        if deps is not None and (not dep_map):
            stats['dnodeps'] += 1
        ir_bytes = None
        if keep_ir and keep_ir in fname:
            try:
                buf = io.BytesIO()
                out[0].operation.write_bytecode(buf)
                ir_bytes = buf.getvalue()
            except Exception as e:
                tlog(f'keep-ir failed {fname}: {e!r}')
        why = None
        pre_parts = last_parts[0]
        try:
            key2, vol2 = call_key(self, funcBody, function_name, func_types, gpu_module_attrs, args, kwonlyargs, sig)
            arg_mods[0] = None
            if key2 != key:
                why = 'trace changed its inputs'
                try:
                    changed = []
                    for i, (a, b) in enumerate(zip(pre_parts or (), last_parts[0] or ())):
                        diff_paths(a, b, PART_NAMES[i] if i < len(PART_NAMES) else str(i), changed)
                    sig_ = (fname, tuple(changed))
                    if sig_ not in impure_seen and len(impure_seen) < 64:
                        impure_seen.add(sig_)
                        tlog(f'impure {fname}: the trace changed {', '.join(changed) or '?'}')
                except Exception as e:
                    tlog(f'impure diff failed {fname}: {e!r}')
        except Exception as e:
            why = f'unkeyable after trace: {e!r}'[:120]
            if (fname, 'unkeyable') not in impure_seen and len(impure_seen) < 64:
                impure_seen.add((fname, 'unkeyable'))
                tlog(f'impure {fname}: {why}')
        if peek_spec() is not None:
            why = why or 'PyIR spec sealed'
        try:
            kinfo = pickle.dumps(self.kernel_info, protocol=4)
        except Exception as e:
            kinfo, why = (None, why or f'kernel_info does not pickle: {e!r}')
        if why:
            stats['timpure'] += 1
            why_counts['impure: ' + why[:70]] = why_counts.get('impure: ' + why[:70], 0) + 1
        state.tc = dict(key=key, vol=vol, fn=fname, entry=entry, verify=usable and (not shadow), shadow=shadow and usable, new=dict(pure=why is None, why=why, kernel_info=kinfo, num_kernels=self.num_kernels, launch_inner_count=self.launch_inner_count, fn=fname, function_name=function_name, trace_s=round(trace_s, 4), vol=vol, created=time.time(), label=label, ir=ir_bytes))
        if deps is not None:
            state.tc['new'].update(deps=dep_map, depsig=k3_trace_deps.depsig(dep_map) if dep_map else None, tree=tree_hash())
        if share and skey and dep_map:
            sdeps = dict(dep_map)
            if any((r.startswith('vllm|') and (not r.startswith('vllm|models/kimi_k3/')) for r in dep_map)):
                sdeps['cfg|'] = (k3_trace_deps._h16(os.environ.get('K3_TRACE_KEY', '')), '')
                stats['scfg'] += 1
            state.tc.update(skey=skey, sentry=sentry, snew=dict(state.tc['new'], deps=sdeps, depsig=k3_trace_deps.depsig(sdeps)))
        return out
    BaseDSL.generate_original_ir = generate_original_ir

    class _Mismatch(RuntimeError):
        pass

    def finish(module_hash):
        try:
            _finish(module_hash)
        except _Mismatch:
            raise
        except Exception as e:
            tlog(f'finish failed: {e!r}')

    def _finish(module_hash):
        tc = getattr(state, 'tc', None)
        state.tc = None
        state.skip_hash = None
        if not tc or 'key' not in tc or tc.get('skip') or (not module_hash):
            return
        new, entry = (tc['new'], tc.get('entry'))
        if tc.get('skey'):
            try:
                share_finish(tc, module_hash)
            except Exception as e:
                tlog(f'share finish failed: {e!r}')
        if entry is not None and (tc.get('verify') or tc.get('shadow')) and entry.get('pure'):
            same = entry['module_hash'] == module_hash and entry['kernel_info'] == new['kernel_info'] and (entry['num_kernels'] == new['num_kernels'])
            if not same and entry['kernel_info'] == new['kernel_info'] and (entry['num_kernels'] == new['num_kernels']) and lowered_equal(entry['module_hash'], module_hash):
                stats['tequiv'] += 1
                tlog(f'equivalent {tc['fn']} key={tc['key']} cached={entry['module_hash'][:12]} traced={module_hash[:12]} (lowered modules identical)')
                same = True
            if tc.get('verify'):
                stats['tverify'] += 1
            if same:
                if tc.get('shadow'):
                    stats['tshadow_ok'] += 1
                return
            stats['tmismatch'] += 1
            extra = dict(vol_changed=entry.get('vol') != new['vol'], cached_label=entry.get('label'), kinfo_same=entry['kernel_info'] == new['kernel_info'])
            dump_ir(tc['key'], entry, new)
            if tc.get('shadow'):
                mismatch('SHADOW MISMATCH', tc['fn'], tc['key'], entry, module_hash, extra)
                return
            mismatch('MISMATCH', tc['fn'], tc['key'], entry, module_hash, extra)
            try:
                os.replace(entry_path(tc['key']), entry_path(tc['key']) + f'.bad.{int(time.time())}')
            except OSError:
                pass
            raise _Mismatch(f'[k3-trace] MISMATCH {tc['fn']} key={tc['key']}: a cached trace differs from a fresh one (see {mismatch_log}); rerun with K3_TRACE_CACHE unset')
        if deps is not None:
            if new.get('deps'):
                new['module_hash'] = module_hash
                write_variant(tc['key'], new)
            return
        if entry is None or entry.get('module_hash') != module_hash or (not entry.get('pure')):
            new['module_hash'] = module_hash
            write_entry(tc['key'], new)

    def share_finish(tc, module_hash):
        snew, sentry = (tc['snew'], tc.get('sentry'))
        if sentry is not None and sentry.get('pure') and snew.get('pure'):
            stats['shits'] += 1
            stats['s_would_save'] += snew.get('trace_s', 0.0)
            same = sentry['module_hash'] == module_hash and sentry['kernel_info'] == snew['kernel_info'] and (sentry['num_kernels'] == snew['num_kernels'])
            if same:
                stats['sok'] += 1
            elif sentry['kernel_info'] == snew['kernel_info'] and sentry['num_kernels'] == snew['num_kernels'] and lowered_equal(sentry['module_hash'], module_hash):
                stats['sequiv'] += 1
            else:
                stats['smismatch'] += 1
                mismatch('SHARE MISMATCH', tc['fn'], tc['skey'], sentry, module_hash, dict(cached_label=sentry.get('label'), kinfo_same=sentry['kernel_info'] == snew['kernel_info'], cached_cfg=sentry.get('cfg'), cfg=os.environ.get('K3_TRACE_KEY', '')), log=share_log)
        snew = dict(snew, module_hash=module_hash, cfg=os.environ.get('K3_TRACE_KEY', ''))
        share_write(tc['skey'], snew)

    def report():
        s = stats
        return f'trace-cache[{mode}] keyed={s['tkeyed']} unkeyable={s['tunkeyed']} impure={s['timpure']} skips={s['skips']} saved~{s['tsaved']:.1f}s verify={s['tverify']} shadow_ok={s['tshadow_ok']} mismatch={s['tmismatch']} equiv={s['tequiv']} written={s['twritten']}' + (' why=' + json.dumps(dict(sorted(why_counts.items(), key=lambda kv: -kv[1])[:6])) if why_counts else '') + (f' deps[stale={s['dstale']} xov={s['dxov']} nodeps={s['dnodeps']} t={s['t_deps']:.1f}s (check {s['t_dpick']:.1f}s){('' if deps.use_globals else ' files-only')}]' + (' stale=' + json.dumps(dict(sorted(stale_counts.items(), key=lambda kv: -kv[1])[:5])) if stale_counts else '') if deps is not None else '') + (f' share[shadow] hits={s['shits']} ok={s['sok']} equiv={s['sequiv']} mismatch={s['smismatch']} would_save~{s['s_would_save']:.1f}s stale={s['sstale']} cfgdep={s['scfg']} written={s['swritten']}' if share else '')
    return dict(finish=finish, report=report, lowered_equal=lowered_equal)
