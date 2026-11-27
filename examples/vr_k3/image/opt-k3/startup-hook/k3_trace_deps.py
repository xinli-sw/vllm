#
# Whole-tree keys re-trace every kernel on the first launch of each new overlay (39% of server items), although an
# overlay usually changes one or two kernel families. With K3_TRACE_DEPS=1 an entry stays valid while the files its
# trace depends on are unchanged:
#   - every Python file that ran during the trace (sys.monitoring PY_START, one event per code object per trace: the
#     callback returns DISABLE and each recorded trace restarts events). The DSL compiles its preprocessed kernel code
#     under the original file name, so kernel sources count as run;
#   - the files of the modules of every class / function / object in the call's arguments;
#   - the files that define what those import (import statements resolved through sys.modules: a module's file, or
#     for `from m import n` the module defining n): one level for any file, transitively within kimi_k3 (module
#     constants pulled in with `from X import C` bind at import and never run); package __init__ re-exports are not
#     followed (kimi_k3/__init__.py imports the whole model);
# each as (sha256 of the file, and for kimi_k3 modules a digest of the module's simple globals and its classes' simple
# attributes; "simple" covers scalars, enums, types / functions by name, ranges, containers of those and plain objects
# made of them, e.g. config dataclasses). The digest is taken after the trace when recording and before the trace when checking, so constants set
# from another file (setattr / monkeypatch at import or at runtime) invalidate the entry like an edit would. Other
# modules (vllm core, scripts, this hook) count by file content only: their globals hold runtime state (e.g. vllm's
# platform init trace), which would make every entry stale.
#   - every environment variable the trace read (os.environ lookups: os.getenv, os.environ.get / [] / `in`, and
#     vllm.envs, which reads through os.getenv), as a digest of its value (or of "unset").
# Only files under the vllm package dir or a non-site-packages PYTHONPATH entry count; installed packages (cutlass,
# torch, ...) are covered by the version string in the process key. Missing / changed files: the entry is stale and
# the call traces (and records a new variant).
import ast
import enum
import hashlib
import os
import sys
import types

TOOL_ID = 4
TOOL_NAME = "k3-trace-deps"
SIMPLE = (bool, int, float, str, bytes, type(None))
FUNCS = (types.FunctionType, types.BuiltinFunctionType)


def _simple(v, depth=0):
    """A stable text form for constant-like values; None for anything else (ignored): scalars, enums, types and
    functions by qualified name, modules by name, torch dtypes / devices, ranges, containers of those (<= 256 items,
    4 levels) and plain objects whose attributes are all such values (e.g. config dataclasses)."""
    if isinstance(v, enum.Enum):
        return f"E:{type(v).__module__}.{type(v).__qualname__}.{v.name}"
    if isinstance(v, SIMPLE):
        return repr(v)
    if isinstance(v, type):
        return f"C:{v.__module__}.{v.__qualname__}"
    if isinstance(v, FUNCS):
        return f"F:{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', '')}"
    if isinstance(v, types.ModuleType):
        return f"M:{v.__name__}"
    if isinstance(v, range):
        return repr(v)
    t = type(v)
    if t.__module__ == "torch" and t.__name__ in ("dtype", "device"):
        return repr(v)
    if depth >= 4:
        return None
    if isinstance(v, (tuple, list, set, frozenset)) and len(v) <= 256:
        parts = [_simple(e, depth + 1) for e in v]
        if any(p is None for p in parts):
            return None
        if isinstance(v, (set, frozenset)):
            parts.sort()
        return f"{t.__name__}({','.join(parts)})"
    if isinstance(v, dict) and len(v) <= 256:
        items = []
        for k, e in v.items():
            pk, pe = _simple(k, depth + 1), _simple(e, depth + 1)
            if pk is None or pe is None:
                return None
            items.append(f"{pk}:{pe}")
        return "dict(" + ",".join(sorted(items)) + ")"
    d = getattr(v, "__dict__", None)
    if isinstance(d, dict) and 0 < len(d) <= 64 and not callable(v):
        parts = []
        for k in sorted(d):
            pe = _simple(d[k], depth + 1)
            if pe is None:
                return None
            parts.append(f"{k}={pe}")
        return f"O:{t.__module__}.{t.__qualname__}(" + ",".join(parts) + ")"
    return None


class Rec:
    """What one trace touched: the files whose code ran, the environment variables it read."""

    __slots__ = ("files", "env")

    def __init__(self):
        self.files, self.env = set(), {}


def _h16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Deps:
    def __init__(self, log, use_globals=True):
        self.log = log
        self.use_globals = use_globals  # test knob (K3_TRACE_DEPS_GLOBALS=0): file content only
        self.mon = getattr(sys, "monitoring", None)
        self.ok = self.mon is not None
        self.active = []  # recording sets of the traces in progress (nested cute.compile calls)
        self._roots = None
        self._rel = {}
        self._sha = {}
        self._names = {}
        self._modfile = {}
        self._nmods = -1
        self.last_stale = None
        if self.ok:
            self._env_hook()

    def _env_hook(self):
        """Record environment reads of traces in progress: os.environ's __getitem__ is what os.getenv,
        os.environ.get, `in os.environ` and vllm.envs all end in. Installed once per process; a no-op list check
        while no trace records."""
        cls = type(os.environ)
        if getattr(cls, "_k3_deps_orig", None) is not None:
            return
        orig, active = cls.__getitem__, self.active

        def __getitem__(env, key):
            try:
                value = orig(env, key)
            except KeyError:
                for r in active:
                    r.env.setdefault(key, None)
                raise
            for r in active:
                r.env.setdefault(key, value)
            return value

        cls._k3_deps_orig = orig
        cls.__getitem__ = __getitem__

    # ---------------------------------------------------------------- tracked files and their stable names
    def roots(self):
        if self._roots is None:
            import importlib.util

            rs = []
            spec = importlib.util.find_spec("vllm")
            pkg = os.path.abspath(list(spec.submodule_search_locations)[0])
            rs.append(("vllm", pkg))
            ovroot = os.path.dirname(pkg)
            here = os.path.dirname(os.path.abspath(__file__))  # this hook: its files are in the code hash
            for e in os.environ.get("PYTHONPATH", "").split(os.pathsep):
                a = os.path.abspath(e) if e else ""
                if a and "site-packages" not in a and "dist-packages" not in a and a not in (ovroot, here):
                    rs.append(("pp:" + a, a))
            self._roots = sorted(rs, key=lambda r: -len(r[1]))
        return self._roots

    def rel_of(self, path):
        """'<root tag>|<path under it>' for a tracked file (the same name in every overlay); None otherwise."""
        r = self._rel.get(path, False)
        if r is not False:
            return r
        r = None
        if path and not path.startswith("<"):
            for tag, prefix in self.roots():
                if path.startswith(prefix + os.sep):
                    r = f"{tag}|{path[len(prefix) + 1:]}"
                    break
        self._rel[path] = r
        return r

    def abs_of(self, rel):
        tag, _, sub = rel.partition("|")
        for t, prefix in self.roots():
            if t == tag:
                p = os.path.join(prefix, sub)
                return p if os.path.isfile(p) else None
        return None

    # ---------------------------------------------------------------- fingerprints
    def _modules_of(self, path):
        mods = self._modfile.get(path)
        if mods is None and len(sys.modules) != self._nmods:
            mf = {}
            for name, mod in list(sys.modules.items()):
                try:
                    f = mod.__dict__.get("__file__") if isinstance(mod, types.ModuleType) else None
                except Exception:  # noqa: BLE001
                    f = None
                if f:
                    mf.setdefault(os.path.abspath(f), []).append(mod)
            self._modfile, self._nmods = mf, len(sys.modules)
            mods = mf.get(path)
        return mods or ()

    def _gdigest(self, path):
        if f"{os.sep}kimi_k3{os.sep}" not in path:
            return ""
        h = hashlib.sha256()
        for mod in self._modules_of(path):
            name = mod.__name__
            d = mod.__dict__
            h.update(name.encode())
            for k in sorted(d):
                if k.startswith("__"):
                    continue
                v = d[k]
                s = _simple(v)
                if s is not None:
                    h.update(f"{k}={s};".encode())
                if isinstance(v, type) and v.__module__ == name:
                    cd = v.__dict__
                    for ck in sorted(cd):
                        if not ck.startswith("__"):
                            cs = _simple(cd[ck])
                            if cs is not None:
                                h.update(f"{k}.{ck}={cs};".encode())
        return h.hexdigest()[:16]

    def fingerprint(self, path):
        sha = self._sha.get(path)
        if sha is None:
            try:
                with open(path, "rb") as f:
                    sha = hashlib.sha256(f.read()).hexdigest()[:16]
            except OSError:
                sha = "missing"
            self._sha[path] = sha
        return (sha, self._gdigest(path))

    # ---------------------------------------------------------------- static imports
    def _import_specs(self, path):
        """[(module, names)] of a file's import statements (relative ones resolved; names None for `import m`)."""
        specs = self._names.get(path)
        if specs is not None:
            return specs
        specs = []
        try:
            with open(path, "rb") as f:
                tree = ast.parse(f.read())
        except Exception:  # noqa: BLE001
            self._names[path] = specs
            return specs
        mods = self._modules_of(path)
        me = mods[0].__name__ if mods else None
        is_pkg = os.path.basename(path) == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                specs += [(a.name, None) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not me:
                        continue
                    base = me.split(".") if is_pkg else me.split(".")[:-1]
                    base = base[: len(base) - (node.level - 1)]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target:
                    specs.append((target, tuple(a.name for a in node.names)))
        self._names[path] = specs
        return specs

    @staticmethod
    def _file_of_module(name):
        m = sys.modules.get(name)
        f = m.__dict__.get("__file__") if isinstance(m, types.ModuleType) else None
        return os.path.abspath(f) if f else None

    def _imports(self, path):
        """The files that define what a file imports: `import m` -> m's file; `from m import n` -> the module n is
        (a submodule), or the module that defines n (classes / functions by __module__), else m's own file (constants;
        their values are in the importing module's globals digest). Parent packages are not followed: a package
        __init__ that re-exports the whole model would pull every file in."""
        out = set()
        for target, names in self._import_specs(path):
            tfile = self._file_of_module(target)
            if names is None or "*" in names:
                if tfile:
                    out.add(tfile)
                continue
            mod = sys.modules.get(target)
            d = mod.__dict__ if isinstance(mod, types.ModuleType) else {}
            for n in names:
                obj = d.get(n)
                f = None
                if isinstance(obj, types.ModuleType):
                    f = obj.__dict__.get("__file__")
                    f = os.path.abspath(f) if f else None
                elif obj is None:
                    f = self._file_of_module(f"{target}.{n}")  # a submodule not bound (yet) as an attribute
                elif isinstance(obj, (type, types.FunctionType)) and getattr(obj, "__module__", None):
                    f = self._file_of_module(obj.__module__)
                f = f or tfile
                if f:
                    out.add(f)
        return out

    # ---------------------------------------------------------------- recording
    def begin(self):
        """Start recording the files that run and the environment read; returns the Rec to pass to end() and
        collect(), or None (no recording)."""
        mon = self.mon
        files = Rec()
        try:
            cur = mon.get_tool(TOOL_ID)
            if cur is None:
                mon.use_tool_id(TOOL_ID, TOOL_NAME)
            elif cur != TOOL_NAME:
                return None
            if not self.active:
                active, disable = self.active, mon.DISABLE

                def on_start(code, offset):
                    fn = code.co_filename
                    for r in active:
                        r.files.add(fn)
                    return disable

                mon.register_callback(TOOL_ID, mon.events.PY_START, on_start)
                mon.set_events(TOOL_ID, mon.events.PY_START)
            self.active.append(files)
            mon.restart_events()
        except Exception as e:  # noqa: BLE001
            self.log(f"deps recording unavailable: {e!r}")
            return None
        return files

    def end(self, files):
        if files is None:
            return
        for i, s in enumerate(self.active):
            if s is files:
                del self.active[i]
                break
        if not self.active:
            try:
                self.mon.set_events(TOOL_ID, 0)
                self.mon.register_callback(TOOL_ID, self.mon.events.PY_START, None)
            except Exception as e:  # noqa: BLE001
                self.log(f"deps recording stop failed: {e!r}")

    # ---------------------------------------------------------------- dependency sets
    def collect(self, rec, modules):
        """{rel: (sha, globals digest)} for a finished trace: the tracked files that ran, the argument modules' files,
        and their imports (transitively within kimi_k3), plus {"env|NAME": (value digest, "")} for every environment
        variable it read. None if nothing tracked ran (never key on nothing)."""
        start = set()
        for f in rec.files:
            if f and not f.startswith("<"):
                a = os.path.abspath(f)
                if self.rel_of(a):
                    start.add(a)
        for name in modules:
            m = sys.modules.get(name)
            if isinstance(m, types.ModuleType):
                f = m.__dict__.get("__file__")
                if f and self.rel_of(os.path.abspath(f)):
                    start.add(os.path.abspath(f))
        if not start:
            return None
        out, todo = set(start), list(start)
        while todo:
            f = todo.pop()
            for g in self._imports(f):
                if g not in out and self.rel_of(g):
                    out.add(g)
                    if f"{os.sep}kimi_k3{os.sep}" in g:
                        todo.append(g)
        deps = {self.rel_of(f): self.fingerprint(f) for f in sorted(out)}
        for k, v in rec.env.items():
            deps[f"env|{k}"] = (_h16("\0unset" if v is None else v), "")
        return deps

    def valid(self, deps, memo):
        """True if every dependency still has the recorded fingerprint (memo: per-call cache of current ones)."""
        if not deps:
            return False
        for rel, fp in deps.items():
            cur = memo.get(rel)
            if cur is None:
                if rel.startswith("env|"):
                    v = os.environ.get(rel[4:])
                    cur = (_h16("\0unset" if v is None else v), "")
                elif rel.startswith("cfg|"):  # (the trace cache's share key: the config key, see there)
                    cur = (_h16(os.environ.get("K3_TRACE_KEY", "")), "")
                else:
                    a = self.abs_of(rel)
                    cur = self.fingerprint(a) if a else ("missing", "")
                memo[rel] = cur
            if cur[0] != fp[0] or (self.use_globals and cur[1] != fp[1]):
                self.last_stale = (rel, "file" if cur[0] != fp[0] else "globals")
                return False
        return True


def depsig(deps):
    return hashlib.sha256(repr(sorted(deps.items())).encode()).hexdigest()[:16]
