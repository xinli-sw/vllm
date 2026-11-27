#
# A module-cache hit still builds the host launcher with LLVM: the DSL wraps the cached lowered module in an MLIR
# ExecutionEngine (MCJIT), ~25 ms of a ~30-40 ms hit; TVM-FFI launchers then also export the module to a PIC object
# (opt 3) and load it with BinaryExecutionEngine (JITLink: the DSL's own fix for MCJIT crashes in multi-process
# runs). This caches those object files per module hash and loads them with the same BinaryExecutionEngine:
#   - our cute.compile module-cache hits get their engine from the object (export at the JIT's opt level, PIC) behind
#     an ExecutionEngine-compatible adapter (lookup = raw_lookup("_mlir_ciface_" + name) as a ctypes callable,
#     raw_lookup = the binary engine's symbol lookup); a missing entry symbol or load error falls back to the JIT;
#   - the TVM-FFI provider's export_module_to_bytes is served from the same cache.
# Everything else is unchanged: the same lowered module is passed on (cubins still load from it), the same function
# type, signature, kernel_info and execution args. Only hits of our cute.compile calls are touched (the object key is
# the module hash the module cache already keys on). Objects: $K3_CUTE_CACHE/obj/<salt>/<module hash>.<fmt><opt>[p].
import ctypes
import os
import sys
import time
import uuid


def install(hook):
    from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support
    from cutlass._mlir._mlir_libs._cutlass_ir._execution_engine import BinaryExecutionEngine
    from cutlass.base_dsl import compiler as compiler_mod

    state, stats, salt, root = hook["state"], hook["stats"], hook["salt"], hook["root"]
    host = os.uname().nodename.split(".")[0]
    for k in ("oload", "oexport", "owritten", "ofail", "t_oread", "t_oexport", "t_oengine"):
        stats.setdefault(k, 0 if not k.startswith("t_") else 0.0)
    cache = {}
    orig_export = _aot_support.export_module_to_bytes

    def olog(msg):
        print(f"[k3-obj] {host} pid={os.getpid()} {msg}", file=sys.stderr, flush=True)

    def obj_dir():
        if "dir" not in cache:
            d = os.path.join(root, "obj", salt())
            os.makedirs(d, exist_ok=True)
            cache["dir"] = d
        return cache["dir"]

    def ours():
        key = getattr(state, "key", None)
        return key if key and getattr(state, "cached", False) and getattr(state, "hit", False) else None

    def cached_export(module, format="o", opt_level=3, enable_pic=True, **kwargs):  # noqa: A002
        key = ours()
        if key is None or kwargs:
            return orig_export(module, format=format, opt_level=opt_level, enable_pic=enable_pic, **kwargs)
        path = os.path.join(obj_dir(), f"{key}.{format}{opt_level}{'p' if enable_pic else ''}")
        t0 = time.perf_counter()
        try:
            with open(path, "rb") as f:
                data = f.read()
            if data:
                stats["oload"] += 1
                stats["t_oread"] += time.perf_counter() - t0
                return data
        except OSError:
            pass
        data = orig_export(module, format=format, opt_level=opt_level, enable_pic=enable_pic)
        stats["oexport"] += 1
        stats["t_oexport"] += time.perf_counter() - t0
        try:
            tmp = f"{path}.tmp.{host}.{os.getpid()}.{uuid.uuid4().hex[:6]}"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            stats["owritten"] += 1
        except OSError as e:
            olog(f"object write failed {path}: {e!r}")
        return data

    class _AotProxy:
        """``_aot_support`` with export_module_to_bytes served from the cache (for the TVM-FFI provider)."""

        def __getattr__(self, name):
            return getattr(_aot_support, name)

        @staticmethod
        def export_module_to_bytes(module, format="o", opt_level=3, enable_pic=True, **kwargs):  # noqa: A002
            return cached_export(module, format=format, opt_level=opt_level, enable_pic=enable_pic, **kwargs)

    def patch_tvm(module):
        module._aot_support = _AotProxy()

    # The provider imports cutlass.cutlass_dsl, which is not initialized yet while the DSL's base module is loading:
    # patch it when it is first imported (or now, if it already is).
    tvm_name = "cutlass.cutlass_dsl.tvm_ffi_provider"
    if tvm_name in sys.modules:
        patch_tvm(sys.modules[tvm_name])
    else:
        hook["on_import"]({tvm_name: patch_tvm})

    class ObjEngine:
        """The MLIR ExecutionEngine interface the DSL uses (lookup, raw_lookup) over a BinaryExecutionEngine."""

        def __init__(self, binary):
            self._binary = binary

        def raw_lookup(self, name):
            return self._binary.lookup(name)

        def lookup(self, name):
            func = self.raw_lookup("_mlir_ciface_" + name)
            if not func:
                raise RuntimeError("Unknown function " + name)
            return ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func)

    orig_jit = compiler_mod.Compiler.jit

    def jit(self, module, opt_level=2, shared_libs=()):
        if ours() is None:
            return orig_jit(self, module, opt_level=opt_level, shared_libs=shared_libs)
        t0 = time.perf_counter()
        try:
            data = cached_export(module, format="o", opt_level=opt_level, enable_pic=True)
            engine = ObjEngine(BinaryExecutionEngine(data, list(shared_libs), True))
            name = getattr(state, "function", None)
            if name and not engine.raw_lookup("_mlir_ciface_" + name):
                raise RuntimeError(f"entry _mlir_ciface_{name} not in the object")
            stats["t_oengine"] += time.perf_counter() - t0
            return engine
        except Exception as e:  # noqa: BLE001  (today's path)
            stats["ofail"] += 1
            olog(f"object engine failed for {ours()[:16]}: {e!r}; JIT instead")
            return orig_jit(self, module, opt_level=opt_level, shared_libs=shared_libs)

    compiler_mod.Compiler.jit = jit

    def report():
        s = stats
        return (f"obj-cache loaded={s['oload']} exported={s['oexport']} written={s['owritten']} failed={s['ofail']} "
                f"read {s['t_oread']:.1f}s export {s['t_oexport']:.1f}s engines {s['t_oengine']:.1f}s")

    return dict(report=report)
