#
# Why: on the D head node EngineCore, the Dynamo decode process, the exporters and four busy-spinning worker main
# threads share CPUs (cpu-bind none). A worker whose host thread loses its CPU enqueues its step late, and every rank
# then waits for it (rank 0 sees it as g_prep stalls: 7.6% of B1 steps, -31 P90 at C1).
#
# What: a daemon thread per process polls /proc/self/cmdline (1 s) until vLLM's process title shows the role, then
# calls sched_setaffinity on every thread of the process (and again 60 s later for threads created in between):
#   VLLM::Worker_TP<r>_*  -> K3_PIN_WORKER_<r % 4> (default: 8 cores + SMT siblings on the GPU's NUMA node)
#   VLLM::EngineCore      -> K3_PIN_ENGINE   (32-39,208-215)
#   dynamo.vllm main      -> K3_PIN_DYNAMO   (40-55,216-231); this process also pins the job's node_exporter /
#                            process-exporter (same user) to K3_PIN_EXPORTERS (56-63,232-239) every 60 s.
# sched_setaffinity is bounded by the job's cpuset only, not by the parent's mask, so the order in which processes pin
# does not matter. reference node: NUMA 0 = CPUs 0-87,176-263 (GPUs 0-1), NUMA 1 = 88-175,264-351 (GPUs 2-3); job cpuset
# 0-149,176-325. Every action prints one "[k3pin]" line to stderr. Nothing touches numerics or CUDA.
import os
import sys

_DEFAULTS = {
    "K3_PIN_WORKER_0": "8-15,184-191",
    "K3_PIN_WORKER_1": "16-23,192-199",
    "K3_PIN_WORKER_2": "96-103,272-279",
    "K3_PIN_WORKER_3": "104-111,280-287",
    "K3_PIN_ENGINE": "32-39,208-215",
    "K3_PIN_DYNAMO": "40-55,216-231",
    "K3_PIN_EXPORTERS": "56-63,232-239",
}


def _chain_next_sitecustomize():
    import importlib.machinery
    import importlib.util

    seen = sys.__dict__.setdefault("_k3_sitecustomize_origins", set())
    seen.add(os.path.abspath(__file__))
    first = getattr(sys.modules.get("sitecustomize"), "__file__", None)
    if first:
        seen.add(os.path.abspath(first))
    for entry in list(sys.path):
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry or os.getcwd()])
        if spec is None or spec.origin is None or os.path.abspath(spec.origin) in seen:
            continue
        seen.add(os.path.abspath(spec.origin))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


def _cpus(spec):
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def _setting(name):
    return os.environ.get(name) or _DEFAULTS[name]


def _cmdline(pid="self"):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def _pin_pid(pid, cpus):
    ok = bad = 0
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return 0, 0
    for tid in tids:
        try:
            os.sched_setaffinity(int(tid), cpus)
            ok += 1
        except OSError:
            bad += 1
    return ok, bad


def _role():
    cl = _cmdline()
    if cl.startswith("VLLM::Worker_TP"):
        try:
            tp = int(cl[len("VLLM::Worker_TP"):].split("_")[0])
        except ValueError:
            return None, None
        return f"worker{tp}", _setting(f"K3_PIN_WORKER_{tp % 4}")
    if cl.startswith("VLLM::EngineCore"):
        return "enginecore", _setting("K3_PIN_ENGINE")
    if "dynamo.vllm" in cl:
        return "dynamo", _setting("K3_PIN_DYNAMO")
    return None, None


def _exporters(cpus):
    uid = os.getuid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
            with open(f"/proc/{name}/comm") as f:
                comm = f.read().strip()
        except OSError:
            continue
        if comm in ("node_exporter", "process-exporte"):
            _pin_pid(name, cpus)


def _run():
    import time

    host = os.uname().nodename.split(".")[0]
    role = spec = None
    for _ in range(3600):  # the title appears within minutes of start; give up after an hour
        role, spec = _role()
        if role:
            break
        time.sleep(1)
    if not role:
        return
    cpus = _cpus(spec)
    for attempt in range(2):
        ok, bad = _pin_pid(os.getpid(), cpus)
        print(f"[k3pin] {host} pid={os.getpid()} {role} -> {spec} threads ok={ok} failed={bad} pass={attempt}",
              file=sys.stderr, flush=True)
        time.sleep(60)
    if role == "dynamo":
        ex = _cpus(_setting("K3_PIN_EXPORTERS"))
        while True:
            _exporters(ex)
            time.sleep(60)


if os.environ.get("K3_PIN_CPUS", "") not in ("", "0") and not getattr(sys, "_k3pin_started", False):
    sys._k3pin_started = True
    try:
        import threading

        threading.Thread(target=_run, name="k3pin", daemon=True).start()
    except Exception as e:  # noqa: BLE001
        print(f"[k3pin] not started: {e!r}", file=sys.stderr, flush=True)
_chain_next_sitecustomize()
