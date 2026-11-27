#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch one InferenceX multi-node AgentX point the way InferenceX's workflow does.

Sets the environment of InferenceX's multi-node benchmark workflow, with a local
runner name, so that `python -m infx.launch run` binds, submits, follows and collects
the point unchanged. The launcher runs detached on the submit host; its log is
<run dir>/launch.log.

Environment:
  INFX_DIR      InferenceX launcher checkout prepared by setup_inferencex.py, with its
                `uv sync --locked` .venv
  K3VR_WORK     work directory holding matrix.json; runs go to $K3VR_WORK/runs
                (default: cwd)
  RUNNER_PREFIX runner names of the cluster label in configs/runners.yaml
                (default: vrnvl72-local, i.e. vrnvl72-local_0 .. _7)

Usage: run_point.py EXP_NAME [--image IMAGE] [--tag TAG] [--runner N] [--eval-only]
  EXP_NAME is an exp-name from matrix.json (see run_pareto.sh); use one --runner per
  concurrent job. A point runs its throughput benchmark only. --eval-only runs an eval
  job instead, on the recipe's eval variant: the point's GSM8K entry if it has one,
  otherwise the eval its matrix entry schedules (for Kimi-K3, the kimi-vendor
  tool-call schema suite).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

E2E = Path(os.environ["INFX_DIR"]).resolve()
K3VR = Path(os.environ.get("K3VR_WORK", ".")).resolve()
PY = E2E / ".venv/bin/python"
RUNNER_PREFIX = os.environ.get("RUNNER_PREFIX", "vrnvl72-local")


def matrix_entry(exp_name: str, eval_only: bool) -> dict:
    """The point's throughput entry, or with eval_only its eval-only entry if any."""
    entries = [
        e
        for e in json.loads((K3VR / "matrix.json").read_text())
        if e["exp-name"] == exp_name
    ]
    for entry in entries:
        if bool(entry.get("eval-only")) == eval_only:
            return entry
    if entries:
        return entries[0]
    raise SystemExit(
        f"no matrix entry {exp_name!r}; have {[e['exp-name'] for e in entries]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name")
    parser.add_argument("--image", help="override the master config image")
    parser.add_argument(
        "--tag", default="ref", help="run label, part of the run directory"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="run the point's eval instead"
    )
    parser.add_argument(
        "--runner",
        type=int,
        default=0,
        help="runner index N (RUNNER_PREFIX_N); one per concurrent job",
    )
    args = parser.parse_args()
    runner = f"{RUNNER_PREFIX}_{args.runner}"

    cfg = matrix_entry(args.exp_name, args.eval_only)
    # Evals run as their own jobs on the recipe's eval variant (see the README); the
    # workflow's eval after the benchmark, against the benchmark servers, is not run.
    run_eval = args.eval_only
    image = args.image or cfg["image"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = K3VR / "runs" / f"{stamp}-{args.tag}-{cfg['exp-name']}"
    run_dir.mkdir(parents=True)
    conc_list = [str(c) for c in cfg["conc"]]
    p, d = cfg["prefill"], cfg["decode"]
    offload = cfg.get("kv-offload-backend")
    agentic = cfg["scenario-type"] == "agentic-coding"

    env = dict(os.environ)
    env.update(
        {
            # workflow-level env
            "PYTHONPATH": str(E2E),
            "INFERENCEX_E2E_ROOT": str(run_dir),
            "PORT": "8888",
            "INFMAX_CONTAINER_WORKSPACE": "/workspace",
            "MODEL_DOWNLOAD_LOCK_TIMEOUT": "21600",
            "ENROOT_IMPORT_TIME_LIMIT": "120",
            "SALLOC_TIME_LIMIT": "480",
            "KEEP_LOGS": "0",
            "IS_MULTINODE": "true",
            "RANDOM_RANGE_RATIO": "0.8",
            "EXP_NAME": cfg["exp-name"],
            "RECIPE_FINGERPRINT": cfg.get("recipe-fingerprint", ""),
            "SRT_RECIPE": cfg.get("srt-recipe", ""),
            "EVAL_SRT_RECIPE": cfg.get("eval-srt-recipe", ""),
            "IMAGE": image,
            "THINKING_MODE": "thinking_on",
            "MODEL_PREFIX": cfg["model-prefix"],
            "MODEL": cfg["model"],
            "FRAMEWORK": cfg["framework"],
            "PRECISION": cfg["precision"],
            "ISL": "0" if agentic else str(cfg["isl"]),
            "OSL": "0" if agentic else str(cfg["osl"]),
            "MAX_MODEL_LEN": "0" if agentic else str(cfg["max-model-len"]),
            "CONC_LIST": " ".join(conc_list),
            "SPEC_DECODING": cfg["spec-decoding"],
            "DISAGG": str(cfg["disagg"]).lower(),
            "RUN_EVAL": "true" if run_eval else "false",
            "EVAL_ONLY": "true" if args.eval_only else "false",
            "EVAL_FRAMEWORK": cfg.get("eval-framework", "") if run_eval else "",
            "EVAL_SUITE": (cfg.get("eval-suite") or "") if run_eval else "",
            "EVAL_CONC": str(cfg.get("eval-conc", "")) if run_eval else "",
            "EVAL_LIMIT": "",
            "POWER": "0",
            "REQUIRE_POWER": "0",
            "SCENARIO_TYPE": cfg["scenario-type"],
            "SCENARIO_SUBDIR": "agentic/" if agentic else "fixed_seq_len/",
            "IS_AGENTIC": "1" if agentic else "0",
            "CONC": conc_list[0] if len(conc_list) == 1 else "",
            "DURATION": str(cfg.get("duration", "")),
            "AIPERF_EXPERIMENTAL_FAST": "0",
            "KV_OFFLOADING": cfg.get("kv-offloading", "") if agentic else "",
            "KV_OFFLOAD_BACKEND": offload["name"] if agentic and offload else "",
            "KV_OFFLOAD_BACKEND_METADATA": json.dumps(offload)
            if agentic and offload
            else "",
            "ROUTER_METADATA": json.dumps(cfg["router"]) if cfg.get("router") else "",
            "KV_P2P_TRANSFER": cfg.get("kv-p2p-transfer", ""),
            "TOTAL_CPU_DRAM_GB": str(cfg.get("total-cpu-dram-gb", 0)),
            "DECODE_CPU_DRAM_GB": str(cfg.get("decode-cpu-dram-gb", 0)),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/tmp/inferencex-pycache",
            "NODE_COUNT": str(cfg["node-count"]),
            "BENCHMARK_REF": "vr_k3-tutorial",
            "PR_NUMBER": "",
            "TEST_NAME": f"vr_k3-{args.tag}",
            "PREFILL_NUM_WORKERS": str(p["num-worker"]),
            "PREFILL_TP": str(p["tp"]),
            "PREFILL_PP_SIZE": str(p["pp"]),
            "PREFILL_DCP_SIZE": str(p["dcp-size"]),
            "PREFILL_PCP_SIZE": str(p["pcp-size"]),
            "PREFILL_DWDP_SIZE": str(p.get("dwdp-size", 1)),
            "PREFILL_EP": str(p["ep"]),
            "PREFILL_DP_ATTN": str(p["dp-attn"]).lower(),
            "DECODE_NUM_WORKERS": str(d["num-worker"]),
            "DECODE_TP": str(d["tp"]),
            "DECODE_PP_SIZE": str(d["pp"]),
            "DECODE_DCP_SIZE": str(d["dcp-size"]),
            "DECODE_PCP_SIZE": str(d["pcp-size"]),
            "DECODE_EP": str(d["ep"]),
            "DECODE_DP_ATTN": str(d["dp-attn"]).lower(),
            # step-level env
            "PREFILL_ADDITIONAL_SETTINGS": json.dumps(
                p.get("additional-settings") or []
            ),
            "DECODE_ADDITIONAL_SETTINGS": json.dumps(
                d.get("additional-settings") or []
            ),
            "RUNNER_NAME": runner,
            "RUNNER_TYPE": cfg["runner"],
            "RUNNER_TEMP": str(run_dir / "tmp"),
            "GITHUB_WORKSPACE": str(run_dir),
            "GITHUB_RUN_ID": stamp.replace("-", ""),
            "GITHUB_RUN_ATTEMPT": "1",
            "INFERENCEX_RESULTS_PYTHON": str(PY),
            "INFERENCEX_LAUNCH_PYTHON": str(PY),
        }
    )
    (run_dir / "tmp").mkdir()
    dwdp = f"-dwdp{p['dwdp-size']}" if p.get("dwdp-size", 1) > 1 else ""
    env["RESULT_FILENAME_BASE"] = (
        f"{cfg['exp-name']}_{cfg['precision']}_{cfg['framework']}"
        f"_prefill-tp{p['tp']}-pp{p['pp']}-dcp{p['dcp-size']}-pcp{p['pcp-size']}{dwdp}"
        f"-ep{p['ep']}-dp{str(p['dp-attn']).lower()}-nw{p['num-worker']}"
        f"_decode-tp{d['tp']}-pp{d['pp']}-dcp{d['dcp-size']}-pcp{d['pcp-size']}"
        f"-ep{d['ep']}-dp{str(d['dp-attn']).lower()}-nw{d['num-worker']}"
        f"_disagg-{str(cfg['disagg']).lower()}_spec-{cfg['spec-decoding']}"
        f"_conc{'x'.join(conc_list)}_a1_{runner}"
    )
    # The workspace is the launcher tree (INFX_DIR), and the job mounts it at
    # /infmax-workspace, so it must be a real copy (symlinks out of it do not resolve in
    # the containers). The srt-slurm submodule is only cloned on the submit host: link
    # it instead of copying.
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=.venv/",
            "--exclude=utils/srt-slurm/",
            "--exclude=__pycache__/",
            f"{E2E}/",
            f"{run_dir}/",
        ],
        check=True,
    )
    (run_dir / "utils/srt-slurm").symlink_to(E2E / "utils/srt-slurm")

    script = f"""
set -eo pipefail
cd {run_dir}
export RESULT_FILENAME=$({PY} -m infx.results.result_filename)
echo "RESULT_FILENAME=$RESULT_FILENAME"
source benchmarks/runtime_settings.sh
source benchmarks/multi_node/runtime_settings.sh
if [[ -z "$EVAL_CONC" ]]; then
  EVAL_CONC=$(tr ' ' '\\n' <<< "$CONC_LIST" | sort -n | tail -1)
fi
export EVAL_CONC IS_MULTINODE=true
{PY} -m infx.launch run
"""
    (run_dir / "launch.sh").write_text(script)
    (run_dir / "point.json").write_text(json.dumps({**cfg, "image": image}, indent=1))
    with open(run_dir / "launch.log", "w") as log:
        proc = subprocess.Popen(
            ["bash", str(run_dir / "launch.sh")],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=run_dir,
        )
    (run_dir / "launch.pid").write_text(str(proc.pid))
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
