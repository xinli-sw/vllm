#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare a public InferenceX checkout for this tutorial on a VR NVL72 cluster.

INFX_DIR is the inferencex-e2e directory of a clone of
https://github.com/SemiAnalysisAI/InferenceX at commit 80dff120 (the directory that
holds infx/, configs/ and benchmarks/). The script adds what this tutorial needs
there, skipping anything already in place:

1. The Kimi-K3 VR NVL72 AgentX recipe and its configuration entry, from inferencex/
   (benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/vrnvl72-fp4/agentx/ and
   configs/nvidia-master.yaml).
2. inferencex/collect.patch: the launcher collects AgentX results from the job's
   logs/agentic directory when they are there.
3. configs/runners.yaml: a cluster record `vrnvl72-local`, with runners
   vrnvl72-local_0 .. _7 (one per concurrent job).
4. infx/launch/drivers/srt/lanes.py: that cluster's multi-node Dynamo + vLLM lane.

Usage: setup_inferencex.py INFX_DIR --partition P --account A --hf-hub-cache DIR
           --aiperf-cache-dir DIR --image-dir DIR --dspark-dir DIR
           [--no-gpu-gres] [--segment]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

VENDORED = Path(__file__).resolve().parent / "inferencex"
NAME = "vrnvl72-local"
RUNNERS = "configs/runners.yaml"
LANES = "infx/launch/drivers/srt/lanes.py"
MASTER = "configs/nvidia-master.yaml"
CONFIG_KEY = "kimik3-fp4-vrnvl72-dynamo-vllm-agentic-dspark-mooncake-p8d16-disagg"
RECIPE = (
    "benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/vrnvl72-fp4/agentx/"
    "disagg-p8-d16-variants.yaml"
)
# The drafter's path in the recipe, inside the containers.
DRAFTER = "/raid/common-models/kimi-k3-dspark"
KIMI_K3_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"

LABEL = f"  cluster:{NAME}:\n" + "".join(f"    - {NAME}_{i}\n" for i in range(8))
RECORD = """\
  # VR NVL72 cluster for vLLM's examples/vr_k3 (written by setup_inferencex.py).
  {name}:
    gpus-per-node: 4
    available-cpu-dram-mib: 1455293
    arch: aarch64
    models:
      entries:
        Kimi-K3:
          root: hf-hub-cache
          dir: models--moonshotai--Kimi-K3/snapshots/{revision}
    scheduler: slurm
    slurm:
      partition: {partition}
      account: {account}
      exclusive: true
      cpus-per-task: 352
      volumes:
        hf-hub-cache:
          path: {hf_hub_cache}
        aiperf-cache:
          path: {aiperf_cache_dir}
      squash:
        dir: {image_dir}
        visibility: shared
        import: unchecked
      srt-slurm:
        network-interface: ''
        job-tag: vrnvl72
        preflight: false
        default-time-limit: 04:00:00
        # false when the nodes define no GPU GRES; exclusive jobs get all 4 GPUs anyway.
        gpus-per-node-directive: {gres}
        # true when Slurm needs --segment to keep a job's nodes in one NVLink domain.
        segment-directive: {segment}
        env:
          ENROOT_ROOTFS_WRITABLE: '1'
        mounts:
          {dspark_dir}: {drafter}
        volume-mounts:
          aiperf-cache: /aiperf_mmap_cache
          hf-hub-cache: /hf_hub_cache
"""
LANE = f"""\
    ("{NAME}", LaunchPath.SRT_MULTI): SrtLane(
        frameworks=any_of("dynamo-vllm"),
        write_eval_meta=True,
    ),
"""


def insert_after(text: str, anchor: str, block: str, path: str) -> str:
    match = re.search(anchor, text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"{path}: found no {anchor!r}; make this edit by hand")
    return text[: match.end()] + block + text[match.end() :]


def edit(root: Path, path: str, change) -> None:
    file = root / path
    old = file.read_text() if file.exists() else ""
    new = change(old)
    print(f"{'updated' if new != old else 'unchanged'}: {path}")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(new)


def apply_patch(root: Path, patch: Path) -> None:
    """Apply a -p1 patch to root unless it is already applied."""
    common = ["patch", "-p1", "-s", "-f", "-d", str(root), "-i", str(patch)]
    if (
        subprocess.run([*common, "-R", "--dry-run"], capture_output=True).returncode
        == 0
    ):
        print(f"unchanged: {patch.name}")
        return
    if subprocess.run(common).returncode != 0:
        raise SystemExit(f"{patch.name} does not apply; use InferenceX commit 80dff120")
    print(f"applied: {patch.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("infx_dir", type=Path)
    for name in ("partition", "account", "hf-hub-cache", "aiperf-cache-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--image-dir", required=True, help="holds <image>.sqsh files")
    parser.add_argument("--dspark-dir", required=True, help="the DSpark drafter")
    parser.add_argument(
        "--no-gpu-gres",
        action="store_true",
        help="the nodes define no GPU GRES: request no GPUs per node",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="pass --segment so that a job's nodes share one NVLink domain",
    )
    args = parser.parse_args()
    root = args.infx_dir.resolve()

    record = RECORD.format(
        name=NAME,
        revision=KIMI_K3_REVISION,
        partition=args.partition,
        account=args.account,
        hf_hub_cache=args.hf_hub_cache,
        aiperf_cache_dir=args.aiperf_cache_dir,
        image_dir=args.image_dir,
        dspark_dir=args.dspark_dir,
        drafter=DRAFTER,
        gres=str(not args.no_gpu_gres).lower(),
        segment=str(args.segment).lower(),
    )

    def recipe(text: str) -> str:
        return (VENDORED / "disagg-p8-d16-variants.yaml").read_text()

    def master(text: str) -> str:
        if re.search(rf"^{re.escape(CONFIG_KEY)}:$", text, re.MULTILINE):
            return text
        block = (VENDORED / "kimik3-vrnvl72.yaml").read_text()
        return text.rstrip("\n") + "\n\n" + block

    def runners(text: str) -> str:
        if f"\n  cluster:{NAME}:\n" not in text:
            text = insert_after(text, r"^labels:\n", LABEL, RUNNERS)
        if f"\n  {NAME}:\n" not in text:
            text = insert_after(text, r"^clusters:\n", record, RUNNERS)
        return text

    def lanes(text: str) -> str:
        if f'("{NAME}", LaunchPath.SRT_MULTI)' in text:
            return text
        return insert_after(text, r"^SRT_LANES: dict\[.*\] = \{\n", LANE, LANES)

    edit(root, RECIPE, recipe)
    edit(root, MASTER, master)
    apply_patch(root, VENDORED / "collect.patch")
    edit(root, RUNNERS, runners)
    edit(root, LANES, lanes)
    try:
        import yaml
    except ImportError:
        return 0
    clusters = yaml.safe_load((root / RUNNERS).read_text())["clusters"]
    assert clusters[NAME]["slurm"]["partition"] == args.partition
    assert CONFIG_KEY in yaml.safe_load((root / MASTER).read_text())
    yaml.safe_load((root / RECIPE).read_text())
    compile((root / LANES).read_text(), LANES, "exec")
    print(f"checked: {RUNNERS}, {MASTER} and the recipe parse, {LANES} compiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
