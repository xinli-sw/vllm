#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare measured Kimi-K3 VR AgentX points with the published InferenceX rows.

Reads every AgentX result (<run dir>/<result name>_conc<N>.json, written by InferenceX's
launcher) under the given run directories, and prints one Markdown row per point:
p90 interactivity (tok/s/user) and throughput per GPU (all GPUs, input + output), each
next to the published value and the difference. With matplotlib, it also writes a
Pareto plot.

Usage: summarize.py RUNS_DIR [RUNS_DIR ...] [--targets inferencex_vr_targets.json]
                    [--label-by image|tag] [--csv out.csv] [--plot pareto.png]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Display names of run tags (run_point.py --tag); other labels print as they are.
NAMES = {"branch": "this branch", "ref": "published image, rerun"}


def load_points(dirs: list[Path]) -> list[dict]:
    points = []
    for base in dirs:
        for f in sorted(base.glob("**/*_conc[0-9]*.json")):
            if f.name.startswith("agg_"):
                continue
            d = json.loads(f.read_text())
            if "request_metrics" not in d:
                continue
            lat = d["request_metrics"]["latency"]
            tput = d["request_metrics"]["throughput"]
            run = f.parent.name
            points.append(
                {
                    "run": run,
                    "tag": run.split("-")[2] if run.count("-") >= 2 else "",
                    "image": d.get("image", ""),
                    "conc": int(d["conc"]),
                    "gpus": int(d["num_gpus"]),
                    "p90_intvty": lat["intvty"]["p90"],
                    "tput_per_gpu": tput["per_gpu"]["total_tput_tps"],
                    "output_tput_per_decode_gpu": tput["output"]["tokens_per_second"]
                    / d["num_decode_gpu"],
                    "input_tput_per_prefill_gpu": tput["input"]["tokens_per_second"]
                    / d["num_prefill_gpu"],
                    "ttft_p50": lat["ttft"]["p50"],
                    "ttft_p90": lat["ttft"]["p90"],
                    "ok": d["num_requests_successful"],
                    "total": d["num_requests_total"],
                    "errors": d.get("request_accounting", {}).get(
                        "records_error_dropped", 0
                    ),
                }
            )
    return points


def pct(measured: float, published: float) -> str:
    return f"{100.0 * (measured - published) / published:+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--targets", type=Path, default=HERE / "inferencex_vr_targets.json"
    )
    parser.add_argument("--label-by", choices=("image", "tag"), default="image")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    targets = {p["conc"]: p for p in json.loads(args.targets.read_text())["points"]}
    points = sorted(load_points(args.runs), key=lambda p: (p[args.label_by], p["conc"]))
    if not points:
        raise SystemExit("no AgentX results found")

    print(
        "| Series | Conc | GPUs | p90 interactivity (tok/s/user) | InferenceX | diff "
        "| tok/s/GPU | InferenceX | diff | TTFT p50 / p90 (s) "
        "| requests measured | errors |"
    )
    print("| --- |" + " ---: |" * 8 + " --- | ---: | ---: |")
    for p in points:
        t = targets.get(p["conc"])
        ref_i = f"{t['p90_interactivity']:.1f}" if t else "-"
        ref_t = f"{t['tput_per_gpu']:,.1f}" if t else "-"
        d_i = pct(p["p90_intvty"], t["p90_interactivity"]) if t else "-"
        d_t = pct(p["tput_per_gpu"], t["tput_per_gpu"]) if t else "-"
        label = NAMES.get(p[args.label_by], p[args.label_by])
        print(
            f"| {label} | {p['conc']} | {p['gpus']} | {p['p90_intvty']:.1f} | {ref_i} "
            f"| {d_i} | {p['tput_per_gpu']:,.1f} | {ref_t} | {d_t} "
            f"| {p['ttft_p50']:.2f} / {p['ttft_p90']:.2f} | {p['ok']:,} "
            f"| {p['errors']:,} ({100.0 * p['errors'] / p['total']:.1f}%) |"
        )

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping the plot")
            return 0
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ref = sorted(targets.values(), key=lambda p: p["p90_interactivity"])
        ax.plot(
            [p["p90_interactivity"] for p in ref],
            [p["tput_per_gpu"] for p in ref],
            "o--",
            color="0.45",
            label="InferenceX (published)",
        )
        for p in ref:
            ax.annotate(
                f"c{p['conc']}",
                (p["p90_interactivity"], p["tput_per_gpu"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color="0.45",
            )
        for label in dict.fromkeys(p[args.label_by] for p in points):
            series = sorted(
                (p for p in points if p[args.label_by] == label),
                key=lambda p: p["p90_intvty"],
            )
            ax.plot(
                [p["p90_intvty"] for p in series],
                [p["tput_per_gpu"] for p in series],
                "o-",
                label=NAMES.get(label, label),
            )
        ax.set_xlabel("p90 interactivity (tok/s/user)")
        ax.set_ylabel("throughput (tok/s/GPU)")
        ax.set_yscale("log")
        ticks = [1000, 2000, 5000, 10000, 20000, 50000]
        ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
        ax.minorticks_off()
        ax.set_title("Kimi-K3 FP4, VR NVL72, Dynamo + vLLM, AgentX")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
