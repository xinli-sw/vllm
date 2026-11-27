#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assemble the image's /opt/k3 tree in OUT_DIR, on the build host (layer.sh copies it).

/opt/k3 holds what InferenceX's Kimi-K3 VR recipe puts on PYTHONPATH or runs by path:
- dynamo, dynamo-conv, dynamo-eval, dynamo-conv-eval: Dynamo 1.5.0 from its public PyPI
  wheels (ai-dynamo, ai-dynamo-runtime for aarch64), each with the patch in dynamo/
  (vLLM API compatibility, conversation-affine routing, eval-side tool-call handling);
- bin/dloan.sh (Mooncake memory lender), pinhook/ (decode CPU pinning) and startup-hook/
  (CuTe compile cache), copied from opt-k3/;
- fe-hf-view/mxfp4: the frontend's tokenizer view of the model mounted at /model;
- jit-cache/*, startup-cache: cache directories the recipe points at, writable by all.

Usage: stage_opt_k3.py OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DYNAMO_VERSION = "1.5.0"
WHEELS = {
    "ai-dynamo": "ai_dynamo-1.5.0-py3-none-any.whl",
    "ai-dynamo-runtime": (
        "ai_dynamo_runtime-1.5.0-cp310-abi3-manylinux_2_28_aarch64.whl"
    ),
}
VARIANTS = ("dynamo", "dynamo-conv", "dynamo-eval", "dynamo-conv-eval")
VIEW_FILES = (
    "LICENSE", "README.md", "assets", "config.json", "configuration_kimi_k3.py",
    "encoding_k3.py", "generation_config.json", "kimi_k3_processor.py",
    "kimi_k3_vision_processing.py", "media_utils.py", "model.safetensors.index.json",
    "modeling_kimi_k3.py", "modeling_kimi_linear.py", "preprocessor_config.json",
    "tiktoken.model", "tokenization_kimi.py", "tokenizer_config.json",
)  # fmt: skip
CACHES = ("cute", "deepgemm", "flashinfer", "inductor", "nvcache", "triton", "vllm")


def download(project: str, filename: str, dest: Path) -> Path:
    """Fetch one release file from PyPI and check it against PyPI's sha256."""
    meta = json.load(
        urllib.request.urlopen(
            f"https://pypi.org/pypi/{project}/{DYNAMO_VERSION}/json", timeout=60
        )
    )
    entry = next(u for u in meta["urls"] if u["filename"] == filename)
    path = dest / filename
    with urllib.request.urlopen(entry["url"], timeout=600) as r, path.open("wb") as f:
        shutil.copyfileobj(r, f)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["digests"]["sha256"]:
        raise SystemExit(f"{filename}: sha256 {digest} does not match PyPI")
    return path


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    downloads = out.parent / "wheels"
    downloads.mkdir(parents=True, exist_ok=True)
    wheels = [download(p, f, downloads) for p, f in WHEELS.items()]
    for variant in VARIANTS:
        target = out / variant
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as zf:
                zf.extractall(target)
        patch = HERE / "dynamo" / f"{variant}.patch"
        subprocess.run(
            ["patch", "-s", "-p1", "-d", str(target / "dynamo"), "-i", str(patch)],
            check=True,
        )
        print(f"{variant}: Dynamo {DYNAMO_VERSION} + {patch.name}")
    shutil.copytree(HERE / "opt-k3", out, dirs_exist_ok=True)
    (out / "bin/dloan.sh").chmod(0o755)
    view = out / "fe-hf-view/mxfp4/models--moonshotai--Kimi-K3"
    (view / "refs").mkdir(parents=True)
    (view / "refs/main").write_text("served")
    (view / "snapshots/served").mkdir(parents=True)
    for name in VIEW_FILES:
        (view / "snapshots/served" / name).symlink_to(f"/model/{name}")
    for name in (*(f"jit-cache/{c}" for c in CACHES), "jit-cache", "startup-cache"):
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name).chmod(0o1777)
    (out / "provenance").mkdir()
    print(f"staged {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
