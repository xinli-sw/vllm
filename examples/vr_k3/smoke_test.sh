#!/bin/bash
# One-node check of a Kimi-K3 VR image before spending a rack on it (about 20 minutes):
#   - device, torch, CUDA and vLLM versions; the image's provenance
#   - imports of the Kimi-K3 model and of branch-only modules
#   - JIT build of the branch's CUDA ops on a VR GPU
#   - imports of the image's Dynamo frontend and worker builds
#   - the unit tests this branch adds or changes (they download two small public models)
#
# Usage (Slurm + Pyxis), from the repository root:
#   sbatch -N1 --exclusive -t 00:45:00 -A <account> -p <partition> \
#     --export=ALL,IMAGE=/path/to/vllm-k3-vr-branch-<sha>.sqsh,OUT=/path/to/smoke-out \
#     examples/vr_k3/smoke_test.sh
# Optional: TESTS="tests/... tests/..." to run only those test files.
# Results: $OUT/pytest.log and $OUT/junit.xml.
set -eo pipefail
: "${IMAGE:?}" "${OUT:?}"
# sbatch runs a copy of this script; find the checkout from the submit directory.
SRC=${SRC:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
UPSTREAM=$(git -C "$SRC" log --format=%H --grep='^Upstream vLLM snapshot at' -1)
TESTS=${TESTS:-$(git -C "$SRC" diff --name-only "$UPSTREAM" HEAD -- tests | grep '/test_.*\.py$' | tr '\n' ' ')}
export TESTS
mkdir -p "$OUT"
# The script runs inside the container; it expands its own variables there.
# shellcheck disable=SC2016
srun --ntasks=1 --container-image="$IMAGE" --container-writable \
  --container-mounts="$SRC:/src:ro,$OUT:/out" --no-container-mount-home \
  --container-env=TESTS bash -c '
set -eo pipefail
export VLLM_CACHE_ROOT=/tmp/vllm-cache HF_HUB_OFFLINE=1
cat /opt/k3/provenance/BRANCH 2>/dev/null || cat /opt/k3/provenance/BUILD
python3 - <<EOF
import time, torch
print("device", torch.cuda.get_device_name(), "cc", torch.cuda.get_device_capability(),
      "torch", torch.__version__, "cuda", torch.version.cuda)
import vllm
print("vllm", vllm.__version__, vllm.__file__)
import vllm.models.kimi_k3.nvidia.model  # noqa: F401
import vllm.v1.core.conversation_state  # noqa: F401  (branch-only)
from vllm.models.kimi_k3.nvidia.ops import kda_segscan, mla_qkv_quant
t = time.time(); kda_segscan.load_extension()
print(f"JIT k3_flashkda_split ok in {time.time() - t:.0f}s")
t = time.time(); ok = mla_qkv_quant.load_extension()
print(f"JIT k3_mla_qkv_quant loaded={ok} in {time.time() - t:.0f}s")
assert ok, "MLA QKV quant extension did not build"
EOF
PYTHONPATH=/opt/k3/dynamo-conv python3 -c "import dynamo.frontend, dynamo.vllm; print(\"dynamo frontend ok\")"
PYTHONPATH=/opt/k3/dynamo python3 -c "import dynamo.vllm; print(\"dynamo worker ok\")"
# Test from a copy so that "import vllm" is the installed, compiled package, not /src/vllm.
mkdir -p /tmp/t && cp -r /src/tests /src/pyproject.toml /tmp/t/ && cd /tmp/t
python3 -m pip install -q tblib ray 2>&1 | grep -v notice || true
HF_HUB_OFFLINE=0 HF_HOME=/out/hf python3 -m pytest -q -p no:cacheprovider \
  --continue-on-collection-errors -rfE --junitxml=/out/junit.xml $TESTS > /out/pytest.log 2>&1 \
  && echo "unit tests passed" || { tail -30 /out/pytest.log; exit 1; }
tail -1 /out/pytest.log
'
