#!/bin/bash
# Submit the six Kimi-K3 VR AgentX points (one Slurm job each, 6 or 8 nodes) with one image.
#
# Usage:
#   INFX_DIR=<INFX_DIR> K3VR_WORK=/path/to/work \
#     bash examples/vr_k3/run_pareto.sh vllm-k3-vr-branch-<sha> [TAG [run_point.py options]]
# The image name resolves to <cluster squash dir>/<name>.sqsh (see the cluster record).
# Further options go to every run_point.py call.
# Watch progress with examples/vr_k3/status.sh; results land in $K3VR_WORK/runs/*/.
set -eo pipefail
: "${INFX_DIR:?}" "${K3VR_WORK:?}"
IMAGE=${1:?image name}
TAG=${2:-branch}
shift $(( $# < 2 ? $# : 2 ))
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$INFX_DIR/.venv/bin/python
KEY=kimik3-fp4-vrnvl72-dynamo-vllm-agentic-dspark-mooncake-p8d16-disagg

cd "$K3VR_WORK"
if [ ! -s matrix.json ]; then
  (cd "$INFX_DIR" && "$PY" -m infx.matrix.generate test-config \
     --config-files configs/nvidia-master.yaml --config-keys "$KEY") > matrix.json
fi
runner=0
for exp in \
  kimik3_p1x8ep8_d1x16_conc1 \
  kimik3_p1x8ep8_d1x16_conc8 \
  kimik3_p2x8ep8_d1x16_conc48_kvdram-mooncake \
  kimik3_p1x8ep8_d1x16_conc64_kvdram-mooncake \
  kimik3_p2x8ep8_d1x16_conc160_kvdram-mooncake \
  kimik3_p2x8ep8_d1x16_conc256_kvdram-mooncake; do
  "$PY" "$HERE/run_point.py" "$exp" --image "$IMAGE" --tag "$TAG" --runner "$runner" "$@"
  runner=$((runner + 1))
done
