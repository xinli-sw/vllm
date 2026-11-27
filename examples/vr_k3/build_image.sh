#!/bin/bash
# Build the Kimi-K3 Vera Rubin serving image from this vLLM checkout.
#
# This branch is upstream vLLM 3f1bcfd82 plus Kimi-K3 commits that change only Python
# (its two new CUDA ops are JIT-built at first use). vLLM's release pipeline published an
# image of that upstream commit for aarch64 with CUDA 13.4, built from this repository's
# docker/Dockerfile (job "Build release image - aarch64 - Rubin CUDA 13.4" in
# .buildkite/release-pipeline.yaml). The image is built in one job on a compute node:
#   1. start from that image (BASE_IMAGE: its registry URI, or a saved copy),
#   2. check that its vLLM is the upstream snapshot, then patch in this branch's vllm/
#      changes,
#   3. add /opt/k3, assembled from image/ by image/stage_opt_k3.py.
#
# Usage (on a Slurm login node with Pyxis; needs git, python3 and patch):
#   SLURM_ACCOUNT=<SLURM_ACCOUNT> SLURM_PARTITION=<SLURM_PARTITION> OUT_DIR=<IMAGE_DIR> \
#     bash examples/vr_k3/build_image.sh
# Optional: BASE_IMAGE (an enroot image URI or .sqsh), WORK_DIR (staging, default
#           $OUT_DIR/.build), TIME (job time limit, default 01:30:00).
# Prints IMAGE=<name>; the image is $OUT_DIR/<name>.sqsh.
set -eo pipefail
: "${OUT_DIR:?set OUT_DIR}" "${SLURM_ACCOUNT:?}" "${SLURM_PARTITION:?}"
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
HERE=$SRC/examples/vr_k3

if [ -n "$(git -C "$SRC" status --porcelain -- vllm examples/vr_k3/image)" ]; then
  echo "uncommitted changes under vllm/ or examples/vr_k3/image/; commit them first" >&2
  exit 1
fi
UPSTREAM=$(git -C "$SRC" log --format=%H --grep='^Upstream vLLM snapshot at' -1)
UPSTREAM_SHA=$(git -C "$SRC" log -1 --format=%s "$UPSTREAM" | awk '{print $NF}')
BASE_IMAGE=${BASE_IMAGE:-public.ecr.aws#q9t5s3a7/vllm-release-repo:${UPSTREAM_SHA}-aarch64-cu134}
HEAD_SHA=$(git -C "$SRC" rev-parse HEAD)
TAG=vllm-k3-vr-branch-${HEAD_SHA:0:7}
W=${WORK_DIR:-$OUT_DIR/.build}/$TAG
OUT=$OUT_DIR/$TAG.sqsh
echo "base=$BASE_IMAGE branch=$HEAD_SHA upstream=$UPSTREAM_SHA -> $OUT"

echo "[1/2] stage this branch's vllm/ changes and /opt/k3 in $W"
rm -rf "$W" && mkdir -p "$W/stage/overlay" "$W/stage/snapshot"
git -C "$SRC" diff --name-only --diff-filter=D "$UPSTREAM" HEAD -- vllm > "$W/stage/deleted"
git -C "$SRC" diff --name-only --diff-filter=AMR "$UPSTREAM" HEAD -- vllm > "$W/changed"
(cd "$SRC" && tar cf - -T "$W/changed") | tar xf - -C "$W/stage/overlay"
git -C "$SRC" archive "$UPSTREAM" vllm | tar x -C "$W/stage/snapshot"
python3 "$HERE/image/stage_opt_k3.py" "$W/stage/opt-k3"
cp "$HERE/image/layer.sh" "$W/stage/"
cat > "$W/stage/opt-k3/provenance/BRANCH" <<EOF
IMG=$TAG
BASE_IMAGE=$BASE_IMAGE
VLLM_REPO=$(git -C "$SRC" remote get-url origin 2>/dev/null || echo unknown)
VLLM_COMMIT=$HEAD_SHA
VLLM_UPSTREAM=$UPSTREAM_SHA
CHANGED_FILES=$(wc -l < "$W/changed")
BUILT=$(date -u +%FT%TZ)
EOF

echo "[2/2] apply the layer on a compute node and save the image"
mkdir -p "$OUT_DIR"
rm -f "$OUT"
# No GPU for the build container, so that the saved image carries no host driver files.
export NVIDIA_VISIBLE_DEVICES=void
srun -N1 --ntasks=1 --exclusive -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" -t "${TIME:-01:30:00}" \
  --container-image="$BASE_IMAGE" --container-remap-root --container-writable \
  --no-container-mount-home --container-env=NVIDIA_VISIBLE_DEVICES \
  --container-mounts="$W/stage:/mnt:ro" \
  --container-save="$OUT" \
  bash /mnt/layer.sh
# srun succeeds even when Pyxis fails to save the image.
[ -s "$OUT" ] || { echo "no image saved at $OUT" >&2; exit 1; }
ls -la "$OUT"
echo "IMAGE=$TAG"
