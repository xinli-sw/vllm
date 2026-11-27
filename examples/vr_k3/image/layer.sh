#!/bin/bash
# Add this branch and the Kimi-K3 serving layer to the image this script runs in.
# Runs as root inside the image being built (see ../build_image.sh), with the staging
# directory that build_image.sh prepared mounted at /mnt (an existing, empty directory:
# Pyxis cannot save an image with a bind-mount point it had to create):
#   /mnt/overlay   this branch's vllm/ files that differ from the upstream snapshot
#   /mnt/deleted   vllm/ files this branch deletes (one path per line)
#   /mnt/snapshot  the upstream snapshot's vllm/ tree, to check the installed package
#   /mnt/opt-k3    the /opt/k3 tree: Dynamo builds, hooks, tokenizer view, provenance
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
umask 022
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

echo "[layer] checking the installed vLLM against the upstream snapshot"
bad=0
while read -r f; do
  cmp -s "/mnt/snapshot/$f" "$SP/$f" || { echo "differs: $f"; bad=1; }
done < <(cd /mnt/snapshot && find vllm -name '*.py')
[ "$bad" = 0 ] || { echo "the image's vLLM is not this branch's upstream snapshot" >&2; exit 1; }
echo "[layer] applying this branch's vllm/ changes"
while read -r f; do rm -f "$SP/$f"; done < /mnt/deleted
cp -r --no-preserve=mode,ownership /mnt/overlay/vllm/. "$SP/vllm/"
find "$SP/vllm" -name '__pycache__' -prune -exec rm -rf {} +

echo "[layer] installing /opt/k3"
mkdir -p /opt/k3
rm -rf /opt/k3/ov-*
cp -r --no-preserve=mode,ownership /mnt/opt-k3/. /opt/k3/
chmod -R u=rwX,go=rX /opt/k3
chmod 755 /opt/k3/bin/dloan.sh
chmod 1777 /opt/k3/jit-cache /opt/k3/jit-cache/* /opt/k3/startup-cache
python3 -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)"
for d in dynamo dynamo-conv dynamo-eval dynamo-conv-eval; do
  PYTHONPATH=/opt/k3/$d python3 -c "import dynamo, dynamo.frontend; print('$d ok')"
done
echo "[layer] done"
