# MiniMax M3 serving configs

Use with vLLM built from this branch. These configs target B200 TP4 at
concurrency 36 and B300 TP2 at concurrency 24. CPU KV cache budgets are about
1,008 GiB and 749 GiB, respectively; leave additional host memory for the server.

From this directory, start B200 with:

```bash
set -a
. ./b200.env
set +a
vllm serve nvidia/MiniMax-M3-NVFP4 --config b200.yaml
```

For B300, use `b300.env` and `b300.yaml`.
