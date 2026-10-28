# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA prefill fp8 epilogue (q / k / v quant + fp8 latent cache insert),
bitwise equal to ``fused_mla_qkv_quant_kv_cache_fp8_insert`` (the _C kernel
fusedKimiK3MLAQKVQuantKVCacheFp8Kernel), with every lane's loads issued before
its stores (csrc/qkv_quant.cu). JIT-built once per source hash into
VLLM_CACHE_ROOT/k3_mla_qkv_quant/<hash> with the _C build's math flags (no fast
math: IEEE division / sqrt, no FTZ, FMA contraction on).
"""

import hashlib
import os
import threading

import torch

_SRC = os.path.join(os.path.dirname(__file__), "csrc", "qkv_quant.cu")
_lock = threading.Lock()
# None: not tried; True: built and loaded (at warmup); False: the build failed.
_state: bool | None = None


def load_extension() -> bool:
    """Build (first use per source hash) and load torch.ops.k3_mla_qkv_quant.
    Called from the startup warmup only, never from the forward pass: a failure
    (e.g. no nvcc in the image) logs a warning and leaves the _C kernel in use."""
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        try:
            _build()
            _state = True
        except Exception:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "Kimi-K3 MLA fp8 epilogue extension unavailable; using the _C kernel.",
                exc_info=True,
            )
            _state = False
    return _state


def _build() -> None:
    from torch.utils.cpp_extension import load

    import vllm.envs as envs

    h = hashlib.sha256()
    with open(_SRC, "rb") as fh:
        h.update(fh.read())
    h.update(f"{torch.__version__} {torch.version.cuda}".encode())
    build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_mla_qkv_quant", h.hexdigest()[:16])
    os.makedirs(build, exist_ok=True)
    flags = ["-O3", "-DUSE_CUDA"]
    cuda = flags + [
        "--expt-relaxed-constexpr",
        "-gencode",
        "arch=compute_100f,code=sm_100f",
    ]
    load(
        name="k3_mla_qkv_quant",
        sources=[_SRC],
        extra_cflags=flags,
        extra_cuda_cflags=cuda,
        build_directory=build,
        is_python_module=False,
        verbose=False,
    )


def available() -> bool:
    """The extension was loaded at startup (the forward pass never builds it)."""
    return _state is True


def eligible(q: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )


def fused_mla_qkv_quant_kv_cache_fp8_insert(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_scale_inv: torch.Tensor,
    k_scale_inv: torch.Tensor,
    v_scale_inv: torch.Tensor,
    cache_scale_inv: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same contract as ops/fused_mla_key_concat_kv_cache.py's function."""
    assert _state is True, "the MLA fp8 epilogue extension is loaded at warmup"
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, _ = q.shape
    fp8 = torch.float8_e4m3fn
    q_fp8 = torch.empty((tp, num_heads, q.shape[2]), dtype=fp8, device=q.device)
    k_fp8 = torch.empty((tp, num_heads, q.shape[2]), dtype=fp8, device=q.device)
    v_fp8 = torch.empty((tp, num_heads, v.shape[2]), dtype=fp8, device=q.device)
    if tp == 0:
        return q_fp8, k_fp8, v_fp8
    torch.ops.k3_mla_qkv_quant.qkv_quant_cache_fp8(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        kv_cache,
        slot_mapping,
        q_scale_inv,
        k_scale_inv,
        v_scale_inv,
        cache_scale_inv,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return q_fp8, k_fp8, v_fp8
