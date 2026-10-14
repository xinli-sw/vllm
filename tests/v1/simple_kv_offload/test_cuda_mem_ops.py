# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy descriptors must preserve block addressing and pacing across backends."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops as ops


def _buffers():
    sources = [np.arange(16, dtype=np.uint8), np.arange(32, 64, dtype=np.uint8)]
    destinations = [np.zeros(16, dtype=np.uint8), np.zeros(32, dtype=np.uint8)]
    params = ops.BatchMemcpyParams(
        np.array([x.ctypes.data for x in sources], dtype=np.uint64),
        np.array([x.ctypes.data for x in destinations], dtype=np.uint64),
        np.array([4, 8], dtype=np.uint64),
        2,
        ops._CUmemcpyAttributes(srcAccessOrder=ops.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM),
        ctypes.c_size_t(0),
        1,
        ctypes.c_size_t(0),
        123,
    )
    return sources, destinations, params


@pytest.mark.parametrize("opt_out,rocm", [(False, False), (True, False), (True, True)])
def test_copy_blocks_preserves_payload_and_chunk_pacing(monkeypatch, opt_out, rocm):
    """Exercise real descriptor generation against host-memory CUDA stand-ins."""
    sources, destinations, params = _buffers()
    events = []
    calls = {"batch": 0, "per_copy": 0}

    def transfer(dst, src, size, stream):
        assert stream == 123
        ctypes.memmove(dst, src, size)
        events.append(("copy", size))
        return 0

    def single(dst, src, size, stream):
        calls["per_copy"] += 1
        return transfer(dst, src, size, stream)

    def batch(dsts, srcs, sizes, count, attrs, idx, num_attrs, fail, stream):
        calls["batch"] += 1
        pointers = [
            ctypes.cast(x, ctypes.POINTER(ctypes.c_uint64)) for x in (dsts, srcs, sizes)
        ]
        for i in range(count):
            transfer(*(x[i] for x in pointers), stream)
        return 0

    monkeypatch.setattr(ops.envs, "VLLM_KV_OFFLOAD_DISABLE_BATCH_MEMCPY", opt_out)
    monkeypatch.setattr(ops, "current_platform", SimpleNamespace(is_rocm=lambda: rocm))
    monkeypatch.setattr(ops, "_batch_memcpy", (batch, 1))
    monkeypatch.setattr(ops, "_per_copy_memcpy", single)
    monkeypatch.setattr(ops, "_max_batch_descriptors", 0)
    ops.copy_blocks(
        [2, 0],
        [1, 3],
        params,
        max_descriptors=3,
        before_chunk=lambda: events.append(("pace",)),
    )
    assert destinations[0].tolist() == [0] * 4 + [8, 9, 10, 11] + [0] * 4 + [0, 1, 2, 3]
    assert destinations[1].tolist() == [0] * 8 + list(range(48, 56)) + [0] * 8 + list(
        range(32, 40)
    )
    assert events == [
        ("pace",),
        ("copy", 4),
        ("copy", 4),
        ("copy", 8),
        ("pace",),
        ("copy", 8),
    ]
    assert calls == (
        {"batch": 0, "per_copy": 4}
        if opt_out and not rocm
        else {"batch": 2, "per_copy": 0}
    )
    assert sources[0].tolist() == list(range(16))


def test_per_copy_driver_failure_is_reported(monkeypatch):
    sources, destinations, params = _buffers()
    monkeypatch.setattr(ops.envs, "VLLM_KV_OFFLOAD_DISABLE_BATCH_MEMCPY", True)
    monkeypatch.setattr(ops, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    monkeypatch.setattr(ops, "_batch_memcpy", (None, 1))
    monkeypatch.setattr(ops, "_per_copy_memcpy", lambda *args: 7)
    monkeypatch.setattr(ops, "_max_batch_descriptors", 0)
    with pytest.raises(RuntimeError, match="per-copy memcpy failed: err=7 index=0"):
        ops.copy_blocks([2], [1], params)
    assert not any(np.any(x) for x in destinations)
