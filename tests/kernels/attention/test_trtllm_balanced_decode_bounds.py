# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.attention.ops.trtllm_balanced_decode import TrtllmBalancedDecode


@pytest.mark.parametrize(
    "requests,pages,expected",
    [
        (4, 8192, 222),
        (75, 8192, 7980),
        (76, 8192, 8192),
        (151, 8192, 8192),
        (512, 8192, 8192),
        (512, 1, 1),
    ],
)
def test_chunk_table_width(requests: int, pages: int, expected: int) -> None:
    # Exercise host-side sizing without allocating CUDA buffers.
    decode = object.__new__(TrtllmBalancedDecode)
    decode.target_chunks = 152
    decode.min_chunk_pages = 8
    decode.max_pages_per_req = pages
    assert decode.max_chunk_pages(requests) == expected
