# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 fused MLA context reconstruction for the DCP context path
(SM100-family, CuTe DSL).

One persistent kernel per DCP chunk replaces
``dequant_fp8 -> kv_b_proj -> pack_fp8``:

  lat = bf16(float(g8) * k_scale)
      converter warps (the cache's formula)
  acc = lat[:, :512] @ W^T
      tcgen05 BF16 MMA, fp32 accumulation in TMEM
  K[:, h, :128] = fp8(bf16(acc)), V[:, h, :] = fp8(bf16(acc))
      epilogue warps (pack_fp8's casts)
  K[:, h, 128:] = fp8(lat[:, 512:])
      k_pe, replicated per head, written by the converter warps

The dequant and the bf16 / fp8 casts are the served formulas (cvt.rn.bf16x2.f32,
cvt.rn.satfinite.e4m3x2.f32); only the fp32 accumulation order of the GEMM can
differ from cuBLAS's.

Schedule: CTA pairs share the chunk's (256-row block, output tile) work evenly:
pair p owns a contiguous range of the num_blocks * 2 * heads output tiles (BN 128
= one head's K-nope or V half), so every pair gets the same tile count to within
one whatever the row count (a row block split between two pairs only repeats its
latent load and dequant). For each row block in its range, a CTA's fp8 latent
(128 x 576 B, contiguous) arrives by one bulk copy into a staging buffer; the
converter warps dequantize it once into an SMEM-resident bf16 A operand (8
K-blocks of 128 x 64, 128B-swizzled K-major) and convert its k_pe in place; each
warp then writes its 32 rows' k_pe to every head, 8 rows x 64 B per store. The
MMA warp sweeps the pair's tiles of the block against TMA-streamed kv_b tiles;
the epilogue warps convert each tile to fp8 into an SMEM staging tile and each
warp writes its 32 rows back as 4 full 128 B rows per store (per-thread row
stores were bound by the store request rate). The next row block's bulk copy
overlaps the current block's MMAs. Structure follows
``model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py`` (TMA B loads, one MMA
warp, TMEM stages, 128-thread epilogue).
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32, cute, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op

from vllm.cute_utils import _tcgen05, simple_tma_copy, to_cta0_smem
from vllm.cute_utils import mbarrier as mbar_ops

NOPE, DV, ROPE, LORA = 128, 128, 64, 512
LAT = LORA + ROPE
BM, BN, BK = 128, 128, 64
NUM_KB = LORA // BK  # 8 resident A K-blocks
A_KB_BYTES = BM * BK * 2  # 16 KB
STG_BYTES = BM * LAT  # 72 KB fp8 row block
B_STAGES = 10


@dsl_user_op
def _fp8x4_to_f32x4(x: Uint32, *, loc=None, ip=None):
    """Four e4m3 bytes -> four fp32 (exact, through f16)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([Float32.mlir_type] * 4),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\t.reg .b32 h01, h23;\n\t.reg .f16 a, b, c, d;\n\t"
        "mov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h01, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 h23, hi;\n\t"
        "mov.b32 {a, b}, h01;\n\t"
        "mov.b32 {c, d}, h23;\n\t"
        "cvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\t"
        "cvt.f32.f16 $2, c;\n\tcvt.f32.f16 $3, d;\n\t}",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(Float32.mlir_type, out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _f32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """(a, b) -> packed bf16x2 (a in the low half), round to nearest even."""
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_to_f32x2(x: Uint32, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([Float32.mlir_type] * 2),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $2;\n\t"
        "mov.b32 $0, {0, lo};\n\tmov.b32 $1, {0, hi};\n\t}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(Float32.mlir_type, out, [i], loc=loc, ip=ip))
        for i in range(2)
    ]


@dsl_user_op
def _f32x4_to_fp8x4(
    a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four fp32 -> four e4m3 bytes (a lowest), round to nearest even, saturate to
    finite."""
    return Uint32(
        llvm.inline_asm(
            Uint32.mlir_type,
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
            "{\n\t.reg .b16 t0, t1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 t0, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 t1, $4, $3;\n\t"
            "mov.b32 $0, {t0, t1};\n\t}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([Uint32.mlir_type] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(Uint32.mlir_type, out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _st_shared_v4(
    addr: Int32, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v4(
    ptr_i64: Int64, v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [Int64(ptr_i64).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(
    dst_smem: Int32,
    src_gmem: Int64,
    nbytes: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """1-D bulk copy global -> shared, completing nbytes of tx on the mbarrier."""
    llvm.inline_asm(
        None,
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


def _tile_range(pb, t_lo, t_hi, n_tiles):
    """Trace-time helper: the [n_lo, n_hi) output tiles of row block pb inside the
    pair's tile range."""
    return cutlass.max(t_lo - pb * n_tiles, 0), cutlass.min(
        t_hi - pb * n_tiles, n_tiles
    )


class Sm100CtxRec:
    def __init__(self, heads: int) -> None:
        self.heads = heads

    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor) -> cpasync.TmaInfo:
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.TWO)
        swizzle_128b = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (BN // 2, BK, B_STAGES), stride=(BK, 1, (BN // 2) * BK)
        )
        layout = cute.make_composed_layout(swizzle_128b, 0, layout)
        return cpasync.make_tiled_tma_atom(tma_op, tensor, layout, (BN // 2, BK))

    @cute.jit
    def __call__(
        self,
        g8: cute.Tensor,
        scale: cute.Tensor,
        W: cute.Tensor,
        k_out: cute.Tensor,
        v_out: cute.Tensor,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        B_tma = self.prepare_tma(W)
        self.kernel(g8, scale, B_tma, k_out, v_out).launch(
            grid=(grid_size, 1, 1),
            block=(10 * 32, 1, 1),
            cluster=(2, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g8: cute.Tensor,
        scale: cute.Tensor,
        B_tma: cpasync.TmaInfo,
        k_out: cute.Tensor,
        v_out: cute.Tensor,
    ) -> None:
        # CTA pairs (cluster of 2, tcgen05 cta_group 2): the pair owns a 256-row
        # block (128 rows per CTA, A resident in each CTA's SMEM), each CTA streams
        # half of every 128-column kv_b tile, and CTA 0's MMA warp issues the
        # 256 x 128 MMAs; each CTA's epilogue drains its own 128 TMEM lanes.
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        cta_rank = raw_bid % 2
        pair = raw_bid // 2
        num_pairs = num_bids // 2
        H = self.heads
        n_tiles = 2 * H
        num_tmem_stages = 512 // BN

        smem = utils.SmemAllocator()
        sA = smem.allocate_array(Uint8, NUM_KB * A_KB_BYTES, byte_alignment=1024)
        sStg = smem.allocate_array(Uint8, STG_BYTES, byte_alignment=128)
        sOut = smem.allocate_array(Uint8, BM * BN, byte_alignment=1024)
        sB = smem.allocate_tensor(
            B_tma.atom.value_type,
            B_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=B_tma.smem_layout.inner,
        )
        b_full = smem.allocate_array(Int64, B_STAGES)
        b_empty = smem.allocate_array(Int64, B_STAGES)
        stg_full = smem.allocate_array(Int64, 1)
        stg_empty = smem.allocate_array(Int64, 1)
        a_full = smem.allocate_array(Int64, 1)
        a_free = smem.allocate_array(Int64, 1)
        tmem_full = smem.allocate_array(Int64, num_tmem_stages)
        tmem_empty = smem.allocate_array(Int64, num_tmem_stages)
        taddr = smem.allocate(Int32, 4)

        BAR_TMEM_ALLOC = 1
        BAR_EPILOGUE = 2
        BAR_CONV = 3

        R = g8.shape[0]
        num_pb = cute.ceil_div(R, 2 * BM)
        # this pair's tiles [t_lo, t_hi) of the num_pb * n_tiles (row block, tile)
        # work, in row-block-major order
        work = num_pb * n_tiles
        t_lo = (pair * work) // num_pairs
        t_hi = ((pair + 1) * work) // num_pairs
        pb_lo = t_lo // n_tiles
        pb_hi = (t_hi + n_tiles - 1) // n_tiles

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(B_STAGES):
                    cute.arch.mbarrier_init(b_full + i, 2)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(stg_full, 1)
                cute.arch.mbarrier_init(stg_empty, 1)
                cute.arch.mbarrier_init(a_full, 2)
                cute.arch.mbarrier_init(a_free, 1)
                for i in cutlass.range_constexpr(num_tmem_stages):
                    cute.arch.mbarrier_init(tmem_full + i, 1)
                    cute.arch.mbarrier_init(tmem_empty + i, 256)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(B_tma.atom)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if warp_id == 9:
            # Producer warp: this CTA's 128 latent rows (bulk copy) and its half of
            # each kv_b tile (2-CTA TMA).
            stage = 0
            b_parity = 1
            stg_parity = 1
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // 2, BK))
            g8_base = Int64(g8.iterator.toint())
            b_full0 = to_cta0_smem(b_full)
            for pb in range(pb_lo, pb_hi):
                # A CTA past the end (odd tail) copies one in-range row; its rows are
                # masked as invalid anyway.
                row0 = cutlass.min(pb * 2 * BM + cta_rank * BM, R - 1)
                nbytes = cutlass.min(R - row0, BM) * LAT
                cute.arch.mbarrier_wait(stg_empty, stg_parity)
                stg_parity ^= 1
                with cute.arch.elect_one():
                    mbar_ops.arrive_expect_tx(stg_full, nbytes)
                    _bulk_g2s(
                        Int32(sStg.toint()),
                        g8_base + Int64(row0) * LAT,
                        nbytes,
                        Int32(stg_full.toint()),
                    )
                n_lo, n_hi = _tile_range(pb, t_lo, t_hi, n_tiles)
                for n in cutlass.range(n_lo, n_hi, unroll=1):
                    for k in cutlass.range(NUM_KB, unroll=1):
                        cute.arch.mbarrier_wait(b_empty + stage, b_parity)
                        with cute.arch.elect_one():
                            mbar_ops.arrive_expect_tx(
                                b_full0 + stage, (BN // 2) * BK * 2, "cluster"
                            )
                        simple_tma_copy(
                            B_tma.atom,
                            gB_tiles[None, (n * 2 + cta_rank, k)],
                            sB[None, None, stage],
                            b_full0 + stage,
                        )
                        stage = (stage + 1) % B_STAGES
                        if stage == 0:
                            b_parity ^= 1

        elif warp_id >= 4 and warp_id < 8:
            # Converter warps: thread t dequantizes row t of this CTA's 128 rows into
            # the resident A (8 swizzled K-blocks) and converts its k_pe (bf16 ->
            # fp8) in place; each warp then writes its rows' k_pe into every head's K
            # (only the pair that owns the block's first tile).
            tid_ = tid % 128
            s = scale[0]
            stg_parity = 0
            free_parity = 1
            sA_base = Int32(sA.toint())
            stg_row = Int32(sStg.toint()) + tid_ * LAT
            k_base = Int64(k_out.iterator.toint())
            sw = tid_ % 8
            a_full0 = to_cta0_smem(a_full)
            for pb in range(pb_lo, pb_hi):
                row = pb * 2 * BM + cta_rank * BM + tid_
                valid = row < R
                n_lo, _ = _tile_range(pb, t_lo, t_hi, n_tiles)
                cute.arch.mbarrier_wait(stg_full, stg_parity)
                stg_parity ^= 1
                cute.arch.mbarrier_wait(a_free, free_parity)
                free_parity ^= 1
                for kb in cutlass.range_constexpr(NUM_KB):
                    dst_row = sA_base + kb * A_KB_BYTES + tid_ * 128
                    for c4 in cutlass.range_constexpr(4):
                        v = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
                        if valid:
                            v = _ld_shared_v4(stg_row + kb * BK + c4 * 16)
                        packed = []
                        for q in cutlass.range_constexpr(4):
                            f = _fp8x4_to_f32x4(v[q])
                            packed.append(_f32x2_to_bf16x2(f[0] * s, f[1] * s))
                            packed.append(_f32x2_to_bf16x2(f[2] * s, f[3] * s))
                        for h2 in cutlass.range_constexpr(2):
                            chunk = 2 * c4 + h2
                            _st_shared_v4(
                                dst_row + ((chunk ^ sw) << 4),
                                packed[4 * h2],
                                packed[4 * h2 + 1],
                                packed[4 * h2 + 2],
                                packed[4 * h2 + 3],
                            )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=128)
                if tid_ == 0:
                    mbar_ops.arrive(a_full0, "cluster", "release")
                if valid and n_lo == 0:
                    for c in cutlass.range_constexpr(4):
                        vp = _ld_shared_v4(stg_row + LORA + c * 16)
                        qk = []
                        for j in cutlass.range_constexpr(4):
                            fk = _fp8x4_to_f32x4(vp[j])
                            bk0 = _bf16x2_to_f32x2(
                                _f32x2_to_bf16x2(fk[0] * s, fk[1] * s)
                            )
                            bk1 = _bf16x2_to_f32x2(
                                _f32x2_to_bf16x2(fk[2] * s, fk[3] * s)
                            )
                            qk.append(_f32x4_to_fp8x4(bk0[0], bk0[1], bk1[0], bk1[1]))
                        _st_shared_v4(
                            stg_row + LORA + c * 16, qk[0], qk[1], qk[2], qk[3]
                        )
                # this warp's 32 rows of fp8 k_pe (in place in the staging rows) ->
                # every head: 8 rows x 64 B per store
                cute.arch.sync_warp()
                if n_lo == 0:
                    lane = tid % 32
                    ch = lane % 4
                    for it in cutlass.range_constexpr(4):
                        rl = (warp_id - 4) * 32 + it * 8 + lane // 4
                        vk = _ld_shared_v4(
                            Int32(sStg.toint()) + rl * LAT + LORA + ch * 16
                        )
                        grow = pb * 2 * BM + cta_rank * BM + rl
                        if grow < R:
                            kd = (
                                k_base
                                + Int64(grow) * (H * (NOPE + ROPE))
                                + NOPE
                                + ch * 16
                            )
                            for h in cutlass.range(H, unroll=1):
                                _st_global_v4(
                                    kd + h * (NOPE + ROPE), vk[0], vk[1], vk[2], vk[3]
                                )
                cute.arch.barrier(barrier_id=BAR_CONV, number_of_threads=128)
                if tid_ == 0:
                    cute.arch.mbarrier_arrive(stg_empty)

        elif warp_id == 8:
            # MMA warp (CTA 0 of the pair issues for both)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)
            if cta_rank == 0:
                stage = 0
                b_parity = 0
                a_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                idesc = _tcgen05.make_bf16_idesc(2 * BM, BN)
                mask = Uint16(3)
                sA_base = Int64(sA.toint())
                for pb in range(pb_lo, pb_hi):
                    n_lo, n_hi = _tile_range(pb, t_lo, t_hi, n_tiles)
                    cute.arch.mbarrier_wait(a_full, a_parity)
                    a_parity ^= 1
                    _tcgen05.fence_after_thread_sync()
                    for n in cutlass.range(n_lo, n_hi, unroll=1):
                        cute.arch.mbarrier_wait(
                            tmem_empty + tmem_stage, tmem_empty_parity
                        )
                        _tcgen05.fence_after_thread_sync()
                        d_tmem = BN * tmem_stage
                        for k in cutlass.range(NUM_KB, unroll=1):
                            a_desc = sdesc | ((sA_base + k * A_KB_BYTES) >> 4)
                            b_desc = sdesc | (
                                sB[None, None, stage].iterator.toint() >> 4
                            )
                            cute.arch.mbarrier_wait(b_full + stage, b_parity)
                            _tcgen05.fence_after_thread_sync()
                            for mma_k in cutlass.range_constexpr(4):
                                enable_d = k > 0 or mma_k > 0
                                _tcgen05.mma_f16(
                                    d_tmem, a_desc, b_desc, idesc, enable_d, 2
                                )
                                a_desc += 32 >> 4
                                b_desc += 32 >> 4
                            _tcgen05.commit(b_empty + stage, mask, 2)
                            stage = (stage + 1) % B_STAGES
                            if stage == 0:
                                b_parity ^= 1
                        _tcgen05.commit(tmem_full + tmem_stage, mask, 2)
                        tmem_stage = (tmem_stage + 1) % num_tmem_stages
                        if tmem_stage == 0:
                            tmem_empty_parity ^= 1
                    _tcgen05.commit(a_free, mask, 2)

        elif warp_id < 4:
            # Epilogue warps: this CTA's TMEM lanes -> bf16 -> fp8 into the SMEM
            # staging tile; each warp then writes its own 32 rows to K-nope / V as 4
            # full 128 B rows per store
            warp_id_ = warp_id % 4
            tid_ = tid % 128
            if warp_id_ == 0:
                _tcgen05.alloc(taddr, 2)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=5 * 32)
            tmem_stage = 0
            parity = 0
            k_base = Int64(k_out.iterator.toint())
            v_base = Int64(v_out.iterator.toint())
            tmem_empty0 = to_cta0_smem(tmem_empty)
            sOut_base = Int32(sOut.toint())
            lane = tid % 32
            for pb in range(pb_lo, pb_hi):
                n_lo, n_hi = _tile_range(pb, t_lo, t_hi, n_tiles)
                for n in cutlass.range(n_lo, n_hi, unroll=1):
                    head = n // 2
                    if warp_id_ == 0:
                        cute.arch.mbarrier_wait(tmem_full + tmem_stage, parity)
                    cute.arch.barrier(barrier_id=BAR_EPILOGUE, number_of_threads=128)
                    _tcgen05.fence_after_thread_sync()
                    for i in cutlass.range_constexpr(BN // 16):
                        regs_ssa = _tcgen05.ld(
                            warp_id_ * 32, tmem_stage * BN + i * 16, "32x32b", 16
                        )
                        _tcgen05.wait_ld()
                        regs = cute.make_rmem_tensor(16, Float32)
                        regs.store(regs_ssa)
                        if cutlass.const_expr(i == BN // 16 - 1):
                            _tcgen05.fence_before_thread_sync()
                            mbar_ops.arrive(tmem_empty0 + tmem_stage, "cluster")
                        q = []
                        for j in cutlass.range_constexpr(4):
                            f = []
                            for p in cutlass.range_constexpr(2):
                                bf = _f32x2_to_bf16x2(
                                    regs[4 * j + 2 * p], regs[4 * j + 2 * p + 1]
                                )
                                f.extend(_bf16x2_to_f32x2(bf))
                            q.append(_f32x4_to_fp8x4(f[0], f[1], f[2], f[3]))
                        _st_shared_v4(
                            sOut_base + tid_ * 128 + ((i ^ (tid_ % 8)) << 4),
                            q[0],
                            q[1],
                            q[2],
                            q[3],
                        )
                    # this warp's 32 rows are staged (by its own lanes): 4 full 128 B
                    # rows per store
                    cute.arch.sync_warp()
                    for it in cutlass.range_constexpr(8):
                        rl = warp_id_ * 32 + it * 4 + lane // 8
                        ch = lane % 8
                        vals = _ld_shared_v4(
                            sOut_base + rl * 128 + ((ch ^ (rl % 8)) << 4)
                        )
                        grow = pb * 2 * BM + cta_rank * BM + rl
                        od = (
                            k_base
                            + Int64(grow) * (H * (NOPE + ROPE))
                            + head * (NOPE + ROPE)
                        )
                        if n % 2 == 1:
                            od = v_base + Int64(grow) * (H * DV) + head * DV
                        if grow < R:
                            _st_global_v4(
                                od + ch * 16, vals[0], vals[1], vals[2], vals[3]
                            )
                    cute.arch.sync_warp()
                    tmem_stage = (tmem_stage + 1) % num_tmem_stages
                    if tmem_stage == 0:
                        parity ^= 1
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp_id_ == 0:
                _tcgen05.dealloc(2)

    @cache
    @staticmethod
    def compile(heads: int):
        R = cute.sym_int()
        g8 = make_fake_tensor(Uint8, (R, LAT), (LAT, 1), assumed_align=16)
        scale = make_fake_tensor(Float32, (1,), (1,), assumed_align=4)
        W = make_fake_tensor(
            BFloat16, (heads * (NOPE + DV), LORA), (LORA, 1), assumed_align=16
        )
        k_out = make_fake_tensor(
            Uint8,
            (R, heads * (NOPE + ROPE)),
            (heads * (NOPE + ROPE), 1),
            assumed_align=16,
        )
        v_out = make_fake_tensor(
            Uint8, (R, heads * DV), (heads * DV, 1), assumed_align=16
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm100CtxRec(heads),
            g8,
            scale,
            W,
            k_out,
            v_out,
            128,
            stream,
            options="--enable-tvm-ffi",
        )


def ctxrec_cute(
    g8: torch.Tensor,
    k_scale: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    grid: int | None = None,
) -> None:
    """g8 [R, 576] fp8 contiguous; k_scale fp32 [1]; w [heads * 256, 512] bf16
    contiguous; k [R, heads, 192] fp8; v [R, heads, 128] fp8 (contiguous)."""
    R = g8.shape[0]
    if R == 0:
        return
    assert (
        g8.is_contiguous()
        and w.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    )
    assert w.shape == (heads * (NOPE + DV), LORA) and g8.shape[1] == LAT
    fn = Sm100CtxRec.compile(heads)
    tiles = (R + 2 * BM - 1) // (2 * BM) * 2 * heads
    sms = torch.cuda.get_device_properties(g8.device).multi_processor_count
    fn(
        g8.view(torch.uint8),
        k_scale.reshape(1).float(),
        w,
        k.view(torch.uint8).reshape(R, -1),
        v.view(torch.uint8).reshape(R, -1),
        2 * min(tiles, (grid or sms) // 2),
    )
