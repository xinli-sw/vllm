/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Kimi-K3 MLA prefill fp8 epilogue (q / k / v quant + fp8 latent cache insert),
 * bitwise equal to fusedKimiK3MLAQKVQuantKVCacheFp8Kernel
 * (csrc/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu). Same grid (one warp
 * per (token, slot), slots [0, H) per head, slot H the cache latent; 256-thread
 * blocks), same per-element arithmetic: the chunk conversion below is that
 * file's copyChunk8 body verbatim, split at its load. What changes: each lane
 * issues all of its global loads (q / k / v chunks, the RoPE cos / sin, the
 * slot id) before any store, through __restrict__ pointers, instead of one
 * load -> convert -> store round trip per chunk.
 */
#include <torch/all.h>
#include <torch/library.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace k3qq {

constexpr int kKvLoraRank = 512;
constexpr int kQkNopeHeadDim = 128;
constexpr int kQkRopeHeadDim = 64;
constexpr int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
constexpr int kVHeadDim = 128;
constexpr int kCacheEntry = kKvLoraRank + kQkRopeHeadDim;  // 576
constexpr int kVecElems = 8;
constexpr float kFp8Max = 448.0f;

// vllm::_typeConvert<BFloat16> (csrc/type_convert.cuh).
__device__ __forceinline__ float2 cvt2(__nv_bfloat162 x) {
  return __bfloat1622float2(x);
}

// copyChunk8<bf16, FP8=true, APPLY_ROPE> after its uint4 load, verbatim.
// cs: the 8 cos / sin values the original reads as cos_sin[pair_idx] and
// cos_sin[pair_idx + 32] for its 4 pairs (cs[i] = cos, cs[4 + i] = sin).
template <bool APPLY_ROPE>
__device__ __forceinline__ void convertChunk8(void* __restrict__ dst,
                                              uint4 const v, float scale_inv,
                                              float const* cs) {
  auto const* p = reinterpret_cast<__nv_bfloat162 const*>(&v);
  float f[kVecElems];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 x = cvt2(p[i]);
    f[2 * i] = x.x;
    f[2 * i + 1] = x.y;
  }
  if constexpr (APPLY_ROPE) {
#pragma unroll
    for (int i = 0; i < kVecElems / 2; i++) {
      float const cos = cs[i];
      float const sin = cs[4 + i];
      float const x = f[2 * i];
      float const y = f[2 * i + 1];
      f[2 * i] = x * cos - y * sin;
      f[2 * i + 1] = x * sin + y * cos;
    }
  }
  uint2 out;
  auto* o2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 s = make_float2(f[2 * i] * scale_inv, f[2 * i + 1] * scale_inv);
    s.x = fminf(fmaxf(s.x, -kFp8Max), kFp8Max);
    s.y = fminf(fmaxf(s.y, -kFp8Max), kFp8Max);
    o2[i] = __nv_cvt_float2_to_fp8x2(s, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint2*>(dst) = out;
}

__device__ __forceinline__ uint4 ld16(const __nv_bfloat16* __restrict__ p) {
  return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void loadCosSin(float* cs,
                                           const float* __restrict__ rope,
                                           int rope_e) {
  // cos_sin[pair_idx] / cos_sin[pair_idx + 32], pair_idx = rope_e / 2 + i, i <
  // 4
  float4 const c = *reinterpret_cast<const float4*>(rope + rope_e / 2);
  float4 const s =
      *reinterpret_cast<const float4*>(rope + rope_e / 2 + kQkRopeHeadDim / 2);
  cs[0] = c.x;
  cs[1] = c.y;
  cs[2] = c.z;
  cs[3] = c.w;
  cs[4] = s.x;
  cs[5] = s.y;
  cs[6] = s.z;
  cs[7] = s.w;
}

template <bool APPLY_ROPE>
__global__ void __launch_bounds__(256) qkvQuantCacheFp8Kernel(
    const __nv_bfloat16* __restrict__ q, int64_t const q_tok_stride,
    int64_t const q_head_stride, const __nv_bfloat16* __restrict__ k_nope,
    int64_t const kn_tok_stride, int64_t const kn_head_stride,
    const __nv_bfloat16* __restrict__ k_pe, int64_t const k_pe_tok_stride,
    const __nv_bfloat16* __restrict__ kv_c, int64_t const kv_c_tok_stride,
    const __nv_bfloat16* __restrict__ v, int64_t const v_tok_stride,
    int64_t const v_head_stride, uint8_t* __restrict__ q_fp8,
    int64_t const qo_tok_stride, int64_t const qo_head_stride,
    uint8_t* __restrict__ k_fp8, int64_t const ko_tok_stride,
    int64_t const ko_head_stride, uint8_t* __restrict__ v_fp8,
    int64_t const vo_tok_stride, int64_t const vo_head_stride,
    uint8_t* __restrict__ k_cache, int64_t const cache_block_stride,
    int64_t const cache_token_stride, const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ q_scale_inv,
    const float* __restrict__ k_scale_inv,
    const float* __restrict__ v_scale_inv,
    const float* __restrict__ cache_scale_inv, int const num_tokens,
    int const num_heads, int const cache_block_size,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache) {
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  int const e = laneId * kVecElems;  // this lane's first element
  if (slotIdx < num_heads) {
    int const h = slotIdx;
    const __nv_bfloat16* qh = q + tokenIdx * q_tok_stride + h * q_head_stride;
    const __nv_bfloat16* kn =
        k_nope + tokenIdx * kn_tok_stride + h * kn_head_stride;
    const __nv_bfloat16* kp = k_pe + tokenIdx * k_pe_tok_stride;
    const __nv_bfloat16* vh = v + tokenIdx * v_tok_stride + h * v_head_stride;
    // Loads first (lanes 0-15: NoPE / V chunks; lanes 16-23: RoPE chunks).
    uint4 qv = make_uint4(0, 0, 0, 0), kv = qv, vv = qv;
    float cs[8];
    bool const has_qk = e < kQkHeadDim;
    bool const rope = e >= kQkNopeHeadDim && has_qk;
    bool const has_v = e < kVHeadDim;
    if (has_qk) {
      qv = ld16(qh + e);
      kv = rope ? ld16(kp + (e - kQkNopeHeadDim)) : ld16(kn + e);
    }
    if (has_v) vv = ld16(vh + e);
    float const qsi = __ldg(q_scale_inv);
    float const ksi = __ldg(k_scale_inv);
    float const vsi = __ldg(v_scale_inv);
    if constexpr (APPLY_ROPE) {
      if (rope) {
        loadCosSin(cs, cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim,
                   e - kQkNopeHeadDim);
      }
    }
    uint8_t* qo = q_fp8 + tokenIdx * qo_tok_stride + h * qo_head_stride;
    uint8_t* ko = k_fp8 + tokenIdx * ko_tok_stride + h * ko_head_stride;
    uint8_t* vo = v_fp8 + tokenIdx * vo_tok_stride + h * vo_head_stride;
    if (has_qk) {
      if (rope) {
        convertChunk8<APPLY_ROPE>(qo + e, qv, qsi, cs);
        convertChunk8<APPLY_ROPE>(ko + e, kv, ksi, cs);
      } else {
        convertChunk8<false>(qo + e, qv, qsi, cs);
        convertChunk8<false>(ko + e, kv, ksi, cs);
      }
    }
    if (has_v) convertChunk8<false>(vo + e, vv, vsi, cs);
  } else {
    int64_t const slot_id = slot_mapping[tokenIdx];
    if (slot_id >= 0) {
      float const ksi = __ldg(cache_scale_inv);
      uint8_t* row = k_cache +
                     (slot_id / cache_block_size) * cache_block_stride +
                     (slot_id % cache_block_size) * cache_token_stride;
      const __nv_bfloat16* a512 = kv_c + tokenIdx * kv_c_tok_stride;
      const __nv_bfloat16* b64 = k_pe + tokenIdx * k_pe_tok_stride;
      // writeLatent576(lane, 32): e = lane*8, +256, +512 (< 576); e >= 512 is
      // RoPE.
      uint4 c0 = ld16(a512 + e);
      uint4 c1 = ld16(a512 + e + 256);
      bool const has2 = e + 512 < kCacheEntry;  // lanes 0-7: the RoPE chunk
      uint4 c2 = make_uint4(0, 0, 0, 0);
      float cs[8];
      if (has2) {
        c2 = ld16(b64 + e);
        if constexpr (APPLY_ROPE) {
          loadCosSin(
              cs, cos_sin_cache + position_ids[tokenIdx] * kQkRopeHeadDim, e);
        }
      }
      convertChunk8<false>(row + e, c0, ksi, cs);
      convertChunk8<false>(row + e + 256, c1, ksi, cs);
      if (has2) convertChunk8<APPLY_ROPE>(row + e + 512, c2, ksi, cs);
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void qkv_quant_cache_fp8(
    torch::Tensor const& q, torch::Tensor const& k_nope,
    torch::Tensor const& k_pe, torch::Tensor const& kv_c_normed,
    torch::Tensor const& v, torch::Tensor& q_fp8, torch::Tensor& k_fp8,
    torch::Tensor& v_fp8, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& q_scale_inv,
    torch::Tensor const& k_scale_inv, torch::Tensor const& v_scale_inv,
    torch::Tensor const& cache_scale_inv, int64_t cache_block_size,
    std::optional<torch::Tensor> const& position_ids,
    std::optional<torch::Tensor> const& cos_sin_cache) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(q.size(2) == 192 && k_nope.size(2) == 128 && v.size(2) == 128);
  TORCH_CHECK(k_pe.dim() == 2 && k_pe.stride(1) == 1 && k_pe.size(1) == 64);
  TORCH_CHECK(kv_c_normed.is_contiguous() && kv_c_normed.size(1) == 512);
  TORCH_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous() &&
              v_fp8.is_contiguous());
  TORCH_CHECK(k_cache.size(2) == 576 && k_cache.stride(2) == 1 &&
              k_cache.size(1) == cache_block_size);
  TORCH_CHECK(position_ids.has_value() == cos_sin_cache.has_value());
  bool const apply_rope = position_ids.has_value();
  if (apply_rope) {
    TORCH_CHECK(cos_sin_cache->size(1) == 64 && cos_sin_cache->stride(1) == 1 &&
                cos_sin_cache->stride(0) == 64 &&
                cos_sin_cache->scalar_type() == at::kFloat);
  }
  int const num_tokens = static_cast<int>(k_nope.size(0));
  int const num_heads = static_cast<int>(k_nope.size(1));
  if (num_tokens == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int kBlockSize = 256;
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens) * (num_heads + 1);
  int const grid = static_cast<int>((total_warps + 7) / 8);
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  auto bf = [](torch::Tensor const& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr());
  };
  auto launch = [&](auto kernel) {
    cudaLaunchKernelEx(
        &config, kernel, bf(q), q.stride(0), q.stride(1), bf(k_nope),
        k_nope.stride(0), k_nope.stride(1), bf(k_pe), k_pe.stride(0),
        bf(kv_c_normed), kv_c_normed.stride(0), bf(v), v.stride(0), v.stride(1),
        reinterpret_cast<uint8_t*>(q_fp8.data_ptr()), q_fp8.stride(0),
        q_fp8.stride(1), reinterpret_cast<uint8_t*>(k_fp8.data_ptr()),
        k_fp8.stride(0), k_fp8.stride(1),
        reinterpret_cast<uint8_t*>(v_fp8.data_ptr()), v_fp8.stride(0),
        v_fp8.stride(1), reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
        k_cache.stride(0), k_cache.stride(1), slot_mapping.data_ptr<int64_t>(),
        q_scale_inv.data_ptr<float>(), k_scale_inv.data_ptr<float>(),
        v_scale_inv.data_ptr<float>(), cache_scale_inv.data_ptr<float>(),
        num_tokens, num_heads, static_cast<int>(cache_block_size),
        apply_rope ? position_ids->data_ptr<int64_t>() : nullptr,
        apply_rope ? cos_sin_cache->data_ptr<float>() : nullptr);
  };
  if (apply_rope) {
    launch(qkvQuantCacheFp8Kernel<true>);
  } else {
    launch(qkvQuantCacheFp8Kernel<false>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace k3qq

TORCH_LIBRARY(k3_mla_qkv_quant, m) {
  m.def(
      "qkv_quant_cache_fp8(Tensor q, Tensor k_nope, Tensor k_pe, Tensor "
      "kv_c_normed, Tensor v, "
      "Tensor(a!) q_fp8, Tensor(b!) k_fp8, Tensor(c!) v_fp8, Tensor(d!) "
      "k_cache, Tensor slot_mapping, "
      "Tensor q_scale_inv, Tensor k_scale_inv, Tensor v_scale_inv, Tensor "
      "cache_scale_inv, "
      "int cache_block_size, Tensor? position_ids, Tensor? cos_sin_cache) -> "
      "()");
}

TORCH_LIBRARY_IMPL(k3_mla_qkv_quant, CUDA, m) {
  m.impl("qkv_quant_cache_fp8", &k3qq::qkv_quant_cache_fp8);
}
