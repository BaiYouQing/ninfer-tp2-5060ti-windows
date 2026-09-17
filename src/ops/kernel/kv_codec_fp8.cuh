#pragma once

// ninfer::ops - E4M3FN row-scaled D256 KV-cache codec（每 256 维 1 个 fp16 scale）。
//
// 来源：上游 Neroued/ninfer 的 src/ops/kv_cache/fp8_e4m3_row_codec.cuh
//       （本地快照 71832af9 / d4929686），原样移植到本 fork 的 ops/kernel 目录下。
//       移植理由：我们的 KV 读侧在 ops/kernel（gqa_attention_*），而该 codec 只依赖
//       ops/kernel/paged_kv_address.cuh，在本树已存在，改路径即可复用，无需重写。
// 用途：KV 档位 `fp8`（K/V 都 e4m3）与 `k16v8`（K=BF16、V=e4m3）的 V 侧读写；
//       K 侧的 s8/f8 MMA 路径见 gqa_attention_decode_i8.cuh 的说明。
//
// 语义（与上游一致，改动前先读上游对应提交）：
//   * 每个 256 维向量一个 fp16 scale，scale 本身被 clamp 到 [2^-24, 65504] 再取 fp16，
//     之后用"取整后的 scale"算 inverse_scale，保证 write/read 用同一个 scale 值。
//   * 量化用 __nv_cvt_float*_to_fp8(..., __NV_SATFINITE, __NV_E4M3)：超出 ±448 饱和，
//     absmax==0 时 scale/inverse_scale 都取 0 且 code 写 0。
//   * 反量化 code2 -> half2 后乘 scale（half 运算）。

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

inline constexpr int kKVCacheFp8HeadDim        = 256;
inline constexpr int kKVCacheFp8Group          = 256;
inline constexpr int kKVCacheFp8Groups         = 1;
inline constexpr float kKVCacheFp8MaxFinite    = 448.0F;
inline constexpr float kKVCacheFp8ScaleMinimum = 0x1p-24F;
inline constexpr float kKVCacheFp8ScaleMaximum = 65504.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheFp8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_scale_index(int physical_page, int kv_head,
                                                                 int page_offset) {
    return paged_kv_element_offset<kKVCacheFp8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, 0);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheFp8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheFp8QuantParams {
    __half scale;
    float inverse_scale;
};

__device__ __forceinline__ KVCacheFp8QuantParams kv_cache_fp8_quant_params(float absmax) {
    if (absmax == 0.0F) { return {.scale = __float2half_rn(0.0F), .inverse_scale = 0.0F}; }
    const float raw_scale = absmax / kKVCacheFp8MaxFinite;
    const float bounded   = fminf(kKVCacheFp8ScaleMaximum, fmaxf(kKVCacheFp8ScaleMinimum, raw_scale));
    const __half scale    = __float2half_rn(bounded);
    const float represented_scale = __half2float(scale);
    return {.scale = scale, .inverse_scale = 1.0F / represented_scale};
}

__device__ __forceinline__ std::uint8_t kv_cache_fp8_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float_to_fp8(x * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint16_t kv_cache_fp8_quant_code2(float x0, float x1,
                                                                  float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float2_to_fp8x2(make_float2(x0 * inverse_scale, x1 * inverse_scale),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __half2 kv_cache_fp8_code2_to_half2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
}

__device__ __forceinline__ __half2 kv_cache_fp8_dequant_code2_to_half2(std::uint16_t storage,
                                                                      __half scale) {
    return __hmul2(kv_cache_fp8_code2_to_half2(storage), __halves2half2(scale, scale));
}

// 写侧：一个 warp 覆盖一个 token 的完整 256 维向量时（256/8 = 32 个 8 元素 chunk ↔ 32 lane），
// lane 先在自己那 8 个元素上求 absmax，再 warp 归约得到整向量 absmax。
// 供 kernel 的融合 append 用：归约完 broadcast 出 scale，各 lane 量化自己的 8 个元素。
__device__ __forceinline__ float kv_cache_fp8_warp_absmax(float lane_absmax) {
    float m = lane_absmax;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    return m;
}

// 读侧：8 个 e4m3 code（8 B，对应 8 维）→ 8 个 bf16，打包成 int4（16 B），可直接 store_vec
// 到 bf16 的 smem tile —— 与 bf16 路径的 16 B chunk 粒度一致，所以读循环的结构不用改。
__device__ __forceinline__ int4 kv_cache_fp8_dequant_code8_to_bf16x8(const std::uint8_t* codes8,
                                                                     __half scale) {
    const __half2 scale2 = __halves2half2(scale, scale);
    int4 packed;
    std::int32_t* words = reinterpret_cast<std::int32_t*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        std::uint16_t code2 = 0;
        code2 |= static_cast<std::uint16_t>(codes8[2 * i]);
        code2 |= static_cast<std::uint16_t>(codes8[2 * i + 1]) << 8;
        const __half2 values = __hmul2(kv_cache_fp8_code2_to_half2(code2), scale2);
        const __nv_bfloat162 packed_bf16 = __float22bfloat162_rn(__half22float2(values));
        std::memcpy(&words[i], &packed_bf16, sizeof(packed_bf16));
    }
    return packed;
}

} // namespace ninfer::ops
