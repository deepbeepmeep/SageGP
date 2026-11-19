// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include "qattn/wide_attention.cuh"

template <bool compact>
__device__ __forceinline__ void run(int8_t *q, int8_t *k, half *v, nv_bfloat16 *out, float *qs, float *ks,
                                    float *vs, uint8_t *flags, int n, int heads) {
  if (bool(flags[blockIdx.y]) != compact)
    return;
  constexpr int d = 256;
  qk_int_sv_f16_attn_kernel<64, 64, 16, 64, d, DataType::kInt8, QuantGranularity::kPerWarp,
                            QuantGranularity::kPerWarp, float, true, nv_bfloat16, ComputeUnit::kTensorCore,
                            MaskMode::kNone, false, false, compact>(
      q, k, v, out, nullptr, qs, ks, reinterpret_cast<nv_bfloat16 *>(vs), n, n, 1, n * heads * d, d, n * d,
      n * heads * d, d, n * d, n * heads * d, d, n * d, n * heads * d, heads * d, d, 0.0625f);
}

#define WIDE_ARGUMENTS                                                                                         \
  int8_t *q, int8_t *k, half *v, nv_bfloat16 *out, float *qs, float *ks, float *vs, uint8_t *flags, int n,     \
      int heads
extern "C" __global__ void dense_compact(WIDE_ARGUMENTS) {
  run<true>(q, k, v, out, qs, ks, vs, flags, n, heads);
}
extern "C" __global__ void dense_full(WIDE_ARGUMENTS) { run<false>(q, k, v, out, qs, ks, vs, flags, n, heads); }
