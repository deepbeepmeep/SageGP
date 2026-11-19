// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// The running maxima include SageAttention's FP8 exponent offset. Most later
// KV tiles fit the current range. In that case a single warp vote skips the
// row reductions and the rescaling of the entire FP32 output fragment.
template<uint32_t q_tiles, uint32_t k_tiles, uint32_t value_tiles,
         bool half_output, bool fp8_offset, bool scaled>
__device__ __forceinline__ void dense_update(float scores[][k_tiles][8],
    float output[][value_tiles][8], float maxima[][2], float sums[][2], float scale) {
  static_assert(!half_output && fp8_offset);
  float peaks[q_tiles][2];
  bool update = false;
  #pragma unroll
  for (int q = 0; q < q_tiles; ++q) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
      float peak = -3.4e38f;
      #pragma unroll
      for (int k = 0; k < k_tiles; ++k) {
        const float* s = scores[q][k] + 2 * row;
        peak = fmaxf(peak, fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5])));
      }
      peaks[q][row] = (scaled ? peak : peak * scale) - S_FP8_OFFSET;
      update |= peaks[q][row] > maxima[q][row];
    }
  }
  if (__any_sync(0xffffffff, update)) {
    #pragma unroll
    for (int q = 0; q < q_tiles; ++q) {
      #pragma unroll
      for (int row = 0; row < 2; ++row) {
        float peak = peaks[q][row];
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, 1));
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, 2));
        const float old = maxima[q][row];
        // Two bits of exponent headroom avoid repeated rescales on small increases.
        const float next = peak > old ? peak + 2.f : old;
        const float factor = math::ptx_exp2(old - next);
        sums[q][row] *= factor;
        maxima[q][row] = next;
        #pragma unroll
        for (int v = 0; v < value_tiles; ++v) {
          float* p = output[q][v] + 2 * row;
          p[0] *= factor;
          p[1] *= factor;
          p[4] *= factor;
          p[5] *= factor;
        }
      }
    }
  }
  #pragma unroll
  for (int q = 0; q < q_tiles; ++q) {
    #pragma unroll
    for (int k = 0; k < k_tiles; ++k) {
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        scores[q][k][i] = math::ptx_exp2(fmaf(scores[q][k][i], scaled ? 1.f : scale, -maxima[q][(i % 4) / 2]));
    }
  }
}
