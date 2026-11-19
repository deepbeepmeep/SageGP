// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// The three parts of dense_update (online_softmax.cuh), with the same expressions, so that the exponentials
// of a tile can be interleaved with the P.V products of the previous tile.
template <uint32_t q_tiles, uint32_t k_tiles>
__device__ __forceinline__ bool xt_peaks(float scores[][k_tiles][8], float maxima[][2], float peaks[][2], float scale) {
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
      peaks[q][row] = peak * scale - S_FP8_OFFSET;
      update |= peaks[q][row] > maxima[q][row];
    }
  }
  return update;
}

template <uint32_t q_tiles, uint32_t value_tiles>
__device__ __forceinline__ void xt_rescale(float peaks[][2], float output[][value_tiles][8], float maxima[][2], float sums[][2]) {
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

// Exponentials of one 16 x 16 score fragment.
__device__ __forceinline__ void xt_exp(float scores[8], const float maxima[2], float scale) {
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    scores[i] = math::ptx_exp2(fmaf(scores[i], scale, -maxima[(i % 4) / 2]));
}

// FP8 probabilities of two neighbouring score fragments (32 keys), as RS_32_to_8 packs them.
__device__ __forceinline__ void xt_pack(uint32_t packed[4], float even[8], float odd[8]) {
  floatx4_to_e4m3x4(packed, even, even + 4);
  floatx4_to_e4m3x4(packed + 1, even + 2, even + 6);
  floatx4_to_e4m3x4(packed + 2, odd, odd + 4);
  floatx4_to_e4m3x4(packed + 3, odd + 2, odd + 6);
}
