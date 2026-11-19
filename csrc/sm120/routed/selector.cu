// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include <cuda_bf16.h>

__device__ float warp_sum(float x) {
#pragma unroll
  for (int i = 16; i; i /= 2) x += __shfl_xor_sync(0xffffffff, x, i);
  return x;
}

// Positive-diagonal QR of (shifted covariance)^3, in the original variance
// pivot order. This is the same Q as three consecutive orthogonal iterations in
// exact arithmetic. Modified Gram-Schmidt avoids squaring the condition number.
extern "C" __global__ void qr128(const float* cov, const float* power,
                                 float* q) {
  extern __shared__ float scratch[];
  __shared__ float diag[128], pivot[128];
  __shared__ int rank[128];
  int t = threadIdx.x, w = t / 32, lane = t % 32, h = blockIdx.x;
  cov += h * 16384;
  power += h * 16384;
  q += h * 16384;
  if (t < 128) diag[t] = cov[t * 129];
  __syncthreads();
  if (t < 128) {
    int pos = 0;
    for (int i = 0; i < 128; ++i)
      pos += diag[i] > diag[t] || (diag[i] == diag[t] && i < t);
    rank[t] = pos;
  }
  __syncthreads();
  for (int i = t; i < 16384; i += blockDim.x)
    scratch[rank[i % 128] * 129 + i / 128] = power[i];
  __syncthreads();
  for (int j = 0; j < 128; ++j) {
    if (w == 0) {
      float a[4], n = 0;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        a[g] = scratch[j * 129 + 32 * g + lane];
        n += a[g] * a[g];
      }
      float inv = rsqrtf(fmaxf(warp_sum(n), 1e-35f));
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        pivot[32 * g + lane] = a[g] * inv;
        scratch[j * 129 + 32 * g + lane] = a[g] * inv;
      }
    }
    __syncthreads();
    float p[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) p[g] = pivot[32 * g + lane];
    for (int col = j + 1 + w; col < 128; col += blockDim.x / 32) {
      float a[4], dot = 0;
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        a[g] = scratch[col * 129 + 32 * g + lane];
        dot += a[g] * p[g];
      }
      dot = warp_sum(dot);
#pragma unroll
      for (int g = 0; g < 4; ++g)
        scratch[col * 129 + 32 * g + lane] = a[g] - dot * p[g];
    }
    __syncthreads();
  }
  for (int i = t; i < 16384; i += blockDim.x)
    q[i] = scratch[(i % 128) * 129 + i / 128];
}

// Three independent leading-principal ridge regressions per head. Pivot columns
// need no inter-CTA synchronization. The covariance plus ridge is positive
// definite.
extern "C" __global__ void regress(const float* cov, const float* mean,
                                   float* weights, float* bias) {
  extern __shared__ float a[];
  __shared__ float col[96], row[128], ridge;
  int t = threadIdx.x, h = blockIdx.x, kind = blockIdx.y,
      width = (kind + 1) * 32;
  cov += h * 16384;
  mean += h * 128;
  weights += (h * 3 + kind) * 16384;
  bias += (h * 3 + kind) * 128;
  if (t == 0) {
    float d = 0.f;
    for (int i = 0; i < width; ++i) d += cov[i * 129];
    ridge = d * (0.06f / width) + 1e-8f;
  }
  __syncthreads();
  for (int i = t; i < width * 128; i += blockDim.x)
    a[(i / 128) * 129 + i % 128] = cov[i] + (i / 128 == i % 128 ? ridge : 0.f);
  __syncthreads();
  for (int k = 0; k < width; ++k) {
    float inv = 1.f / fmaxf(a[k * 129 + k], 1e-20f);
    if (t < width) col[t] = a[t * 129 + k];
    if (t < 128) row[t] = a[k * 129 + t] * inv;
    __syncthreads();
    for (int i = t; i < width * 128; i += blockDim.x) {
      int r = i / 128, d = i % 128;
      if (d > k)
        a[r * 129 + d] =
            r == k ? row[d] : fmaf(-col[r], row[d], a[r * 129 + d]);
    }
    __syncthreads();
  }
  for (int i = t; i < width * 128; i += blockDim.x)
    if (i % 128 >= width) weights[i] = a[(i / 128) * 129 + i % 128];
  if (t >= width && t < 128) {
    float x = mean[t];
    for (int i = 0; i < width; ++i) x = fmaf(-mean[i], a[i * 129 + t], x);
    bias[t] = x;
  }
}
