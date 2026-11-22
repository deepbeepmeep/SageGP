// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

// SageAttention 2's sm89 kernel (INT8 QK per-thread scales, FP8 PV, fused V scale, instruction buffer: the kernel of
// qk_int8_sv_f8_accum_f{16,32}_fuse_v_scale_attn_inst_buf) over a range of the keys: blockIdx.z is batch * splits + range, each range writes
// its normalized output and log2-sum-exp from query block out_first (sagegp.sm89.split merges them), so that the query blocks of a
// part-filled last wave fill it. Layout NHD for Q, K and O, V (batch, heads, dim, keys padded).
template<int HD, typename DTypeOut, bool pv_fp16>
__device__ __forceinline__ void attend_split(int8_t* q, int8_t* k, int8_t* v, DTypeOut* out, float* lse, float* qs, float* ks, float* vs,
    int qo_len, int kv_len, int groups, int sbq, int ssq, int shq, int sbk, int ssk, int shk, int sbv, int shv, int sdv,
    int sbo, int sso, int sho, float sm_scale, int first, int splits, int split_len, int out_first, int out_rows) {
  qk_int_sv_f8_attn_kernel<128, 64, 32, 64, HD, DataType::kInt8, QuantGranularity::kPerThread, QuantGranularity::kPerThread,
      float, true, DTypeOut, ComputeUnit::kCudaCore, MaskMode::kNone, true, true, false, pv_fp16>(
      q, k, v, out, lse, qs, ks, vs, nullptr, qo_len, kv_len, groups, sbq, ssq, shq, sbk, ssk, shk, sbv, shv, sdv, sbo, sso, sho,
      sm_scale, first, splits, split_len, out_first, out_rows);
}

#define WGP_SPLIT_ENTRY(NAME, HD, TYPE, PV)                                                                                              \
  extern "C" __global__ void NAME(int8_t* q, int8_t* k, int8_t* v, TYPE* out, float* lse, float* qs, float* ks, float* vs,               \
      int qo_len, int kv_len, int groups, int sbq, int ssq, int shq, int sbk, int ssk, int shk, int sbv, int shv, int sdv,                  \
      int sbo, int sso, int sho, float sm_scale, int first, int splits, int split_len, int out_first, int out_rows) {                      \
    attend_split<HD, TYPE, PV>(q, k, v, out, lse, qs, ks, vs, qo_len, kv_len, groups, sbq, ssq, shq, sbk, ssk, shk, sbv, shv, sdv,      \
                               sbo, sso, sho, sm_scale, first, splits, split_len, out_first, out_rows);                                   \
  }

WGP_SPLIT_ENTRY(wangp_sm89_split_d64_bf16_pv16, 64, nv_bfloat16, true)
WGP_SPLIT_ENTRY(wangp_sm89_split_d64_bf16_pv32, 64, nv_bfloat16, false)
WGP_SPLIT_ENTRY(wangp_sm89_split_d64_fp16_pv16, 64, half, true)
WGP_SPLIT_ENTRY(wangp_sm89_split_d64_fp16_pv32, 64, half, false)
WGP_SPLIT_ENTRY(wangp_sm89_split_d128_bf16_pv16, 128, nv_bfloat16, true)
WGP_SPLIT_ENTRY(wangp_sm89_split_d128_bf16_pv32, 128, nv_bfloat16, false)
WGP_SPLIT_ENTRY(wangp_sm89_split_d128_fp16_pv16, 128, half, true)
WGP_SPLIT_ENTRY(wangp_sm89_split_d128_fp16_pv32, 128, half, false)
