// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

// Q's packed bytes occupy the start of their original BF16 row. Each CTA
// finishes loading its own query tile before replacing those rows with output.
template<bool mixed>
__device__ __forceinline__ void attend_strided(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out,
    float* qs, float* ks, float* vs, int t, int n, int h, int vrows, int ost, int osh, int qst, int qsh, int first) {
  constexpr int d = 128;
  qk_int_sv_f8_attn_kernel<128,64,32,64,d,DataType::kInt8,QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,float,true,nv_bfloat16,ComputeUnit::kTensorCore,
      MaskMode::kNone,false,true,false,true,mixed>(q,k,v,out,nullptr,qs,ks,vs,nullptr,t,n,1,
      0,qst,qsh,n*h*d,d,n*d,h*d*vrows,d*vrows,vrows,0,ost,osh,0.08838834764831845f,first);
}

extern "C" __global__ void wangp_dense_nvfp4_strided_q(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out,
    float* qs, float* ks, float* vs, uint8_t* mixed_heads, int t, int n, int h, int vrows,
    int ost, int osh, int qst, int qsh, int first) {
  if (mixed_heads[blockIdx.y])
    attend_strided<true>(q,k,v,out,qs,ks,vs,t,n,h,vrows,ost,osh,qst,qsh,first);
  else
    attend_strided<false>(q,k,v,out,qs,ks,vs,t,n,h,vrows,ost,osh,qst,qsh,first);
}
