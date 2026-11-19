// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

// Each head independently selects its score precision on-device. Ordinary
// INT8 and mixed heads share the same launch and output layout.
template<bool mixed>
__device__ __forceinline__ void attend(int8_t* q, int8_t* k, int8_t* v,
    nv_bfloat16* out, float* qs, float* ks, float* vs,
    int t, int n, int h, int vrows, int out_token_stride, int out_head_stride, int query_block) {
  constexpr int d = 128;
  qk_int_sv_f8_attn_kernel<128,64,32,64,d,DataType::kInt8,QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,float,true,nv_bfloat16,ComputeUnit::kTensorCore,
      MaskMode::kNone,false,true,false,true,mixed>(q,k,v,out,nullptr,qs,ks,vs,nullptr,t,n,1,
      t*h*d,d,t*d,n*h*d,d,n*d,h*d*vrows,d*vrows,vrows,
      0,out_token_stride,out_head_stride,0.08838834764831845f,query_block);
}

extern "C" __global__ void wangp_dense_nvfp4(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out,
    float* qs, float* ks, float* vs, uint8_t* mixed_heads, int t, int n, int h, int vrows,
    int out_token_stride, int out_head_stride, int query_block) {
  if (mixed_heads[blockIdx.y])
    attend<true>(q,k,v,out,qs,ks,vs,t,n,h,vrows,out_token_stride,out_head_stride,query_block);
  else
    attend<false>(q,k,v,out,qs,ks,vs,t,n,h,vrows,out_token_stride,out_head_stride,query_block);
}
