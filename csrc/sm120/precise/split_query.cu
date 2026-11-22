// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

// The precise kernel over a range of the keys: blockIdx.z is the range, each range writes its normalized output and its log2-sum-exp
// (lse) at its own index from query block out_first (out_rows rows per range and head), merged afterwards (sagegp.sm120.dense: split
// keys), so that the query blocks of a part-filled last wave fill it.
template<bool mixed>
__device__ __forceinline__ void attend_split(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out, float* lse,
    float* qs, float* ks, float* vs, int t, int n, int h, int vrows, int obz, int ost, int osh, int qst, int qsh, int first,
    int splits, int split_len, int out_first, int out_rows) {
  constexpr int d = 128;
  qk_int_sv_f8_attn_kernel<128,64,32,64,d,DataType::kInt8,QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,float,true,nv_bfloat16,ComputeUnit::kTensorCore,
      MaskMode::kNone,true,true,false,true,mixed>(q,k,v,out,lse,qs,ks,vs,nullptr,t,n,1,
      0,qst,qsh,n*h*d,d,n*d,h*d*vrows,d*vrows,vrows,obz,ost,osh,0.08838834764831845f,first,splits,split_len,out_first,out_rows);
}

extern "C" __global__ void wangp_dense_precise_split(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out, float* lse,
    float* qs, float* ks, float* vs, uint8_t* mixed_heads, int t, int n, int h, int vrows,
    int obz, int ost, int osh, int qst, int qsh, int first, int splits, int split_len, int out_first, int out_rows) {
  if (mixed_heads[blockIdx.y])
    attend_split<true>(q,k,v,out,lse,qs,ks,vs,t,n,h,vrows,obz,ost,osh,qst,qsh,first,splits,split_len,out_first,out_rows);
  else
    attend_split<false>(q,k,v,out,lse,qs,ks,vs,t,n,h,vrows,obz,ost,osh,qst,qsh,first,splits,split_len,out_first,out_rows);
}
