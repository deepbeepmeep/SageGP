// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// The compiled images include modified SageAttention CUDA code,
// Copyright (c) 2024 by SageAttention team. See ../LICENSE-SageAttention.txt.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

// attention.cu's routed kernel over a range of the keys: blockIdx.z is the range, each range writes its normalized output and its
// log2-sum-exp (lse) at its own index from query block out_first (out_rows rows per range and head), merged afterwards
// (sagegp.sm120.dense: split keys), so that the query blocks of a part-filled last wave fill it. The arguments are the precise split
// kernel's, the heads' choices in place of its mixed flags.
template<int mode>
__device__ __forceinline__ void attend_split(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out, float* lse,
    float* qs, float* ks, float* vs, int t, int n, int h, int vrows, int obz,
    int ost, int osh, int qst, int qsh, int first, int splits, int split_len, int out_first, int out_rows) {
  constexpr int d = 128;
  qk_int_sv_f8_attn_kernel<128,64,32,64,d,DataType::kInt8,QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,float,true,nv_bfloat16,ComputeUnit::kTensorCore,
      MaskMode::kNone,true,true,(VALUE_WIDTH != 0),true,(mode == 2)
#if VALUE_WIDTH
      ,mode
#endif
      >(q,k,v,out,lse,qs,ks,vs,VALUE_WIDTH ? vs+h*d : nullptr,t,n,1,
        0,qst,qsh,n*h*176,176,n*176,h*d*vrows,d*vrows,vrows,obz,ost,osh,0.08838834764831845f,first,splits,split_len,out_first,out_rows);
}

extern "C" __global__ void wangp_dense_routed_split(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out, float* lse,
    float* qs, float* ks, float* vs, uint8_t* choices, int t, int n, int h, int vrows,
    int obz, int ost, int osh, int qst, int qsh, int first, int splits, int split_len, int out_first, int out_rows) {
  unsigned choice = choices[blockIdx.y];
#define ATTEND(mode) attend_split<mode>(q,k,v,out,lse,qs,ks,vs,t,n,h,vrows,obz,ost,osh,qst,qsh,first,splits,split_len,out_first,out_rows)
#if VALUE_WIDTH
  if (choice / 4 != VALUE_WIDTH / 32 - 1) return;
  switch (choice % 4) {
    case 0: ATTEND(0); break;
    case 1: ATTEND(1); break;
    case 2: ATTEND(2); break;
    case 3: ATTEND(3); break;
  }
#else
  if (choice == SCORE_MODE) ATTEND(SCORE_MODE);
#endif
#undef ATTEND
}
