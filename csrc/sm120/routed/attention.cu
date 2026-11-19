// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// The compiled images include modified SageAttention CUDA code,
// Copyright (c) 2024 by SageAttention team. See ../LICENSE-SageAttention.txt.
#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

template<int mode>
__device__ __forceinline__ void attend(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out,
    float* qs, float* ks, float* vs, int t, int n, int h, int vrows,
    int ost, int osh, int qst, int qsh, int first) {
  constexpr int d = 128;
  qk_int_sv_f8_attn_kernel<128,64,32,64,d,DataType::kInt8,QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,float,true,nv_bfloat16,ComputeUnit::kTensorCore,
      MaskMode::kNone,false,true,(VALUE_WIDTH != 0),true,(mode == 2)
#if VALUE_WIDTH
      ,mode
#endif
      >(q,k,v,out,nullptr,qs,ks,vs,VALUE_WIDTH ? vs+h*d : nullptr,t,n,1,
        0,qst,qsh,n*h*176,176,n*176,h*d*vrows,d*vrows,vrows,0,ost,osh,0.08838834764831845f,first);
}

extern "C" __global__ void wangp_dense_nvfp4_strided_q(int8_t* q, int8_t* k, int8_t* v, nv_bfloat16* out,
    float* qs, float* ks, float* vs, uint8_t* choices, int t, int n, int h, int vrows,
    int ost, int osh, int qst, int qsh, int first) {
  unsigned choice = choices[blockIdx.y];
#define ATTEND(mode) attend<mode>(q,k,v,out,qs,ks,vs,t,n,h,vrows,ost,osh,qst,qsh,first)
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
