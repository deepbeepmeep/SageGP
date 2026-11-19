// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// Native SM120a NVFP4 QK. Each k64 slice has four E4M3 scales per row.
#ifndef INT8_PREFIX_FRAGMENTS
#define INT8_PREFIX_FRAGMENTS 0
#endif
template<uint32_t wq, uint32_t wk, uint32_t tq, SwizzleMode sw, uint32_t stride>
__device__ __forceinline__ void read_qscales(const smem_t<sw,stride>& q, uint32_t (&scales)[tq][SCORE_PARTS]) {
  const int lane=get_lane_id();
  #pragma unroll
  for(int i=0;i<tq;++i){
    const int row=(get_warp_idx_q<wq,wk>()*tq+i)*16+lane/4+(lane%2)*8;
    const auto* s=reinterpret_cast<const uint32_t*>(q.base+q.get_permuted_offset(row,6));
    #pragma unroll
    for(int part=0;part<SCORE_PARTS;++part)scales[i][part]=s[part];
  }
}

template<uint32_t wq,uint32_t wk,uint32_t tq,uint32_t tk,uint32_t inner,SwizzleMode sw,uint32_t stride,DataType type>
__device__ __forceinline__ void scaled_qk(const smem_t<sw,stride>& q,const smem_t<sw,stride>& k,
    int32_t scores[][tk][8],uint32_t qo,uint32_t ko,const uint32_t (&qscale)[tq][SCORE_PARTS]){
  const int lane=get_lane_id();
  #pragma unroll
  for(int part=0;part<INT8_PREFIX_FRAGMENTS+SCORE_PARTS;++part){
    uint32_t qr[tq][4];
    #pragma unroll
    for(int i=0;i<tq;++i){
      q.ldmatrix_m8n8x4(qo,qr[i]);
      qo=q.template advance_offset_by_row<16>(qo);
    }
    qo=q.template advance_offset_by_column<2>(qo-tq*16*stride,part);
    #pragma unroll
    for(int j=0;j<tk;++j){
      uint32_t kr[4];
      k.ldmatrix_m8n8x4(ko,kr);
      ko=k.template advance_offset_by_row<16>(ko);
      if(part<INT8_PREFIX_FRAGMENTS){
        static_assert(INT8_PREFIX_FRAGMENTS<=1);
        #pragma unroll
        for(int i=0;i<tq;++i)
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(scores[i][j],qr[i],kr);
        continue;
      }
      const int fp4_part=part-INT8_PREFIX_FRAGMENTS;
      #pragma unroll
      for(int half=0;half<2;++half){
        const int row=(get_warp_idx_k<wq,wk>()*tk+j)*16+half*8+lane/4;
        const uint32_t ks=reinterpret_cast<const uint32_t*>(k.base+k.get_permuted_offset(row,6))[fp4_part];
        #pragma unroll
        for(int i=0;i<tq;++i){
          float a[4];
          #pragma unroll
          for(int z=0;z<4;++z){
            if(fp4_part) a[z]=__int_as_float(scores[i][j][half*4+z]);
            else if constexpr(INT8_PREFIX_FRAGMENTS) a[z]=__int2float_rn(scores[i][j][half*4+z]);
            else a[z]=0.f;
          }
          asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%12,%12},{%11},{%12,%12};"
             : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3])
             : "r"(qr[i][0]),"r"(qr[i][1]),"r"(qr[i][2]),"r"(qr[i][3]),
               "r"(kr[2*half]),"r"(kr[2*half+1]),"r"(qscale[i][fp4_part]),"r"(ks),"h"(uint16_t(0)));
          #pragma unroll
          for(int z=0;z<4;++z)scores[i][j][half*4+z]=__float_as_int(a[z]);
        }
      }
    }
    ko=k.template advance_offset_by_column<2>(ko-tk*16*stride,part);
  }
}
