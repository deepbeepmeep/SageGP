// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// Uses SageAttention's shared-memory layout and INT8 fragment conventions.
// One query fragment consumes the same scale word for every key tile.
template <uint32_t warps_q, uint32_t warps_k, uint32_t tiles_q, SwizzleMode swizzle, uint32_t stride>
__device__ __forceinline__ void load_query_scales(
    const smem_t<swizzle, stride>& q, uint32_t (&scales)[tiles_q]) {
  const int lane = get_lane_id();
  const int warp = get_warp_idx_q<warps_q, warps_k>();
  #pragma unroll
  for (int i = 0; i < tiles_q; ++i) {
    const int row = (warp * tiles_q + i) * 16 + lane / 4 + (lane % 2) * 8;
    scales[i] = *reinterpret_cast<const uint32_t*>(q.base + q.get_permuted_offset(row, 6));
  }
}

template <uint32_t warps_q, uint32_t warps_k, uint32_t tiles_q, uint32_t tiles_k,
          uint32_t inner, SwizzleMode swizzle, uint32_t stride, DataType type>
__device__ __forceinline__ void mixed_qk(
    const smem_t<swizzle, stride>& q, const smem_t<swizzle, stride>& k,
    int32_t scores[][tiles_k][8], uint32_t qoff, uint32_t koff, const uint32_t (&qscales)[tiles_q]) {
  static_assert(inner == 4 && type == DataType::kInt8);
  // Each 128-byte row holds 64 INT8s, 64 packed E2M1s and four UE4M3 scales.
  // Offsets are local: successive KV tiles start at the same shared addresses.
  #pragma unroll
  for (int part = 0; part < 3; ++part) {
    uint32_t qr[tiles_q][4];
    #pragma unroll
    for (int i = 0; i < tiles_q; ++i) {
      q.ldmatrix_m8n8x4(qoff, qr[i]);
      qoff = q.template advance_offset_by_row<16>(qoff);
    }
    qoff = q.template advance_offset_by_column<2>(qoff - tiles_q * 16 * stride, part);
    #pragma unroll
    for (int j = 0; j < tiles_k; ++j) {
      uint32_t kr[4];
      k.ldmatrix_m8n8x4(koff, kr);
      koff = k.template advance_offset_by_row<16>(koff);
      #pragma unroll
      for (int i = 0; i < tiles_q; ++i) {
        if (part == 0) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(scores[i][j], qr[i], kr);
        } else if (part == 1) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(scores[i][j], qr[i], kr);
        } else {
          const int lane = get_lane_id();
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int krow = (get_warp_idx_k<warps_q, warps_k>() * tiles_k + j) * 16 + half * 8 + lane / 4;
            const uint32_t ks = *reinterpret_cast<const uint32_t*>(k.base + k.get_permuted_offset(krow, 6));
            float acc[4];
            #pragma unroll
            for (int x = 0; x < 4; ++x) acc[x] = __int2float_rn(scores[i][j][4 * half + x]);
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%12,%12},{%11},{%12,%12};"
                         : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                         : "r"(qr[i][0]), "r"(qr[i][1]), "r"(qr[i][2]), "r"(qr[i][3]),
                           "r"(kr[2 * half]), "r"(kr[2 * half + 1]), "r"(qscales[i]), "r"(ks), "h"(uint16_t(0)));
            #pragma unroll
            for (int x = 0; x < 4; ++x) scores[i][j][4 * half + x] = __float_as_int(acc[x]);
          }
        }
      }
    }
    koff = k.template advance_offset_by_column<2>(koff - tiles_k * 16 * stride, part);
  }
}
