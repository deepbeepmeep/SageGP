// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// One 128-byte row represents 96 INT8 channels, 48 scaled E2M1 channels,
// and four microscales. Q and K put scales in different, zero-opposed
// lanes of the last MMA, preserving the selected 144 feature products.
template <uint32_t wq, uint32_t wk, uint32_t tq, SwizzleMode sw, uint32_t st>
__device__ __forceinline__ void load_query_scales(const smem_t<sw, st> &q, uint32_t (&s)[tq]) {
  const int lane = get_lane_id(), warp = get_warp_idx_q<wq, wk>();
#pragma unroll
  for (int i = 0; i < tq; ++i) {
    int r = (warp * tq + i) * 16 + lane / 4 + (lane % 2) * 8;
    s[i] = reinterpret_cast<const uint32_t *>(q.base + q.get_permuted_offset(r, 7))[2];
  }
}
template <uint32_t wq, uint32_t wk, uint32_t tq, uint32_t tk, uint32_t inner, SwizzleMode sw, uint32_t st,
          DataType type>
__device__ __forceinline__ void mixed_qk(const smem_t<sw, st> &q, const smem_t<sw, st> &k, int32_t o[][tk][8],
                                         uint32_t qo, uint32_t ko, const uint32_t (&qs)[tq]) {
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    uint32_t a[tq][4];
#pragma unroll
    for (int i = 0; i < tq; ++i) {
      q.ldmatrix_m8n8x4(qo, a[i]);
      qo = q.template advance_offset_by_row<16>(qo);
    }
    qo = q.template advance_offset_by_column<2>(qo - tq * 16 * st, part);
#pragma unroll
    for (int j = 0; j < tk; ++j) {
      uint32_t b[4];
      k.ldmatrix_m8n8x4(ko, b);
      ko = k.template advance_offset_by_row<16>(ko);
#pragma unroll
      for (int i = 0; i < tq; ++i) {
        if (part == 0)
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(o[i][j], a[i], b);
        else if (part < 3)
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(o[i][j], a[i], b);
        else {
#pragma unroll
          for (int side = 0; side < 2; ++side) {
            const int row = (get_warp_idx_k<wq, wk>() * tk + j) * 16 + side * 8 + get_lane_id() / 4;
            const uint32_t ks = reinterpret_cast<const uint32_t *>(k.base + k.get_permuted_offset(row, 7))[3];
            float c[4];
#pragma unroll
            for (int x = 0; x < 4; ++x)
              c[x] = __int2float_rn(o[i][j][4 * side + x]);
            asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
                         "e2m1.f32.ue4m3 "
                         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%12,%12},{%11},{%12,%12};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]), "r"(b[2 * side]),
                           "r"(b[2 * side + 1]), "r"(qs[i]), "r"(ks), "h"(uint16_t(0)));
#pragma unroll
            for (int x = 0; x < 4; ++x)
              o[i][j][4 * side + x] = __float_as_int(c[x]);
          }
        }
      }
    }
    ko = k.template advance_offset_by_column<2>(ko - tk * 16 * st, part);
  }
}
