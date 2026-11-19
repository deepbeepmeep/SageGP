// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// P.V of one 64-key tile, one 16-channel value fragment per basic block (round 2 of pv_steps.cuh).
// Same products and sums as SageAttention's compute_fp8_sv_inst_buf_fp16_accu: each FP16 accumulator receives its
// fk = 0 product, then its fk = 1 product, and is added once to the FP32 output. Only the order between accumulators
// changes: an accumulator lives for 8 products instead of 64, its conversion overlaps the next fragment's products
// and V fragments are read two fragments ahead.

// A never-taken branch. ptxas schedules inside a basic block and moves instructions without later readers in the block
// to its end: ending the block keeps the work of a stage between that stage's products. Each boundary tests another
// value, since ptxas removes a branch whose condition an earlier branch already decided.
#define WGP_STAGE(n, value, never) asm volatile("{ .reg .pred p; .reg .u32 t; add.u32 t, %1, " #n "; setp.ne.u32 p, %0, t; @p bra wgp_stage; trap; wgp_stage: }" :: "r"(value), "r"(never) : "memory")
__device__ __forceinline__ void wgp_stage(uint32_t n, uint32_t value, uint32_t never) {
  switch (n) {
    case 0: WGP_STAGE(0, value, never); break; case 1: WGP_STAGE(1, value, never); break; case 2: WGP_STAGE(2, value, never); break;
    case 3: WGP_STAGE(3, value, never); break; case 4: WGP_STAGE(4, value, never); break; case 5: WGP_STAGE(5, value, never); break;
    case 6: WGP_STAGE(6, value, never); break; case 7: WGP_STAGE(7, value, never); break; case 8: WGP_STAGE(8, value, never); break;
    case 9: WGP_STAGE(9, value, never); break; case 10: WGP_STAGE(10, value, never); break; case 11: WGP_STAGE(11, value, never); break;
    case 12: WGP_STAGE(12, value, never); break; case 13: WGP_STAGE(13, value, never); break; case 14: WGP_STAGE(14, value, never); break;
    default: WGP_STAGE(15, value, never); break;
  }
}
// A value >= 2^31 that the compiler does not know (lane and tile indices stay below 2^31).
__device__ __forceinline__ uint32_t wgp_never(uint32_t seed) {
  uint32_t never;
  asm volatile("or.b32 %0, %1, 0x80000000;" : "=r"(never) : "r"(seed));
  return never;
}

template <uint32_t tiles_q>
__device__ __forceinline__ void wgp_pv_products(uint32_t acc[][4], uint32_t P8[][2][4], uint32_t* RV, uint32_t fk) {
  #pragma unroll
  for (uint32_t fq = 0; fq < tiles_q; fq++) {
    if (fk == 0) mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInit>(acc[fq], P8[fq][0], RV);
    else mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInplaceUpdate>(acc[fq], P8[fq][1], RV);
  }
}

template <uint32_t tiles_q, uint32_t tiles_v>
__device__ __forceinline__ void wgp_pv_add(float RO[][tiles_v][8], uint32_t acc[][4], uint32_t fv) {
  float pair[2];
  #pragma unroll
  for (uint32_t fq = 0; fq < tiles_q; fq++) {
    #pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      unpack_half2_from_uint32_to_float(pair, acc[fq][k]);
      RO[fq][fv][k * 2 + 0] += pair[0];
      RO[fq][fv][k * 2 + 1] += pair[1];
    }
  }
}

// Shared-memory offset of the value fragment (fv, fk) for this lane (64-byte rows, one row per channel).
template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ uint32_t wgp_v_offset(uint32_t fv, uint32_t fk) {
  const uint32_t row = get_lane_id() % 8 + (get_lane_id() / 16) * 8, col = (get_lane_id() / 8) % 2;
  return smem_t<swizzle_mode, stride>::get_permuted_offset(row, col + fk * 2) + fv * 16 * stride;
}

// The first two value fragments, read before their tile's softmax when V is already in shared memory.
template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void wgp_pv_read_first(const smem_t<swizzle_mode, stride>& smem_V, uint32_t first[2][2][4]) {
  #pragma unroll
  for (uint32_t s = 0; s < 2; s++) {
    smem_V.ldmatrix_m8n8x4(wgp_v_offset<swizzle_mode, stride>(s, 0), first[s][0]);
    smem_V.ldmatrix_m8n8x4(wgp_v_offset<swizzle_mode, stride>(s, 1), first[s][1]);
  }
}

// `used` value fragments of `tiles_v` (the routed Fast kernels keep VALUE_WIDTH / 16 of them).
// With `hold`, the products of the last fragment are left to the caller (after its next barrier): its two V halves,
// read here, are returned in `held`. With `preloaded`, `first` holds the fragments 0 and 1 (wgp_pv_read_first).
template <uint32_t tiles_q, uint32_t tiles_v, uint32_t used, SwizzleMode swizzle_mode, uint32_t stride, bool hold, bool preloaded = false>
__device__ __forceinline__ void wgp_pv_staged(const smem_t<swizzle_mode, stride>& smem_V, uint32_t P8[][2][4], float RO[][tiles_v][8], uint32_t held[2][4],
                                              uint32_t first[2][2][4] = nullptr) {
  static_assert(used >= 2 && used <= tiles_v);
  const uint32_t lane = get_lane_id(), never = wgp_never(lane);
  constexpr uint32_t issued = hold ? used - 1 : used;  // fragments whose products are issued here
  uint32_t RV[3][2][4], acc[2][tiles_q][4];
  if constexpr (preloaded) {
    #pragma unroll
    for (uint32_t s = 0; s < 2; s++) {
      #pragma unroll
      for (uint32_t k = 0; k < 4; k++) {
        RV[s][0][k] = first[s][0][k];
        RV[s][1][k] = first[s][1][k];
      }
    }
  } else {
    wgp_pv_read_first<swizzle_mode, stride>(smem_V, RV);
  }
  #pragma unroll
  for (uint32_t fv = 0; fv < issued; fv++) {
    if (fv > 0) wgp_stage(fv, lane, never);
    if (fv + 2 < used) {
      smem_V.ldmatrix_m8n8x4(wgp_v_offset<swizzle_mode, stride>(fv + 2, 0), RV[(fv + 2) % 3][0]);
      smem_V.ldmatrix_m8n8x4(wgp_v_offset<swizzle_mode, stride>(fv + 2, 1), RV[(fv + 2) % 3][1]);
    }
    wgp_pv_products<tiles_q>(acc[fv % 2], P8, RV[fv % 3][0], 0);
    if (fv > 0) wgp_pv_add<tiles_q, tiles_v>(RO, acc[(fv + 1) % 2], fv - 1);
    wgp_pv_products<tiles_q>(acc[fv % 2], P8, RV[fv % 3][1], 1);
  }
  wgp_stage(issued, lane, never);
  wgp_pv_add<tiles_q, tiles_v>(RO, acc[(issued - 1) % 2], issued - 1);
  if constexpr (hold) {
    #pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      held[0][k] = RV[(used - 1) % 3][0][k];
      held[1][k] = RV[(used - 1) % 3][1][k];
    }
  }
}

// The held fragment of the previous tile: its 8 products and its sums.
template <uint32_t tiles_q, uint32_t tiles_v, uint32_t used>
__device__ __forceinline__ void wgp_pv_held(uint32_t P8[][2][4], float RO[][tiles_v][8], uint32_t held[2][4]) {
  uint32_t acc[tiles_q][4];
  wgp_pv_products<tiles_q>(acc, P8, held[0], 0);
  wgp_pv_products<tiles_q>(acc, P8, held[1], 1);
  wgp_pv_add<tiles_q, tiles_v>(RO, acc, used - 1);
}

#ifndef WGP_VALUE_TILES
#define WGP_VALUE_TILES 0  // 0: all the fragments
#endif
// Call-compatible with compute_fp8_sv_inst_buf_fp16_accu, with the choice as first template argument.
template <bool staged, uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          SwizzleMode swizzle_mode, uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void wgp_pv(const smem_t<swizzle_mode, stride>& smem_V, uint32_t RS_f8[][num_tiles_k / 2][4],
                                       DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
  static_assert(num_tiles_k == 4 && std::is_same<DTypeSVAccum, float>::value);
  if constexpr (staged)
    wgp_pv_staged<num_tiles_q, num_tiles_v, WGP_VALUE_TILES ? WGP_VALUE_TILES : num_tiles_v, swizzle_mode, stride, false>(smem_V, RS_f8, RO, nullptr);
  else
    compute_fp8_sv_inst_buf_fp16_accu<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode, stride>(smem_V, RS_f8, RO, d);
}
