// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// INT8 Q.K of one 64-key tile with the K fragments read ahead (replaces SageAttention's compute_int_qk: the same
// fragment loads and the same products, in the same order on each accumulator).
// The 16-key x 32-channel columns are issued two per basic block. ptxas places a fragment read at the end of its
// block, so a block reads the K fragments of the second column of the next block and of the first column of the
// block after (lead: 4 products or more), and the Q fragments of the next pass in the first block of a pass.
// done(fk) is called when the scores of 16 keys are final (during the next block's products, or after the last
// one); stage(b) ends the block before block b.
template <uint32_t tiles_q, uint32_t tiles_k, uint32_t inner, SwizzleMode swizzle, uint32_t stride, bool fine, class Done, class Stage>
__device__ __forceinline__ void wgp_int_qk(const smem_t<swizzle, stride>& q, const smem_t<swizzle, stride>& k,
    int32_t scores[][tiles_k][8], uint32_t qoff, uint32_t koff, Done&& done, Stage&& stage) {
  static_assert(tiles_k == 4 && inner >= 2);
  constexpr uint32_t columns = inner * tiles_k;
  // Shared-memory offsets of the passes, as compute_int_qk walks them.
  uint32_t qo[inner], ko[inner];
  #pragma unroll
  for (uint32_t p = 0; p < inner; p++) {
    qo[p] = qoff;
    ko[p] = koff;
    qoff = q.template advance_offset_by_column<2>(qoff, p);
    koff = k.template advance_offset_by_column<2>(koff, p);
  }
  uint32_t RQ[2][tiles_q][4], RK[5][4];
  auto read_q = [&](uint32_t p) {
    #pragma unroll
    for (uint32_t fq = 0; fq < tiles_q; fq++) q.ldmatrix_m8n8x4(qo[p] + fq * 16 * stride, RQ[p % 2][fq]);
  };
  auto read_k = [&](uint32_t c) { k.ldmatrix_m8n8x4(ko[c / tiles_k] + (c % tiles_k) * 16 * stride, RK[c % 5]); };
  auto products = [&](uint32_t c) {
    const uint32_t p = c / tiles_k, fk = c % tiles_k;
    #pragma unroll
    for (uint32_t fq = 0; fq < tiles_q; fq++) {
      if (p == 0) mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(scores[fq][fk], RQ[p % 2][fq], RK[c % 5]);
      else mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(scores[fq][fk], RQ[p % 2][fq], RK[c % 5]);
    }
  };
  read_q(0);
  read_k(0);
  read_k(1);
  read_k(2);
  // `fine`: the last pass is issued one column per block, so that three of its four columns are handed over
  // (done) between products instead of two.
  constexpr uint32_t coarse = fine ? columns - tiles_k : columns;
  #pragma unroll
  for (uint32_t c = 0; c < coarse; c += 2) {
    if (c > 0) stage(c / 2);
    if (c + 3 < columns) read_k(c + 3);
    if (c + 4 < columns) read_k(c + 4);
    if (c % tiles_k == 0 && c / tiles_k + 1 < inner) read_q(c / tiles_k + 1);
    products(c);
    if (c >= columns - tiles_k + 2) {  // the previous block finished two columns of the last pass
      done((c - 2) % tiles_k);
      done((c - 1) % tiles_k);
    }
    products(c + 1);
  }
  if constexpr (fine) {
    #pragma unroll
    for (uint32_t c = coarse; c < columns; c++) {
      stage(columns / 2 + c - coarse);
      if (c == coarse) read_k(columns - 1);  // the other columns of the pass were read by the blocks before
      products(c);
      if (c > coarse) done((c - 1) % tiles_k);
    }
    done(tiles_k - 1);
  } else {
    done(tiles_k - 2);
    done(tiles_k - 1);
  }
}
