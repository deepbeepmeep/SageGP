// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// Key-tile loop of the dense SM120 kernel for all-INT8 heads (replaces the Sage2 loop when the head is not mixed).
// Sage2 order per tile (Q.K, softmax, P.V); every product, sum and conversion keeps its operands and its order on
// each accumulator. Compile-time choices:
//   I8_BARRIERS 2: [K(t) ready, V(t-1) read] and [V(t) ready, K(t) read] instead of Sage2's four block barriers.
//   I8_BARRIERS 1: K and V double buffered (48 KB of shared memory: Q, K0, K1, V0, V1), one barrier per tile.
//   I8_KA      : Q.K with the K fragments read ahead, two columns per basic block (int_qk.cuh).
//   I8_CM      : (with I8_KA) FP32 scores and row maxima of a column during the next block's products.
//   I8_VEARLY  : the second barrier (two-barrier mode) and the first V fragment reads come before the softmax.
//   I8_HOLD    : the products of the last value fragment of tile t-1 are issued after the first barrier of tile t,
//                while the first Q and K fragments of tile t are read.
// The half-rate FP32 row sums of tile t-1 (denominator) are issued at the start of tile t, before a rescale can scale them.
{
  static_assert(!mixed && num_tiles_q == 2 && num_tiles_k == 4 && use_inst_buffer && use_pv_fp16_accu);
  static_assert(std::is_same<DTypeSVAccum, float>::value && DenominatorAccumUnit == ComputeUnit::kTensorCore && mask_mode == MaskMode::kNone);
  static_assert(I8_BARRIERS == 1 || I8_BARRIERS == 2);
  constexpr uint32_t QK_stride = QK_SMEM_STRIDE / PACK_SIZE_QK, V_stride = V_SMEM_STRIDE / PACK_SIZE_V;
  constexpr uint32_t used_v = WGP_VALUE_TILES ? WGP_VALUE_TILES : num_tiles_v;
  const uint32_t T = num_iterations;
  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;
  const float* K_scales = K_scale + k_scale_idx;
  const uint32_t K_load_idx0 = K_load_idx_lane_base, K_mask_idx0 = K_idx_lane_base;
  [[maybe_unused]] const uint32_t i8_never = wgp_never(kv_len);
  // FP8 probabilities of the previous tile and the V halves of its held fragment (zero before the first tile:
  // the row sums and the held products then add +0).
  uint32_t P8[num_tiles_q][num_tiles_k / 2][4];
  [[maybe_unused]] uint32_t held[2][4];
  #pragma unroll
  for (uint32_t k = 0; k < 4; k++) {
    #pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) P8[fq][0][k] = P8[fq][1][k] = 0;
    held[0][k] = held[1][k] = 0;
  }
  using k_buffer = smem_t<swizzle_mode_QK, QK_stride>;
  using v_buffer = smem_t<swizzle_mode_V, V_stride>;
#if I8_BARRIERS == 1
  auto k_slot = [&](uint32_t tile) { return k_buffer(smem + (CTA_Q + CTA_K * (tile & 1)) * QK_SMEM_STRIDE); };
  auto v_slot = [&](uint32_t tile) { return v_buffer(smem + (CTA_Q + 2 * CTA_K + CTA_K * (tile & 1)) * QK_SMEM_STRIDE); };
#else
  auto k_slot = [&](uint32_t) { return smem_K; };
  auto v_slot = [&](uint32_t) { return smem_V; };
#endif
  auto request_k = [&](uint32_t tile, bool bounded) {
    const k_buffer slot = k_slot(tile);
    if (bounded)
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_stride, CTA_K>(
          &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, slot, K_load_idx0 + tile * CTA_K, kv_len);
    else
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_stride, CTA_K>(
          &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, slot);
  };
  auto request_v = [&](uint32_t tile) {
    const v_buffer slot = v_slot(tile);
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_stride, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_d_v, slot);
  };
  auto arrived = [&]() { cp_async::wait_group<0>(); __syncthreads(); };

  auto tile_step = [&](const uint32_t tile, auto late_tag, auto last_tag) {
    constexpr bool late = decltype(late_tag)::value, last = decltype(last_tag)::value;
    // Sage2 folds the dequantization into the softmax scale except on its last two tiles.
    const float dequant = q_scale * K_scales[tile * k_scale_advance_offset];
    const float scale = late ? original_sm_scale : original_sm_scale * dequant;
    float S[num_tiles_q][num_tiles_k][8], peak[num_tiles_q][2], peaks[num_tiles_q][2];
    #pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) peak[fq][0] = peak[fq][1] = -3.4e38f;
    // FP32 scores and row maxima of 16 keys (first part of dense_update, one column at a time).
    // With I8_CM the conversions and maxima are written as volatile PTX (the instructions the compiler emits for
    // __int2float_rz and fmaxf): the PTX emitter moves plain arithmetic next to its first reader, past the block boundaries.
    auto column = [&](uint32_t fk) {
      #pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          float score;
          if (I8_CM) asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(score) : "r"(RS[fq][fk][k]));
          else score = __int2float_rz(RS[fq][fk][k]);
          if constexpr (late) score *= dequant;
          if constexpr (last) {  // apply_out_of_bound_mask
            const uint32_t kv_idx = K_mask_idx0 + tile * CTA_K + fk * 16 + 8 * (k / 4) + k % 2;
            score = kv_idx >= kv_len ? -5000000.0f : score;
          }
          S[fq][fk][k] = score;
        }
        #pragma unroll
        for (uint32_t row = 0; row < 2; row++) {
          const float* s = S[fq][fk] + 2 * row;
          if (I8_CM) {
            float a, b;
            asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(a) : "f"(s[0]), "f"(s[1]));
            asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(b) : "f"(s[4]), "f"(s[5]));
            asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(a) : "f"(a), "f"(b));
            asm volatile("max.ftz.f32 %0, %0, %1;" : "+f"(peak[fq][row]) : "f"(a));
          } else {
            peak[fq][row] = fmaxf(peak[fq][row], fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5])));
          }
        }
      }
    };
    const k_buffer smem_Kt = k_slot(tile);
    const v_buffer smem_Vt = v_slot(tile);
    arrived();  // K(tile) is in shared memory and every warp has issued its reads of V(tile-1); one barrier: V(tile) too
#if I8_BARRIERS == 1
    if constexpr (!last) {
      request_k(tile + 1, late);
      request_v(tile + 1);
      cp_async::commit_group();
    }
#else
    request_v(tile);
    cp_async::commit_group();
#endif
    // Tensor work that only needs registers, while the first Q and K fragments are read: the row sums of the previous
    // tile (before a rescale of this tile can scale them) and the held value fragment of the previous tile.
    accumulate_d_f8<num_tiles_q, num_tiles_k>(P8, d);
#if I8_HOLD
    wgp_pv_held<num_tiles_q, num_tiles_v, used_v>(P8, RO, held);
#endif
#if I8_KA
    wgp_int_qk<num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_stride, bool(WGP_QK_FINE)>(smem_Q, smem_Kt, RS, Q_smem_offset_mma, K_smem_offset_mma,
        [&](uint32_t fk) { if (I8_CM) column(fk); }, [&](uint32_t block) { wgp_stage(block, tile, i8_never); });
    if (!I8_CM) {
      #pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) column(fk);
    }
#else
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_stride, DTypeQK>(
        smem_Q, smem_Kt, RS, Q_smem_offset_mma, K_smem_offset_mma);
    #pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) column(fk);
#endif
#if I8_VEARLY
    // V(tile) is read before the softmax: no barrier and no fragment read between the exponentials and P.V.
#if I8_BARRIERS == 2
    arrived();  // V(tile) is in shared memory; every warp has read K(tile)
    if constexpr (!last) {
      request_k(tile + 1, late);
      cp_async::commit_group();
    }
#endif
    uint32_t first_v[2][2][4];
    wgp_pv_read_first<swizzle_mode_V, V_stride>(smem_Vt, first_v);
#endif
    bool update = false;
    #pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      #pragma unroll
      for (uint32_t row = 0; row < 2; row++) {
        peaks[fq][row] = peak[fq][row] * scale - S_FP8_OFFSET;
        update |= peaks[fq][row] > m[fq][row];
      }
    }
    if (__any_sync(0xffffffff, update)) xt_rescale<num_tiles_q, num_tiles_v>(peaks, RO, m, d);
    #pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      #pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) xt_exp(S[fq][fk], m[fq], scale);
      #pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) xt_pack(P8[fq][fk], S[fq][2 * fk], S[fq][2 * fk + 1]);
    }
#if I8_VEARLY
    wgp_pv_staged<num_tiles_q, num_tiles_v, used_v, swizzle_mode_V, V_stride, bool(I8_HOLD), true>(smem_Vt, P8, RO, held, first_v);
#else
#if I8_BARRIERS == 2
    arrived();  // V(tile) is in shared memory; every warp has read K(tile)
    if constexpr (!last) {
      request_k(tile + 1, late);
      cp_async::commit_group();
    }
#endif
    wgp_pv_staged<num_tiles_q, num_tiles_v, used_v, swizzle_mode_V, V_stride, bool(I8_HOLD)>(smem_Vt, P8, RO, held);
#endif
  };

  request_k(0, true);
#if I8_BARRIERS == 1
  request_v(0);
#endif
  cp_async::commit_group();
  #pragma unroll 1
  for (uint32_t tile = 0; tile + 2 < T; tile++) tile_step(tile, std::false_type(), std::false_type());
  if (T > 1) tile_step(T - 2, std::true_type(), std::false_type());
  tile_step(T - 1, std::true_type(), std::true_type());
#if I8_HOLD
  wgp_pv_held<num_tiles_q, num_tiles_v, used_v>(P8, RO, held);
#endif
  accumulate_d_f8<num_tiles_q, num_tiles_k>(P8, d);
  __syncthreads();
}
