// Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
// Device loop for the shared SageAttention SM120 kernel. See ../LICENSE-SageAttention.txt.
// Each 64-key record contains 64 common INT8 channels. If the score mode is
// already known, it stores just the selected tail (INT8 or scaled FP4).
// Late selection keeps both tails. The joint layout also holds all 128 FP8
// value channels next to the mixed head's K bytes.
constexpr int slots = 2;
// In shared memory the uncompressed tail needs 4096 bytes. The FP4 tail
// takes only 2048 bytes of codes and 256 bytes of scales in the device record.
constexpr unsigned common_in_shared = ScoreMode == 3 ? 4096 : 2304;
constexpr unsigned value_in_shared = common_in_shared + 4096;
__shared__ unsigned long long ready[slots];
const bool leader = threadIdx.x == 0 && threadIdx.y == 0;
auto request = [&](unsigned tile, unsigned slot) {
  unsigned dest = static_cast<uint32_t>(__cvta_generic_to_shared(smem + CTA_Q * 128 + slot * 16384));
  unsigned bar = static_cast<uint32_t>(__cvta_generic_to_shared(ready + slot));
  constexpr unsigned vbytes = VALUE_WIDTH * 64;
  constexpr unsigned kbytes = ScoreMode == 2 ? 6400 : ScoreMode == 3 ? 8192 : 4096;
  const size_t record = size_t(head_id) * num_iterations + tile;
#if SINGLE_K
  const int8_t *src = K + record * 8192;
#else
  const int8_t *src = K + record * (SPLIT_KV ? 10496 : 18688);
#endif
  auto fetch = [&](unsigned to, const int8_t *from, unsigned n) {
    // All consumers are in this CTA. Cluster addressing makes SM120 reserve
    // a large per-thread stack for the driver's remote shared-memory path.
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::"r"(to),
        "l"(from), "r"(n), "r"(bar)
        : "memory");
  };
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(bar), "r"(kbytes + vbytes)
               : "memory");
  constexpr unsigned source_start = ScoreMode == 2 ? 0 : 2304;
  constexpr unsigned shared_start = ScoreMode == 2 ? 0 : common_in_shared;
  constexpr unsigned main_keys = 6400 - source_start;
#if SINGLE_K
  // The head's score mode was known at pack time. A full INT8 tile
  // already matches the shared layout; a mixed tile begins with its FP4 tail.
  if constexpr (ScoreMode == 0 || ScoreMode == 1)
    fetch(dest + common_in_shared, src + 4096, kbytes);
  else
    fetch(dest, src, kbytes);
  fetch(dest + value_in_shared, V + record * 8192, vbytes);
#else
  if constexpr (SPLIT_KV) {
    fetch(dest + shared_start, src + source_start, main_keys);
    fetch(dest + value_in_shared, V + record * 8192, vbytes);
  } else if (K == V) {
    fetch(dest + shared_start, src + source_start, main_keys + vbytes);
  } else {
    fetch(dest + shared_start, src + source_start, main_keys);
    fetch(dest + value_in_shared, V + record * 18688 + 6400, vbytes);
  }
  if constexpr (ScoreMode == 3)
    fetch(dest, src + (SPLIT_KV ? 6400 : 14592), 4096);
#endif
};
if (leader) {
#pragma unroll
  for (int i = 0; i < slots; i++)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(uint32_t(__cvta_generic_to_shared(ready + i)))
                 : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
  for (int i = 0; i < slots; i++)
    if (i < num_iterations)
      request(i, i);
}
__syncthreads();
const float scale_q = Q_scale[q_scale_idx];
#pragma unroll 1
for (unsigned tile = 0; tile < num_iterations; tile++) {
  unsigned slot = tile % slots;
  unsigned bar = uint32_t(__cvta_generic_to_shared(ready + slot));
  asm volatile("{.reg .pred ok; waitbundled: mbarrier.try_wait.parity.shared::cta.b64 ok,[%0],%1; @!ok "
               "bra.uni waitbundled;}" ::"r"(bar),
               "r"((tile / slots) % 2)
               : "memory");
  int8_t *base = smem + CTA_Q * 128 + slot * 16384;
  decltype(smem_V) values(base + value_in_shared);
  constexpr int parts = ScoreMode == 0 ? 1 : ScoreMode == 1 ? 2 : ScoreMode == 2 ? 3 : 4;
  unsigned qo = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16,
                                           lane_id / 16);
#pragma unroll
  for (unsigned p = 0; p < parts; p++) {
    unsigned qr[num_tiles_q][4];
#pragma unroll
    for (unsigned i = 0; i < num_tiles_q; i++) {
      smem_Q.ldmatrix_m8n8x4(qo, qr[i]);
      qo = smem_Q.advance_offset_by_row<16>(qo);
    }
#pragma unroll
    for (unsigned j = 0; j < num_tiles_k; j++) {
      unsigned rk[4];
      unsigned location = uint32_t(
          __cvta_generic_to_shared(base + (p < 2 ? common_in_shared + p * 2048 : (p - 2) * 2048) + j * 512 +
                                   (lane_id / 16) * 256 + ((lane_id / 8) % 2) * 128 + (lane_id % 8) * 16));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                   : "=r"(rk[0]), "=r"(rk[1]), "=r"(rk[2]), "=r"(rk[3])
                   : "r"(location));
#pragma unroll
      for (unsigned i = 0; i < num_tiles_q; i++) {
        if (p == 0)
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[i][j], qr[i], rk);
        else if (ScoreMode != 2 || p != 2)
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[i][j], qr[i], rk);
        else {
#pragma unroll
          for (int half = 0; half < 2; half++) {
            const unsigned row = j * 16 + half * 8 + lane_id / 4;
            const unsigned ks = *reinterpret_cast<const unsigned *>(base + 2048 + row * 4);
            float acc[4];
#pragma unroll
            for (int z = 0; z < 4; z++)
              acc[z] = __int2float_rn(RS[i][j][4 * half + z]);
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
                "f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%12,%12},{%11},{%12,%12};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(qr[i][0]), "r"(qr[i][1]), "r"(qr[i][2]), "r"(qr[i][3]), "r"(rk[2 * half]),
                  "r"(rk[2 * half + 1]), "r"(query_scales[i]), "r"(ks), "h"(uint16_t(0)));
#pragma unroll
            for (int z = 0; z < 4; z++)
              RS[i][j][4 * half + z] = __float_as_int(acc[z]);
          }
        }
      }
    }
    qo = smem_Q.advance_offset_by_column<2>(qo - num_tiles_q * 16 * (QK_SMEM_STRIDE / PACK_SIZE_QK), p);
  }
  const float dequant = scale_q * K_scale[k_scale_idx + tile];
  const bool edge = tile + 2 >= num_iterations;
  float s[num_tiles_q][num_tiles_k][8];
#pragma unroll
  for (unsigned i = 0; i < num_tiles_q; i++)
#pragma unroll
    for (unsigned j = 0; j < num_tiles_k; j++)
#pragma unroll
      for (unsigned c = 0; c < 8; c++) {
        float x = mixed ? __int_as_float(RS[i][j][c]) : __int2float_rz(RS[i][j][c]);
        s[i][j][c] = edge ? x * dequant : x;
      }
  if (edge)
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base + tile * CTA_K, s, kv_len);
  dense_update<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
      s, RO, m, d, edge ? sm_scale : sm_scale * dequant);
  uint32_t p[num_tiles_q][num_tiles_k / 2][4];
  RS_32_to_8<num_tiles_q, num_tiles_k>(s, p);
  accumulate_d_f8<num_tiles_q, num_tiles_k>(p, d);
  compute_fp8_sv_inst_buf_fp16_accu<num_warps_q, 1, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V,
                                    V_SMEM_STRIDE / PACK_SIZE_V>(values, p, RO, d);

  __syncthreads();
  if (leader && tile + slots < num_iterations)
    request(tile + slots, slot);
}
