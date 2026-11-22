# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""Build the kernel of the precise mode, sagegp/sm120/cubins/precise_sm120a.cubin: the dense kernel (../strided_query.cu) for heads
that keep all their scores in INT8, with the same products and sums as SageAttention 2 in a faster schedule (bit-identical output):
- P.V one 16-channel value fragment per basic block (pv_stages.cuh): each FP16 accumulator lives for 8 products instead of 64, its
  conversion overlaps the next fragment's products and the V fragments are read two fragments ahead;
- the CTA of the odd shared-memory window of an SM starts 700 cycles late and the even one exits 700 cycles late, so that the two
  CTAs of an SM do not reach their tensor-core phases together.

--split builds precise_split_sm120a.cubin instead: the same kernel over a range of the keys (blockIdx.z), each range writing its output
and its log2-sum-exp at its own index (split_query.cu), merged by sagegp.sm120.dense for the query blocks of a part-filled last wave.

python build.py [/path/to/SageAttention] [--split] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
"""
import argparse
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE.parent / "build.py"))
SM89 = "qattn/qk_int_sv_f8_cuda_sm89.cuh"

DEFINES = """#define WGP_M_PVS 0
#define WGP_I_PVS 1
#define WGP_I_KA 0
#define WGP_I8_LOOP 0
#define I8_BARRIERS 2
#define I8_KA 0
#define WGP_QK_FINE 0
#define I8_CM 0
#define I8_HOLD 0
#define I8_VEARLY 0
#define WGP_SEL(mixed, X) ((mixed) ? WGP_M_##X : WGP_I_##X)
"""

STAGGER = r"""#define WGP_STAGGER 700
#define WGP_SLOT_BIT 1

// Index of this CTA's shared-memory window on its SM: the generic address of shared memory is window << 24 | offset.
__device__ __forceinline__ uint32_t wgp_cta_slot() {
  extern __shared__ int8_t wgp_window[];
  uint64_t generic;
  asm volatile("cvta.shared.u64 %0, %1;" : "=l"(generic) : "l"(static_cast<uint64_t>(__cvta_generic_to_shared(wgp_window))));
  return static_cast<uint32_t>(generic >> 24);
}
// Wait for a number of SM clock cycles.
__device__ __forceinline__ void wgp_wait(uint32_t cycles) {
  uint32_t start, now;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start));
  do {
    asm volatile("mov.u32 %0, %%clock;" : "=r"(now));
  } while (now - start < cycles);
}
#define WGP_ENTER if (wgp_cta_slot() & WGP_SLOT_BIT) wgp_wait(WGP_STAGGER)
#define WGP_EXIT if (!(wgp_cta_slot() & WGP_SLOT_BIT)) wgp_wait(WGP_STAGGER)
"""


def replace_once(text, old, new, count=1):
    if text.count(old) != count:
        raise RuntimeError(f"Unexpected source around {old[:60]!r}")
    return text.replace(old, new)


def schedule(kernel):
    """The P.V of every path through pv_stages.cuh (selected per path by WGP_SEL), the other loop changes left off."""
    kernel = replace_once(kernel, '#include "online_softmax.cuh"',
                          DEFINES + '#include "online_softmax.cuh"\n#include "pv_stages.cuh"\n#include "xt_helpers.cuh"\n#include "int_qk.cuh"')
    kernel = replace_once(kernel, "compute_fp8_sv_inst_buf_fp16_accu<", "wgp_pv<WGP_SEL(mixed, PVS), ", 3)
    qk = ("else compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(\n"
          "        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);")
    ahead = ("else if constexpr (WGP_I_KA) wgp_int_qk<num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, bool(WGP_QK_FINE)>(\n"
             "        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma, [](uint32_t) {}, [&](uint32_t block) { wgp_stage(block, lane_id, wgp_qk_never); });\n      ")
    kernel = replace_once(kernel, qk, ahead + qk, 3)
    ready = "  uint32_t query_scales[num_tiles_q];"
    kernel = replace_once(kernel, ready, "  [[maybe_unused]] const uint32_t wgp_qk_never = wgp_never(kv_len);\n" + ready)
    start = kernel.index("  // load K with predicate")
    end = kernel.index("  // TODO: thread block sync mdo", start)
    return kernel[:start] + "  if constexpr (!mixed && WGP_I8_LOOP)\n#include \"i8_loop.cuh\"\n  else {\n" + kernel[start:end] + "  }\n" + kernel[end:]


def split_keys(kernel):
    """The kernel over the keys [kv_start, kv_start + kv_split_len) of blockIdx.z's range (all of them when kv_split_len is 0), its output
    and lse at index blockIdx.z, the key scales indexed over all the keys; the output and lse rows from query block out_first (out_rows
    rows per range and head, qo_len when 0)."""
    kernel = replace_once(kernel, "const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,",
                          "const uint32_t qo_len, const uint32_t kv_len_all, const uint32_t num_kv_groups,")
    kernel = replace_once(kernel, "float sm_scale, uint32_t query_block)",
                          "float sm_scale, uint32_t query_block, const uint32_t num_splits = 1, const uint32_t kv_split_len = 0,"
                          " const uint32_t out_first = 0, const uint32_t out_rows = 0)")
    kernel = replace_once(kernel, "  const uint32_t batch_id = blockIdx.z;", """  const uint32_t out_batch = blockIdx.z;
  const uint32_t batch_id = blockIdx.z / num_splits;
  const uint32_t kv_start = kv_split_len ? (blockIdx.z % num_splits) * kv_split_len : 0;
  const uint32_t kv_len = kv_split_len ? min(kv_len_all - kv_start, kv_split_len) : kv_len_all;""")
    kernel = replace_once(kernel, "const uint32_t num_block_k = div_ceil(kv_len, CTA_K);", "const uint32_t num_block_k = div_ceil(kv_len_all, CTA_K);")
    kernel = replace_once(kernel, "const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);",
                          "const uint32_t num_warp_block_k = div_ceil(kv_len_all, CTA_K) * (CTA_K / WARP_K);", 2)
    advance = "  constexpr uint32_t k_scale_advance_offset = "
    line_end = kernel.index("\n", kernel.index(advance))
    kernel = kernel[:line_end + 1] + "  k_scale_idx += (kv_start / CTA_K) * k_scale_advance_offset;\n" + kernel[line_end + 1:]
    kernel = replace_once(kernel, "K + batch_id * stride_bz_k + ", "K + batch_id * stride_bz_k + kv_start * stride_seq_k + ")
    kernel = replace_once(kernel, "V + batch_id * stride_bz_v + ", "V + batch_id * stride_bz_v + kv_start + ")
    kernel = replace_once(kernel, "O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q",
                          "O + out_batch * stride_bz_o + head_id * stride_h_o + ((bx - out_first) * CTA_Q")
    return replace_once(kernel, "Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx",
                        "Lse + out_batch * ((out_rows ? out_rows : qo_len) * num_qo_heads) + head_id * (out_rows ? out_rows : qo_len) + lse_idx - out_first * CTA_Q")


def entry(text):
    body_start = text.index("  if (mixed_heads[blockIdx.y])")
    body_end = text.index("\n}", body_start) + 1
    head = text.index('extern "C" __global__')
    body = text[body_start:body_end]
    text = text[:body_start] + "  WGP_ENTER;\n" + body + "  WGP_EXIT;\n" + text[body_end:]
    return text[:head] + STAGGER + text[head:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sage_source", type=Path, nargs="?", default=BASE["ROOT"])
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ccbin")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    source, image = ("split_query.cu", "precise_split_sm120a.cubin") if args.split else ("strided_query.cu", "precise_sm120a.cubin")
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        BASE["write_headers"](work, args.sage_source)
        kernel = work / SM89
        scheduled = schedule(kernel.read_text(encoding="utf-8"))
        kernel.write_text(split_keys(scheduled) if args.split else scheduled, encoding="utf-8")
        for name in ("pv_stages.cuh", "xt_helpers.cuh", "int_qk.cuh", "i8_loop.cuh"):
            shutil.copyfile(HERE / name, work / "qattn" / name)
        entry_source = (HERE / source) if args.split else (HERE.parent / source)
        (work / source).write_text(entry(entry_source.read_text(encoding="utf-8")), encoding="utf-8")
        command = [args.nvcc, "--cubin", "-arch=sm_120a", "-O3", "--use_fast_math", "-std=c++17"]
        if args.ccbin:
            command += ["-ccbin", args.ccbin]
        subprocess.run([*command, str(work / source), "-o", str(work / image)], cwd=work, check=True)
        shutil.copyfile(work / image, BASE["CUBINS"] / image)


if __name__ == "__main__":
    main()
