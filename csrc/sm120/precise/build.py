# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""Build the kernel of the precise mode, sagegp/sm120/cubins/precise_sm120a.cubin: the dense kernel (../strided_query.cu) for heads
that keep all their scores in INT8, with the same products and sums as SageAttention 2 in a faster schedule (bit-identical output):
- P.V one 16-channel value fragment per basic block (pv_stages.cuh): each FP16 accumulator lives for 8 products instead of 64, its
  conversion overlaps the next fragment's products and the V fragments are read two fragments ahead;
- the CTA of the odd shared-memory window of an SM starts 700 cycles late and the even one exits 700 cycles late, so that the two
  CTAs of an SM do not reach their tensor-core phases together.

python build.py [/path/to/SageAttention] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
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
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        BASE["write_headers"](work, args.sage_source)
        kernel = work / SM89
        kernel.write_text(schedule(kernel.read_text(encoding="utf-8")), encoding="utf-8")
        for name in ("pv_stages.cuh", "xt_helpers.cuh", "int_qk.cuh", "i8_loop.cuh"):
            shutil.copyfile(HERE / name, work / "qattn" / name)
        (work / "strided_query.cu").write_text(entry((HERE.parent / "strided_query.cu").read_text(encoding="utf-8")), encoding="utf-8")
        command = [args.nvcc, "--cubin", "-arch=sm_120a", "-O3", "--use_fast_math", "-std=c++17"]
        if args.ccbin:
            command += ["-ccbin", args.ccbin]
        subprocess.run([*command, str(work / "strided_query.cu"), "-o", str(work / "precise_sm120a.cubin")], cwd=work, check=True)
        shutil.copyfile(work / "precise_sm120a.cubin", BASE["CUBINS"] / "precise_sm120a.cubin")


if __name__ == "__main__":
    main()
