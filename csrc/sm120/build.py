"""Rebuild the SM120a device images of sagegp/sm120/cubins from the pinned Apache-2.0 SageAttention CUDA headers.

python build.py [/path/to/SageAttention] [--routed | --joint-tiles | --split-tiles | --single-key] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
The headers are read from this repository's git history at REVISION (SageAttention 2.2), by default.
The build reads the named git objects so local changes cannot affect the device image.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

REVISION = "4993c866a60a802354c86d599298454e60a640d6"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUBINS = ROOT / "sagegp" / "sm120" / "cubins"
HEADERS = ("utils.cuh", "mma.cuh", "numeric_conversion.cuh", "permuted_smem.cuh", "cp_async.cuh",
           "math.cuh", "dispatch_utils.h", "qattn/attn_utils.cuh", "qattn/qk_int_sv_f8_cuda_sm89.cuh")


def write_headers(work, sage_source):
    """The pinned SageAttention headers with the dense-kernel substitutions, and the SM120 headers of this folder, under work."""
    (work / "qattn").mkdir()
    for name in HEADERS:
        header = subprocess.check_output(["git", "-C", str(sage_source), "show", f"{REVISION}:csrc/{name}"]).decode()
        header = header.replace("#include <torch/extension.h>", "#include <type_traits>\n#include <cstdint>")
        if name.endswith("cuda_sm89.cuh"):
            header = header.replace('#include "attn_utils.cuh"', '#include "attn_utils.cuh"\n#include "mixed_qk.cuh"\n#include "online_softmax.cuh"')
            header = header.replace("update_mdo<", "dense_update<")
            header = header.replace("normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>",
                                    "normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>")
            header = header.replace("bool use_pv_fp16_accu=false>", "bool use_pv_fp16_accu=false, bool mixed=false>")
            header = header.replace("__global__ void qk_int", "__device__ __forceinline__ void qk_int")
            header = header.replace("float sm_scale)\n{", "float sm_scale, uint32_t query_block)\n{")
            header = header.replace("const uint32_t bx = blockIdx.x;", "const uint32_t bx = blockIdx.x + query_block;")
            header = header.replace("= gridDim.x", "= div_ceil(qo_len, CTA_Q)")
            ready = "  // for num_tiles_qk_inner = 1, we load all Qs in register"
            assert header.count(ready) == 1
            header = header.replace(ready, """  uint32_t query_scales[num_tiles_q];
  if constexpr (mixed)
    load_query_scales<num_warps_q, num_warps_k, num_tiles_q>(smem_Q, query_scales);
""" + ready)
            call = "compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>"
            original = call + "(\n        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);"
            assert header.count(original) == 3
            mixed_call = original.replace("compute_int_qk", "mixed_qk").replace("K_smem_offset_mma);", "K_smem_offset_mma, query_scales);")
            header = header.replace(original, "if constexpr (mixed) " + mixed_call + "\n      else " + original)
            cast = "__int2float_rz(RS[fq][fk][k])"
            header = header.replace(cast, f"(mixed ? __int_as_float(RS[fq][fk][k]) : {cast})")
        (work / name).write_text(header, encoding="utf-8")
    shutil.copyfile(HERE / "mixed_qk.cuh", work / "qattn/mixed_qk.cuh")
    shutil.copyfile(HERE / "online_softmax.cuh", work / "qattn/online_softmax.cuh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sage_source", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ccbin")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--strided-query", action="store_true", help="Build the variant that reads Q from the recycled projection")
    variant.add_argument("--routed", action="store_true", help="Build all eight Sage2 quality kernels")
    variant.add_argument("--joint-tiles", action="store_true", help="Build the dense Fast TMA kernels")
    variant.add_argument("--split-tiles", action="store_true", help="Build the dense Fast TMA kernels for separately packed K and V")
    variant.add_argument("--single-key", action="store_true", help="Build the dense Fast TMA kernels for keys packed after score selection")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        write_headers(work, args.sage_source)
        command = [args.nvcc, "--cubin", "-arch=sm_120a", "-O3", "--use_fast_math", "-std=c++17"]
        if args.ccbin:
            command.extend(["-ccbin", args.ccbin])
        if args.routed or args.joint_tiles or args.split_tiles or args.single_key:
            from routed.build_variants import build_variants
            build_variants(work, command, joint_tiles=args.joint_tiles or args.split_tiles or args.single_key,
                           split_kv=args.split_tiles or args.single_key, single_key=args.single_key)
            return
        source = "strided_query.cu" if args.strided_query else "dense.cu"
        image = "strided_query_sm120a.cubin" if args.strided_query else "dense_sm120a.cubin"
        shutil.copyfile(HERE / source, work / source)
        subprocess.run([*command, str(work / source), "-o", str(work / image)], cwd=work, check=True)
        shutil.copyfile(work / image, CUBINS / image)


if __name__ == "__main__":
    main()
