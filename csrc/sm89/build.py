# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""Build sagegp/sm89/cubins/split_sm89.fatbin: SageAttention 2's sm89 kernel (the pinned Apache-2.0 SageAttention 2.2 header, as the
wheel's sm89 extension compiles it) over a range of the keys (split_query.cu), for the query blocks of a part-filled last wave (merged by
sagegp.sm89.split). The key-range changes are those of ../sm120/precise/build.py (split_keys); the header is otherwise SageAttention's.
SASS for sm_89 and PTX of compute_89 (the device image of later GPUs, which compile it at load).

python build.py [/path/to/SageAttention] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
"""
import argparse
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SM120 = runpy.run_path(str(HERE.parent / "sm120" / "build.py"))
PRECISE = runpy.run_path(str(HERE.parent / "sm120" / "precise" / "build.py"))
CUBINS = ROOT / "sagegp" / "sm89" / "cubins"
HEADERS = ("utils.cuh", "mma.cuh", "numeric_conversion.cuh", "permuted_smem.cuh", "cp_async.cuh", "math.cuh", "dispatch_utils.h",
           "qattn/attn_utils.cuh", "qattn/qk_int_sv_f8_cuda_sm89.cuh")


def write_headers(work, sage_source):
    """The pinned headers, the kernel made a device function with a first query block and its key-range parameters."""
    (work / "qattn").mkdir()
    replace = PRECISE["replace_once"]
    for name in HEADERS:
        header = subprocess.check_output(["git", "-C", str(sage_source), "show", f"{SM120['REVISION']}:csrc/{name}"]).decode().replace("\r\n", "\n")
        header = header.replace("#include <torch/extension.h>", "#include <type_traits>\n#include <cstdint>")
        if name.endswith("cuda_sm89.cuh"):
            header = replace(header, "__global__ void qk_int_sv_f8_attn_kernel(", "__device__ __forceinline__ void qk_int_sv_f8_attn_kernel(")
            header = replace(header, "float sm_scale)\n{", "float sm_scale, uint32_t query_block)\n{")
            header = replace(header, "const uint32_t bx = blockIdx.x;", "const uint32_t bx = blockIdx.x + query_block;")
            header = header.replace("= gridDim.x", "= div_ceil(qo_len, CTA_Q)")
            header = PRECISE["split_keys"](header)
        (work / name).write_text(header, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sage_source", type=Path, nargs="?", default=ROOT)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ccbin")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        write_headers(work, args.sage_source)
        shutil.copyfile(HERE / "split_query.cu", work / "split_query.cu")
        command = [args.nvcc, "--fatbin", "-gencode=arch=compute_89,code=[sm_89,compute_89]", "-O3", "--use_fast_math", "-std=c++17"]
        if args.ccbin:
            command += ["-ccbin", args.ccbin]
        subprocess.run([*command, str(work / "split_query.cu"), "-o", str(work / "split_sm89.fatbin")], cwd=work, check=True)
        CUBINS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "split_sm89.fatbin", CUBINS / "split_sm89.fatbin")


if __name__ == "__main__":
    main()
