"""Build the two SM120a wide-head specializations from the pinned SageAttention headers.

python build.py [/path/to/SageAttention] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
The headers are read from this repository's git history at the revision of ../build.py, by default.
"""
import argparse
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE.parent / "build.py"))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected pinned header around {old!r}")
    return source.replace(old, new)


def patch_forward(header):
    header = header.split("torch::Tensor qk_int8_sv_f16_accum_f32_attn")[0]
    header = replace_once(header, '#include "../stream.cuh"', "")
    header = replace_once(header, '#include "attn_utils.cuh"',
                          '#include "attn_utils.cuh"\n#include "compact_scores.cuh"\n#include "online_softmax.cuh"')
    header = replace_once(header, "bool fuse_v_mean=false>", "bool fuse_v_mean=false, bool compact=false>")
    header = replace_once(header, "__global__ void qk_int", "__device__ __forceinline__ void qk_int")
    header = replace_once(header, "(head_dim / MMA_QK_K)", "((compact ? 128 : head_dim) / MMA_QK_K)")
    header = replace_once(header, "? (head_dim) : (head_dim / 2)", "? (compact ? 128 : head_dim) : (head_dim / 2)")
    marker = "  // for num_tiles_qk_inner = 1"
    header = replace_once(header, marker, """  uint32_t query_scales[num_tiles_q];
  if constexpr (compact) load_query_scales<num_warps_q,num_warps_k,num_tiles_q>(smem_Q,query_scales);
""" + marker)
    call = ("compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, "
            "swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(\n"
            "        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);")
    if header.count(call) != 3:
        raise RuntimeError("Unexpected number of score products")
    mixed = call.replace("compute_int_qk", "mixed_qk").replace("K_smem_offset_mma);", "K_smem_offset_mma,query_scales);")
    header = header.replace(call, "if constexpr (compact) " + mixed + "\n      else " + call)
    cast = "__int2float_rz(RS[fq][fk][k])"
    header = header.replace(cast, f"(compact ? __int_as_float(RS[fq][fk][k]) : {cast})")
    header = header.replace("update_mdo<", "dense_update<")
    marker = "  normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, m, d);"
    return replace_once(header, marker, marker + "\n" + """  const float* value_scale=reinterpret_cast<float*>(V_mean)+head_id*head_dim;
  #pragma unroll
  for(int col=0;col<num_tiles_v;++col)
    #pragma unroll
    for(int j=0;j<8;++j){
      float scale=value_scale[col*16+(lane_id%4)*2+(j/4)*8+j%2];
      #pragma unroll
      for(int row=0;row<num_tiles_q;++row) RO[row][col][j]*=scale;
    }
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sage_source", type=Path, nargs="?", default=BASE["ROOT"])
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ccbin")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "qattn").mkdir()
        for name in (*BASE["HEADERS"][:-1], "qattn/qk_int_sv_f16_cuda_sm80.cu"):
            header = subprocess.check_output(["git", "-C", str(args.sage_source), "show",
                                              f"{BASE['REVISION']}:csrc/{name}"]).decode()
            header = header.replace("#include <torch/extension.h>", "#include <type_traits>\n#include <cstdint>")
            if name.endswith(".cu"):
                header = patch_forward(header)
                name = "qattn/wide_attention.cuh"
            (work / name).write_text(header, encoding="utf-8")
        softmax = (HERE.parent / "online_softmax.cuh").read_text()
        softmax = replace_once(softmax, "!half_output && fp8_offset", "!half_output")
        softmax = replace_once(softmax, "- S_FP8_OFFSET", "- (fp8_offset ? S_FP8_OFFSET : 0.f)")
        (work / "qattn/online_softmax.cuh").write_text(softmax, encoding="utf-8")
        shutil.copyfile(HERE / "compact_scores.cuh", work / "qattn/compact_scores.cuh")
        shutil.copyfile(HERE / "forward.cu", work / "forward.cu")
        command = [args.nvcc, "--cubin", "-arch=sm_120a", "-O3", "--use_fast_math", "-std=c++17"]
        if args.ccbin:
            command.extend(["-ccbin", args.ccbin])
        subprocess.run([*command, str(work / "forward.cu"), "-o", str(work / "wide_sm120a.cubin")], cwd=work, check=True)
        shutil.copyfile(work / "wide_sm120a.cubin", BASE["CUBINS"] / "wide" / "wide_sm120a.cubin")

if __name__ == "__main__":
    main()
