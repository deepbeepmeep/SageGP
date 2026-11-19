"""Build the all-NVFP4 score kernel (FP8 values) of the Fast precision (sagegp/sm120/fast.py).

python build.py [/path/to/SageAttention] [--nvcc /path/to/nvcc] [--ccbin /path/to/compiler]
It specializes the production dense kernel (../build.py, pinned SageAttention revision) so that each score is one block-scaled
NVFP4 product over all 128 channels (scaled_qk.cuh, two k64 parts), with constant row units (the packer writes 0.125)."""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sage_source", type=Path, nargs="?", default=PARENT.parents[1])
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ccbin")
    args = parser.parse_args()
    script = (PARENT / "build.py").read_text(encoding="utf-8")
    replacements = (("CUBINS = ROOT / \"sagegp\" / \"sm120\" / \"cubins\"", "CUBINS = HERE"),
                    ("uint32_t query_scales[num_tiles_q];", "uint32_t query_scales[num_tiles_q][SCORE_PARTS];"),
                    ("load_query_scales<num_warps_q", "read_qscales<num_warps_q"),
                    ('"compute_int_qk", "mixed_qk"', '"compute_int_qk", "scaled_qk"'))
    for old, new in replacements:
        assert script.count(old) == 1, old
        script = script.replace(old, new)
    extra = '''
        if name.endswith("cuda_sm89.cuh"):
            # Four 16-byte lanes fetch the FP4 bytes of a key row, one fetches its scale factors.
            load = "load_global_to_share<global_to_shared_line_lanes_QK"
            assert header.count(load) == 4
            header = header.replace(load, "if ((lane_id % 8 < 4) || (lane_id % 8 == 6)) " + load)
            for i in ("0", "iter", "(num_iterations - 1)"):
                scale = "q_scale * K_scale[k_scale_idx + " + i + " * k_scale_advance_offset]"
                assert header.count(scale) == 1
                header = header.replace(scale, "0.015625f")  # both row units are 0.125
'''
    marker = "        (work / name)"
    assert script.count(marker) == 1
    script = script.replace(marker, extra + marker)
    entry = (PARENT / "strided_query.cu").read_text(encoding="utf-8")
    first = entry.index("  if (mixed_heads")
    call = entry[entry.index("    attend_strided<true>", first):].splitlines()[0]
    entry = entry[:first] + call + "\n}\n"
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "build.py").write_text(script, encoding="utf-8")
        (work / "strided_query.cu").write_text(entry, encoding="utf-8")
        header = (HERE / "scaled_qk.cuh").read_text(encoding="utf-8")
        (work / "mixed_qk.cuh").write_text("#define SCORE_PARTS 2\n#define INT8_PREFIX_FRAGMENTS 0\n" + header, encoding="utf-8")
        shutil.copyfile(PARENT / "online_softmax.cuh", work / "online_softmax.cuh")
        command = [sys.executable, str(work / "build.py"), str(args.sage_source), "--strided-query", "--nvcc", args.nvcc]
        if args.ccbin:
            command += ["--ccbin", args.ccbin]
        subprocess.run(command, check=True)
        shutil.copyfile(work / "strided_query_sm120a.cubin", PARENT.parents[1] / "sagegp" / "sm120" / "cubins" / "fixed_sm120a.cubin")


if __name__ == "__main__":
    main()
