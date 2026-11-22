# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""Specialize the Sage2 tile loads and tensor-core loops for the routed head widths."""
from pathlib import Path
import re
import runpy
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
CUBINS = HERE.parents[2] / "sagegp" / "sm120" / "cubins" / "routed"


def specialize(kernel, utils, mode, value_width):
    utils = utils.replace("uint32_t &offset_Q, uint32_t &offset_K)", "uint32_t offset_Q, uint32_t offset_K)")
    depth = (1, 2, 4, 4)[mode]
    lanes = (2, 4, 7, 8)[mode]
    if value_width:
        kernel = kernel.replace("bool mixed=false>", "bool mixed=false, int ScoreMode=3>")
        depth = "(ScoreMode == 0 ? 1 : ScoreMode == 1 ? 2 : 4)"
        lanes = "(ScoreMode == 0 ? 2 : ScoreMode == 1 ? 4 : ScoreMode == 2 ? 7 : 8)"

    qk_tiles = "(DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K)"
    load = "load_global_to_share<global_to_shared_line_lanes_QK"
    assert qk_tiles in kernel and kernel.count(load) == 4
    kernel = kernel.replace(qk_tiles, str(depth)).replace(load, f"if (lane_id % 8 < {lanes}) " + load)
    if value_width:
        # Fast lays each 32-channel K plane contiguously across a 64-key tile.
        tiles = iter(("0", "iter", "num_iterations-1"))
        expr = r"if \(lane_id % 8 < [^\n]+?\) load_global_to_share<[^\n]+CTA_K>\(\s*&K_lane_base_ptr,[^;]+;"
        kernel, count = re.subn(expr, lambda _: f"load_planes<ScoreMode>(K,smem_K,head_id,kv_len,{next(tiles)});", kernel)
        assert count == 3
        kernel = kernel.replace('#include "mixed_qk.cuh"', '#include "mixed_qk.cuh"\n#include "router_planes.cuh"')
        load_v = "load_fp8_V_global_to_share<global_to_shared_line_lanes_V"
        assert kernel.count(load_v) == 3
        kernel = kernel.replace(load_v, f"if (warp_id < {value_width // 32}) " + load_v)
        start = utils.index("void compute_fp8_sv_inst_buf_fp16_accu(")
        end = utils.index("\n}", start)
        loop, count = re.subn(r"(fv|j) < num_tiles_v;", rf"\1 < {value_width // 16};", utils[start:end])
        assert count >= 3
        utils = utils[:start] + loop + utils[end:]
    elif mode == 2:
        # Mixed heads read the FP4 tail and its scale after the full INT8 representation.
        pos = kernel.index("  //                                                                for fp16: CTA_K")
        kernel = kernel[:pos] + "  if (lane_id % 8 >= 4) K_lane_base_ptr += 64;\n" + kernel[pos:]
    return kernel, utils


def with_joint_tiles(kernel):
    # The shortened value fragment also shortens its shared output. The public
    # epilogue must still write zero into every discarded channel before decode.
    kernel = kernel.replace("constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;",
                            "constexpr uint32_t num_tiles_v = VALUE_WIDTH / MMA_SV_N;")
    store = "smem_O.store_128b(offset_O, O_lane_ptr);"
    assert store in kernel
    kernel = kernel.replace(store, """
        if ((j * global_to_shared_line_lanes_O + lane_id % global_to_shared_line_lanes_O) * PACK_SIZE_O < VALUE_WIDTH)
          smem_O.store_128b(offset_O, O_lane_ptr);
        else
          *reinterpret_cast<uint4*>(O_lane_ptr) = make_uint4(0, 0, 0, 0);""")
    start = kernel.index("  // load K with predicate")
    end = kernel.index("  // TODO: thread block sync mdo", start)
    return kernel[:start] + '#include "joint_tiles.cuh"\n' + kernel[end:]


def with_split_keys(kernel, value_width):
    """The precise split's patch (../precise/build.py: the kernel over blockIdx.z's range of the keys, its output and lse at that
    index), and the key planes the value variants address themselves from the head's first tile."""
    kernel = runpy.run_path(str(HERE.parent / "precise" / "build.py"))["split_keys"](kernel)
    planes = "load_planes<ScoreMode>(K,smem_K,head_id,kv_len,"
    assert kernel.count(planes) == (3 if value_width and "joint_tiles.cuh" not in kernel else 0)
    return kernel.replace(planes, "load_planes<ScoreMode>(K,smem_K,head_id,kv_len_all,kv_start/CTA_K+")


def split_joint_tiles(text):
    """joint_tiles.cuh over a range of the keys: its records counted from the head's first tile."""
    record = "const size_t record = size_t(head_id) * num_iterations + tile;"
    assert text.count(record) == 1
    return text.replace(record, "const size_t record = size_t(head_id) * div_ceil(kv_len_all, CTA_K) + kv_start / CTA_K + tile;")


def build_variants(work, command, *, joint_tiles=False, split_kv=False, single_key=False, split_keys=False):
    """split_keys: the images over ranges of the keys (split_attention.cu, `<name>_keys_sm120a.cubin`) instead."""
    kernel = work / "qattn/qk_int_sv_f8_cuda_sm89.cuh"
    utils = work / "qattn/attn_utils.cuh"
    original = kernel.read_text(), utils.read_text()
    shutil.copyfile(HERE / "router_planes.cuh", work / "qattn/router_planes.cuh")
    source = work / ("split_attention.cu" if split_keys else "attention.cu")
    shutil.copyfile(HERE / source.name, source)
    images = []
    modes = [(w, 2) for w in (32, 64, 96, 128)]
    if joint_tiles:
        tiles = (HERE / "joint_tiles.cuh").read_text(encoding="utf-8")
        (work / "qattn/joint_tiles.cuh").write_text(split_joint_tiles(tiles) if split_keys else tiles, encoding="utf-8")
    if joint_tiles or split_keys:
        if split_keys and not joint_tiles:
            modes = [(0, m) for m in range(4)] + modes
    else:
        selector = work / "selector_sm120a.cubin"
        subprocess.run([*command, str(HERE / "selector.cu"), "-o", str(selector)], cwd=work, check=True)
        images.append(selector)
        modes = [(0, m) for m in range(4)] + modes
    for value_width, score_mode in modes:
        code, helpers = specialize(*original, score_mode, value_width)
        if joint_tiles:
            code = with_joint_tiles(code)
        if split_keys:
            code = with_split_keys(code, value_width)
        kernel.write_text(code, encoding="utf-8")
        utils.write_text(helpers, encoding="utf-8")
        name = f"value{value_width}" if value_width else f"score{score_mode}"
        if joint_tiles:
            name = ("single_" if single_key else "split_" if split_kv else "joint_") + name
        image = work / f"{name}{'_keys' if split_keys else ''}_sm120a.cubin"
        subprocess.run([*command, f"-DVALUE_WIDTH={value_width}", f"-DSCORE_MODE={score_mode}", f"-DSPLIT_KV={int(split_kv)}", f"-DSINGLE_K={int(single_key)}",
                        str(source), "-o", str(image)], cwd=work, check=True)
        images.append(image)
    for image in images:
        shutil.copyfile(image, CUBINS / image.name)
