# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""SageAttention 2's sm89 attention (INT8 QK with per-thread scales, FP8 PV: qk_int8_sv_f8_accum_f{16,32}_fuse_v_scale_attn_inst_buf)
with the query blocks of its last, part-filled wave split into ranges of the keys (cubins/split_sm89.fatbin, built by
csrc/sm89/build.py) and merged (sagegp.split_merge): the full waves run SageAttention's own kernel unchanged, then the tail's ranges,
then their merge. Pays where a launch has few CTAs per wave (few heads); nothing otherwise."""
import struct
from functools import lru_cache
from pathlib import Path

import torch
import triton

from sagegp import split_merge

CTA_Q, CTA_K, THREADS = 128, 64, 128
Q_SCALES_PER_BLOCK = (CTA_Q // 32) * 8  # per-thread granularity: 8 scales per 32-row warp


@lru_cache(None)
def _kernel(device, head_dim, dtype, pv_fp16):
    name = f"wangp_sm89_split_d{head_dim}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}_pv{16 if pv_fp16 else 32}"
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / "split_sm89.fatbin"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), [name])[name]


def _shared_mem(head_dim):
    return max(CTA_Q * head_dim + 2 * CTA_K * head_dim, CTA_Q * head_dim * 2)


@lru_cache(None)
def _slots(device, head_dim, dtype, pv_fp16):
    return torch.cuda.get_device_properties(device).multi_processor_count * split_merge.occupancy(_kernel(device, head_dim, dtype, pv_fp16), THREADS, _shared_mem(head_dim))


def supported(q, out):
    return q.is_cuda and q.dim() == 4 and q.shape[-1] in (64, 128) and out.dtype in (torch.bfloat16, torch.float16)


def attention(forward, q, k, v, out, q_scale, k_scale, v_scale, sm_scale, pv_fp16):
    """out = SageAttention 2 (layout NHD: q, k, out (batch, tokens, heads, dim), v FP8 (batch, dim, heads, keys padded) as per_channel_fp8
    packs it; unmasked, per-thread scales) with the last wave's query blocks split; forward: the sm89 op the
    full blocks run (qk_int8_sv_f8_accum_f16/f32_fuse_v_scale_attn_inst_buf). False, nothing done, when the last wave has nothing to split."""
    batch, tokens, heads, dim = q.shape
    keys = k.shape[1]
    blocks = triton.cdiv(tokens, CTA_Q)
    main, splits = split_merge.tail_plan(_slots(q.device, dim, out.dtype, pv_fp16), blocks, heads * batch, triton.cdiv(keys, CTA_K))
    if splits == 1:
        return False
    rows = main * CTA_Q
    if main:
        forward(q[:, :rows], k, v, out[:, :rows], q_scale[..., :main * Q_SCALES_PER_BLOCK].contiguous(), k_scale, v_scale, 0, 0, 3, sm_scale, 0)
    tail_rows, split_len = tokens - rows, triton.cdiv(triton.cdiv(keys, CTA_K), splits) * CTA_K
    parts = torch.empty((batch * splits, tail_rows, heads, dim), device=q.device, dtype=out.dtype)
    lse = torch.empty((batch * splits, heads, tail_rows), device=q.device, dtype=torch.float32)
    scale_bits = struct.unpack("<i", struct.pack("<f", sm_scale))[0]  # passed as a 32-bit float (Python floats go as doubles)
    _kernel(q.device, dim, out.dtype, pv_fp16)((blocks - main, heads, batch * splits), (32, 4, 1),
        [q, k, v, parts, lse, q_scale, k_scale, v_scale, tokens, keys, heads // k.shape[2], q.stride(0), q.stride(1), q.stride(2),
         k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(2), v.stride(1), tail_rows * heads * dim, heads * dim, dim,
         scale_bits, main, splits, split_len, main, tail_rows], shared_mem=_shared_mem(dim), stream=torch.cuda.current_stream(q.device))
    split_merge.merge(parts, lse, out, rows)
    return True
