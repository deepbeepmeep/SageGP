# Copyright (c) 2024 by SageAttention team; changes Copyright 2026 DeepBeepMeep. Licensed under the Apache License, Version 2.0.
"""SageAttention's Triton per-block attention (INT8 QK, one scale per 128 queries and per 64 keys, FP16 PV: sageattention.triton.
attn_qk_int8_per_block._attn_fwd, the kernel of GPUs without the INT8 / FP8 CUDA kernels) for the query blocks of a part-filled last
wave, over ranges of the keys: each program's range writes its normalized output and log2-sum-exp, merged by sagegp.split_merge. The
arithmetic of a range is that kernel's, unchanged; the full blocks run the kernel itself (launched over the first query blocks only)."""
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sagegp import split_merge

_INTS = ["qo_len", "kv_len", "FIRST", "ROWS", "SPLIT_LEN"]


@triton.jit
def _range_inner(acc, l_i, m_i, q, q_scale, kv_len, lo, hi, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, BLOCK_N: tl.constexpr,
                 offs_n: tl.constexpr):
    """attn_qk_int8_per_block._attn_fwd_inner (unmasked) over the keys [lo, hi)."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
        qk += tl.where(k_mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit(do_not_specialize=_INTS, do_not_specialize_on_alignment=_INTS)
def _attn_range(Q, K, V, Q_scale, K_scale, Parts, Lse, stride_qz, stride_qh, stride_qn, stride_kz, stride_kh, stride_kn, stride_vz, stride_vh,
                stride_vn, qo_len, kv_len, FIRST, ROWS, SPLIT_LEN, H: tl.constexpr, num_kv_groups: tl.constexpr, SPLITS: tl.constexpr,
                HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Query block FIRST + program 0 of head program 1, keys of range program 2 % SPLITS of batch item program 2 // SPLITS."""
    start_m = FIRST + tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_z = (tl.program_id(2) // SPLITS).to(tl.int64)
    split = tl.program_id(2) % SPLITS
    lo = split * SPLIT_LEN
    hi = tl.minimum(kv_len, lo + SPLIT_LEN)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N) + lo // BLOCK_N
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + (lo + offs_n[None, :]) * stride_kn + offs_k[:, None]
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + (lo + offs_n[:, None]) * stride_vn + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale + q_scale_offset + start_m)
    acc, l_i, m_i = _range_inner(acc, l_i, m_i, q, q_scale, kv_len, lo, hi, K_ptrs, K_scale + k_scale_offset, V_ptrs, stride_kn, stride_vn,
                                 BLOCK_N, offs_n)
    acc = acc / l_i[:, None]
    local = (offs_m - FIRST * BLOCK_M).to(tl.int64)
    inside = offs_m < qo_len
    out = tl.program_id(2).to(tl.int64)
    tl.store(Parts + ((out * ROWS + local[:, None]) * H + off_h) * HEAD_DIM + offs_k[None, :], acc.to(Parts.dtype.element_ty), inside[:, None])
    tl.store(Lse + (out * H + off_h) * ROWS + local, tl.log2(l_i) + m_i, inside)


def tail(q, k, v, q_scale, k_scale, out, first_block, splits, block_m=128, num_warps=8, num_stages=4):
    """out's rows from first_block * block_m: attention over `splits` ranges of the keys, merged. q, k (batch, tokens, heads, dim) INT8
    and v (batch, keys, heads, dim) FP16, with the scales of sageattention.triton.quant_per_block (one per block_m queries, per 64 keys)."""
    batch, tokens, heads, dim = q.shape
    keys = k.shape[1]
    rows = tokens - first_block * block_m
    split_len = triton.cdiv(triton.cdiv(keys, 64), splits) * 64
    parts = torch.empty((batch * splits, rows, heads, dim), device=q.device, dtype=out.dtype)
    lse = torch.empty((batch * splits, heads, rows), device=q.device, dtype=torch.float32)
    _attn_range[(triton.cdiv(tokens, block_m) - first_block, heads, batch * splits)](
        q, k, v, q_scale, k_scale, parts, lse, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1), tokens, keys, first_block, rows, split_len, heads, heads // k.shape[2], splits, dim, block_m, 64,
        num_warps=num_warps, num_stages=num_stages)
    split_merge.merge(parts, lse, out, first_block * block_m)


@lru_cache(None)
def slots(device, dim, block_m=128, num_warps=8, num_stages=4, dtype=torch.float16):
    """CTAs the GPU runs at once of this attention (SMs x the range kernel's occupancy, compiled once for this tile: the per-block kernel
    has its tile, warps and stages)."""
    q = torch.zeros((1, block_m, 1, dim), device=device, dtype=torch.int8)
    v = torch.zeros((1, 64 * 16 * 2, 1, dim), device=device, dtype=torch.float16)
    k = torch.zeros((1, v.shape[1], 1, dim), device=device, dtype=torch.int8)
    scales = torch.ones(64, device=device, dtype=torch.float32)
    parts = torch.empty((2, block_m, 1, dim), device=device, dtype=dtype)
    lse = torch.empty((2, 1, block_m), device=device, dtype=torch.float32)
    compiled = _attn_range[(1, 1, 2)](q, k, v, scales, scales, parts, lse, *q.stride()[:1], q.stride(2), q.stride(1), k.stride(0), k.stride(2),
                                      k.stride(1), v.stride(0), v.stride(2), v.stride(1), block_m, k.shape[1], 0, block_m, 64 * 16, 1, 1, 2,
                                      dim, block_m, 64, num_warps=num_warps, num_stages=num_stages)
    return split_merge.triton_slots(device, compiled, num_warps)


def plan(device, q, k, block_m=128, num_warps=8, num_stages=4):
    """(query blocks the per-block kernel runs whole, ranges of the others): split_merge.tail_plan for this call."""
    batch, tokens, heads, dim = q.shape
    return split_merge.tail_plan(slots(device, dim, block_m, num_warps, num_stages), triton.cdiv(tokens, block_m), heads * batch,
                                 triton.cdiv(k.shape[1], 64))

