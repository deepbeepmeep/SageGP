# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Unmasked SM120 attention with 256-dimensional values and data-dependent score width."""
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .common import _pack_pairs, supported


def eligible(batch, queries, keys, heads, kv_heads, dim, dtype):
    # The reduction of the range table is specialized for up to 256 row groups.
    driver = getattr(torch.cuda, "_utils", None)
    return (batch == 1 and dim == 256 and dtype == torch.bfloat16 and 0 < heads == kv_heads <= 65535
            and 2048 <= queries == keys <= 16384 and queries * heads >= 32768 and queries * heads * dim < 2**31
            and hasattr(driver, "_cuda_load_module") and hasattr(getattr(driver, "_CudaKernel", None), "set_shared_memory_config"))


def preferred(device, dtype, batch, queries, keys, heads, kv_heads, dim):
    return dim == 256 and supported(device, dtype) and eligible(batch, queries, keys, heads, kv_heads, dim, dtype)


@lru_cache(None)
def _kernels(device):
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / "wide" / "wide_sm120a.cubin"
        functions = torch.cuda._utils._cuda_load_module(image.read_bytes(), ["dense_compact", "dense_full"])
        compact, full = functions["dense_compact"], functions["dense_full"]
        compact.set_shared_memory_config(49152)
        full.set_shared_memory_config(65536)
        return compact, full


@triton.jit
def _center(K, MEAN, N, ST, SH):
    head = tl.program_id(0)
    row = tl.arange(0, 64) * (N - 1) // 63
    col = tl.arange(0, 256)
    k = tl.load(K + row[:, None].to(tl.int64) * ST + head.to(tl.int64) * SH + col[None, :]).to(tl.float32)
    tl.store(MEAN + head * 256 + col, tl.sum(k * (1. / 64), 0))


@triton.jit
def _ranges(Q, K, V, MEAN, PARTS, N, H, QT, QH, KT, KH, VT, VH, GROUPS):
    tile, head = tl.program_id(0), tl.program_id(1)
    group = tile // 4
    row = group * 64 + tl.arange(0, 64)
    col = tile % 4 * 64 + tl.arange(0, 64)
    valid = row[:, None] < N
    r, h = row[:, None].to(tl.int64), head.to(tl.int64)
    q = tl.load(Q + r * QT + h * QH + col[None, :], valid, 0.).to(tl.float32)
    k = tl.load(K + r * KT + h * KH + col[None, :], valid, 0.).to(tl.float32)
    v = tl.load(V + r * VT + h * VH + col[None, :], valid, 0.).to(tl.float32)
    k = tl.where(valid, k - tl.load(MEAN + head * 256 + col)[None, :], 0.)
    out = PARTS + (head * GROUPS + group) * 256 + col
    step = H * GROUPS * 256
    tl.store(out, tl.max(tl.abs(q), 0))
    tl.store(out + step, tl.max(tl.abs(k), 0))
    tl.store(out + 2 * step, tl.max(tl.abs(v), 0))


@triton.jit
def _bounds(PARTS, BOUNDS, VS, H, GROUPS, BLOCK: tl.constexpr):
    head = tl.program_id(1)
    row = tl.arange(0, BLOCK)
    col = tl.program_id(0) * 32 + tl.arange(0, 32)
    src = PARTS + (head * GROUPS + row[:, None]) * 256 + col[None, :]
    stride = H * GROUPS * 256
    q = tl.max(tl.load(src, row[:, None] < GROUPS, 0.), 0)
    k = tl.max(tl.load(src + stride, row[:, None] < GROUPS, 0.), 0)
    v = tl.max(tl.load(src + 2 * stride, row[:, None] < GROUPS, 0.), 0)
    tl.store(BOUNDS + head * 256 + col, (q * .0625) * k)
    # A 64-key FP16 partial cannot overflow at this scale. Preserve exponent range
    # for small BF16 values too. exp2(-126) is the smallest normal FP32 scale.
    exponent = tl.maximum(-126., tl.ceil(tl.log2(tl.maximum(v, 1.e-37))) - 9.)
    tl.store(VS + head * 256 + col, tl.exp2(exponent))


@triton.jit
def _select(BOUNDS, ORDER, FLAGS):
    head = tl.program_id(0)
    d = tl.arange(0, 256)
    bounds = tl.load(BOUNDS + head * 256 + d)
    keyed = (bounds.to(tl.uint32, bitcast=True) & 0xffffff00) | d.to(tl.uint32)
    order = (tl.sort(keyed, descending=True) & 255).to(tl.int32)
    sorted_bounds = tl.gather(bounds, order, 0)
    # Sum of coordinatewise absolute bounds applies to every query-key pair.
    omitted = tl.sum(tl.where(d >= 144, sorted_bounds, 0.), 0)
    lower = tl.sum(tl.where(d >= 96, sorted_bounds, 0.), 0)
    tl.store(FLAGS + head, (omitted <= .0625) & (lower <= 1.25))
    tl.store(ORDER + head * 256 + d, order.to(tl.uint8))


@triton.jit
def _pack(X, V, ORDER, FLAGS, CENTER, O, S, VO, VS, N, ST, SH, VT, VH,
          ROWS: tl.constexpr, PAD, KEY: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    row = tile * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, 256)
    r, h = row[:, None].to(tl.int64), head.to(tl.int64)
    x = tl.load(X + r * ST + h * SH + col[None, :], r < N, 0.).to(tl.float32)
    if KEY:
        x = tl.where(r < N, x - tl.load(CENTER + head * 256 + col)[None, :], 0.)
    else:
        v = tl.load(V + r * VT + h * VH + col[None, :], r < N, 0.).to(tl.float32)
        v /= tl.load(VS + head * 256 + col)[None, :]
        tl.store(VO + (head * N + r) * 256 + col[None, :], v, r < N)
    dst = O + (head * N + r) * 256
    if tl.load(FLAGS + head):
        order = tl.load(ORDER + head * 256 + col).to(tl.int32)
        x = tl.gather(x, tl.broadcast_to(order[None, :], (ROWS, 256)), 1)
        hi_col = tl.arange(0, 128)
        high = tl.gather(x, tl.broadcast_to(hi_col[None, :], (ROWS, 128)), 1)
        high = tl.where(hi_col[None, :] < 96, high, 0.)
        lo_col = tl.arange(0, 64)
        low = tl.gather(x, tl.broadcast_to((96 + lo_col)[None, :], (ROWS, 64)), 1)
        groups = tl.reshape(tl.where(lo_col[None, :] < 48, low, 0.), (ROWS, 4, 16))
        peaks = tl.max(tl.abs(groups), 2)
        high_unit = tl.max(tl.max(tl.abs(high), 1), 0) / 127.
        low_unit = tl.max(tl.max(peaks, 1), 0) / (448. * 6.)
        unit = tl.maximum(tl.maximum(high_unit, low_unit), tl.full((), 2.0 ** -126, tl.float32))
        tl.store(dst + hi_col[None, :], libdevice.rint(high / unit).to(tl.int8), (r < N) & (hi_col[None, :] < 96))
        sf = tl.minimum(peaks / (6. * unit), 448.).to(tl.float8e4nv)
        factor = sf.to(tl.float32) * unit
        codes = _pack_pairs(tl.reshape(groups / tl.where(factor > 0., factor, 1.)[:, :, None], (ROWS, 64)), ROWS, 64)
        encoded = tl.arange(0, 32)
        tl.store(dst + 96 + encoded[None, :], codes.to(tl.int8, bitcast=True), (r < N) & (encoded[None, :] < 24))
        suffix = tl.arange(0, 8)
        scales = tl.gather(sf.to(tl.int8, bitcast=True), tl.broadcast_to((suffix % 4)[None, :], (ROWS, 8)), 1)
        # The other operand has zero in these slots, including when the MMA uses
        # the scale bytes as data for its final 16 logical lanes.
        live = suffix >= 4 if KEY else suffix < 4
        tl.store(dst + 120 + suffix[None, :], tl.where(live[None, :], scales, 0), r < N)
    else:
        unit = tl.maximum(tl.max(tl.max(tl.abs(x), 1), 0) / 127., tl.full((), 2.0 ** -126, tl.float32))
        tl.store(dst + col[None, :], libdevice.rint(x / unit).to(tl.int8), r < N)
    tl.store(S + head * PAD + tile, unit)


def _plan(q, k, v):
    _, n, h, d = q.shape
    groups = triton.cdiv(n, 64)
    center = torch.empty((h, d), dtype=torch.float32, device=q.device)
    bounds = torch.empty_like(center)
    vs = torch.empty_like(center)
    order = torch.empty_like(center, dtype=torch.uint8)
    flags = torch.empty((h,), dtype=torch.uint8, device=q.device)
    parts = torch.empty((3, h, groups, d), dtype=torch.float32, device=q.device)
    _center[(h,)](k, center, n, *k.stride()[1:3], num_warps=4)
    _ranges[(groups * 4, h)](q, k, v, center, parts, n, h, *q.stride()[1:3], *k.stride()[1:3],
                              *v.stride()[1:3], groups, num_warps=4)
    _bounds[(8, h)](parts, bounds, vs, h, groups, triton.next_power_of_2(groups), num_warps=4)
    _select[(h,)](bounds, order, flags, num_warps=4)
    return center, order, flags, vs


def attention(q, k, v):
    """Run direct NHD attention on one SM120 GPU. Callers select unmasked, default-scale inference."""
    return attention_owned([q, k, v], recycle_q=False)


def attention_owned(inputs, *, recycle_q):
    q, k, v = inputs
    inputs.clear()
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 256
            or not 2048 <= q.shape[1] <= 16384 or not 0 < q.shape[2] <= 65535 or q.numel() >= 2**31
            or q.device.type != "cuda" or any(t.device != q.device or t.dtype != torch.bfloat16 or t.stride(-1) != 1
                                               for t in (q, k, v))):
        raise ValueError("Expected equal, square BF16 NHD inputs with head width 256 in the supported range")
    _, n, h, d = q.shape
    groups = triton.cdiv(n, 64)
    with torch.cuda.device(q.device):
        compact, full = _kernels(q.device)
        center, order, flags, vs = _plan(q, k, v)
        def encode(x, key, values):
            rows, pad = (64, groups) if key else (16, groups * 4)
            data = torch.empty((h, n, d), dtype=torch.int8, device=q.device)
            scales = torch.empty((h, pad), dtype=torch.float32, device=q.device)
            _pack[(triton.cdiv(n, rows), h)](x, v, order, flags, center, data, scales, values, vs, n,
                                              *x.stride()[1:3], *v.stride()[1:3], rows, pad, key, num_warps=4)
            return data, scales

        key, ks = encode(k, True, None)
        del k
        values = torch.empty((h, n, d), dtype=torch.float16, device=q.device)
        query, qs = encode(q, False, values)
        del v
        out = q if recycle_q and q.is_contiguous() and q.data_ptr() % 16 == 0 else torch.empty(q.shape, dtype=q.dtype, device=q.device)
        args = [query, key, values, out, qs, ks, vs, flags, n, h]
        stream = torch.cuda.current_stream()
        compact((groups, h, 1), (32, 4, 1), args, shared_mem=49152, stream=stream)
        full((groups, h, 1), (32, 4, 1), args, shared_mem=65536, stream=stream)
    torch.autograd.graph.increment_version(out)
    return out
