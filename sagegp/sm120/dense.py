# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Dense SM120 attention with adaptive INT8 / NVFP4 score channels and FP8 values.

Each head ranks channels once from sample rows. The same permutation packs Q and K
for every score in that head. Heads with a diffuse energy distribution keep all
128 channels in INT8. The selector and all graph-replay decisions stay on-device.
When the caller owns the query projection, its BF16 rows can hold the packed
query until the attention CTA replaces those same rows with its BF16 output.
"""
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .common import _pack_pairs, recyclable_query
from .rows import _rows
from .values import pack_value


def eligible(batch, queries, keys, heads, kv_heads, dim, dtype):
    return (batch == 1 and 0 < heads == kv_heads <= 65535 and dim == 128 and dtype == torch.bfloat16
            and 2048 <= queries == keys <= 1048576
            and keys * heads * dim < 2**31
            and hasattr(getattr(torch.cuda, "_utils", None), "_cuda_load_module"))


@lru_cache(None)
def _kernel(device, strided_query=False):
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / ("strided_query_sm120a.cubin" if strided_query else "dense_sm120a.cubin")
        name = "wangp_dense_nvfp4_strided_q" if strided_query else "wangp_dense_nvfp4"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), [name])[name]


@lru_cache(None)
def _precise_kernel(device):
    """The INT8 path in a faster schedule with the same arithmetic (csrc/sm120/precise): heads without NVFP4 tails only."""
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / "precise_sm120a.cubin"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), ["wangp_dense_nvfp4_strided_q"])["wangp_dense_nvfp4_strided_q"]


# Split keys (precise mode, comparison switch): the query blocks of a launch's part-filled last wave run over ranges of the keys
# (precise_split_sm120a.cubin), each range's normalized output and log2-sum-exp merged after (_merge_splits); the full waves before
# them run as before. Only the tail's ranges take VRAM: a wave of CTAs' outputs (about 11 MB on an RTX 5090).
SPLIT_KEYS = False
SPLIT_RANGES = 8  # ranges of the keys of the last wave's blocks
SPLIT_MIN_KEY_BLOCKS = 16  # 64-key blocks per range at least
CTAS_PER_SM = 2  # the precise kernel's occupancy (128 threads, 32 KB of shared memory)


@lru_cache(None)
def _split_kernel(device):
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / "precise_split_sm120a.cubin"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), ["wangp_dense_precise_split"])["wangp_dense_precise_split"]


@lru_cache(None)
def _tail_split(device, blocks, heads, key_blocks):
    """(query blocks run whole, ranges of keys of the blocks after them): the blocks of the last, part-filled wave of the GPU's CTA slots
    split in SPLIT_RANGES (measured on an RTX 5090 at 220k tokens: 8 ranges 4 to 19 % faster for 1 to 4 heads, splitting earlier waves
    too never faster)."""
    slots = torch.cuda.get_device_properties(device).multi_processor_count * CTAS_PER_SM
    main = blocks * heads // slots * slots // heads
    splits = min(SPLIT_RANGES, key_blocks // SPLIT_MIN_KEY_BLOCKS)
    return (main, splits) if main < blocks and splits > 1 else (blocks, 1)


@triton.jit
def _merge_splits(P, L, O, T, H, OST, OSH, S: tl.constexpr, BT: tl.constexpr):
    """O[t, h] = sum_s 2^(L[s, h, t] - m) P[s, t, h] / sum_s 2^(L[s, h, t] - m), m = max_s L[s, h, t], in FP32."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1).to(tl.int64)
    inside = t < T
    t64 = t.to(tl.int64)
    d = tl.arange(0, 128)
    m = tl.full((BT,), float("-inf"), tl.float32)
    for s in tl.static_range(S):
        m = tl.maximum(m, tl.load(L + (s * H + h) * T + t64, inside, float("-inf")))
    acc = tl.zeros((BT, 128), tl.float32)
    den = tl.zeros((BT,), tl.float32)
    for s in tl.static_range(S):
        w = tl.exp2(tl.load(L + (s * H + h) * T + t64, inside, float("-inf")) - m)
        w = tl.where(inside, w, 0.0)
        acc += w[:, None] * tl.load(P + ((s * T + t64[:, None]) * H + h) * 128 + d[None, :], inside[:, None], 0).to(tl.float32)
        den += w
    out = tl.div_rn(acc, tl.where(inside, den, 1.0)[:, None])
    tl.store(O + t64[:, None] * OST + h * OSH + d[None, :], out.to(O.dtype.element_ty), inside[:, None])


@triton.jit
def _read(X, W, ROPE, row, head, N, ST, SH, RT, RH, RP: tl.constexpr, RC: tl.constexpr, WH, EPS: tl.constexpr,
          PAIRS: tl.constexpr, INTERLEAVED: tl.constexpr, ROPE_FIRST: tl.constexpr, R: tl.constexpr):
    channel = tl.arange(0, 128)
    return _rows(X, W, None, ROPE, None, row.to(tl.int64) * ST + head.to(tl.int64) * SH, row, row < N, channel, 0, head,
                 0, RT, RH, 0, 0, WH, EPS, RP, RC, 128, PAIRS, W is not None, ROPE is not None,
                 False, R, True, INTERLEAVED, False, ROPE_FIRST, True).to(X.dtype.element_ty).to(tl.float32)


@triton.jit
def _rank(K, Q, W, ROPE, ORDER, MIXED, MEAN, N, ST, SH, QT, QH, RT, RH, RP: tl.constexpr, RC: tl.constexpr,
          WH, EPS: tl.constexpr, PAIRS: tl.constexpr, INTERLEAVED: tl.constexpr, ROPE_FIRST: tl.constexpr, SAMPLE: tl.constexpr):
    head = tl.program_id(0)
    row = tl.arange(0, SAMPLE) * (N - 1) // (SAMPLE - 1)
    d = tl.arange(0, 128)
    keys = _read(K, W, ROPE, row, head, N, ST, SH, RT, RH, RP, RC, WH, EPS, PAIRS, INTERLEAVED, ROPE_FIRST, SAMPLE)
    center = tl.sum(keys, 0) / SAMPLE
    keys -= center[None, :]
    importance = tl.sum(keys * keys, 0)
    if Q is not None:
        queries = tl.load(Q + row[:, None].to(tl.int64) * QT + head.to(tl.int64) * QH + d[None, :]).to(tl.float32)
        importance *= tl.sum(queries * queries, 0)
    # Encode the index in low mantissa bits for a unique and deterministic ordering.
    ranked = tl.sort((importance.to(tl.uint32, bitcast=True) & 0xffffff00) | d.to(tl.uint32), descending=True)
    energy = (ranked & 0xffffff00).to(tl.float32, bitcast=True)
    tail = tl.sum(tl.where(d >= 64, energy, 0.), 0)
    cutoff: tl.constexpr = 0.06 if Q is not None else 0.10
    tl.store(MIXED + head, tail < cutoff * tl.sum(energy, 0))
    tl.store(ORDER + head * 128 + d, (ranked & 255).to(tl.uint8))
    tl.store(MEAN + head * 128 + d, center)


@triton.jit
def _encode(X, W, ROPE, ORDER, MIXED, MEAN, OUT, SCALES, N, ST, SH, RT, RH, RP: tl.constexpr, RC: tl.constexpr,
            WH, EPS: tl.constexpr, PAIRS: tl.constexpr, INTERLEAVED: tl.constexpr, ROPE_FIRST: tl.constexpr,
            R: tl.constexpr, PAD: tl.constexpr, RECYCLE: tl.constexpr = False):
    head, tile = tl.program_id(1), tl.program_id(0)
    row = tile * R + tl.arange(0, R)
    d = tl.arange(0, 128)
    x = _read(X, W, ROPE, row, head, N, ST, SH, RT, RH, RP, RC, WH, EPS, PAIRS, INTERLEAVED, ROPE_FIRST, R)
    if MEAN is not None:
        x = tl.where(row[:, None] < N, x - tl.load(MEAN + head * 128 + d)[None, :], 0.)
    if RECYCLE:
        # One program owns each BF16 row. Finish all reads, including RoPE
        # partners, before writing packed bytes into the same rows.
        tl.debug_barrier()
        dest = OUT + 2 * (head.to(tl.int64) * SH + row[:, None].to(tl.int64) * ST)
    else:
        dest = OUT + (head * N + row[:, None]) * 128
    if tl.load(MIXED + head):
        order = tl.load(ORDER + head * 128 + d).to(tl.int32)
        permuted = tl.gather(x, tl.broadcast_to(order[None, :], (R, 128)), 1)
        high, low = tl.split(tl.permute(tl.reshape(permuted, (R, 2, 64)), 0, 2, 1))
        scale = tl.maximum(tl.max(tl.max(tl.abs(high), 1), 0) / 127., 1.e-30)
        low = tl.reshape(low, (R, 4, 16))
        sub = tl.minimum(tl.max(tl.abs(low), 2) / (6. * scale), 448.).to(tl.float8e4nv)
        factor = sub.to(tl.float32) * scale
        codes = _pack_pairs(tl.reshape(low / tl.where(factor > 0., factor, 1.)[:, :, None], (R, 64)), R, 64)
        tl.store(dest + tl.arange(0, 64)[None, :], libdevice.rint(high / scale).to(tl.int8), row[:, None] < N)
        tl.store(dest + 64 + tl.arange(0, 32)[None, :], codes.to(tl.int8, bitcast=True), row[:, None] < N)
        tl.store(dest + 96 + tl.arange(0, 4)[None, :], sub.to(tl.int8, bitcast=True), row[:, None] < N)
    else:
        scale = tl.maximum(tl.max(tl.max(tl.abs(x), 1), 0) / 127., 1.e-30)
        tl.store(dest + d[None, :], libdevice.rint(x / scale).to(tl.int8), row[:, None] < N)
    tl.store(SCALES + head * PAD + tile, scale)


def _geometry(rope, norm, interleaved, rope_first):
    weight, eps, _ = norm
    head_stride = weight.stride(0) if weight is not None and weight.ndim == 2 and weight.size(0) > 1 else 0
    angles = (0, 0, 0, 0) if rope is None else (rope.stride(1), rope.stride(2) if rope.size(2) > 1 else 0,
                                                 rope.stride(3), rope.stride(4))
    return (*angles, head_stride, eps or 0., rope.size(3) if rope is not None else 0, interleaved, rope_first)


def _quantize(x, settings, key, q=None, rope=None, norm=(None, 0., None), interleaved=False, rope_first=False):
    _, n, heads, dim = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    strides = x.stride()[1:3]
    offsets = _geometry(rope, norm, interleaved, rope_first)
    if key:
        order = torch.empty((heads, dim), dtype=torch.uint8, device=x.device)
        mixed = torch.empty((heads,), dtype=torch.uint8, device=x.device)
        center = torch.empty((heads, dim), dtype=torch.float32, device=x.device)
        qs = (0, 0) if q is None else q.stride()[1:3]
        sample = 32 if q is None else 128
        _rank[(heads,)](x, q, norm[0], rope, order, mixed, center, n, *strides, *qs, *offsets, sample, num_warps=4)
        if settings.get("int8_scores", False):  # precise: every score in INT8
            mixed.zero_()
        settings.update(order=order, mixed=mixed)
    else:
        order, mixed, center = settings["order"], settings["mixed"], None
    rows = 64 if key else 32
    pad = triton.cdiv(n, 64) if key else triton.cdiv(n, 128) * 4
    recycle = not key and settings.get("recycle_query", False) and _recyclable_query(x)
    if recycle:
        out = x.view(torch.int8)[..., :dim].transpose(1, 2)
    else:
        out = torch.empty((1, heads, n, dim), dtype=torch.int8, device=x.device)
    scales = torch.empty((heads, pad), dtype=torch.float32, device=x.device)
    _encode[(triton.cdiv(n, rows), heads)](x, norm[0], rope, order, mixed, center, out, scales, n, *strides,
                                            *offsets, rows, pad, recycle, num_warps=4)
    if recycle:
        torch.autograd.graph.increment_version(x)
    return out, scales


def _recyclable_query(x):
    return x.dtype == torch.bfloat16 and recyclable_query(x)


def staged_quantize_v(value_list, settings):
    v = value_list.pop()
    return pack_value((v if v.stride(-1) == 1 else v.contiguous()).transpose(1, 2), False)


def staged_quantize_k(key_list, settings):
    return _quantize(key_list.pop(), settings, True)


def staged_quantize_q(q, settings):
    return _quantize(q, settings, False)


def _fuse(x, rope, norm, settings, key, interleaved, rope_first, quantize=None):
    weight, _, segments = norm
    if (segments is not None or x.stride(-1) != 1
            or weight is not None and (weight.ndim not in (1, 2) or weight.shape[-1] != 128 or weight.stride(-1) != 1)
            or rope is not None and (rope.ndim not in (5, 6) or rope.shape[3] > 64)):
        return None
    encode = _quantize if quantize is None else quantize
    return encode(x, settings, key, rope=rope, norm=norm, interleaved=interleaved, rope_first=rope_first)


def fused_warp_query(x, rope, norm, settings, *, interleaved, rope_first):
    return _fuse(x, rope, norm, settings, False, interleaved, rope_first)


def fused_warp_key(x, rope, norm, settings, *, interleaved, rope_first):
    return _fuse(x, rope, norm, settings, True, interleaved, rope_first)


def staged_attention(out, packed, settings, mask=None):
    kernels = (_precise_kernel(out.device),) if settings.get("int8_scores", False) else None
    return _staged_attention(out, packed, settings, mask, kernels)


def _staged_attention(out, packed, settings, mask, kernels=None, restore=None, shared_mem=32768):
    if mask is not None:
        raise ValueError("The dense NVFP4 kernel requires unmasked attention")
    (q, qs), (k, ks), (v, vs, _mean) = packed
    packed.clear()
    _, h, t, _ = q.shape
    n = k.size(2)
    # The vectorized epilogue may recycle a sliced NHD or HND projection.
    # Expanded, interleaved and unaligned views need their own output storage.
    st, sh = out.stride()[1:3]
    aligned = out.stride(3) == 1 and not (out.data_ptr() % 16 or st % 8 or sh % 8)
    span = 127 * out.stride(3) + 1
    disjoint = out.stride(3) > 0 and ((sh >= span and st >= (h - 1) * sh + span)
                                       or (st >= span and sh >= (t - 1) * st + span))
    borrowed = out
    if not (aligned and disjoint and (t - 1) * st + (h - 1) * sh + 128 < 2**31):
        out = torch.empty(out.shape, device=out.device, dtype=out.dtype)
    with torch.cuda.device(q.device):
        strided_query = kernels is not None or not q.is_contiguous()
        if kernels is None:
            kernels = (_kernel(q.device, strided_query),)
        blocks = triton.cdiv(t, 128)
        # Bound the work per driver submission on long videos. A query tile
        # still visits every key and shares the same prepared K and V.
        if blocks * 128 * n * h > (1 << 36):
            processors = torch.cuda.get_device_properties(q.device).multi_processor_count
            per_launch = max(1, (processors << 32) // (128 * n * h))
        else:
            per_launch = blocks
        args = [q, k, v, out, qs, ks, vs, settings["mixed"], t, n, h, v.size(3), *out.stride()[1:3]]
        if strided_query:
            args.extend((q.stride(2), q.stride(1)))
        stream = torch.cuda.current_stream()
        full_blocks = triton.cdiv(settings.get("full_query_rows", 0), 128)
        if full_blocks:
            # Prefix queries keep the full score and value width. They read
            # the same K/V and write directly into their own output rows.
            full_args = args.copy()
            full_args[7] = settings["full_query_modes"]
            for first in range(0, full_blocks, per_launch):
                kernels[-1]((min(full_blocks - first, per_launch), h, 1), (32, 4, 1),
                            [*full_args, first], shared_mem=shared_mem, stream=stream)
        main, splits = (_tail_split(q.device, blocks, h, triton.cdiv(n, 64)) if SPLIT_KEYS and settings.get("int8_scores", False) and not full_blocks
                        and strided_query else (blocks, 1))
        for first in range(full_blocks, main, per_launch):
            for kernel in kernels:
                kernel((min(main - first, per_launch), h, 1), (32, 4, 1), [*args, first], shared_mem=shared_mem, stream=stream)
        if splits > 1:
            # the tail's ranges of keys into their own outputs and lse, then merged into out's tail rows (which may hold the tail's packed
            # queries: written after)
            rows, split_len = t - main * 128, triton.cdiv(triton.cdiv(n, 64), splits) * 64
            parts = torch.empty((splits, rows, h, 128), device=q.device, dtype=out.dtype)
            lse = torch.empty((splits, h, rows), device=q.device, dtype=torch.float32)
            split_args = [q, k, v, parts, lse, qs, ks, vs, settings["mixed"], t, n, h, v.size(3), rows * h * 128, h * 128, 128, q.stride(2), q.stride(1),
                          main, splits, split_len, main, rows]
            _split_kernel(q.device)((blocks - main, h, splits), (32, 4, 1), split_args, shared_mem=shared_mem, stream=stream)
            _merge_splits[(triton.cdiv(rows, 64), h)](parts, lse, out[:, main * 128:], rows, h, out.stride(1), out.stride(2), splits, 64, num_warps=4)
            del parts, lse
    torch.autograd.graph.increment_version(out)
    if restore is not None:
        restore(out, settings)
    if out is not borrowed and disjoint:
        return borrowed.copy_(out)
    return out


def attention(q, k, v):
    return attention_owned([q, k, v], recycle_q=False)


def attention_owned(inputs, *, recycle_q, out=None, int8_scores=False):
    """int8_scores: every head keeps its 128 score channels in INT8 (the precise mode)."""
    q, k, v = inputs
    inputs.clear()
    recycle = recycle_q and _recyclable_query(q)
    settings = {"recycle_query": recycle, "int8_scores": int8_scores}
    key = _quantize(k, settings, True, q=q)
    del k
    values = staged_quantize_v([v], settings)
    del v
    query = staged_quantize_q(q, settings)
    if out is None:
        out = q if recycle else torch.empty_like(q, memory_format=torch.contiguous_format)
    return staged_attention(out, [query, key, values], settings)
