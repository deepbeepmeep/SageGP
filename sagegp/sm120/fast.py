# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Dense SM120 attention with every score channel in NVFP4 and FP8 values (Sage2 quality "fast").

The scores run on the block-scaled NVFP4 tensor cores (twice the INT8 rate) for every head, so the speed does not depend on
how compressible the activations are. Packing keeps their error well below SageAttention 3's: Q and centred K are rotated by
one signed Hadamard matrix (exact in Q.K), then each 16-channel block is rounded and its error spread over the later channels
of the row by the matrix that minimises the score error in the partner's second-moment metric (GPTQ order, per block). K uses
Q's statistics when the queries are available (direct calls), otherwise its own. The spread matrices come from matrix products
only (Newton-Schulz inverses and Schur downdates), so a call can be recorded in a CUDA graph.
"""
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from .. import hooks
from . import dense
from .common import _pack_pairs, _unpack_pairs, supported

UNIT = 0.125  # row unit of both packed operands; the kernel's dequantization is the constant UNIT * UNIT
DAMP = 0.02  # metric damping, relative to its mean diagonal
SAMPLES = 1024  # rows sampled for the key mean and the second moments
ROWS = 64


def eligible(device, dtype, batch, queries, keys, heads, kv_heads, dim):
    return (queries == keys and queries > 8192
            and dense.eligible(batch, queries, keys, heads, kv_heads, dim, dtype) and supported(device, dtype))


def settings(recycle_query):
    return {"recycle_query": recycle_query}


@lru_cache(None)
def _kernel(device):
    image = Path(__file__).with_name("cubins") / "fixed_sm120a.cubin"
    with torch.cuda.device(device):
        name = "wangp_dense_nvfp4_strided_q"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), [name])[name]


@lru_cache(None)
def _signs(device):
    generator = torch.Generator().manual_seed(1234)
    return hooks.register_constant((torch.randint(0, 2, (128,), generator=generator).float() * 2 - 1).to(device))


@lru_cache(None)
def _rotation(device):
    """The signed Hadamard matrix the packer applies (rows scaled by the signs), for the sampled statistics."""
    h = torch.ones((1, 1))
    while h.size(0) < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return hooks.register_constant((_signs(device)[:, None] * h.to(device) * 128 ** -0.5).contiguous())


@triton.jit
def _ns_inverse(a, eye):
    """Newton-Schulz inverse of an SPD tile, FP16 products with FP32 accumulation, fixed iteration count."""
    x = eye / tl.sum(tl.sum(tl.where(eye > 0, a, 0.), 1), 0)
    for _ in range(30):
        x = tl.dot(x.to(tl.float16), (2. * eye - tl.dot(a.to(tl.float16), x.to(tl.float16))).to(tl.float16))
    return x


@triton.jit
def _spread(M, OUT, DAMP: tl.constexpr):
    # Rows of block b: the correction of the later channels per unit of block b's rounding error, -(H_ll^-1 H_lb)^T, from the
    # precision of the channels not yet rounded (P_bb^-1 P_bl), downdated block by block.
    head = tl.program_id(0)
    i = tl.arange(0, 128)
    j = tl.arange(0, 16)
    eye = (i[:, None] == i[None, :]).to(tl.float32)
    eye16 = (j[:, None] == j[None, :]).to(tl.float32)
    m = tl.load(M + head * 16384 + i[:, None] * 128 + i[None, :])
    mean_diag = tl.maximum(tl.sum(tl.sum(m * eye, 1), 0) / 128., 1e-30)
    p = _ns_inverse(m / mean_diag + DAMP * eye, eye)
    blocks = i // 16
    for b in tl.static_range(7):
        rows = tl.sum(tl.where((tl.arange(0, 8) == b)[:, None, None], tl.reshape(p, (8, 16, 128)), 0.), 0)
        pbb = tl.sum(tl.where((blocks == b)[None, None, :] & (j[None, :, None] == (i % 16)[None, None, :]), rows[:, None, :], 0.), 2)
        pbl = tl.where((blocks > b)[None, :], rows, 0.)
        s = tl.dot(_ns_inverse((pbb + tl.trans(pbb)) * 0.5, eye16).to(tl.float16), pbl.to(tl.float16))
        tl.store(OUT + head * 16384 + (b * 16 + j[:, None]) * 128 + i[None, :], s)
        p -= tl.dot(tl.trans(pbl).to(tl.float16), s.to(tl.float16))
    tl.store(OUT + head * 16384 + (112 + j[:, None]) * 128 + i[None, :], tl.zeros((16, 128), tl.float32))


@triton.jit(do_not_specialize=["N", "ST", "SH", "OT", "OH"])
def _pack(X, MEAN, SIGNS, SPREAD, OUT, N, ST, SH, OT, OH, R: tl.constexpr, CENTER: tl.constexpr, RECYCLE: tl.constexpr):
    # Block b of the rotated row is x @ H[:, b] (H the +-1 Hadamard matrix, exact in BF16) minus the errors of the blocks
    # already rounded, spread by SPREAD. Row layout of the kernel: 64 bytes of E2M1 codes, eight E4M3 block scales at byte 96.
    tile, head = tl.program_id(0), tl.program_id(1)
    row = tile * R + tl.arange(0, R)
    d = tl.arange(0, 128)
    c = tl.arange(0, 16)
    valid = row[:, None] < N
    x = tl.load(X + row[:, None].to(tl.int64) * ST + head * SH + d[None, :], valid, 0.)
    if CENTER:
        x = x.to(tl.float32) - tl.load(MEAN + head * 128 + d)[None, :]
    x = tl.where(valid, x * tl.load(SIGNS + d)[None, :], 0.).to(tl.bfloat16)
    if RECYCLE:
        tl.debug_barrier()
    dest = OUT + row[:, None].to(tl.int64) * OT + head * OH
    errors = tl.zeros((R, 128), tl.float32)
    for b in tl.static_range(8):
        col = b * 16 + c
        bits = d[:, None] & col[None, :]  # Sylvester Hadamard: H[i, j] = (-1)^popcount(i & j)
        bits = bits ^ (bits >> 1)
        bits = bits ^ (bits >> 2)
        bits = bits ^ (bits >> 4)
        hb = tl.where((bits & 1) == 0, 1., -1.).to(tl.bfloat16)
        blk = tl.dot(x, hb) * (1. / (11.313708498984761 * 0.125))  # / sqrt(128), / UNIT
        if b > 0:
            blk -= tl.dot(errors.to(tl.float16), tl.load(SPREAD + head * 16384 + d[:, None] * 128 + col[None, :]).to(tl.float16))
        sf = tl.minimum(tl.max(tl.abs(blk), 1) * (1. / 6.), 448.).to(tl.float8e4nv).to(tl.float32)
        codes = _pack_pairs(blk / tl.where(sf > 0., sf, 1.)[:, None], R, 16)
        tl.store(dest + b * 8 + tl.arange(0, 8)[None, :], codes.to(tl.int8, bitcast=True), valid)
        tl.store(dest + 96 + b, sf.to(tl.float8e4nv).to(tl.int8, bitcast=True)[:, None], valid)
        if b < 7:
            err = (blk - _unpack_pairs(codes, R, 16) * sf[:, None]).to(tl.float16)
            errors += tl.dot(err, (c[:, None] + b * 16 == d[None, :]).to(tl.float16))


def _moments(x):
    """Second moments [h, 128, 128] (FP32) and means [h, 128] of evenly spaced rows of x [1, n, h, 128], from BF16 rows."""
    n = x.size(1)
    count = min(SAMPLES, n)
    ids = torch.linspace(0, n - 1, count, device=x.device).round().long()
    rows = x[0].index_select(0, ids).transpose(0, 1)  # [h, s, 128]
    second = torch.bmm(rows.mT, rows, out_dtype=torch.float32).div_(count)
    return second, rows.sum(1, dtype=torch.float32).div_(count)


def _spread_from(second):
    """Spread matrices [h, 128, 128] for second moments [h, 128, 128] of the unrotated rows."""
    rot = _rotation(second.device)
    metric = (rot.mT @ second @ rot).contiguous()
    out = torch.empty_like(metric)
    _spread[(metric.size(0),)](metric, out, DAMP, num_warps=8)
    return out


def _pack_rows(x, out, mean, spread, recycle):
    _, n, h, _ = x.shape
    _pack[(triton.cdiv(n, ROWS), h)](x, mean, _signs(x.device), spread, out, n, x.stride(1), x.stride(2), out.stride(2), out.stride(1),
                                     ROWS, mean is not None, recycle, num_warps=4, num_stages=1)
    torch.autograd.graph.increment_version(out)


def staged_quantize_v(values, settings):
    return dense.staged_quantize_v(values, settings)


def _quantize_k(x, settings, query_moments=None):
    if x.stride(-1) != 1:
        x = x.contiguous()
    _, n, h, _ = x.shape
    second, mean = _moments(x)  # any key vector common to all keys cancels in the softmax: centre on the mean
    settings["key_spread"] = _spread_from(second - mean[:, :, None] * mean[:, None, :])  # for the queries
    spread = _spread_from(query_moments) if query_moments is not None else settings["key_spread"]
    out = torch.empty((1, h, n, 128), dtype=torch.int8, device=x.device)  # the kernel's key head stride is n * 128
    _pack_rows(x, out, mean, spread, False)
    return out, torch.full((h, triton.cdiv(n, 64)), UNIT, device=x.device)


@torch.autocast(device_type="cuda", enabled=False)
def staged_quantize_k(keys, settings):
    return _quantize_k(keys.pop(), settings)


@torch.autocast(device_type="cuda", enabled=False)
def staged_quantize_q(q, settings):
    _, n, h, _ = q.shape
    recycle = settings["recycle_query"] and dense._recyclable_query(q)
    if not recycle and q.stride(-1) != 1:
        q = q.contiguous()
    out = q.view(torch.int8)[..., :128].transpose(1, 2) if recycle else torch.empty((1, h, n, 128), dtype=torch.int8, device=q.device)
    _pack_rows(q, out, None, settings.pop("key_spread"), recycle)
    settings["mixed"] = torch.zeros(h, dtype=torch.uint8, device=q.device)
    return out, torch.full((h, triton.cdiv(n, 128) * 4), UNIT, device=q.device)


@lru_cache(None)
def _split_kernel(device):
    """The kernel over ranges of the keys (dense.SPLIT_KEYS: the last wave's query blocks)."""
    with torch.cuda.device(device):
        image = Path(__file__).with_name("cubins") / "fixed_split_sm120a.cubin"
        return torch.cuda._utils._cuda_load_module(image.read_bytes(), ["wangp_dense_fast_split"])["wangp_dense_fast_split"]


def staged_attention(out, packed, settings, mask=None):
    return dense._staged_attention(out, packed, settings, mask, kernels=(_kernel(out.device),), split_kernels=(_split_kernel(out.device),))


def attention_owned(inputs, *, recycle_q, out=None):
    q, k, v = inputs
    inputs.clear()
    state = settings(recycle_q and dense._recyclable_query(q))
    key = _quantize_k(k, state, _moments(q)[0])
    del k
    values = staged_quantize_v([v], state)
    del v
    query = staged_quantize_q(q, state)
    if out is None:
        out = q if state["recycle_query"] else torch.empty_like(q, memory_format=torch.contiguous_format)
    return staged_attention(out, [query, key, values], state)
