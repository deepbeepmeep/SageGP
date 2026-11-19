# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Per-channel E4M3 packing of the values in the layout of SageAttention's CUDA kernels
(https://github.com/thu-ml/SageAttention, Apache-2.0)."""
import torch
import triton
import triton.language as tl


@triton.jit
def scaled_e4m3(x, peak, BF16: tl.constexpr):
    # A finite BF16 subnormal can have an infinite FP32 reciprocal. Lift those channels by an
    # exact power of two before forming their multiplier; the output scale uses the original peak.
    if BF16:
        tiny = (peak > 0.) & (peak < 1.e-37)
        if tl.max(tiny.to(tl.int32), 0):
            lift = tl.where(tiny, 18446744073709551616., 1.)
            peak *= lift
            x *= lift[None, :]
    return (x * tl.where(peak == 0., 0., 2.25 / peak)[None, :]).to(tl.float8e4nv)


@triton.jit(do_not_specialize=["N", "H"])
def _value_amax(V, PARTIAL, N, H, D: tl.constexpr, VZ, VT, VH, ROWS: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    row = block.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    channel = tl.arange(0, D)
    x = tl.load(V + batch * VZ + head * VH + row[:, None] * VT + channel[None, :], row[:, None] < N, 0)
    peak = tl.max(tl.abs(x.to(tl.float32)), 0)
    tl.store(PARTIAL + ((batch * H + head) * tl.cdiv(N, ROWS) + block) * D + channel, peak)


@triton.jit(do_not_specialize=["N", "H"])
def _pack_value(V, PARTIAL, SCALE, OUT, N, H, D: tl.constexpr, VZ, VT, VH, OZ, OD, OH,
                ROWS: tl.constexpr, CHANNELS: tl.constexpr, GROUPS: tl.constexpr, PEAK_ROWS: tl.constexpr,
                TILE_WIDTH: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    block = tile // (D // CHANNELS)
    channel = tile % (D // CHANNELS) * CHANNELS + tl.arange(0, CHANNELS)
    row = block.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    # Sage2's FP8 MMA consumes each 16-token group as 0, 1, 8, 9, 2, 3, 10, 11, ...
    src = row // 16 * 16 + row % 4 // 2 * 8 + row % 16 // 4 * 2 + row % 2
    x = tl.load(V + batch * VZ + head * VH + src[:, None] * VT + channel[None, :], src[:, None] < N, 0).to(tl.float32)
    count = tl.cdiv(N, PEAK_ROWS)
    if GROUPS:
        group = tl.arange(0, GROUPS)
        peak = tl.max(tl.load(PARTIAL + ((batch * H + head) * count + group[:, None]) * D + channel[None, :],
                              group[:, None] < count, 0.), 0)
    else:
        peak = tl.load(PARTIAL + (batch * H + head) * D + channel)
    if block == 0:
        tl.store(SCALE + (batch * H + head) * D + channel, peak / 2.25)
    fp8 = scaled_e4m3(x, peak, V.dtype.element_ty == tl.bfloat16)
    if TILE_WIDTH:
        base = ((batch * H + head) * tl.cdiv(N, 64) + row // 64) * (64 * TILE_WIDTH)
        if TILE_WIDTH == 292:
            base += 6400
        swizzle = ((row[:, None] % 64 // 16) ^ (channel[None, :] // 2 % 4)) * 16
        tl.store(OUT + base[:, None] + channel[None, :] * 64 + swizzle + row[:, None] % 16, fp8,
                 row[:, None] < tl.cdiv(N, 64) * 64)
    else:
        tl.store(OUT + batch * OZ + head * OH + channel[:, None] * OD + row[None, :], tl.trans(fp8),
                 row[None, :] < tl.cdiv(N, 64) * 64)


@triton.jit(do_not_specialize=["N", "H"])
def _channel_peak(PARTIAL, PEAK, N, H, D: tl.constexpr, GROUPS: tl.constexpr, PEAK_ROWS: tl.constexpr):
    head, batch = tl.program_id(1), tl.program_id(2).to(tl.int64)
    channel = tl.program_id(0) * 16 + tl.arange(0, 16)
    group = tl.arange(0, GROUPS)
    count = tl.cdiv(N, PEAK_ROWS)
    peak = tl.max(tl.load(PARTIAL + ((batch * H + head) * count + group[:, None]) * D + channel[None, :],
                          group[:, None] < count, 0.), 0)
    tl.store(PEAK + (batch * H + head) * D + channel, peak)


def pack_value(v, nhd, *, partial=None, peak_rows=128, tiles=None):
    """Pack 16-bit V, optionally using maxima from its producer. Routed HND calls can supply their K/V tiles."""
    x = v if nhd else v.transpose(1, 2)
    batch, tokens, heads, dim = x.shape
    padded = triton.cdiv(tokens, 64) * 64
    groups = triton.cdiv(tokens, peak_rows)
    if partial is None:
        partial = torch.empty((batch, heads, groups, dim), dtype=torch.float32, device=x.device)
        _value_amax[(groups, heads, batch)](x, partial, tokens, heads, dim, *x.stride()[:3], peak_rows, num_warps=4)
    scale = torch.empty((batch, heads, dim), dtype=torch.float32, device=x.device)
    shape = (batch, dim, heads, padded) if nhd else (batch, heads, dim, padded)
    if tiles is None:
        out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        out = tiles.view(torch.float8_e4m3fn).as_strided(shape, (tiles.stride(0), tiles.stride(1), padded, 1))
    logical = out if nhd else out.transpose(1, 2)
    reduction = triton.next_power_of_2(groups)
    if groups >= 32:
        peak = torch.empty_like(scale)
        _channel_peak[(dim // 16, heads, batch)](partial, peak, tokens, heads, dim, reduction, peak_rows, num_warps=4)
        partial, reduction = peak, 0
    _pack_value[(padded // 32 * (dim // 64), heads, batch)](
        x, partial, scale, out, tokens, heads, dim, *x.stride()[:3], *logical.stride()[:3],
        32, 64, reduction, peak_rows, 0 if tiles is None else tiles.size(-1), num_warps=4)
    if tiles is not None:
        torch.autograd.graph.increment_version(tiles)
    return out, scale, None
