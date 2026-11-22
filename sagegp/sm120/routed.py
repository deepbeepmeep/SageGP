# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Dense SM120 Sage2 attention with score and value widths selected on current activations.

Each head can use 32, 64 or 128 INT8 score channels, or 64 INT8 plus 64 scaled FP4.
Strict and Balanced keep all FP8 value channels. Fast also tests four value widths,
fits the discarded components on pilot queries and scores them on held-out queries.
The estimated error budget applies across the heads of the current launch.
"""
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .. import hooks
from . import dense, pilot
from .common import _pack_pairs, _unpack_pairs, supported
from .values import pack_value


# Packed bytes per row averaged over a 64-key tile: 128 INT8 codes plus an
# alternate 64-channel FP4 tail (32 bytes of codes and 4 bytes of scales).
_TILED_KEY_WIDTH = 164
_JOINT_WIDTH = _TILED_KEY_WIDTH + 128


def split_keys_eligible(tokens, settings):
    # A split projection with a norm over all heads also needs a Q statistics pass and one calibration per group.
    # Across an LTX step on SM120 these cost more than they save at 27,280 tokens. At 35,712 only Fast saved over 1%.
    return tokens > (32768 if settings["quality"] == "adaptive" else 36864)


def eligible(device, dtype, batch, queries, keys, heads, kv_heads, dim):
    # At these lengths even the compressed core saves less time than calibration
    # and packing cost on the tested SM120 workloads. Use the regular staged core.
    return (queries > 8192
            and dense.eligible(batch, queries, keys, heads, kv_heads, dim, dtype)
            and triton.cdiv(keys, 64) * 64 * kv_heads * 176 < 2**31 and supported(device, dtype))


def settings(recycle_query, preview=None):
    """preview: (query rows kept at full width, error budget) of an experimental schedule, None for the standard budget.
    Each split head group gets a shallow copy. Positions depend only on the sequence, whereas
    bases, pilot activations and routing decisions belong to the individual group."""
    state = {"quality": "adaptive", "recycle_query": recycle_query, "recycle_value": False, "sample_positions": {}}
    if preview is not None:
        state["full_query_rows"], state["step_budget"] = preview
    return state


def rows(n, count, device, salt):
    i = torch.arange(count, device=device, dtype=torch.int64)
    start = i * n // count
    span = ((i + 1) * n // count - start).clamp_min(1)
    noise = ((i + salt) * 1664525 + 1013904223) & 0xFFFFFFFF
    return (start + noise % span).clamp_max(n - 1)


@triton.jit(do_not_specialize=["N", "ST", "SH"])
def _sample(
    X,
    W,
    ROPE,
    IDS,
    Y,
    N,
    ST,
    SH,
    RT,
    RH,
    RP: tl.constexpr,
    RC: tl.constexpr,
    WH,
    EPS: tl.constexpr,
    PAIRS: tl.constexpr,
    INTER: tl.constexpr,
    FIRST: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
):
    h = tl.program_id(1)
    j = tl.program_id(0) * R + tl.arange(0, R)
    idx = tl.load(IDS + j)
    x = dense._read(X, W, ROPE, idx, h, N, ST, SH, RT, RH, RP, RC, WH, EPS, PAIRS, INTER, FIRST, R)
    tl.store(Y + (h * S + j[:, None]) * 128 + tl.arange(0, 128)[None, :], x)


@triton.jit
def _tail(x, R: tl.constexpr):
    g = tl.reshape(x, (R, 4, 16))
    sf = tl.minimum(tl.max(tl.abs(g), 2) * (1.0 / 6), 448.0).to(tl.float8e4nv).to(tl.float32)
    c = _pack_pairs(tl.reshape(g / tl.where(sf > 0, sf, 1.0)[:, :, None], (R, 64)), R, 64)
    levels = tl.reshape(_unpack_pairs(c, R, 64), (R, 4, 16))
    den = tl.sum(levels * levels, 2)
    fitted = (
        tl.minimum(tl.sum(levels * g, 2) / tl.maximum(den, 1.0e-20), 448.0).to(tl.float8e4nv).to(tl.float32)
    )
    f = _pack_pairs(tl.reshape(g / tl.where(fitted > 0, fitted, 1.0)[:, :, None], (R, 64)), R, 64)
    return f, fitted


@triton.jit(do_not_specialize=["N", "PAD", "ST", "SH", "OST", "OSH"])
def _pack(
    X,
    W,
    ROPE,
    BASIS,
    MEAN,
    FLAGS,
    Y,
    SF,
    N,
    ST,
    SH,
    RT,
    RH,
    RP: tl.constexpr,
    RC: tl.constexpr,
    WH,
    EPS: tl.constexpr,
    PAIRS: tl.constexpr,
    INTER: tl.constexpr,
    FIRST: tl.constexpr,
    PAD,
    OST,
    OSH,
    KEY: tl.constexpr,
    R: tl.constexpr,
    PLANES: tl.constexpr = False,
    TILE_WIDTH: tl.constexpr = 0,
    FULL_QUERY_ROWS=0,
):
    tile, h = tl.program_id(0), tl.program_id(1)
    row = tile * R + tl.arange(0, R)
    d = tl.arange(0, 128)
    x = dense._read(X, W, ROPE, row, h, N, ST, SH, RT, RH, RP, RC, WH, EPS, PAIRS, INTER, FIRST, R)
    if KEY:
        x = tl.where(row[:, None] < N, x - tl.load(MEAN + h * 128 + d)[None, :], 0.0)
    b = tl.load(BASIS + h * 16384 + d[:, None] * 128 + d[None, :])
    x = tl.dot(x.to(tl.bfloat16), b, out_dtype=tl.float32)
    peaks = tl.max(tl.abs(x), 1)
    scale = tl.maximum(tl.max(peaks, 0) * (1.0 / 127), 1.0e-20)
    tl.store(SF + h * PAD + tile, scale)
    normed = x / scale
    if KEY and PLANES:
        base = Y + (h.to(tl.int64) * tl.cdiv(N, 64) + tile) * (TILE_WIDTH * 64 if TILE_WIDTH else 11264)
        o = base + tl.arange(0, R)[:, None] * 32
    elif KEY:
        o = Y + (h.to(tl.int64) * N + row[:, None]) * 176
    else:
        tl.debug_barrier()
        o = Y + row[:, None].to(tl.int64) * OST + h * OSH
    u = libdevice.rint(normed).to(tl.int8)
    if KEY:
        if TILE_WIDTH == 128:
            mix = tl.load(FLAGS + h) % 4 == 2
        else:
            mix = True
        if TILE_WIDTH:
            r = tl.arange(0, R)[:, None]
            c = d[None, :]
            addr = (c % 64 // 32) * 2048 + r // 8 * 256 + c % 32 // 16 * 128 + r % 8 * 16 + c % 16
            if TILE_WIDTH == 128:
                tl.store(base + tl.where(c < 64, tl.where(mix, 2304, 4096), 0) + addr, u, (c < 64) | (~mix))
            else:
                tl.store(base + tl.where(c < 64, 2304, TILE_WIDTH * 64 - 4096) + addr, u)
        elif PLANES:
            tl.store(o + (d // 32 * 2048 + d % 32)[None, :], u)
        else:
            tl.store(o + d[None, :], u, row[:, None] < N)
    else:
        mix = (tl.load(FLAGS + h) % 4 == 2) & (tile * R >= FULL_QUERY_ROWS)
        tl.store(o + d[None, :], u, (row[:, None] < N) & ((d[None, :] < 64) | (~mix)))
    if mix:
        ix = 64 + tl.arange(0, 64)
        weak = tl.gather(normed, tl.broadcast_to(ix[None, :], (R, 64)), 1)
        codes, sub = _tail(weak, R)
        if KEY and TILE_WIDTH:
            r = tl.arange(0, R)[:, None]
            byte = tl.arange(0, 32)[None, :]
            tl.store(base + r // 8 * 256 + byte // 16 * 128 + r % 8 * 16 + byte % 16, codes.to(tl.int8, bitcast=True))
            tl.store(base + 2048 + r * 4 + tl.arange(0, 4)[None, :], sub.to(tl.float8e4nv).to(tl.int8, bitcast=True))
        elif KEY and PLANES:
            tl.store(o + 8192 + tl.arange(0, 32)[None, :], codes.to(tl.int8, bitcast=True))
            tl.store(
                base + 10240 + tl.arange(0, R)[:, None] * 16 + tl.arange(0, 4)[None, :],
                sub.to(tl.float8e4nv).to(tl.int8, bitcast=True),
            )
        else:
            offset: tl.constexpr = 128 if KEY else 64
            tl.store(
                o + offset + tl.arange(0, 32)[None, :], codes.to(tl.int8, bitcast=True), row[:, None] < N
            )
            tl.store(
                o + offset + 32 + tl.arange(0, 4)[None, :],
                sub.to(tl.float8e4nv).to(tl.int8, bitcast=True),
                row[:, None] < N,
            )


@triton.jit
def _fake_tail(X, Y, N: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    i = (tl.program_id(1) * N + r[:, None]) * 128 + 64 + tl.arange(0, 64)[None, :]
    x = tl.load(X + i).to(tl.float32)
    codes, sf = _tail(x, R)
    decoded = tl.reshape(_unpack_pairs(codes, R, 64), (R, 4, 16)) * sf[:, :, None]
    tl.store(
        Y + (tl.program_id(1) * N + r[:, None]) * 64 + tl.arange(0, 64)[None, :], tl.reshape(decoded, (R, 64))
    )


@triton.jit(do_not_specialize=["H"])
def _choose(
    ERR, ENERGY, FLAGS, H, HP: tl.constexpr, BUDGET: tl.constexpr, MODES: tl.constexpr = 4, STEP_BUDGET=None
):
    h = tl.arange(0, HP)
    energy = tl.sum(tl.load(ENERGY + h, h < H, 0.0), 0)
    raw = tl.load(ERR + tl.arange(0, MODES)[:, None] * H + h[None, :], h[None, :] < H, 0.0)
    risk = raw * (H / tl.maximum(energy, 1.0e-20))
    mode = tl.arange(0, MODES)[:, None]
    score = mode % 4
    cost = tl.where(score == 0, 0.0, tl.where(score == 1, 0.3, tl.where(score == 2, 0.65, 1.0))) + 0.35 * (
        mode // 4
    )
    lo, hi = 0.0, 1048576.0
    budget = BUDGET if STEP_BUDGET is None else tl.load(STEP_BUDGET)
    for _ in tl.static_range(22):
        price = (lo + hi) * 0.5
        best = tl.argmin(cost + price * risk, 0)
        total = tl.sum(tl.sum(tl.where(mode == best[None, :], risk, 0.0), 0), 0)
        over = total > H * budget * budget
        lo, hi = tl.where(over, price, lo), tl.where(over, hi, price)
    chosen = tl.argmin(cost + hi * risk, 0)
    tl.store(FLAGS + h, chosen, h < H)


@lru_cache(None)
def rot(device):
    x = torch.ones(1, 1, device=device, dtype=torch.float32)
    for _ in range(5):
        x = torch.cat((torch.cat((x, x), 1), torch.cat((x, -x), 1)), 0)
    return hooks.register_constant(x * (32**-0.5))


def sample(x, ids, rope, norm, offsets):
    _, n, h, d = x.shape
    y = x.new_empty((h, ids.numel(), d))
    _sample[(ids.numel() // 16, h)](
        x, norm[0], rope, ids, y, n, *x.stride()[1:3], *offsets, ids.numel(), 16, num_warps=4
    )
    return y


def rank_basis(k):
    return pilot.rank_basis(k, rot(k.device))


def fake(x):
    h, n, _ = x.shape
    y = x.new_empty((h, n, 64))
    _fake_tail[(n // 16, h)](x.contiguous(), y, n, 16, num_warps=4)
    return y


def select(q, kc, ks, v, basis, flags, budget, weights, vbasis, vmean):
    return pilot.select(q, kc, ks, v, basis, flags, budget, weights, vbasis, vmean, fake=fake, choose=_choose)


@triton.jit(do_not_specialize=["N", "ST", "SH", "OT", "OH"])
def _values(X, BASIS, MU, Y, N, ST, SH, OT, OH, UNDO: tl.constexpr, RECYCLE: tl.constexpr = False,
            PARTIAL=None, ROWS: tl.constexpr = 32):
    tile, h = tl.program_id(0), tl.program_id(1)
    r = tile * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)
    x = tl.load(X + r[:, None].to(tl.int64) * ST + h * SH + d[None, :], r[:, None] < N, 0.0)
    m = tl.load(MU + h * 128 + d)
    b = tl.load(BASIS + h * 16384 + d[:, None] * 128 + d[None, :])
    if UNDO:
        y = tl.dot(x, tl.trans(b), out_dtype=tl.float32) + m[None, :]
    else:
        y = tl.dot((x.to(tl.float32) - m[None, :]).to(tl.bfloat16), b, out_dtype=tl.float32)
    if PARTIAL is not None:
        rounded = y.to(Y.dtype.element_ty).to(tl.float32)
        peak = tl.max(tl.where(r[:, None] < N, tl.abs(rounded), 0.), 0)
        tl.store(PARTIAL + (h * tl.cdiv(N, ROWS) + tile) * 128 + d, peak)
    if UNDO or RECYCLE:
        tl.debug_barrier()
    tl.store(Y + r[:, None].to(tl.int64) * OT + h * OH + d[None, :], y, r[:, None] < N)


# IDs are shared between the K and V stages: callers can release either projection first.
def _pilot_ids(x, settings):
    positions = settings.setdefault("sample_positions", {})
    if "keys" not in positions:
        n = x.size(1)
        edge = torch.arange(256, device=x.device)
        middle = rows(n - 512, 2048, x.device, 67) + 256
        ids = torch.cat((edge, middle, n - 256 + edge))
        mass = torch.ones(ids.numel(), device=x.device, dtype=torch.float32)
        mass[256:-256] = (n - 512) / middle.numel()
        positions["keys"] = ids, mass.log()
    ids, settings["pilot_mass"] = positions["keys"]
    return ids


def _joint_storage(x, settings, stage, n, h):
    # K and V can arrive in either order. Each split head group owns its record.
    # A later query needs both forms of K. An early selection needs only its chosen form.
    if not (settings["quality"] == "adaptive" and 49152 <= n <= 131072 and h >= 4):
        return None
    shape = (1, h, triton.cdiv(n, 64) * 64)
    pending = settings.pop("_pending_tiles", None)
    if pending is not None and pending[0] == shape:
        partner = pending[1]
        if stage == "value" and partner is not None:
            return partner
        if stage == "key" and partner is None:
            width = 128 if settings.get("key_selected", False) else _TILED_KEY_WIDTH
            return x.new_empty((*shape, width), dtype=torch.int8)
    # V-first: reserving K here overlaps the BF16 K projection and its pilot.
    # Two TMA requests can fill the same on-chip slot from separate allocations.
    key_first = stage == "key"
    tiles = x.new_empty((*shape, _JOINT_WIDTH if key_first else 128), dtype=torch.int8)
    settings["_pending_tiles"] = shape, tiles if key_first else None
    return tiles


def _sample_rows(n, device, settings, training=False):
    positions = settings.setdefault("sample_positions", {})
    name = "values" if training else "queries"
    if name not in positions:
        count = 1536 if training else 384 if settings["quality"] == "adaptive" else 128
        ids = rows(n, count, device, 953 if training else 187)
        positions[name] = ids[::2].contiguous() if not training and n <= 32768 else ids
    return positions[name]


def early_query_ids(settings, n, heads, device):
    """Only a V-first tiled call with a single score mode per head can discard the alternate key."""
    pending = settings.get("_pending_tiles")
    if (pending is None or pending[1] is not None or pending[0] != (1, heads, triton.cdiv(n, 64) * 64)
            or settings.get("full_query_rows", 0) or "step_budget" in settings):
        return None
    return _sample_rows(n, device, settings)


def sample_projected_query(q, ids, rope, norm, *, interleaved=False, rope_first=False):
    if rope is not None and rope.size(1) != 1:
        rope = rope.index_select(1, ids)
    elif rope is not None:
        rope = rope.expand(-1, ids.numel(), *rope.shape[2:])
    geometry = dense._geometry(rope, norm, interleaved, rope_first)
    return sample(q, torch.arange(ids.numel(), device=q.device), rope, norm, geometry)


@torch.autocast(device_type="cuda", enabled=False)
def staged_quantize_v(values, settings):
    x = values.pop()
    if x.stride(-1) != 1:
        x = x.contiguous()
    settings["pilot_v"] = x[0].index_select(0, _pilot_ids(x, settings)).transpose(0, 1).contiguous()
    if settings["quality"] != "adaptive":
        return dense.staged_quantize_v([x], settings)

    n = x.size(1)
    training = x[0].index_select(0, _sample_rows(n, x.device, settings, training=True)).transpose(0, 1).contiguous()
    mean, basis, centered = rank_basis(training)
    del training, centered
    settings.update(v_basis=basis, v_mean=mean)
    recycle = settings.get("recycle_value", False) and dense._recyclable_query(x)
    transformed = x if recycle else torch.empty_like(x)
    fused_peak = n <= 262144
    tile = 64 if fused_peak else 32
    groups = triton.cdiv(n, tile)
    peak = x.new_empty((1, x.size(2), groups, 128), dtype=torch.float32) if fused_peak else None
    _values[(groups, x.size(2))](
        x, basis, mean, transformed, n, *x.stride()[1:3], *transformed.stride()[1:3], False, recycle, peak, tile,
        num_warps=4,
    )
    if recycle:
        torch.autograd.graph.increment_version(x)
    value_list = [transformed]
    del x, transformed
    if fused_peak:
        hooks.notice("Sage2 FP8 Value Packing")
        transformed = value_list.pop()
        tiles = _joint_storage(transformed, settings, "value", n, transformed.size(2))
        value, scale, _ = pack_value(transformed.transpose(1, 2), False, partial=peak, peak_rows=tile, tiles=tiles)
    else:
        value, scale, _ = dense.staged_quantize_v(value_list, settings)
    # The grouped CUDA epilogue accepts one scale and bias per value channel.
    return value, torch.cat((scale, torch.zeros_like(scale)), 0), None


_BUDGETS = {"strict": 0.002, "balanced": 0.009, "adaptive": 0.032}


def _select_query(queries, basis, flags, settings):
    decoder = select(
        queries, settings.pop("pilot_k"), settings.pop("pilot_ks"), settings.pop("pilot_v"), basis, flags,
        settings.get("step_budget", _BUDGETS[settings["quality"]]), settings.pop("pilot_mass", None),
        settings.get("v_basis"), settings.get("v_mean"),
    )
    if settings["quality"] == "adaptive":
        settings["v_basis"], settings["v_mean"] = decoder


# The pilot uses explicit BF16 projections and FP32 factorizations under every model autocast policy.
@torch.autocast(device_type="cuda", enabled=False)
def _quantize(x, settings, key, rope=None, norm=(None, 0.0, None), interleaved=False, rope_first=False, fitted_key=None):
    if x.stride(-1) != 1:
        x = x.contiguous()
    _, n, heads, _ = x.shape
    quality = settings["quality"]
    fast = quality == "adaptive"
    geometry = dense._geometry(rope, norm, interleaved, rope_first)
    recycle = False
    tile_width = 0
    full_rows = 0
    if key:
        if fitted_key is None:
            keys = sample(x, _pilot_ids(x, settings), rope, norm, geometry)
            mean, basis, centered = rank_basis(keys)
            del keys
            rotated = centered @ basis
        else:
            mean, basis, centered, rotated = fitted_key
        settings.update(basis=basis, pilot_k=centered, pilot_ks=rotated)
        flags = settings["mixed"] = torch.empty(heads, dtype=torch.uint8, device=x.device)
        early = settings.pop("early_q", None)
        if early is not None:
            _select_query(early, basis, flags, settings)
            del early, centered, rotated
            settings["key_selected"] = True
        capacity = triton.cdiv(n, 64) * 64 if fast else n
        tiles = _joint_storage(x, settings, "key", n, heads) if fast else None
        tile_width = tiles.size(-1) if tiles is not None else 0
        out = tiles[:, :, :n] if tile_width else x.new_empty((1, heads, capacity, 176), dtype=torch.int8)[:, :, :n]
    else:
        basis = settings.pop("basis")
        flags = settings["mixed"]
        # Long calls can amortize the full pilot. On shorter calls sample the
        # same stratified positions as every other row of that pilot.
        full_rows = min(n, settings.get("full_query_rows", 0))
        if full_rows:
            settings["full_query_rows"] = full_rows
            settings["full_v_basis"] = settings["v_basis"], settings["v_mean"]
            settings["full_query_modes"] = torch.full_like(flags, 15)
        if not settings.pop("key_selected", False):
            queries = sample(x, _sample_rows(n, x.device, settings), rope, norm, geometry)
            _select_query(queries, basis, flags, settings)
        mean = None
        recycle = settings["recycle_query"] and dense._recyclable_query(x)
        packed = x.view(torch.int8)[..., :128] if recycle else x.new_empty(x.shape, dtype=torch.int8)
        out = packed.transpose(1, 2)
    block = 64 if key else 32
    pad = triton.cdiv(n, 64) if key else triton.cdiv(n, 128) * 4
    scales = torch.empty(heads, pad, device=x.device, dtype=torch.float32)
    _pack[(triton.cdiv(n, block), heads)](
        x, norm[0], rope, basis, mean, flags, out, scales, n, *x.stride()[1:3], *geometry,
        pad, out.stride(2), out.stride(1), key, block, fast, tile_width, full_rows, num_warps=4,
    )
    if tile_width:
        torch.autograd.graph.increment_version(out)
    if recycle:
        torch.autograd.graph.increment_version(x)
    return out, scales


def staged_quantize_k(keys, settings):
    return _quantize(keys.pop(), settings, True)


@torch.autocast(device_type="cuda", enabled=False)
def pack_split_keys(keys, settings):
    """Fit the retained heads together, then pack and release each group's BF16 key in order."""
    if not keys:
        return []
    shared = settings.copy()
    ids = _pilot_ids(keys[0], shared)
    widths = [key.size(2) for key in keys]
    samples = torch.cat([key[0].index_select(0, ids).transpose(0, 1) for key in keys])
    means, bases, centered = rank_basis(samples)
    del samples
    rotated = centered @ bases
    groups = []
    keys.reverse()
    for fit in zip(means.split(widths), bases.split(widths), centered.split(widths), rotated.split(widths)):
        state = shared.copy()
        groups.append((_quantize(keys.pop(), state, True, fitted_key=fit), state))
    return groups


def staged_quantize_q(q, settings):
    return _quantize(q, settings, False)


def fused_warp_query(x, rope, norm, settings, *, interleaved, rope_first):
    return dense._fuse(x, rope, norm, settings, False, interleaved, rope_first, _quantize)


def fused_warp_key(x, rope, norm, settings, *, interleaved, rope_first):
    return dense._fuse(x, rope, norm, settings, True, interleaved, rope_first, _quantize)


def _restore(x, settings):
    def restore_rows(rows, basis, mean):
        _values[(triton.cdiv(rows.size(1), 32), rows.size(2))](
            rows, basis, mean, rows, rows.size(1), *rows.stride()[1:3], *rows.stride()[1:3], True, num_warps=4,
        )

    full_rows = settings.get("full_query_rows", 0)
    if full_rows:
        restore_rows(x[:, :full_rows], *settings.pop("full_v_basis"))
    basis, mean = settings.pop("v_basis"), settings.pop("v_mean")
    if full_rows < x.size(1):
        restore_rows(x[:, full_rows:] if full_rows else x, basis, mean)
    torch.autograd.graph.increment_version(x)


@lru_cache(None)
@lru_cache(None)
def _kernels(device, fast, joint=False, split=False, single=False, keys=False):
    """The routed variants' kernels; keys: their images over ranges of the keys (dense.SPLIT_KEYS: the last wave's query blocks)."""
    directory = Path(__file__).with_name("cubins") / "routed"
    variants = ("value32", "value64", "value96", "value128") if fast else ("score0", "score1", "score2", "score3")
    if joint or split or single:
        prefix = "single_" if single else "split_" if split else "joint_"
        variants = tuple(prefix + name for name in variants)
    symbol = "wangp_dense_routed_split" if keys else "wangp_dense_nvfp4_strided_q"
    suffix = "_keys" if keys else ""
    with torch.cuda.device(device):
        kernels = tuple(torch.cuda._utils._cuda_load_module((directory / f"{v}{suffix}_sm120a.cubin").read_bytes(), [symbol])[symbol]
                        for v in variants)
        if joint or split or single:
            for kernel in kernels:
                kernel.set_shared_memory_config(49152)
        return kernels


def staged_attention(out, packed, settings, mask=None):
    fast = settings["quality"] == "adaptive"
    width = packed[1][0].size(-1)
    joint, split = fast and width == _JOINT_WIDTH, fast and width == _TILED_KEY_WIDTH
    single = fast and width == 128
    return dense._staged_attention(out, packed, settings, mask, _kernels(out.device, fast, joint, split, single),
                                   _restore if fast else None, 49152 if joint or split or single else 32768,
                                   split_kernels=_kernels(out.device, fast, joint, split, single, True) if dense.SPLIT_KEYS else ())


def attention_owned(inputs, *, recycle_q, out=None):
    q, keys, values = inputs[0], inputs[1:2], inputs[2:3]
    inputs.clear()
    recycle = recycle_q and dense._recyclable_query(q)
    state = settings(recycle)
    values = staged_quantize_v(values, state)
    ids = early_query_ids(state, q.size(1), q.size(2), q.device)
    if ids is not None and q.stride(-1) == 1:
        norm = (None, 0., None)
        state["early_q"] = sample(q, ids, None, norm, dense._geometry(None, norm, False, False))
    keys = staged_quantize_k(keys, state)
    packed_q = staged_quantize_q(q, state)
    if out is None:
        out = q if recycle else torch.empty_like(q, memory_format=torch.contiguous_format)
    return staged_attention(out, [packed_q, keys, values], state)
