# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""GPU-only calibration of the SM120 dense Sage2 score and value widths."""

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


@lru_cache(None)
def _native(device):
    image = Path(__file__).with_name("cubins") / "routed" / "selector_sm120a.cubin"
    with torch.cuda.device(device):
        kernels = torch.cuda._utils._cuda_load_module(image.read_bytes(), ["qr128", "regress"])
        kernels["qr128"].set_shared_memory_config(128 * 129 * 4)
        kernels["regress"].set_shared_memory_config(96 * 129 * 4)
        return kernels


@triton.jit
def _ieee_product(A, B, C):
    # The cube precedes QR. TensorFloat-32 rounding here can amplify errors in low-energy directions.
    h = tl.program_id(2)
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    j = tl.program_id(1) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 32)
    result = tl.full((16, 32), 0, tl.float32)
    for group in tl.static_range(4):
        a = tl.load(A + (h * 128 + i[:, None]) * 128 + group * 32 + k[None, :])
        b = tl.load(B + (h * 128 + group * 32 + k[:, None]) * 128 + j[None, :])
        result = tl.dot(a, b, result, input_precision="ieee")
    tl.store(C + (h * 128 + i[:, None]) * 128 + j[None, :], result)


def _cov_product(a, b):
    if torch.get_float32_matmul_precision() == "highest":
        return torch.bmm(a, b)
    product = torch.empty_like(a)
    _ieee_product[(8, 4, a.size(0))](a, b, product)
    return product


def rank_basis(samples, rotation):
    mean = samples.float().mean(1, keepdim=True)
    centered = (samples.float() - mean).bfloat16()
    cov = torch.bmm(centered.mT.contiguous(), centered, out_dtype=torch.float32)
    bound = cov.abs().sum(-1).amax(-1).clamp_min(1e-10)[:, None, None]
    shifted = cov / bound
    shifted.diagonal(dim1=1, dim2=2).add_(1 / 128)
    # QR(S cubed) with the same initial column order gives the same subspaces
    # as three successive orthogonal iterations, using only one factorization.
    power = _cov_product(_cov_product(shifted, shifted), shifted)
    basis = torch.empty_like(cov)
    _native(samples.device)["qr128"](
        (samples.size(0), 1, 1),
        (512, 1, 1),
        [cov, power, basis],
        shared_mem=128 * 129 * 4,
        stream=torch.cuda.current_stream(samples.device),
    )
    variance = (basis * (cov @ basis)).sum(1)
    order = variance.argsort(descending=True)
    basis = basis.gather(2, order[:, None, :].expand_as(basis))
    basis = (basis.unflatten(-1, (4, 32)) @ rotation).flatten(-2).bfloat16().contiguous()
    return mean, basis, centered


@triton.jit
def _partial_attention(
    Q,
    K,
    A,
    B,
    FQ,
    FK,
    V,
    MASS,
    OUT,
    LSE,
    H: tl.constexpr,
    R: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    STEP: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWS: tl.constexpr,
):
    # The first three groups cover held-out queries; the reference group also fits the value decoder.
    row, split, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mode = tl.minimum(row // (R // ROWS), 3)
    i = row * ROWS + tl.arange(0, ROWS)
    qi = tl.where(mode == 3, i - 3 * R, i % R + T)
    d = tl.arange(0, 128)
    c = tl.arange(0, 32)
    w = tl.arange(0, 64)
    stride = head * (R + T) * 128
    q = tl.load(Q + stride + qi[:, None] * 128 + d[None, :])
    a0 = tl.load(A + stride + qi[:, None] * 128 + c[None, :])
    a1 = tl.load(A + stride + qi[:, None] * 128 + 32 + c[None, :])
    aq = tl.load(FQ + (head * (R + T) + qi[:, None]) * 64 + w[None, :])
    total = tl.full((ROWS, 128), 0, tl.float32)
    denom = tl.full((ROWS,), 0, tl.float32)
    peak = tl.full((ROWS,), -float("inf"), tl.float32)
    for part in tl.static_range(CHUNK // STEP):
        j = split * CHUNK + part * STEP + tl.arange(0, STEP)
        if mode == 3:
            s = tl.dot(
                q, tl.trans(tl.load(K + (head * N + j[:, None]) * 128 + d[None, :], j[:, None] < N, 0.0))
            )
        else:
            kp = B + (head * N + j[:, None]) * 128 + c[None, :]
            s = tl.dot(a0, tl.trans(tl.load(kp, j[:, None] < N, 0.0)))
            if mode >= 1:
                s = tl.dot(a1, tl.trans(tl.load(kp + 32, j[:, None] < N, 0.0)), s)
            if mode == 2:
                kk = tl.load(FK + (head * N + j[:, None]) * 64 + w[None, :], j[:, None] < N, 0.0)
                s = tl.dot(aq, tl.trans(kk), s)
        s = s * 0.08838834764831845 + tl.load(MASS + j, j < N, -float("inf"))[None, :]
        mx = tl.maximum(peak, tl.max(s, 1))
        p = tl.exp(s - mx[:, None])
        correction = tl.exp(peak - mx)
        denom = denom * correction + tl.sum(p, 1)
        v = tl.load(V + (head * N + j[:, None]) * 128 + d[None, :], j[:, None] < N, 0.0)
        total = tl.dot(p.to(tl.bfloat16), v, total * correction[:, None])
        peak = mx
    ids = (head * (4 * R + T) + i) * SPLITS + split
    tl.store(OUT + ids[:, None] * 128 + d[None, :], total / denom[:, None])
    tl.store(LSE + ids, tl.log(denom) + peak)


@triton.jit
def _merge_attention(OUT, LSE, Z, S: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    j = tl.arange(0, PAD)
    log_mass = tl.load(LSE + row * S + j, j < S, -float("inf"))
    p = tl.exp(log_mass - tl.max(log_mass, 0))
    vals = tl.load(OUT + ((row * S + j[:, None]) * 128 + d[None, :]), j[:, None] < S, 0.0)
    tl.store(Z + row * 128 + d, tl.sum(p[:, None] * vals, 0) / tl.sum(p, 0))


def _attention_outputs(q, k, a, b, fq, fk, v, mass, r, t):
    h, n, _ = k.shape
    chunk = 256
    splits = triton.cdiv(n, chunk)
    # Store 128 output channels per key split instead of writing every query/key probability to VRAM.
    out = q.new_empty((h, 4 * r + t, splits, 128), dtype=torch.float32)
    lse = q.new_empty((h, 4 * r + t, splits), dtype=torch.float32)
    z = q.new_empty((h, 4 * r + t, 128), dtype=torch.float32)
    _partial_attention[((4 * r + t) // 16, splits, h)](
        q, k, a, b, fq, fk, v, mass, out, lse, h, r, t, n, splits, 64, chunk, 16, num_warps=4
    )
    _merge_attention[(h * (4 * r + t),)](out, lse, z, splits, triton.next_power_of_2(splits))
    return z


@triton.jit
def _loss(Z, B, BIAS, E, H: tl.constexpr, R: tl.constexpr, T: tl.constexpr, FAST: tl.constexpr):
    h, score, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    value = group // (R // 32)
    r = group % (R // 32) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    base = Z + (h * (4 * R + T) + r[:, None]) * 128 + d[None, :]
    ref = tl.load(base + (3 * R + T) * 128)
    z = tl.load(base + tl.where(score == 3, 3 * R + T, score * R) * 128)
    if FAST:
        if value < 3:
            width = (value + 1) * 32
            coef = tl.load(
                B + ((h * 3 + value) * 128 + d[:, None]) * 128 + d[None, :],
                (d[:, None] < width) & (d[None, :] >= width),
                0.0,
            )
            bias = tl.load(BIAS + (h * 3 + value) * 128 + d, d >= width, 0.0)
            pred = tl.dot(z.to(tl.bfloat16), coef.to(tl.bfloat16)) + bias[None, :]
            z = tl.where(d[None, :] < width, z, pred)
    sq = (z - ref) * (z - ref)
    tl.store(E + ((4 * value + score) * H + h) * (R // 32) + group % (R // 32), tl.sum(tl.sum(sq, 1), 0))


@triton.jit
def _restore_basis(U, B, BIAS, VM, FLAGS, OUT, MEAN):
    tile_i, tile_j, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i = tile_i * 16 + tl.arange(0, 16)
    j = tile_j * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    choice = tl.load(FLAGS + h) // 4
    width = (choice + 1) * 32
    base = (h * 128 + i[:, None]) * 128
    result = tl.load(U + base + j[None, :])
    if choice < 3:
        if tile_j * 32 < width:
            tail = tl.load(U + base + d[None, :], d[None, :] >= width, 0.0)
            coef = tl.load(
                B + ((h * 3 + choice) * 128 + j[:, None]) * 128 + d[None, :], d[None, :] >= width, 0.0
            )
            correction = tl.dot(tail, tl.trans(coef.to(tl.bfloat16)))
            result += correction.to(tl.bfloat16)
    tl.store(OUT + base + j[None, :], result)
    if tile_j == 0:
        mean = tl.load(VM + h * 128 + i)
        if choice < 3:
            tail = tl.load(U + base + d[None, :], d[None, :] >= width, 0.0)
            bias = tl.load(BIAS + (h * 3 + choice) * 128 + d, d >= width, 0.0)
            mean += tl.sum(tail.to(tl.float32) * bias[None, :], 1)
        tl.store(MEAN + h * 128 + i, mean)


def select(
    q, keys, rotated_keys, values, basis, flags, budget, log_mass, value_basis, value_mean, *, fake, choose
):
    heads = keys.size(0)
    fast = value_basis is not None
    valid = q.size(1) // 3 if fast else q.size(1)
    train = 2 * valid if fast else 0
    if fast:
        # Interleave fit and validation positions throughout the sequence, then
        # group them for GEMM. Only validation queries need all four candidates.
        groups = q.unflatten(1, (-1, 3))
        q = torch.cat((groups[:, :, :2].reshape(heads, train, 128), groups[:, :, 2]), 1)
    rotated_q = q @ basis
    outputs = _attention_outputs(
        q, keys, rotated_q, rotated_keys, fake(rotated_q), fake(rotated_keys), values, log_mass, valid, train
    )
    energy = outputs[:, -valid:].square().sum((1, 2))
    weights = bias = None
    if fast:
        outputs = torch.bmm((outputs - value_mean).bfloat16(), value_basis, out_dtype=torch.float32)
        fit = outputs[:, 3 * valid : 3 * valid + train]
        mean = fit.mean(1, keepdim=True)
        centered = fit - mean
        cov = centered.mT @ centered / train
        weights = q.new_empty((heads, 3, 128, 128), dtype=torch.float32)
        bias = q.new_empty((heads, 3, 128), dtype=torch.float32)
        _native(q.device)["regress"](
            (heads, 3, 1),
            (256, 1, 1),
            [cov, mean, weights, bias],
            shared_mem=96 * 129 * 4,
            stream=torch.cuda.current_stream(q.device),
        )
    value_modes = 4 if fast else 1
    partial = q.new_empty((4 * value_modes, heads, valid // 32), dtype=torch.float32)
    _loss[(heads, 4, value_modes * (valid // 32))](
        outputs, weights, bias, partial, heads, valid, train, fast, num_warps=4
    )
    errors = partial.sum(-1)
    selection = (0.0, errors.size(0), budget) if isinstance(budget, torch.Tensor) else (budget, errors.size(0))
    choose[(1,)](errors, energy, flags, heads, triton.next_power_of_2(heads), *selection, num_warps=4)
    if fast:
        decoder = torch.empty_like(value_basis)
        output_mean = torch.empty_like(value_mean)
        _restore_basis[(8, 4, heads)](value_basis, weights, bias, value_mean, flags, decoder, output_mean)
        return decoder, output_mean
