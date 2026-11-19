# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Row reader of the packers: optional weighted RMSNorm and rotary embedding applied while the rows are read (the plain API reads
the rows as they are)."""
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rows(X, W, WROW, ROPE, MEAN, base, seq, ok, c, batch, head, RB, RS, RH, MB, MH, WH, EPS, RP: tl.constexpr, RC: tl.constexpr, D: tl.constexpr,
          PAIRS: tl.constexpr, NORM: tl.constexpr, ROTATE: tl.constexpr, HAS_MEAN: tl.constexpr, SUB: tl.constexpr, LANES: tl.constexpr,
          INTERLEAVED: tl.constexpr, SEGMENTED: tl.constexpr, ROPE_FIRST: tl.constexpr, FIXED_LANES: tl.constexpr = False):
    """The rows at base (one head; c spans D rounded up to a power of two, the channels beyond D are zeros): weighted RMSNorm in FP32
    rounded to the activation dtype (NORM; the weight row of each token from WROW with SEGMENTED, else that of the head at stride WH,
    0 for one row), the rotation of the first 2 * PAIRS channels in FP32 (ROTATE, cos / sin pairs of ROPE), minus the head's MEAN row
    (HAS_MEAN); with ROPE_FIRST the rotation comes first and the norm of the rotated rows stays in FP32; rows beyond SEQ are zeros. RH:
    the head stride of a table with one rotation per head, 0 for one shared by the heads."""
    inside = ok[:, None] & (c < D)[None, :]
    x = tl.load(X + base[:, None] + c[None, :], inside, 0).to(tl.float32)
    if SEGMENTED:
        wrow = tl.load(WROW + seq, ok, 0).to(tl.int64)[:, None] * D
    else:
        wrow = tl.zeros([SUB, 1], tl.int64) + head * WH
    if NORM and not ROPE_FIRST:
        if LANES and D == 128:  # lane sums of c, c + 32, c + 64, c + 96, then the lanes (shared.kernels.denoiser_triton._rms_rope)
            lanes = tl.reshape(tl.permute(tl.reshape(x, (SUB, 4, 32)), 0, 2, 1), (SUB, 32, 2, 2))
            even, odd = tl.split(lanes)
            x0, x2 = tl.split(even)
            x1, x3 = tl.split(odd)
            squares = tl.fma(x3, x3, tl.fma(x2, x2, tl.fma(x1, x1, x0 * x0)))
            if FIXED_LANES:
                # _rms_rope sums lanes at XOR distances 16, 8, 4, 2, 1. A CTA spanning a whole quantization
                # group otherwise reduces in a different order and can move a rotated value across an INT8 boundary.
                for stage in tl.static_range(5):
                    left, right = tl.split(tl.permute(tl.reshape(squares, (SUB, 2, 16 >> stage)), 0, 2, 1))
                    squares = left + right
                total = tl.reshape(squares, (SUB,))
            else:
                total = tl.sum(squares, 1)
        else:
            total = tl.sum(x * x, 1)
        inv = libdevice.rsqrt(total / D + EPS)[:, None]
        y = (x * inv * tl.load(W + wrow + c[None, :], (c < D)[None, :], 0).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    else:
        y = x
    if ROTATE:
        rotated = c < 2 * PAIRS
        if INTERLEAVED:  # channel 2p with 2p + 1
            leading = c % 2 == 0
            partner = tl.where(rotated, c ^ 1, c)
            pair = c // 2
        else:  # channel p with p + PAIRS
            leading = c < PAIRS
            partner = tl.where(leading, c + PAIRS, tl.where(rotated, c - PAIRS, c))
            pair = tl.where(leading, c, c - PAIRS)
        other = tl.load(X + base[:, None] + partner[None, :], inside, 0).to(tl.float32)
        if NORM and not ROPE_FIRST:
            other = (other * inv * tl.load(W + wrow + partner[None, :], (c < D)[None, :], 0).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        angle = batch * RB + seq[:, None] * RS + head * RH + pair[None, :] * RP
        mask = ok[:, None] & rotated[None, :]
        cos = tl.load(ROPE + angle, mask, 1.0).to(tl.float32)
        sin = tl.load(ROPE + angle + RC, mask, 0.0).to(tl.float32)
        first = leading[None, :]
        out = tl.fma(tl.where(first, cos, sin), tl.where(first, y, other), tl.where(first, -sin, cos) * tl.where(first, other, y))
        y = tl.where(rotated[None, :], out, y)
    if NORM and ROPE_FIRST:
        inv = libdevice.rsqrt(tl.sum(y * y, 1) / D + EPS)[:, None]
        y = y * inv * tl.load(W + wrow + c[None, :], (c < D)[None, :], 0).to(tl.float32)
    if HAS_MEAN:  # rows beyond SEQ stay zeros: they must not reach the scale of their row group
        y = tl.where(ok[:, None], y - tl.load(MEAN + batch * MB + head * MH + c, c < D, 0).to(tl.float32)[None, :], 0.0)
    return y
