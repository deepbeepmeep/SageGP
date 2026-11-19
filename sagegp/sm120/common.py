# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Helpers shared by the SM120 kernels: E2M1 pair conversion, device support and views whose rows a kernel can overwrite."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_pairs(x, ROWS: tl.constexpr, DIM: tl.constexpr):
    even, odd = tl.split(tl.reshape(x, (ROWS, DIM // 2, 2)))
    return tl.inline_asm_elementwise(
        "{ .reg .b8 c; cvt.rn.satfinite.e2m1x2.f32 c, $2, $1; mov.b32 $0, {c, 0, 0, 0}; }",
        constraints="=r,f,f", args=[even, odd], dtype=tl.uint32, is_pure=True, pack=1).to(tl.uint8)


@triton.jit
def _unpack_pairs(code, ROWS: tl.constexpr, DIM: tl.constexpr):
    even, odd = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; .reg .b8 c,x,y,z; .reg .b16 a,b; mov.b32 {c,x,y,z}, $2;"
        " cvt.rn.f16x2.e2m1x2 pair,c; mov.b32 {a,b},pair; cvt.f32.f16 $0,a; cvt.f32.f16 $1,b; }",
        constraints="=f,=f,r", args=[code.to(tl.uint32)], dtype=(tl.float32, tl.float32), is_pure=True, pack=1)
    return tl.reshape(tl.join(even, odd), (ROWS, DIM))


@lru_cache(None)
def _capability(index):
    return torch.cuda.get_device_capability(index)


def supported(device, dtype):
    """GeForce Blackwell (SM120) inference: no autograd, no compilation, a Triton with block-scaled dots."""
    return (device.type == "cuda" and torch.version.hip is None and dtype in (torch.float16, torch.bfloat16)
            and _capability(device.index if device.index is not None else torch.cuda.current_device()) == (12, 0)
            and hasattr(tl, "dot_scaled") and not torch.is_grad_enabled() and not torch.compiler.is_compiling())


def recyclable_query(x):
    """A vector-aligned NHD or head-major view with disjoint rows and 32-bit byte offsets."""
    b, n, h, d = x.shape
    sb, st, sh, sc = x.stride()
    span = (n - 1) * st + (h - 1) * sh + d
    return (x.dtype in (torch.bfloat16, torch.float16) and sc == 1 and x.data_ptr() % 16 == 0
            and st % 8 == 0 and sh % 8 == 0 and (b == 1 or (sb % 8 == 0 and sb >= span))
            and 2 * max(sb, st, sh, (b - 1) * sb + span) < 2**31
            and ((sh >= d and st >= (h - 1) * sh + d) or (st >= d and sh >= (n - 1) * st + d)))
