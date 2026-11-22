# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0.
"""The last wave of an attention launch split into ranges of the keys, for every GPU family (sagegp.sm89.split, sagegp.triton_split):
the plan (which query blocks run whole, how many ranges the others take) and the merge of the ranges' normalized outputs by their
log2-sum-exp, in FP32 with one rounding, in range order (deterministic)."""
import ctypes
from functools import lru_cache

import torch
import triton
import triton.language as tl

SPLIT_RANGES = 8  # ranges of the keys of the last wave's query blocks (RTX 5090, 220k tokens: 8 ranges best or tied for 1 to 4 heads)
SPLIT_MIN_KEY_BLOCKS = 16  # 64-key blocks per range at least
TAIL_FILL_MAX = 0.5  # the last wave is split when it holds at most this share of the slots (sm89 kernel, 4 heads x 50k: 60 % full, +5 % split)


def occupancy_of(function, threads, shared_mem):
    """CTAs of a kernel one SM (compute unit) runs at once: the driver's occupancy query, CUDA's or HIP's (ROCm). function: the CUfunction
    or hipFunction_t (an int or ctypes pointer)."""
    count = ctypes.c_int()
    library = torch.cuda._utils._get_gpu_runtime_library()
    query = library.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor if torch.version.hip else library.cuOccupancyMaxActiveBlocksPerMultiprocessor
    query(ctypes.byref(count), ctypes.c_void_p(function) if isinstance(function, int) else function, ctypes.c_int(threads), ctypes.c_size_t(shared_mem))
    return max(1, count.value)


def occupancy(kernel, threads, shared_mem):
    """occupancy_of a torch.cuda._utils._CudaKernel."""
    return occupancy_of(kernel.func, threads, shared_mem)


def triton_slots(device, compiled, num_warps):
    """CTAs of a compiled Triton kernel the GPU runs at once (SMs x its occupancy; a warp of the device's width: 64 on CDNA)."""
    threads = num_warps * getattr(compiled.metadata, "warp_size", 32)
    return torch.cuda.get_device_properties(device).multi_processor_count * occupancy_of(compiled.function, threads, compiled.metadata.shared)


@lru_cache(None)
def tail_plan(slots, blocks, programs_per_block, key_blocks):
    """(query blocks run whole, ranges of the keys of the blocks after them): the blocks of the last, part-filled wave of `slots` CTAs
    (SMs x CTAs per SM), each query block being programs_per_block CTAs (heads x batch); (blocks, 1) when the last wave is full or the
    keys too few to split, or the last wave more than TAIL_FILL_MAX full."""
    main = blocks * programs_per_block // slots * slots // programs_per_block
    splits = min(SPLIT_RANGES, key_blocks // SPLIT_MIN_KEY_BLOCKS)
    fill = (blocks - main) * programs_per_block / slots
    return (main, splits) if main < blocks and splits > 1 and fill <= TAIL_FILL_MAX else (blocks, 1)


@triton.jit
def _bf16_bits(x):
    """The bits of x (FP32) rounded to BF16, to nearest even (NaN kept quiet): for GPUs whose Triton has no BF16 conversion (Turing)."""
    bits = x.to(tl.uint32, bitcast=True)
    rounded = tl.where(x != x, 0x7FC0, (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    return rounded.to(tl.uint16).to(tl.int16, bitcast=True)


_MERGE_INTS = ["R", "H", "FIRST", "OSB", "OST", "OSH"]


@triton.jit(do_not_specialize=_MERGE_INTS, do_not_specialize_on_alignment=_MERGE_INTS)
def _merge(P, L, O, R, H, FIRST, OSB, OST, OSH, S: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BITS: tl.constexpr):
    """O[b, FIRST + r, h] = sum_s 2^(L[b, s, h, r] - m) P[b, s, r, h] / sum_s 2^(L[b, s, h, r] - m), m = max_s L[b, s, h, r], in FP32: P
    (batch * S, R, H, D) the ranges' normalized outputs (any float dtype), L (batch * S, H, R) their log2-sum-exp. BITS: O is the int16
    view of a BF16 output."""
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    inside = r < R
    r64 = r.to(tl.int64)
    d = tl.arange(0, D)
    m = tl.full((BT,), float("-inf"), tl.float32)
    for s in tl.static_range(S):
        m = tl.maximum(m, tl.load(L + ((b * S + s) * H + h) * R + r64, inside, float("-inf")))
    acc = tl.zeros((BT, D), tl.float32)
    den = tl.zeros((BT,), tl.float32)
    for s in tl.static_range(S):
        w = tl.where(inside, tl.exp2(tl.load(L + ((b * S + s) * H + h) * R + r64, inside, float("-inf")) - m), 0.0)
        acc += w[:, None] * tl.load(P + (((b * S + s) * R + r64[:, None]) * H + h) * D + d[None, :], inside[:, None], 0).to(tl.float32)
        den += w
    out = tl.div_rn(acc, tl.where(inside, den, 1.0)[:, None])
    target = O + b * OSB + (FIRST + r64[:, None]) * OST + h * OSH + d[None, :]
    if BITS:
        tl.store(target, _bf16_bits(out), inside[:, None])
    else:
        tl.store(target, out.to(O.dtype.element_ty), inside[:, None])


def merge(parts, lse, out, first_row, bf16_bits=False):
    """out (batch, tokens, heads, dim; unit dim stride) rows from first_row = the merge of parts (batch * splits, rows, heads, dim) by
    lse (batch * splits, heads, rows). bf16_bits: out is the int16 view of a BF16 tensor (written as bits)."""
    batch, rows, heads, dim = out.shape[0], parts.shape[1], parts.shape[2], parts.shape[3]
    _merge[(triton.cdiv(rows, 64), heads, batch)](parts, lse, out, rows, heads, first_row, out.stride(0), out.stride(1), out.stride(2),
                                                 parts.shape[0] // batch, dim, 64, bf16_bits, num_warps=4)
    torch.autograd.graph.increment_version(out)
