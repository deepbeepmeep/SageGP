# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Public attention entry points.

sageattn(..., precision="precise")   SageAttention 2 accuracy (INT8 scores, FP8 values). On GeForce Blackwell the SageGP kernel.
sageattn(..., precision="adaptive")  GeForce Blackwell: each head keeps the score and value widths its activations need within an
                                     error budget (part of the scores in NVFP4). Elsewhere: precise.
sageattn_fast(...)                   GeForce Blackwell: every score in NVFP4 (SageAttention 3 speed, a fraction of its error).
                                     Calls it cannot serve run adaptive, then precise.
sageattn3_blackwell(...)             sageattn_fast with SageAttention 3's signature (HND tensors).

The SM120 kernels serve forward, non-causal, unmasked BF16 attention with the default softmax scale on equal-length Q and K
(head width 128, or 256 for adaptive and fast); any other call runs SageAttention 2 (the sageattention package of this
distribution), so the functions accept every call sageattention.sageattn accepts.
"""
import torch

try:
    from .sm120 import dense as _dense, fast as _fast, routed as _routed, wide as _wide
    from .sm120.common import supported as _supported
except ImportError:  # a Triton without the features of the SM120 packers
    _dense = None

PRECISIONS = ("precise", "adaptive", "fast")


def _sm120_mode(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, mode):
    """The SM120 kernel that serves this call in this mode ("precise", "adaptive" or "fast"), else None."""
    if (_dense is None or is_causal or return_lse or tensor_layout not in ("HND", "NHD") or q.ndim != 4
            or not (q.shape == k.shape == v.shape) or not (q.dtype == k.dtype == v.dtype == torch.bfloat16)
            or not (q.device == k.device == v.device) or not _supported(q.device, q.dtype)):
        return None
    batch, heads, tokens, dim = q.shape if tensor_layout == "HND" else (q.size(0), q.size(2), q.size(1), q.size(3))
    if sm_scale is not None and sm_scale != dim ** -0.5:
        return None
    if dim == 256:
        return "wide" if mode != "precise" and _wide.eligible(1, tokens, tokens, heads, heads, dim, q.dtype) else None
    if mode == "fast" and _fast.eligible(q.device, q.dtype, 1, tokens, tokens, heads, heads, dim):
        return "fast"
    if mode != "precise" and _routed.eligible(q.device, q.dtype, 1, tokens, tokens, heads, heads, dim):
        return "adaptive"
    return "precise" if _dense.eligible(1, tokens, tokens, heads, heads, dim, q.dtype) else None


def _sm120_attention(q, k, v, tensor_layout, kernel):
    nhd = (lambda x: x) if tensor_layout == "NHD" else (lambda x: x.transpose(1, 2))
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    view = nhd(out)
    q, k, v = nhd(q), nhd(k), nhd(v)
    for i in range(q.size(0)):
        inputs = [q[i:i + 1], k[i:i + 1], v[i:i + 1]]
        if kernel == "wide":
            view[i:i + 1].copy_(_wide.attention_owned(inputs, recycle_q=False))
        elif kernel == "fast":
            _fast.attention_owned(inputs, recycle_q=False, out=view[i:i + 1])
        elif kernel == "adaptive":
            _routed.attention_owned(inputs, recycle_q=False, out=view[i:i + 1])
        else:
            _dense.attention_owned(inputs, recycle_q=False, out=view[i:i + 1], int8_scores=True)
    return out


def _attention(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, mode, **kwargs):
    kernel = _sm120_mode(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, mode)
    if kernel is not None:
        return _sm120_attention(q, k, v, tensor_layout, kernel)
    from sageattention import sageattn as _sage2
    return _sage2(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, **kwargs)


def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, precision="precise", **kwargs):
    """SageAttention 2's sageattn with a precision choice: "precise" (default) or "adaptive". q, k, v: (batch, heads, tokens, dim)
    for tensor_layout "HND", (batch, tokens, heads, dim) for "NHD"."""
    if precision not in ("precise", "adaptive"):
        raise ValueError(f"precision must be 'precise' or 'adaptive', not {precision!r} (sageattn_fast runs the fast mode)")
    return _attention(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, precision, **kwargs)


def sageattn_fast(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs):
    """Every score in NVFP4 on GeForce Blackwell; the calls it cannot serve run adaptive, then precise."""
    return _attention(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, "fast", **kwargs)


def sageattn3_blackwell(q, k, v, attn_mask=None, is_causal=False, per_block_mean=True, **kwargs):
    """Drop-in for SageAttention 3's sageattn3_blackwell: (batch, heads, tokens, dim) tensors."""
    if attn_mask is not None:
        raise ValueError("sageattn3_blackwell does not take a mask")
    return sageattn_fast(q, k, v, tensor_layout="HND", is_causal=is_causal)


def kernel_for(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, precision="precise"):
    """Which kernel a call would run: "fast", "adaptive", "wide", "precise" (the SageGP SM120 kernels) or "sageattention"."""
    return _sm120_mode(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, precision) or "sageattention"
