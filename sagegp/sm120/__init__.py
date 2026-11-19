# Copyright 2026 DeepBeepMeep. Licensed under Apache-2.0 (see LICENSE).
"""Kernels for GeForce Blackwell (SM120: RTX 50xx, RTX PRO Blackwell). The CUDA kernels are prebuilt sm_120a cubins (see csrc/sm120
for their sources and build scripts) loaded through PyTorch; the packing kernels are Triton kernels.

dense:  INT8 scores and FP8 values with per-head NVFP4 score tails (the precise mode keeps all scores in INT8).
routed: Adaptive, the score and value widths of each head chosen on the current activations within an error budget.
fast:   every score in NVFP4 with error-feedback packing, FP8 values: SageAttention 3 speed at a much smaller error.
wide:   256-channel heads with a data-dependent score width.
"""
