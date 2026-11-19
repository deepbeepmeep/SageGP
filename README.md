# SageGP

**SageAttention for the GPU Poor.** SageGP is SageAttention 2 for every NVIDIA GPU it supports, plus three attention modes for GeForce
Blackwell (RTX 50xx): a faster **Precise** mode, an **Adaptive** mode, and a **Fast** mode that runs at SageAttention 3's speed with a
fraction of its error.

SageGP was built for [**WanGP**](https://github.com/deepbeepmeep/Wan2GP), the open-source video, image and audio generator for the GPU
Poor. WanGP shows the full integration: kernels fused with the q / k norms and rotary embeddings, the query packed in place, attention
split by groups of heads to lower peak VRAM, CUDA graphs, and the first denoising step in Precise. If you want these kernels at work
without writing any code, [try WanGP](https://github.com/deepbeepmeep/Wan2GP).

> This repository is a fork of [SageAttention](https://github.com/thu-ml/SageAttention), including the Windows and stable-ABI work of
> [woct0rdho's SageAttention](https://github.com/woct0rdho/SageAttention). The original README, with the papers and their citations, is
> [README_SageAttention.md](README_SageAttention.md).

## Attention modes

| Mode | GPUs | What it does |
|------|------|--------------|
| `sageattn(..., precision="precise")` (default) | all | SageAttention 2: INT8 scores, FP8 values. On RTX 50xx, SageGP's INT8 kernel in a faster schedule (about 8 % faster on long sequences) at about SageAttention 2's error. |
| `sageattn(..., precision="adaptive")` | RTX 50xx | Each head keeps the score and value widths its current activations need within an error budget: part of the scores in NVFP4, part of the values dropped when they carry nothing. The speed depends on the content. |
| `sageattn_fast(...)` / `sageattn3_blackwell(...)` | RTX 50xx | Every score in NVFP4 (the FP4 tensor cores run twice the INT8 rate), values in FP8: SageAttention 3's speed on every head, with error-feedback packing that keeps the error at 50 to 70 % of SageAttention 3's. |

Calls the RTX 50xx kernels do not serve run SageAttention 2, so every function accepts any call `sageattention.sageattn` accepts. Fast
hands the calls it cannot serve to Adaptive, then to Precise.

### Measured

Error against an FP32 reference (relative RMS) and time per call relative to SageAttention 2, on real self-attention inputs captured
from video generations, on a GeForce RTX 50-series GPU:

| Inputs | SageAttention 2 | SageGP Precise | SageGP Adaptive | SageGP Fast | SageAttention 3 |
|--------|-----------------|----------------|-----------------|-------------|-----------------|
| LTX-2.5, 27,280 tokens, 32 heads | 1.43 %, 1.00x | 1.48 %, 1.05x | 3.49 %, 1.00x | 4.27 %, 1.19x | 8.09 %, 1.20x |
| MiniMax H3, 64,391 tokens, 56 heads | 1.76 %, 1.00x | 1.77 %, 1.10x | 3.29 %, 1.26x | 6.43 %, 1.28x | 10.46 %, 1.28x |
| Wan 2.2, 75,600 tokens, 40 heads | 1.07 %, 1.00x | 1.09 %, 1.09x | 3.41 %, 1.18x | 5.66 %, 1.29x | 8.29 %, 1.32x |

## Installation

SageGP contains the `sageattention` package (SageAttention 2.2 and its CUDA kernels) and the `sagegp` package. It replaces SageAttention:
uninstall `sageattention` first.

```
pip uninstall -y sageattention
# Windows
pip install https://github.com/deepbeepmeep/SageGP/releases/download/v1.0.0/sagegp-1.0.0+cu130torch2.10.0andhigher-cp310-abi3-win_amd64.whl
# Linux (glibc 2.28 or newer)
pip install https://github.com/deepbeepmeep/SageGP/releases/download/v1.0.0/sagegp-1.0.0+cu130torch2.10.0andhigher-cp310-abi3-linux_x86_64.whl
```

The wheels work with any Python from 3.10 and PyTorch 2.10 or newer built for CUDA 13 (PyTorch's stable ABI), on GPUs from the RTX 30
series (Ampere: 8.0, 8.6; Ada: 8.9; Hopper: 9.0; Blackwell: 10.0, 12.0, 12.1).

To build from source (CUDA toolkit 12.8 or newer, PyTorch installed):

```
git clone https://github.com/deepbeepmeep/SageGP
cd SageGP
pip install . --no-build-isolation
```

The RTX 50xx kernels are prebuilt (`sagegp/sm120/cubins`) and need no compiler. They need PyTorch 2.10 or newer and Triton 3.6 or newer
(`triton-windows` on Windows). The scripts that rebuild them are in `csrc/sm120`.

## Usage

```python
import torch
from sagegp import sageattn, sageattn_fast, sageattn3_blackwell

q = torch.randn(1, 24, 32768, 128, dtype=torch.bfloat16, device="cuda")  # (batch, heads, tokens, dim): "HND"
k, v = torch.randn_like(q), torch.randn_like(q)

with torch.inference_mode():
    out = sageattn(q, k, v)                           # precise (SageAttention 2's accuracy)
    out = sageattn(q, k, v, precision="adaptive")     # per-head widths within an error budget
    out = sageattn_fast(q, k, v)                      # every score in NVFP4
    out = sageattn3_blackwell(q, k, v)                # the same with SageAttention 3's signature
    out = sageattn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), tensor_layout="NHD")
```

`sageattn` keeps the signature of `sageattention.sageattn` (`tensor_layout`, `is_causal`, `sm_scale`, `return_lse`, ...) and adds
`precision`. `sagegp.kernel_for(q, k, v, ...)` tells which kernel a call would run. Existing code that imports `sageattention` keeps
working unchanged.

### When the RTX 50xx kernels apply

- inference only (no autograd), outside `torch.compile`;
- BF16 inputs, non-causal, no mask, the default softmax scale, queries and keys of the same length;
- 128 channels per head (256 for Adaptive and Fast); Adaptive and Fast from 8,193 tokens, Precise from 2,048.

Other calls run SageAttention 2.

### CUDA graphs

The kernels can be recorded in CUDA graphs. A few constant tensors (a rotation matrix) are created once per device and read at a fixed
address: an application whose graph manager recycles memory between captures registers them through a hook:

```python
import sagegp.hooks
sagegp.hooks.register_constant = my_graph_manager.register_constant  # returns the tensor the kernels will read
sagegp.hooks.notice = lambda operation: print("kernel:", operation)   # optional: log the optional routes taken
```

## How it works

All the RTX 50xx kernels share one CUDA kernel family, built from SageAttention 2.2's SM89 attention template (`csrc/sm120`): INT8 /
NVFP4 Q.K on the tensor cores, an FP32 online softmax, FP8 (E4M3) P.V with SageAttention's two-level accumulation, per-channel value
scales applied in the epilogue. They are compiled once for `sm_120a` and shipped as cubins that PyTorch loads at run time, so installing
SageGP needs no compiler. The packing kernels around them are Triton kernels (`sagegp/sm120`). Every launch runs on PyTorch's current
stream and every decision (which head takes which format) stays on the GPU, with no synchronization with the host, so whole models
can be recorded in CUDA graphs.

### Precise: SageAttention 2 arithmetic in a faster schedule

- **Same quantization as SageAttention 2**: Q to INT8 per 32 rows, K centred on its mean and quantized per 64 rows, V to FP8 per channel.
  The error stays at SageAttention 2's (1.09 % against 1.07 % on Wan 2.2 inputs, 1.77 % against 1.76 % on MiniMax H3).
- **Dense kernel**: a single kernel for the whole sequence instead of SageAttention's general entry point; the packed query can be read
  from strided rows (for example the very rows of a projection), and the output written back to strided rows.
- **P.V by value fragment**: SageAttention computes P.V for a 64-key tile with all its FP16 accumulators alive across 64 products. SageGP
  issues the products one 16-channel value fragment per basic block: each FP16 accumulator lives for 8 products instead of 64, its
  conversion to FP32 overlaps the next fragment's products, and the V fragments are read two fragments ahead. Every accumulator receives
  the same products in the same order, so the output is bit-identical to the plain schedule.
- **Staggered blocks**: the two thread blocks that share a multiprocessor start (and end) 700 cycles apart, so that their tensor-core
  phases interleave instead of competing.

Together these make a long-sequence call 5 to 10 % faster than SageAttention 2.2 on an RTX 50-series GPU.

### Adaptive: a per-head format chosen on the activations

- **Channel ranking**: each head ranks its 128 score channels by their energy on sampled K rows (weighted by the queries' energy when they
  are available), after centring K. The same permutation packs Q and K, so it costs nothing in Q.K.
- **Score widths**: a head keeps 32, 64 or 128 INT8 channels, or 64 INT8 channels plus 64 channels in NVFP4 (E2M1 values with one E4M3
  scale per 16 channels), read by the kernel in the same pass. Value widths of 32, 64, 96 or 128 channels are taken in a ranked basis
  of the values fitted on sampled rows; the discarded components are fitted on pilot queries.
- **Selection**: pilot queries attend to sampled keys in every candidate format; the cheapest format per head whose estimated output
  error, measured on held-out queries, keeps the whole launch within a 3.2 % budget wins. A selector kernel makes the choice on the GPU
  and the attention kernel reads it per head, so no synchronization with the host is needed.
- **Joint tiles**: for 48K to 131K tokens, K and V are packed into one tiled buffer that the kernel loads with TMA (the
  Blackwell tensor memory accelerator).

### Fast: full NVFP4 scores with error feedback

SageAttention 3 also computes the scores in NVFP4 (twice the INT8 tensor-core rate), but rounds each 16-channel block independently.
SageGP keeps the same speed and packs Q and K so that the score error is 50 to 70 % of SageAttention 3's:

- **Rotation**: Q and the centred K are multiplied by one signed Hadamard matrix (Sylvester order, fixed random signs), which is exact in
  Q.K and spreads outlier channels over all channels before rounding.
- **Error feedback in the partner's metric**: each 16-channel block of a row is rounded to NVFP4, then its rounding error is spread over
  the later channels of the same row through the matrix that minimizes the error of the score, measured with the second moments of the
  other operand (the keys' rounding is weighted by the queries' statistics when the queries are available, and conversely), as GPTQ
  does for weights. The second moments
  come from 1,024 sampled rows, with 2 % damping.
- **Graph-safe linear algebra**: the spread matrices are computed with Newton-Schulz inverses and Schur downdates in a Triton kernel (FP16
  products, FP32 accumulation), matrix products only, so no host synchronization or library call breaks a CUDA graph.
- **One scale per row**: each score is one block-scaled NVFP4 product over all 128 channels (two k64 parts); the row scale is a constant
  unit, which removes the per-row dequantization from the inner loop. Values stay in FP8 as in SageAttention 2.

### Wide heads (256 channels)

Adaptive and Fast use a dedicated kernel for 256-channel heads: each head's channel ranges are measured while it is packed, and a row of
128 bytes holds 96 INT8 channels, 48 scaled E2M1 channels and their microscales when the compact form fits; the head keeps all 256
score channels wherever it would not.

## License and attribution

SageGP is licensed under the Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). It is based on SageAttention by the
SageAttention team; please cite their papers ([README_SageAttention.md](README_SageAttention.md)) when you use it in research.

The optimizations SageGP adds on top of SageAttention (the RTX 50xx kernels and their build scripts in `csrc/sm120`, the packing,
selection and error-feedback kernels in `sagegp`, and the techniques described above) were developed by DeepBeepMeep for WanGP. If you
reuse them, in code or as a method, please credit **SageGP by DeepBeepMeep** with a link to this repository
(https://github.com/deepbeepmeep/SageGP) in your documentation, release notes or paper. Redistributions of the code must keep the
[NOTICE](NOTICE) file, as the Apache License requires.

```
@misc{sagegp,
  author       = {DeepBeepMeep},
  title        = {SageGP: faster precise, adaptive and FP4 attention kernels for GeForce Blackwell},
  year         = {2026},
  howpublished = {\url{https://github.com/deepbeepmeep/SageGP}}
}
```
