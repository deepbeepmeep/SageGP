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

- **Precise** quantizes Q and K to INT8 per 32 and 64 rows (K centred on its mean) and V to FP8 per channel, as SageAttention 2 does. Its
  kernel computes the P.V products one 16-channel value fragment at a time, so that each FP16 accumulator lives for 8 products instead
  of 64 and the V fragments are read ahead, and staggers the two blocks of a multiprocessor so that their tensor-core phases interleave.
- **Adaptive** ranks the channels of each head on sample rows and tests, on pilot queries, which score widths (32, 64 or 128 INT8 channels,
  or 64 INT8 + 64 NVFP4) and value widths keep the attention output within a budget; all the decisions stay on the GPU.
- **Fast** rotates Q and the centred K by a signed Hadamard matrix (exact in Q.K), then rounds each 16-channel block to NVFP4 and spreads
  its rounding error over the later channels of the row through the matrix that minimizes the score error in the partner's
  second-moment metric (GPTQ-style feedback). The spread matrices come from matrix products only (Newton-Schulz), so a call can be
  recorded in a CUDA graph.

## License

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). SageGP is based on SageAttention by the SageAttention team; please
cite their papers ([README_SageAttention.md](README_SageAttention.md)) when you use it in research.
