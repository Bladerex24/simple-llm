# SimpleLLM

~950 line, minimal, extensible LLM inference engine built from scratch.

**NOTE:** Currently, this repository ONLY supports `OpenAI/gpt-oss-120b` on a single NVIDIA H100. Why? A complex starting point (MoE + large model + reasoning + good hardware) demonstrates that building something like this from scratch is viable!!

| Component | Lines |
|-----------|-------|
| `llm.py` (engine) | 568 |
| `model/model.py` | 324 |
| `model/tokenizer.py` | 92 |

The codebase is designed to be read, extended, and modified.

## Performance

The engine is async by default. Every request goes through a background inference loop that continuously batches work to keep the GPU saturated & prioritizing throughput.

Single H100 80GB, max_tokens=1000:

| Benchmark | SimpleLLM | vLLM |
|-----------|------------|------|
| batch_size=1 | 145 tok/s | 138 tok/s |
| batch_size=64 | 4,041 tok/s | 3,234 tok/s |
## Why SimpleLLM?

**Researchers trying new ideas.** If you're experimenting with novel inference techniques, you want a simple but performant starting point that already runs on good hardware with a real model. This gives you continuous batching, CUDA graphs, quantized MoE, and all the modern stuff in code you can actually read and hack on.

**Research labs that would otherwise fork vLLM.** If your plan is to fork a production engine and strip it down to implement your own kernels or adapt it to your infrastructure, consider starting here instead. It's already stripped down.

**Students learning how inference engines work.** This is a working implementation of current state-of-the-art techniques in ~760 lines. You can trace through the entire request lifecycle, from tokenization to continuous batching to CUDA graph replay, without getting lost in abstraction layers.

## Always Maximizing Throughput

The engine implements several techniques to squeeze every bit of performance from the hardware:

- **Async by default**: all generation happens in a background thread with a request queue. You submit prompts, get futures back, and the GPU never idles waiting for you.

- **Continuous batching**: new requests join the active batch mid-generation instead of waiting for the current batch to finish.

- **CUDA graphs**: decode steps are captured and replayed as graphs, which eliminates kernel launch overhead.

- **Slot-based KV cache**: pre-allocated cache slots enable zero-copy sequence management. When a sequence finishes, its slot is immediately available for new work.

- **Fused QKV projections**: three matmuls collapsed into one after weight loading.

- **Fused RMSNorm + residual**: a Triton kernel that combines normalization and residual addition in a single memory pass.

- **Fused RoPE**: position encoding applied in-place via Triton during decode.

- **Flash Attention 2**: memory-efficient attention for both prefill (variable-length) and decode (with KV cache).

- **Paged KV cache**: pre-allocated memory pages for KV storage, so sequences can grow without reallocation and completed sequences free their pages instantly.

- **GQA (grouped query attention)**: 8 KV heads shared across 64 query heads, reducing memory bandwidth during decode.

## Installation

Requires Python 3.12+ and an NVIDIA GPU with CUDA 12.8+.

```bash
./setup.sh
source ./venv/bin/activate
```

## Usage

```python
from llm import LLM

engine = LLM("./gpt-oss-120b")
outputs = engine.generate(["What is the meaning of life?"], max_tokens=100).result()
print(outputs[0].text)
```

## Kernels

The `kernels/triton_kernels/` directory contains community-contributed Triton kernels:

- **routing**: expert selection and token dispatch
- **matmul_ogs**: quantized grouped matmuls for MoE
- **numerics**: MXFP4 quantization/dequantization
- **swiglu**: fused SwiGLU activation
- **topk**: efficient top-k selection
- **compaction**: token compaction for sparse routing

Some of these kernels were adapted/copied from open-source libraries like vLLM & Triton.

## What's Next
- [ ] Paged attention (potentially slower on 1xH100, can be faster when max_tokens/user very quite a bit)
- [ ] `OpenAI/gpt-oss-120b` Tensor parallelism on 8x H100s
- [ ] Support for other MoE models
- [ ] Support for other architectures

## License

Apache 2.0
