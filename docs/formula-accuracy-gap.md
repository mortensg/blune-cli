# Current formula accuracy vs. real data — what needs improving

## Three levels of "real" (don't conflate them)

1. **Ground truth**: actual measured tok/s from a fully downloaded model
   running for real. Only **5 data points exist**, all dense or
   "ordinary" MoE architectures on one machine (Apple M4 Pro, 48GB):
   Qwen3-Coder-30B-A3B, gemma-4-26b-a4b, Qwen2.5-Coder-7B, Qwen3.6-35B-A3B,
   gpt-oss-20b-OptiQ.
2. **Zero-download real probe** (`probe_mlx.py`): builds the model's
   actual mlx-lm architecture class with random weights and runs a real
   prefill+decode loop. Validated against level 1 at ~81-85% of real
   speed, but only for 3 of those 5 architectures. Used as a stand-in
   ground truth for architectures we haven't fully downloaded.
3. **Formula** (`probe_formula.py`): pure config.json math, no execution.
   Calibrated (2 free parameters, linear regression) against level 1's
   5 points only.

## What's actually validated

Formula vs. level-1 ground truth, on the 5-point set it was fit to:
**mean error 4.8%, max 7.0%**. This is real, but it is an in-sample fit
result on 5 points with 2 free parameters — not evidence the formula
generalizes.

## What's NOT validated at all

None of the newly implemented architecture features — MLA, hybrid
Mamba/SSM/linear-attention layers, shared experts, sliding-window
attention, dense/MoE layer interleaving — have been checked against a
**single real tok/s measurement** of a model that actually has them.

The MLA and dense/MoE-interleaving code was checked against DeepSeek-V3's
*published parameter count* (671B) as a sanity check — the estimate
landed at 672.1B, which validates the **parameter-counting** formula but
says nothing about whether the **speed** formula is right for an
MLA/interleaved model, since DeepSeek-V3 was never run for real.

## The one concrete data point we do have — and it's not good

Comparing the formula against the zero-download real probe (level 2, NOT
level-1 ground truth) for a hybrid Mamba/attention architecture
(`Youssofal/Qwen3.6-35B-A3B-*`, same family as the model that originally
exposed this gap):

| Model | Zero-download real probe | Formula | Delta |
|---|---|---|---|
| Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit | 66.3 tok/s | 92.1 tok/s | **+38.9%** |
| Qwen3.6-35B-A3B-MTPLX-Optimized-Speed | 69.9 tok/s | 92.1 tok/s | **+31.8%** |

The formula over-predicts speed by ~32-39% on this architecture family.
Root cause suspected: the formula's per-layer *weight-parameter* count
still uses one generic attention+MLP formula for every layer, including
Mamba/SSM/linear-attention ones, which have a materially different
weight structure (in_proj, conv1d, x_proj, dt_proj, out_proj) that isn't
being parsed at all — the formula gets the *KV-cache* side of hybrid
layers right (correctly excludes them from the growing KV term) but not
the *weight-bytes* side.

Note this comparison is formula-vs-level-2 (our own zero-download probe),
not formula-vs-level-1 (genuine measured ground truth) — so even the
32-39% figure has an unquantified error bar of its own, since the
zero-download probe itself was never validated against a real download
for a hybrid architecture.

## What this means for further research

1. **Highest-value next step**: get real level-1 measurements (full
   model download + real generation speed) for at least one model with
   each of: MLA (a DeepSeek variant), a hybrid Mamba/attention
   architecture, and a model with shared experts. Right now the formula
   is unvalidated in exactly the areas it was just extended to cover.
2. **Second priority**: a generic (config-field-based) weight-parameter
   formula for Mamba/SSM layers specifically, since that's the
   suspected root cause of the 32-39% hybrid-architecture gap. Field
   names for this vary across converters (`mamba_d_state`/`d_state`,
   `linear_key_head_dim`, `mamba_expand`, etc.) and would need
   cataloging the way MoE's expert-count field names were.
3. Everything else in the original research brief (engine-specific
   efficiency profiles for llama.cpp/vLLM, concurrent-batch throughput,
   speculative decoding, multi-GPU/interconnect) remains completely
   unimplemented and unvalidated — this document only covers the
   MLX single-stream-decode formula's own internal accuracy.
