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

## Update: root-caused and partially fixed against mlx-lm's real source

Rather than guess, we read mlx-lm's actual layer implementations
(`site-packages/mlx_lm/models/{gated_delta,mamba,qwen3_next}.py`) to get
exact parameter formulas instead of approximations. Two real bugs found
and fixed this way:

1. **SSM/linear-attention layers used the generic attention formula.**
   `size_estimate.py` now has exact param formulas matching
   `GatedDeltaNet.__init__` (Qwen3.5/3.6/3-Next's linear-attention layer:
   `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, depthwise
   `conv1d`, `out_proj`) and `MambaBlock.__init__` (classic Mamba/Jamba-
   style: `in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj`), selected
   by which config fields are present.
2. **Qwen3-Next's always-on shared expert was invisible to the formula.**
   `Qwen3NextSparseMoeBlock` always instantiates one shared expert sized
   by `shared_expert_intermediate_size` — a separate field from the
   generic `n_shared_experts` count this project already checked, so it
   was silently contributing zero bytes. This fix also **improved the
   original 5-point calibration set** (mean error 4.8% → 4.5%), since
   one of those 5 models (`Qwen3.6-35B-A3B-4bit`) has the same
   architecture and was quietly undercounted too.

Effect on the same hold-out comparison (formula vs. our own zero-download
real probe, level 2 — not level-1 ground truth):

| Model | Zero-download real probe | Formula (before fix) | Formula (after fix) |
|---|---|---|---|
| Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit | 68.1 tok/s | 92.1 (+38.9%) | 85.9 (**+26.1%**) |
| Qwen3.6-35B-A3B-MTPLX-Optimized-Speed | 73.0 tok/s | 92.1 (+31.8%) | 85.9 (**+17.7%**) |

Better, not solved. A third bug was found but deliberately **not** fixed:
`Qwen3NextAttention.q_proj` projects to `num_attention_heads * head_dim *
2` (double the standard size) in mlx-lm's real implementation — some
gated-attention variant specific to this architecture family. We checked
whether the `attn_output_gate` config field could generically signal
this (it seemed plausible), but Gemma4 also sets `attn_output_gate: true`
and its real `Attention.q_proj` is the *standard*, non-doubled size — so
using that field as a general rule would have silently broken Gemma4's
(already-validated, in-calibration-set) attention param count to fix
Qwen3-Next's. This is the kind of per-architecture-family quirk that
doesn't generalize from config.json field presence alone and would need
either a `model_type`-keyed lookup table or (better) real measured data
for more architectures to know how much it actually matters.

## What this means for further research

1. **Highest-value next step, unchanged**: get real level-1 measurements
   (full model download + real generation speed) for at least one model
   with each of MLA, a hybrid Mamba/attention architecture, and a model
   with shared experts. The formula is still unvalidated against genuine
   ground truth in every area it was extended to cover — only checked
   against our own zero-download probe, which has its own unquantified
   error for these architectures.
2. **Revised second priority**: rather than one more generic formula,
   the pattern above (real bugs found by reading mlx-lm's actual layer
   source, not by guessing from config field names) generalizes well —
   repeat it for other architecture families as they come up, and
   consider a small `model_type -> known quirks` lookup table for the
   handful of per-architecture-family exceptions (like Qwen3-Next's
   doubled q_proj) that genuinely can't be inferred from config alone.
3. Everything else in the original research brief (engine-specific
   efficiency profiles for llama.cpp/vLLM, concurrent-batch throughput,
   speculative decoding, multi-GPU/interconnect) remains completely
   unimplemented and unvalidated — this document only covers the
   MLX single-stream-decode formula's own internal accuracy.
