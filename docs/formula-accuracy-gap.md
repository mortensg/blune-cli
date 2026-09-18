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

## Update: online research + real source cross-checks for ~25 more architectures

A second, much larger research pass (see `weight-formula-research-prompt.md`)
produced per-architecture layer formulas for the top architectures by
prevalence in our curated cache. Before implementing any of it, every
formula was cross-checked against the actual field values in our own
cached configs -- and several didn't match:

**Implemented and verified correct:**
- **MLA weight params** (not just KV-cache) for DeepSeek-V3-style
  attention (`q_a_proj`/`q_b_proj`, `kv_a_proj_with_mqa`/`kv_b_proj`,
  `o_proj`) -- this was a real gap: only the KV-cache term was
  MLA-aware before, weight params still used the generic GQA formula.
  Effect: DeepSeek-V3's total-param estimate improved from 672.1B to
  **671.0B** against the real published 671B.
- **Mamba-2** (NVIDIA Nemotron-H, IBM Granite hybrid): single fused
  `in_proj`, depthwise conv, `out_proj` -- structurally different from
  Mamba-1 (no separate `x_proj`/`dt_proj`). Verified against both
  `nemotron_h.py` and `granitemoehybrid.py`; despite very different
  field names (`mamba_num_heads`/`ssm_state_size`/`conv_kernel` vs.
  `mamba_n_heads`/`mamba_d_state`/`mamba_d_conv`) the underlying
  structure is identical. Dispatched *before* the Mamba-1 formula in
  the fallback chain, since Nemotron-H's field names would otherwise
  also (wrongly) match Mamba-1's.
- **LFM2's ShortConv block**: turned out to NOT be a state-space model
  at all (despite living in a "hybrid" architecture) -- just a gated
  depthwise convolution (`in_proj`: hidden→3×hidden, conv, `out_proj`).
  The research's guessed field names (`conv_kernel_size`,
  `intermediate_size`-based sizing) didn't match the real config at all
  (`conv_L_cache`, full `hidden_size`-based sizing) -- implemented from
  reading `lfm2.py` directly instead.
- Generalized shared-expert field-name detection: found **three more
  naming variants** in real configs beyond what was already known
  (`moe_shared_expert_intermediate_size` for Nemotron-H,
  `shared_intermediate_size` for Granite, `num_shared_experts` as a
  count field for a Bailing/Ring variant).
- Added `"conv"` to the SSM/no-growing-KV-cache layer-type set (LFM2's
  `layer_types` uses this word, not `"linear_attention"`/`"mamba"`).

**Explicitly NOT implemented, and why:**
- **Nemotron-H's per-layer structure** doesn't fit this project's model
  at all: reading `nemotron_h.py` directly showed each layer is
  *either* a Mamba-2 mixer, an attention mixer, a plain MLP, *or* a MoE
  block (`hybrid_override_pattern` characters `M`/`*`/`-`/`E`) -- never
  a mixer *plus* a separate FFN the way every other architecture here
  works. Forcing it through the current "mixer + MLP per layer" loop
  double-counts the MLP-only/MoE-only layers. Result: a real Nemotron-H
  config estimates at 103.1B against a model named "30B" -- known-wrong,
  not silently trusted. Fixing this needs a real restructuring (a
  single-component-per-layer mode), not another formula patch.
- **`bailing_moe_linear`**: the research described `linear_key_dim`/
  `linear_value_dim` fields that simply aren't in the real config (which
  instead has `head_dim`/`group_norm_size`) -- the described formula
  would silently fail (return `None`, fall back to the generic
  approximation) rather than error, so it was left unimplemented rather
  than encode something unverified.
- **`deepseek_v4`**: the research described the same MLA structure as
  V3 (`kv_lora_rank`, `v_head_dim`), but the real cached config has
  neither -- it has `o_lora_rank`, `q_lora_rank`, and an `index_head_dim`/
  `index_n_heads`/`index_topk` indexer structure instead, closer to
  `glm_moe_dsa`'s sparse-indexer pattern than to V3's MLA. Left
  unimplemented rather than guess.

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
