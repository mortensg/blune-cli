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

## Update: Nemotron-H's single-component structure, now implemented

A third research pass focused specifically on the two biggest documented
gaps from the last round: Nemotron-H's structure and DeepSeek-V4/GLM's
sparse-indexer models. Both were cross-checked against real mlx-lm
source (available locally) before implementing.

**Nemotron-H (`_nemotron_h_estimate`)**: confirmed by reading
`nemotron_h.py` directly -- a Nemotron-H layer is *either* a Mamba-2
mixer, an attention block, a plain 2-matrix MLP (`up_proj`/`down_proj`
with ReLU², **not** SwiGLU's 3-matrix gate/up/down -- a real, separate
discovery), *or* a MoE block, selected per-layer by a single character
in `hybrid_override_pattern` (`M`/`*`/`-`/`E`) -- never a mixer *plus* a
separate FFN the way every other architecture in this file works.
Added a dedicated estimator that parses this pattern directly instead
of forcing it through the generic per-layer loop. Effect on a real
config: **103.1B → 31.6B**, against a model named "30B-A3B" -- from
3.4x too high to within 5%.

**GLM's DSA indexer (`_dsa_indexer_params`)**: confirmed by reading
`deepseek_v32.py` (which `glm_moe_dsa.py` is a thin, unmodified subclass
of) that mlx-lm 0.31.3 does **not** implement the "IndexShare" weight-
sharing the earlier research described -- `glm_moe_dsa.py`'s `ModelArgs`
doesn't even declare an `indexer_types` field, so every MLA layer gets
its own full `Indexer` (`wq_b`/`wk`/`weights_proj`) regardless of what
config.json's `indexer_types` list says. Added the indexer's weight
params to every MLA layer unconditionally, matching what mlx-lm
actually builds (not what the checkpoint format nominally supports).

**Still not implemented**: DeepSeek-V4's grouped output projection +
query LoRA + indexer combination (the research's formula for
`P_grouped_out`/`P_core_kv` wasn't precise enough to implement with
confidence, and DeepSeek-V4's real structure differs enough from V3
that guessing felt riskier than leaving it as a known gap); the
roofline-style multi-term efficiency model (`η_weights`, `η_kv`,
`η_state` as separate degradation curves) proposed as a Phase 4
replacement for the current 2-constant linear fit -- a much larger
change that needs real Level-1 measurements across several more
architectures to calibrate honestly, not just a formula rewrite;
`probe_mlx.py`'s own ~15-19% synthetic-probe deficit (subnormal-float
handling, quantized-kernel dispatch, warmup depth) -- a separate,
substantial investigation into MLX runtime behavior, not a
config-formula change.

## Update: real Level-1 data finally acquired for hybrid architectures

Every research round above ended with "the highest-value next step is
real measurements for hybrid/MLA/etc. architectures." This is that step:
3 hybrid models were actually downloaded and run for real (not DeepSeek,
by request) via `mlx_lm.generate` on the same M4 Pro/48GB machine:

| Model | Real tok/s | probe_mlx.py (zero-download) | probe_formula.py |
|---|---|---|---|
| LFM2.5-1.2B-Instruct-abliterated-8bit | 176.6 | 189.0 (+7.0%) | 83.2 (**-52.9%**) |
| LFM2-8B-A1B-3bit-MLX | 192.1 | 198.9 (+3.5%) | 107.1 (**-44.2%**) |
| granite-4.0-h-tiny-6bit-MLX | 116.9 | 105.5 (-9.8%) | 93.2 (**-20.3%**) |

Two clear, opposite conclusions:

1. **`probe_mlx.py` generalizes well beyond its original calibration.**
   All 3 land within its documented ~81-85%-of-real band despite none
   of these architectures being part of its original validation set.
   This is now real evidence, not just an assumption, that the
   zero-download execution probe is broadly trustworthy across
   architectures -- prefer it over the formula for hybrid/SSM models.
2. **`probe_formula.py`'s fixed-overhead model does not generalize.**
   Tried three different ways to rescue a single formula across all 8
   points (refit ratio+overhead on all 8: 15.8% mean error, worse than
   the original 5-point fit's 4.5%; overhead scaled per total layer
   count: ranged 115-1131us/layer with no consistent constant across
   the 8 models; overhead scaled per attention-layer count only: same
   problem, no consistent constant). None of these are a real physical
   decomposition -- they're curve-fitting attempts that failed, which
   is itself the useful result: the additive-fixed-overhead assumption
   is probably only valid for conventional attention+MoE graphs, and
   hybrid SSM/conv graphs behave qualitatively differently (plausibly
   because MLX's lazy-eval graph fusion behaves differently across
   mixed operation types within one model -- see this project's own
   much earlier "same real layer x N" graph-fusion finding for a
   related mechanism). Rather than ship a worse-fitting universal
   formula, the original 5-point-calibrated constants were left alone
   and `probe_formula.probe()` now detects >20% SSM/conv layer share
   and downgrades its own `confidence` field to `"low"` with an
   explicit pointer to `probe_mlx.py` instead of silently returning an
   estimate now known to be wrong by 20-53%.

All 3 real measurements were added to `measurements.json`, so
`blune test`/`sweep` now returns real measured numbers (not formula
estimates) for these specific repos.

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

## Update: a 4th real measurement complicates "hybrid = unreliable"

Nemotron-3-Nano-30B-A3B (the exact model whose *structure* was fixed
earlier — 103.1B → 31.6B) was downloaded and measured for real: **57.1
tok/s**. Both the zero-download probe (60.2, +5.4%) and the formula
(60.6, **+6.1%**) were accurate — despite Nemotron-H being arguably the
*most* hybrid architecture here (every layer is mixer-or-FFN, never
both), and despite it tripping the `>20% SSM/conv layers` confidence
check's underlying assumption.

So "hybrid architectures break the formula" was too broad a conclusion
from 3 data points. The actual distinction, best guess with 4 points:
Nemotron-H's bytes-per-token comes from `_nemotron_h_estimate()`, a
dedicated formula built by reading `nemotron_h.py`'s real layer classes
line by line — while LFM2/Granite still run through the generic
per-layer loop with an SSM-formula plugged in, which is evidently less
complete for those two architectures specifically than the Nemotron-H
formula is for its own. In other words: the failure mode isn't "hybrid,"
it's "how completely was this specific architecture's real source
verified" — which loops back to this project's actual working method
(read the source, don't guess) rather than a property of hybrid
architectures as a category.

Practical consequence left as-is rather than "fixed": the confidence
check in `probe_formula.py` only inspects `_analyze()`'s output, which
has no notion of `hybrid_override_pattern` and reports 0 SSM layers for
Nemotron-H — so it never downgrades Nemotron-H's confidence. That's the
right answer here, but by accident of code structure (the dedicated
Nemotron-H path is invisible to the confidence check), not because the
check was designed to distinguish "good hybrid" from "bad hybrid."
Worth revisiting if/when more hybrid architectures are added.

## Update: LFM2/Granite's layer structure re-verified line by line -- it's not incomplete

The "how completely was the architecture verified" theory above was
tested directly: `lfm2.py`, `lfm2_moe.py` (including its own
`Lfm2MoeSparseMoeBlock` and the `SwitchGLU` class it uses), and
`granitemoehybrid.py` were all read in full, matching every `nn.Linear`
and `nn.Conv1d` against what `size_estimate.py` already computes.
Result: **the theory was wrong.** Every layer type (`Attention`,
`ShortConv`, dense `MLP`, `SwitchGLU`'s 3-matrix expert MLP) matches
this project's formulas exactly -- confirmed with an actual worked
calculation, not just a code read.

The real picture, decomposed by removing `FIXED_OVERHEAD_SEC` entirely
and comparing bytes/bandwidth alone against real speed:

| Model | raw bytes/bandwidth (no overhead) vs. real |
|---|---|
| LFM2.5-1.2B (dense) | **-1.8%** -- byte estimate is essentially exact |
| LFM2-8B-A1B (MoE) | **+142.7%** -- byte estimate implies ~2.4x too little traffic |
| granite-4.0-h-tiny (MoE) | **+112.3%** -- byte estimate implies ~2.1x too little traffic |
| Nemotron-3-Nano-30B-A3B (MoE) | +48.1% |

This cleanly separates the two failure modes that were previously
conflated as one "hybrid" problem:

1. **LFM2.5-1.2B's only problem is `FIXED_OVERHEAD_SEC` itself.** Its
   real decode time (~5.66ms/token) is *shorter* than the 7.755ms fixed
   overhead this formula unconditionally adds -- a flat additive
   constant cannot represent both a model this fast and the original
   24-48-layer calibration set simultaneously. The byte-counting is
   already correct; only the overhead term is wrong for it.
2. **LFM2-8B-A1B and granite have a genuine ~2-2.4x active-bytes gap
   that isn't explained by any layer-structure omission** -- every
   weight matrix is accounted for. The leading suspect, not yet
   confirmed: both are MoE with a small expert count (32 and 64) and a
   small `num_experts_per_tok` (4 and 6) under single-token (batch=1)
   decode, and mlx_lm's own `SwitchGLU`/`SparseMoeBlock` source
   contains an explicit code-path split on token count ("when we have
   many tokens, sort them... "), implying single-token decode may hit a
   less bandwidth-efficient kernel path than this formula's "read
   exactly `experts_per_tok` experts, nothing more" assumption models.
   The original 5-point calibration set's MoE models (e.g.
   Qwen3-Coder-30B-A3B, 128ish experts / 8 active) did NOT show this
   problem, so it isn't simply "any MoE at batch=1" -- something about
   small total expert counts specifically. This cannot be resolved by
   reading source code further; it needs actual MLX execution profiling
   (Metal counters or timed sub-steps) to confirm or refute, which is a
   different kind of investigation than everything else in this
   document.

One real, independent bug was found and fixed while re-verifying:
`lfm2_moe`'s dense-layer count field is `num_dense_layers`, not
`first_k_dense_replace` -- `size_estimate.py` was silently treating ALL
of its layers as MoE. Confirmed not to be the cause of the 2-2.4x gap
above (fixing it makes the estimate `_smaller`_, the wrong direction),
but a genuine correctness fix on its own.

---

## Next research needed: llama.cpp / GGUF and vLLM calibration

Everything above is MLX-specific (`probe_mlx.py` / `probe_formula.py`).
The other two engines this project supports are in a much worse state
and are the next place to spend effort, in priority order:

### `probe_llamacpp.py` -- `GGUF_CALIBRATION_RATIO = 0.60`

This constant is a placeholder, explicitly documented in the module as
"seeded from the MLX ratio plus llama.cpp's generally-lower Metal
efficiency" -- i.e. never derived from a real llama.cpp measurement.

**A first real measurement was taken this session** (via `llama-bench`,
Apple M4 Pro, Metal backend) on Qwen2.5-0.5B-Instruct-GGUF (630M params)
across 4 quantization levels, isolating the quantization axis on a
single fixed model:

| Quant | File size | Real tg150 (tok/s) | Raw bandwidth-only tps | Implied fixed overhead |
|---|---|---|---|---|
| Q4_0 | 403.2 MiB | 307.6 | 645.6 | 1.70ms |
| Q4_K_M | 463.0 MiB | 269.8 | 562.3 | 1.93ms |
| Q6_K | 614.6 MiB | 256.9 | 423.7 | 1.53ms |
| Q8_0 | 638.7 MiB | 256.1 | 407.7 | 1.45ms |

The implied fixed overhead (1.45-1.93ms) is much more consistent across
quant levels than a single ratio would suggest, and -- as expected for
a compiled C++ binary vs. MLX's Python-orchestrated graph -- much
smaller than MLX's ~7.7ms base overhead. This strongly suggests the
same "bytes/bandwidth + fixed overhead(s)" methodology that worked for
MLX (see above) would work here too, replacing the single ratio.
**Only one model size (630M) has been tested so far** -- a second real
measurement at a larger size (7B) was started but not yet complete at
time of writing; more sizes and at least one MoE GGUF model (e.g. a
Mixtral or Qwen3-MoE GGUF conversion) are needed before fitting real
constants.

**What to research:**
1. Exact bits-per-weight (including block/superblock scale+min
   overhead, not the nominal marketing number) for every common GGUF
   quant type (Q4_0, Q4_1, Q4_K_S/M, Q5_0/1, Q5_K_S/M, Q6_K, Q8_0,
   IQ-series) -- cite `ggml/src/ggml-quants.c` or the block-type structs
   in `ggml/include/ggml.h` directly (block size in elements, bytes per
   block).
2. How llama.cpp's Metal backend (`ggml-metal.m`/`.metal`) dequantizes
   and computes decode (batch=1) -- is there a real per-token
   dispatch/kernel-launch floor, and does it vary by quant type the way
   the table above hints?
3. How llama.cpp handles MoE GGUF models specifically -- does it read
   all experts but only compute active ones (matching this project's
   approach), or does the fused-expert-tensor GGUF format change memory
   traffic in some other way? Look for GitHub issues/discussions
   specifically about MoE decode speed on Metal.
4. Any public `llama-bench` result datasets (GitHub, r/LocalLLaMA
   benchmark megathreads) across model sizes and quant levels on Apple
   Silicon, usable as additional calibration points without downloading
   everything ourselves.

### `probe_vllm.py` -- `VLLM_SINGLE_STREAM_RATIO = 0.55`

Worse off than llama.cpp: based on exactly ONE real comparison (a
Gemma-4 MLX-vs-vLLM measurement from earlier in this project), and it
doesn't even run its own probe -- it calls the MLX probe and multiplies
by this ratio, which conflates vLLM's actual serving engine with
mlx-lm's, architecturally.

**What to research:**
1. Is there a real, current vLLM Metal/MPS backend for Apple Silicon as
   of now? Does it actually reuse mlx-lm's layer implementations (this
   project's standing assumption) or run its own PyTorch MPS kernels --
   find and cite the actual source.
2. How does vLLM's PagedAttention/continuous batching behave at
   concurrency=1 specifically -- does single-stream vLLM degrade toward
   native speed, or is it structurally handicapped by batching machinery
   designed for high concurrency?
3. Any public vLLM-on-Apple-Silicon benchmarks (GitHub, blog posts,
   Reddit/HN) usable as additional real calibration points beyond this
   project's single Gemma-4 measurement.

## Update: llama.cpp recalibrated with real data; vLLM research landed, not yet acted on

A background research agent (general-purpose, web+source access) completed
the llama.cpp/vLLM research brief above while 5 real `llama-bench`
measurements were taken in parallel. Two independently useful outcomes:

**llama.cpp: done.** Found and fixed a real bug in `_GGML_TYPE_BITS`
(Q2_K/Q3_K were priced at ~2x their real bytes -- an ID-mapping error,
not an imprecision, verified against ggml's own block-struct source),
and replaced the flat `GGUF_CALIBRATION_RATIO=0.60` guess with a
bandwidth-ratio + fixed-overhead model fit against 5 real measurements
(Qwen2.5-0.5B across 4 quant levels + Qwen2.5-7B Q4_K_M, M4 Pro,
`llama-bench`): mean error 17.1% -> 6.2%, max 28.7% -> 10.4%. See
`probe_llamacpp.py`'s docstring for the full table and methodology --
it deliberately mirrors `probe_formula.py`'s MLX approach. Same
honesty caveat as MLX's early calibration: 5 points, one architecture
family (Qwen2 dense), two sizes -- a real anchor, not a finished
formula. The agent's research also *confirmed* (by reading llama.cpp's
real Metal MoE kernel, `mul_mv_id`) that this project's existing
MoE-active-bytes approximation is mechanistically correct, not just
convenient -- no change needed there.

**vLLM: researched, not yet implemented.** The single biggest finding:
**there are two separate, real, unrelated "vLLM on Apple Silicon"
packages** -- `vllm-project/vllm-metal` (official org, wraps vLLM's
actual CUDA-lineage scheduler around mlx-lm's layers with a custom
Metal attention kernel) and `waybarrios/vllm-mlx` (independent,
ground-up MLX-native reimplementation of vLLM's *ideas*, not its code).
This project's existing `VLLM_SINGLE_STREAM_RATIO=0.55` was almost
certainly measured against `vllm-metal` (the docstring's own wording
matches that repo's README), but `vllm-mlx`'s own published benchmarks
suggest it runs much closer to native mlx-lm speed at concurrency=1 --
meaning **one ratio cannot represent both packages**, and `probe_vllm.py`
either needs to ask/detect which one a user has, or needs two ratios.
Not implemented yet because: (a) it needs a real controlled
measurement, not another guess layered on top of third-party numbers,
and (b) the third-party numbers found have real internal
inconsistency -- `vllm-mlx`'s own docs report >50% different
single-stream tok/s for the identical model/chip in two different
tables in the same file, which the agent flagged as a genuine
data-quality finding in its own right (any benchmark of this kind is
noisy enough that single-run numbers shouldn't be trusted at face
value). The agent also caught and explicitly discarded its own
hallucinated claim mid-research (a fabricated "45-90 Metal command
buffers per token" figure that doesn't exist in the source it claimed
to cite) -- worth noting as a demonstrated failure mode to watch for
when delegating this kind of research generally, not specific to this
finding.

Full agent output (GGUF bit-width table, verified Metal dispatch
mechanics, the ferrox third-party llama.cpp benchmark comparison, and
the complete prioritized follow-up list) is in this session's transcript
but was not saved to a separate doc file -- the actionable parts are
captured above and in `probe_llamacpp.py`'s docstring; re-run a similar
research pass if the full source-citation detail is needed again later.
