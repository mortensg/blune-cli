# Research brief: what's missing to get the MLX speed formula to ~97% accuracy

## Context: what this project already has (don't re-derive)

`blune-cli` estimates MLX decode speed (tokens/sec) on Apple Silicon from
`config.json` alone, no weight download. The current formula
(`blune_cli/probe_formula.py`) is:

    time_per_token = bytes_per_token / (bandwidth * BANDWIDTH_CALIBRATION_RATIO)
                      + BASE_OVERHEAD_SEC
                      + n_moe_layers * MOE_LAYER_OVERHEAD_SEC

Fit via linear regression against **9 real measurements** (M4 Pro, 48GB,
273 GB/s spec bandwidth) spanning 5 architecture families: dense
(Qwen2.5-Coder-7B), conventional MoE (Qwen3-Coder-30B-A3B,
gemma-4-26b-a4b, gpt-oss-20b), GatedDeltaNet hybrid+MoE
(Qwen3.6-35B-A3B), LFM2 ShortConv hybrid (dense and MoE variants),
IBM Granite Mamba-2 hybrid+MoE, and NVIDIA Nemotron-H's
single-component-per-layer architecture. Current result: **9.4% mean
error, 21.1% max error** (in-sample, on the same 9 points used to fit
the 3 constants).

`bytes_per_token` itself comes from `blune_cli/size_estimate.py`, which
has real, source-verified per-architecture weight formulas (not
approximations) for: standard GQA/MHA, MLA (DeepSeek-V3-style, formula
verified against the real 671B published param count), GatedDeltaNet
linear attention, classic Mamba-1 and Mamba-2 SSM blocks, LFM2's
ShortConv, Nemotron-H's single-component-per-layer structure, GLM's DSA
indexer, and bailing_moe_linear's Lightning Attention. Every one of
these was built by reading mlx-lm's actual layer source code
(`site-packages/mlx_lm/models/*.py`) rather than guessed from
documentation, after multiple earlier research passes got specific
structural details wrong (see `docs/formula-accuracy-gap.md` for the
full track record of what was tried, verified, and rejected).

**A held-out test already exists and currently fails badly**: two
`Youssofal/Qwen3.6-35B-A3B-*` repos (same architecture family as one
calibration point, different fine-tunes) show +26-30% error versus
`probe_mlx.py`'s own zero-download execution probe (itself validated at
~81-85% of true downloaded-weight speed for 3 architectures, so not
perfect ground truth either, but the best available proxy for
architectures not fully downloaded).

## The core honest question to answer first

**Is 97% accuracy (≈3% error) actually achievable, or is there a lower
real ceiling?** Address this explicitly, with reasoning, before
proposing how to get there. Specifically:

1. What is MLX's own run-to-run measurement noise on identical hardware
   running the identical model? (This project has never measured this
   directly -- every real data point so far is a single run, not a
   repeated-trials average with a reported variance.) If run-to-run
   noise is itself, say, ±3-5%, then "97% accurate" is not a meaningful
   target for a point estimate -- it would need to be reframed as a
   confidence interval or range.
2. Does `probe_mlx.py`'s own ~15-19% gap from true downloaded-weight
   speed impose a hard ceiling on formula accuracy for any architecture
   not directly, really measured? (Today, most of the 9 calibration
   points ARE real Level-1 measurements, not probe-vs-formula
   comparisons -- but the held-out test above is formula-vs-probe, one
   level removed from ground truth. Clarify how much of the 26-30% gap
   could be probe-imprecision vs. genuine formula error.)
3. With 9 real data points and 3 already-fit free parameters, is there
   headroom left to add more parameters without overfitting, or is more
   *data* (not more parameters) the actual bottleneck? Recommend a
   specific minimum number of real measurements needed per architecture
   class to responsibly calibrate further.

## What's missing, by category

### A. `probe_mlx.py`'s own ~15-19% deficit vs. real downloaded-weight generation

One candidate cause (insufficient warmup / DVFS clock-ramp / JIT
pipeline compilation) was tested directly this session and **ruled
out**: running the same real model at warmup depths of 5, 15, 25, and
40 decode iterations before timing gave identical speed within noise
(49.0-49.2 tok/s) at every depth. Do not re-propose this.

Untested candidate mechanisms, in need of real investigation (ideally
via Instruments/Metal System Trace comparing the two code paths, not
just more benchmarking scripts):
1. `probe_mlx.py` builds a model with `mx.random.normal` /
   `mx.random.randint`-initialized arrays, then calls `nn.quantize()`
   on them. A real checkpoint load instead reads pre-quantized integer
   weight data directly via `mx.load`/`safetensors`. Does building a
   model this way and then quantizing produce a *different* array
   layout, dtype path, or kernel dispatch than loading pre-quantized
   weights from disk -- even after both are fully materialized and
   evaluated? (i.e., is there a persistent structural difference, not
   just a one-time construction-cost difference already excluded from
   the timed region?)
2. Real weight loading uses `mmap` (via `safetensors`/`mx.load`),
   mapping the file directly into a page-aligned, contiguous virtual
   address range. `probe_mlx.py`'s randomly-constructed arrays are
   heap-allocated by MLX's own allocator. Does this cause a measurable
   TLB-miss-rate or memory-locality difference during decode
   specifically on Apple Silicon's unified memory architecture? Is
   there a way to test this directly (e.g., write the probe's random
   weights to a temp file and `mx.load` them back, to compare identical
   *values* under real mmap-backed loading vs. heap allocation)?
3. Do randomly-initialized (untrained) weight distributions trigger
   different numerical behavior in Metal's ALUs (e.g., more subnormal
   floats, different branch patterns in dequantization kernels) than
   trained weight distributions, in a way that's measurably slower?
4. Is there a way to directly instrument and compare the actual Metal
   command buffer / kernel dispatch sequence between a probe run and a
   real checkpoint run of the *identical* architecture, to see if they
   diverge at the kernel level rather than just the wall-clock level?

### B. The unresolved Qwen3.6-35B-A3B-family hold-out gap (worst known error, +26-30%)

This is a GatedDeltaNet-hybrid + MoE architecture (same family as one
calibration point, `Qwen3.6-35B-A3B-4bit`, which fits well at +0.8%).
The held-out repos are different fine-tunes of the same base
architecture. Since the same architecture's formula fits well for one
checkpoint and poorly for others of the *same* architecture family:
1. Is this actually a formula problem at all, or could it be
   *checkpoint-specific* (e.g., different quantization group_size,
   different per-tensor quantization overrides, a different exact
   `num_experts_per_tok`/routing config between the fine-tunes)?
   Compare the two held-out repos' `config.json` field-by-field against
   the well-fitting calibration point to find what's actually different
   about them structurally, not just assume it's the same architecture
   family behaving identically.
2. If it is a genuine formula gap: `Qwen3NextAttention.q_proj` was
   already found to double its output size (fixed). Are there other,
   still-undiscovered per-layer quirks in `mlx_lm/models/qwen3_next.py`
   or `gated_delta.py` specifically for the "full attention" or
   MoE-routing sub-layers that a full, exhaustive line-by-line read
   (not a targeted search for one known issue) would surface? Do the
   same exercise that resolved `bailing_moe_linear` and Nemotron-H:
   read every `nn.Linear`/`nn.Conv1d` in the relevant classes and diff
   against what `size_estimate.py` currently computes for this
   architecture, not just the one field already fixed.
3. Get a genuine Level-1 (real download + real generation) measurement
   for one of these two specific held-out repos, so the +26-30% number
   can be checked against true ground truth instead of only against
   `probe_mlx.py`'s own imperfect proxy.

### C. Generalizing the MoE occupancy/bandwidth-efficiency finding

This session found, via a chained-layer micro-benchmark (N=1..32
identical MoE layers, 4-bit quantized) at LFM2-8B-A1B's *exact*
dimensions (32 experts, 4 active, `moe_intermediate_size=1792`,
`hidden=2048`), that effective bandwidth utilization converges to
~45.2% of spec (123 GB/s of 273 GB/s) as layer count grows -- currently
captured in the formula as a flat `MOE_LAYER_OVERHEAD_SEC` additive
term, fit from a 9-point regression, not from this specific
occupancy-efficiency framing directly.

**Not yet known:** what does this ceiling depend on?
1. Is it a function of `moe_intermediate_size` (expert width) --
   narrower experts (e.g. granite-4.0-h-tiny's `intermediate_size=512`
   fallback) should, per an "occupancy starvation" theory, show an even
   *lower* ceiling than LFM2's 1792. Does that actually hold, or does
   the ceiling saturate/plateau above some width and granite's real
   behavior is different?
2. Is it a function of `num_experts_per_tok` (top_k) independent of
   width? Of total expert count? Of `hidden_size`?
3. Design a systematic sweep (not just 2 anecdotal configs) varying one
   dimension at a time (e.g. fix experts=32/top_k=4/hidden=2048, vary
   `moe_intermediate_size` across {256, 512, 1024, 1792, 4096, 8192})
   using the same chained-N methodology, to find the actual functional
   form `η_MoE(moe_intermediate_size, top_k, hidden, ...)` rather than
   a flat constant -- or determine that a flat constant is actually
   fine within measurement noise and further parameterization isn't
   worth the added complexity/overfitting risk.
4. Double-check `granite-4.0-h-tiny`'s real expert width: this
   project's config parsing currently falls back from an absent
   `moe_intermediate_size` field to `intermediate_size=512` for this
   repo. Verify from `mlx_lm/models/granitemoehybrid.py`'s real
   `GraniteMoeHybridMoE`/expert-MLP class whether that fallback is
   actually correct, or whether Granite's real experts use a different
   field/default this project hasn't found.

### D. Real ground-truth (Level 1) coverage gaps

No real (downloaded + measured) data point exists for:
1. **MLA** (any architecture -- DeepSeek is off-limits per project
   constraints, but are there other real, small, MLX-available MLA
   models? e.g. Kimi-K2/Kimi-Linear variants, MiniMax-M2, or GLM's
   DSA-MLA-hybrid family -- anything genuinely small enough to download
   quickly that uses `kv_lora_rank`/`qk_rope_head_dim`-style MLA).
2. **Classic Mamba-1** SSM (Jamba-style, as opposed to the Mamba-2 this
   project already measured via Nemotron-H/Granite) -- is there a
   small, real, MLX-quantized Falcon-H1 or Jamba-family model available
   to test?
3. **Per-layer mixed quantization** (a repo whose config.json shows
   different `bits` for different tensor paths, e.g. `lm_head` at 6-bit
   while the rest is 4-bit) at real, measured speed -- the formula's
   `_infer_bits` path-override logic exists but has never been checked
   against a real speed measurement of such a repo specifically.
4. **Long-context decode** -- every real measurement so far used a
   short prompt (~20-40 tokens). The formula's KV-cache-growth term
   (`context_length` parameter) has never been validated against real
   measured speed degradation at, say, 4K/16K/32K context on an actual
   model. Find a real, testable way to measure this (e.g. `mlx_lm.generate`
   with a long fixed prompt) and compare to the formula's prediction at
   that context length.
5. **Other Apple Silicon chips** -- every measurement so far is on one
   M4 Pro. Do the fitted constants (bandwidth ratio, overheads) hold on
   M1/M2/M3 or M4 Max/Ultra, or are they specific to this exact chip's
   GPU core count / thermal envelope? (Lower priority if this project's
   scope stays single-machine, but relevant if the tool is meant to
   generalize across users' different Macs -- which is its stated
   purpose.)

### E. Measurement methodology and noise floor (foundational -- do this first if possible)

This project has never quantified its own measurement noise. For at
least 2-3 of the existing real calibration models, run the *identical*
generation (same prompt, same model, same machine) 5-10 times and
report the real observed variance in tok/s. This directly answers
whether "97%" is a coherent target (see "core honest question" above)
and should inform every other recommendation in this research.

### F. MLX runtime internals not yet investigated

1. Does MLX batch multiple graph nodes into one Metal command buffer
   per decode step (analogous to llama.cpp's confirmed `n_cb=1`
   default, found in a separate research pass this project ran), or
   does it dispatch more granularly? Is this configurable, and does it
   affect the per-layer dispatch overhead this project has been
   measuring empirically?
2. Does GPU core count (M4 Pro's 20 vs. M4 Max's 40 vs. M4's fewer)
   change the *marginal per-layer dispatch overhead* specifically (as
   opposed to just raw bandwidth, which this project's `hardware_data.py`
   already accounts for per-chip)? This matters for whether
   `BASE_OVERHEAD_SEC`/`MOE_LAYER_OVERHEAD_SEC` need to be per-chip
   constants rather than universal ones.
3. Does the quantization `group_size` (this project has only tested
   `group_size=64`, the near-universal default in the MLX community's
   own conversions) affect kernel dispatch efficiency at other values
   (32, 128)? Low priority unless real repos using non-default group
   sizes are common enough to matter.

## Deliverable format

For each section (A-F), report: what's verified from source/experiment
(cite exactly how), what's a plausible-but-untested hypothesis, and
what would need to be *measured* (not just reasoned about) to resolve
it -- specifying the minimum concrete experiment (which model, how many
repeats, what's held constant) rather than a general description. Where
a claim can be checked against mlx-lm's actual installed source
(`site-packages/mlx_lm/models/*.py` on the target machine, or
`github.com/ml-explore/mlx-lm` at a pinned version) rather than
inferred from documentation or blog posts, do that -- this project has
had multiple research passes get specific structural details wrong when
relying on descriptions instead of the real source (see
`docs/formula-accuracy-gap.md`'s track record), so source-verification
is weighted much more heavily than plausible-sounding narrative.

Most importantly: answer the "core honest question" section with a
specific number or range (e.g. "given X noise floor and Y probe-gap,
the realistic ceiling is roughly Z% error, not 3%") rather than treating
97% as a given target to design toward regardless of whether it's
achievable.
