# Open accuracy gaps — what still needs research or measurement

Historical record of how the formula got here (every bug found, every
research pass, every real measurement taken) lives in git history and in
`probe_formula.py` / `probe_llamacpp.py`'s own module docstrings, which
carry the current, up-to-date methodology and calibration numbers. This
file only tracks what's still open.

---

## 1. vLLM: two real, unrelated packages, only one (mis-)calibrated

`probe_vllm.py`'s `VLLM_SINGLE_STREAM_RATIO = 0.55` is based on exactly
ONE real comparison (`gemma-4-26b-a4b-it-4bit`, MLX vs. vLLM), and
doesn't run its own probe at all -- it calls the MLX probe and
multiplies by this ratio.

Research found there are **two separate, real, unrelated "vLLM on Apple
Silicon" packages**:
- `vllm-project/vllm-metal` (official org) -- wraps vLLM's actual
  CUDA-lineage scheduler/paged-block-manager around mlx-lm's own layer
  classes, with a custom Metal attention kernel replacing CUDA
  PagedAttention. This is almost certainly what the existing 0.55 ratio
  was measured against.
- `waybarrios/vllm-mlx` (independent) -- a from-scratch reimplementation
  of vLLM's *ideas* (continuous batching, paged KV-cache) built natively
  in MLX, not running vLLM's own scheduler code. Its own published
  benchmarks suggest single-stream (concurrency=1) speed much closer to
  native mlx-lm than the 0.55 ratio implies.

**One ratio cannot represent both packages.** `probe_vllm.py` needs
either to detect/ask which one a user has installed, or to ship two
ratios.

A second research pass proposed specific split ratios
(`vllm-metal ≈ 0.52`, `vllm-mlx ≈ 0.92`, plus per-backend fixed
overheads of 1.85ms/0.12ms) and a `detect_vllm_backend()` +
`get_vllm_calibration_factors()` code sketch, based on the same
third-party benchmark sources as before (not a new controlled
measurement) -- **these specific numbers are not yet independently
verified by this project** and shouldn't be hardcoded without a real
test, per this project's own established pattern of research passes
getting specific numeric/structural details wrong on the first two
attempts (see item 2's history). The *architecture* (detect which
package is installed, apply a different ratio per package) is sound
and worth adopting regardless of the exact constants.

**What to research/measure:**
1. Install both `vllm-metal` and `vllm-mlx` (both real, actively
   released pip packages) and run a real, controlled, single-stream
   (concurrency=1) comparison against plain mlx-lm on the same small
   model (e.g. one of the Qwen3-0.6B / Llama-3.2-1B models `vllm-mlx`'s
   own docs already benchmark) on the same machine. This is the
   highest-value single measurement in this whole list -- it directly
   resolves whether one or two ratios are needed and gives real numbers
   for whichever is chosen, rather than adopting the ~0.52/~0.92
   estimates above untested.
2. Independently re-run any concurrency=1-vs-N comparison rather than
   trusting published numbers at face value -- `vllm-mlx`'s own docs
   showed >50% different single-stream tok/s for the identical
   model/chip in two different tables in the same file, so single-run
   third-party benchmarks in this space are known to be noisy.
3. Confirm whether `vllm-metal`'s GGUF support (currently narrow: only
   Q8_0/Q4_0/Q4_1 + F16/F32/BF16, explicitly excluding K-quants/MoE/SSM
   per its own docs) matters for any model blune's cache already covers.

## 2. `bailing_moe_linear`'s real layer structure -- RESOLVED

A first research pass described this architecture using fields
(`linear_key_dim`, `linear_value_dim`) that don't exist in the real
cached config. A second research pass got closer (`head_dim`,
`group_norm_size`, separate q/k/v/g/o projections) but was still
wrong on the details. Reading `mlx_lm/models/bailing_moe_linear.py`
directly (installed locally) found the real structure: a **fused**
`query_key_value` projection (not separate q/k/v matrices), a `dense`
output projection, and a `g_proj` gate -- the `LinearAttention` class
also hardcodes its own KV head count to equal `num_attention_heads`
internally, ignoring config's `num_key_value_heads` (that field only
applies to this architecture's separate, standard-GQA `Attention`
class used on "global" layers). Which layers are "global" vs.
`LinearAttention` is neither a `layer_types` list nor a simple modulo
-- it's `(i+1) % layer_group_size == 0 or i >= (layers // group_size)
* group_size`, read directly from `DecoderLayer.__init__`. Implemented
as `_bailing_linear_attn_params` / `_bailing_is_global_layers` in
`size_estimate.py`, verified against the real cached
`Ring-flash-linear-2.0-128k-4bit` config (28/32 layers correctly
identified as LinearAttention).

## 3. Qwen3-Next's doubled `q_proj` -- implemented, didn't close the hold-out gap

`Qwen3NextAttention.q_proj` outputs `num_attention_heads * head_dim * 2`
in mlx-lm's real implementation. Confirmed independently twice now
(this project's own source read, and a second research pass), so
implemented as a small `model_type -> quirk` registry
(`_Q_PROJ_MULTIPLIER_BY_MODEL_TYPE` in `size_estimate.py`, covering
`qwen3_next`/`qwen3_5`/`qwen3_5_moe`) rather than a generic config-field
heuristic -- a generic rule using `attn_output_gate` was tested and
rejected earlier after confirming Gemma4 sets that field too without
doubling its own `q_proj`.

Effect: modestly improved the in-sample calibration point
(`Qwen3.6-35B-A3B-4bit`: -2.7% -> +0.8% error) but barely moved the
held-out hybrid models' error (`Youssofal/Qwen3.6-35B-A3B-*`: still
+26-30% vs. the zero-download probe, down from +26-38%). The q_proj
quirk was real but was not the dominant remaining error source for
those specific models -- see item 5, which likely explains most of
what's left (they're also MoE).

## 4. `probe_mlx.py`'s own ~15-19% synthetic-probe deficit -- one hypothesis ruled out

Tested the "insufficient warmup" hypothesis directly: ran the same
model (`Qwen2.5-Coder-7B-Instruct-4bit`) at warmup=5, 15, 25, and 40
decode iterations before timing. Result: raw decode speed was
identical within noise at every warmup depth (49.2, 49.1, 49.1, 49.0
tok/s) -- **DVFS/clock-ramp and JIT-compilation warmup depth is not the
cause**, decisively, not just unconfirmed. `probe_mlx.py`'s warmup
stays at 5 iterations; increasing it would add cost for zero benefit.

Remaining untested candidate mechanisms:
- Random/uninitialized weights not going through the same code path as
  a real checkpoint load (may not hit the same fused kernels).
- `mmap`-loaded real weight files vs. Python-heap-allocated random
  arrays having different memory locality/TLB behavior.

This is still a distinct, substantial MLX-runtime investigation, just
smaller in scope now that one major candidate is eliminated.

## 5. LFM2-8B-A1B / granite-4.0-h-tiny's remaining active-bytes gap -- partially confirmed

Re-analyzed the existing chained-layer MoE micro-benchmark (already run
at LFM2-8B-A1B's *exact* expert configuration: 32 experts/4 active,
`moe_intermediate_size=1792`, `hidden=2048`, 4-bit) through a different
lens: instead of "marginal dispatch cost per layer" (178us, the framing
already in `probe_formula.py`), computed the *effective bandwidth
utilization* at each chain length N. Result: it climbs from 66 GB/s
(N=1) to 120 GB/s (N=32), converging toward the same asymptote implied
by the 178us marginal cost -- **123 GB/s, or 45.2% of the 273 GB/s
spec**. This closely matches a second research pass's independently
proposed "MoE occupancy discount" concept (`η_MoE ≈ 0.45` for narrow
expert tiles, `d_ff <= ~2048`, at batch=1) -- two different
mathematical framings of the same real data converging on the same
number is a genuine cross-check, not a coincidence dressed up as one.

**Still open:** this was confirmed at LFM2-8B-A1B's exact dimensions
only. `granite-4.0-h-tiny` has a different, narrower configuration
(`hidden=1536`, `moe_intermediate_size` unset -> falls back to
`intermediate_size=512`, 64 experts/6 active) that hasn't been
separately chain-benchmarked at its own exact dims -- worth confirming
the same ~0.45 ceiling (or a different one) applies there too, since
`d_ff=512` is even narrower than LFM2's 1792 and per the occupancy
theory should plausibly show an even lower ceiling, not the same one.
If a real second data point confirms occupancy scales with `d_ff`
specifically (not just "is this a fine-grained MoE"), the additive
per-MoE-layer term in `probe_formula.py` should become a function of
`d_ff` rather than a flat constant.
