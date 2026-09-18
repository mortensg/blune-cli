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

## 3. Qwen3-Next's doubled `q_proj` -- implemented; hold-out gap now mostly resolved via item 3b

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
(`Qwen3.6-35B-A3B-4bit`: -2.7% -> +0.8% error) but on its own barely
moved the held-out hybrid models' error. The real dominant cause was
found separately -- see 3b below.

### 3b. Held-out `Youssofal/Qwen3.6-35B-A3B-*` gap -- root-caused and mostly closed

A third research pass (asked to explain the remaining +26-30% gap)
fabricated a plausible-looking `quantization.overrides` JSON schema with
wildcard tensor-name patterns -- **that specific schema does not exist**
in mlx-lm/MLX's real quantization manifest format. But reading the
actual cached `Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit`
config.json (already in this project's own cache) confirmed the
*underlying claim* was real, just described with the wrong shape: a
**flat, per-tensor** `quantization` dict (515 keys: 3 global defaults +
512 explicit per-tensor `{group_size, bits, mode}` entries), 147 of
which are explicitly 6-bit against a 4-bit repo default -- concentrated
in `mlp.switch_mlp.*` (routed experts), `mlp.shared_expert.*`, `lm_head`,
and a non-uniform subset of attention/linear-attn output projections.

`size_estimate.py`'s `_infer_bits()` already *had* per-tensor-manifest
lookup support (via its `path_hint` parameter) but no call site ever
used it for anything beyond `embed_tokens`/`lm_head` in the RAM-sizing
path -- the actual speed-formula path (`estimate_active_bytes_per_token`)
used one flat repo-wide `bits` for everything, silently treating all 147
of those 6-bit tensors as 4-bit. `probe_mlx.py`'s real
`nn.quantize(..., class_predicate=...)` already honors these per-tensor
overrides correctly (confirmed by reading its source) -- so this
specific model's formula-vs-probe comparison was a bytes mismatch
between the two, not a discovery about formula correctness at the size
it appeared to be.

Fixed by looking up bits separately for routed-expert (`switch_mlp`),
shared-expert (`shared_expert`), and `lm_head` weights when a
per-tensor manifest is present. Not extended to individual mixer
sub-projections (q/k/v/o) or the router -- those overrides in this repo
are non-uniform *within* a component across layers (e.g. only 6 of 10
full-attention layers' `o_proj` are bumped to 6-bit, not all of them),
which `_infer_bits`'s current first-match-wins lookup can't represent
correctly; estimated impact of adding that too is a further ~2-3
points on this one model, not pursued given the added per-architecture
fragility for a shrinking return.

Alongside this, found and fixed a second, more universal bug (see item
6) -- quantization metadata bytes (scale/bias) were missing everywhere,
not just on mixed-precision repos. With both fixes and a required
refit of the 3 global constants (see `probe_formula.py`'s docstring):
this held-out model's error vs. `probe_mlx.py` fell from a freshly
remeasured **+33.6% to +14.9%**, and the in-sample 9-point mean error
fell from 9.4% to **7.2%** (max 21.1% -> 20.1%) -- a real, verified
improvement on the calibration set itself, not just the one outlier
that motivated the investigation.

**Still open:** re-running this comparison after item 4's hybrid-aware
probe fix (below) shows Youssofal's `probe_mlx.py` "ground truth"
estimate itself moved from 66.6 to 88.2 tok/s (it's a GatedDeltaNet+MoE
hybrid, so it was being miscalibrated by the same flat-ratio bug found
in item 4) -- formula-vs-corrected-probe is now **-13.3%** (formula
under-predicting, having flipped sign from the pre-item-4 +14.9%). The
remaining gap is most plausibly this project's general 7.2%-mean formula
error (item 6) plus the still-unhandled partial mixer-projection
overrides noted above (~2-3 points, not implemented), not a new bug.

## 6. Quantization metadata bytes (scale/bias) missing from every byte estimate -- RESOLVED

Every `bits/8` computation in `size_estimate.py` ignored the per-group
scale + bias metadata that affine quantization (MLX's default) always
stores alongside packed weights -- one fp16 scale and one fp16 bias per
group of `group_size` weights. At the common `group_size=64`, 4-bit:
nominal 0.5 bytes/weight vs. real 0.5+4/64=0.5625 bytes/weight, a flat
12.5% miss on *every* quantized model, not just mixed-precision ones.
Fixed via `_effective_bits()` (bits + 32/group_size), applied at every
call site that used to call `_infer_bits()` directly for byte math.
Required refitting `probe_formula.py`'s 3 constants against the same 9
measurements (a systematic bytes-per-token change shifts what
`BANDWIDTH_CALIBRATION_RATIO` should be) -- see item 3b for the combined
before/after numbers from this fix plus the mixed-quantization fix.

## 4. `probe_mlx.py`'s own synthetic-probe deficit -- hybrid architectures split out, 2 hypotheses ruled out

Tested the "insufficient warmup" hypothesis directly: ran the same
model (`Qwen2.5-Coder-7B-Instruct-4bit`) at warmup=5, 15, 25, and 40
decode iterations before timing. Result: raw decode speed was
identical within noise at every warmup depth (49.2, 49.1, 49.1, 49.0
tok/s) -- **DVFS/clock-ramp and JIT-compilation warmup depth is not the
cause**, decisively, not just unconfirmed. `probe_mlx.py`'s warmup
stays at 5 iterations; increasing it would add cost for zero benefit.

**New finding: the deficit is not a flat ~15-19% -- it's much larger
and architecture-dependent.** Downloaded two real GatedDeltaNet-hybrid
checkpoints (not previously in this project's real-download set, which
was all dense/MoE) and measured real speed directly via 10 and 5
repeated `mlx_lm.generate` trials respectively (both under 1% relative
std, confirming this is a real effect, not run-to-run noise):
- `Qwen3.6-35B-A3B-4bit` (GatedDeltaNet+MoE hybrid): real mean 89.1
  tok/s, `probe_mlx.py` raw 59.4 tok/s -- a **33% raw deficit**, still
  -18% error after the flat 0.82 `CALIBRATION_RATIO`.
- `Josiefied-Qwen3.5-0.8B-gabliterated-v1-4bit` (GatedDeltaNet, dense,
  no MoE): real mean 339.7 tok/s, `probe_mlx.py` raw 204.6 tok/s -- a
  **40% raw deficit**, -27% error after the flat ratio.

Both far exceed the ~15-19% figure the flat ratio was validated
against (which used only dense/MoE checkpoints -- see `probe_mlx.py`'s
own docstring). Added `HYBRID_CALIBRATION_RATIO = 0.63` (probe_mlx.py
now picks it automatically via `_analyze(config).n_ssm_layers > 0`),
which brought both hybrid points to +6.9% and -5.7% error respectively
without touching the existing dense/MoE ratio or its own accuracy
(`Qwen3-Coder-30B-A3B-Instruct-4bit` still -5.0% with 0.82). Fit from
only 2 real points, below this project's own stated 5-per-family bar --
treat as a real, directionally-confirmed improvement, not a precisely
calibrated constant; more hybrid ground truth would firm this up.

Also directly tested the **mmap/TLB-locality** hypothesis (a "deep
research" pass's proposed mechanism: heap-fragmented random arrays vs.
mmap-backed contiguous real weight files causing GPU MMU/TLB misses).
Built `probe_mlx.py`'s exact random+quantized model, timed decode, then
saved those exact values to safetensors and reloaded via `mx.load`
(mmap-backed, numerically identical) and re-timed. **Result: the
mmap-reloaded version was 4.9% SLOWER, not faster** -- the wrong
direction for TLB thrashing to be the explanation. Ruled out, same
decisive way the warmup hypothesis was.

Remaining untested candidate mechanism: something GatedDeltaNet's
custom Metal kernel (`gated_delta.py`'s `_gated_delta_kernel`) does
differently with random vs. real gating/decay values -- would need
Metal System Trace (`xctrace`) to actually confirm at the kernel-dispatch
level, not just narrow by elimination like the two ruled-out hypotheses
above.

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

## 7. Small/fast dense-hybrid models don't fit the same global 3-parameter model -- confirmed, not just LFM2.5-specific

Downloaded and measured `Josiefied-Qwen3.5-0.8B-gabliterated-v1-4bit`
(dense GatedDeltaNet hybrid, 0.8B params, real mean 339.7 tok/s over 5
trials) as a genuinely new calibration point -- smaller and faster than
anything previously measured, and the first *pure dense* GatedDeltaNet
point (previously only represented combined with MoE, in
`Qwen3.6-35B-A3B-4bit`). Re-running the same 3-parameter least-squares
fit with this 10th point added made the overall fit WORSE, not better:
mean error 7.2% -> 10.5%, and this new point itself came out at +43.9%
error -- the refit's `BASE_OVERHEAD_SEC` flipped from negative to
positive trying to accommodate it, which then hurt several mid-size
models that were previously well-fit.

This is not new-point noise -- it's the same failure mode already
flagged for `Huihui-LFM2.5-1.2B` (also small/fast/dense-hybrid, also the
worst-fit point even before this addition), now confirmed on a second,
even smaller/faster model. **A single global `BASE_OVERHEAD_SEC` cannot
be simultaneously right for models this fast (where fixed overhead is a
large fraction of total decode time) and for the 7B-35B range the rest
of the calibration set covers.** The 10-point refit's constants were
NOT adopted for production (`probe_formula.py` keeps the 9-point-fit
values, 7.2%/20.1% mean/max) -- this new measurement is kept in
`measurements.json` as real ground truth for whenever this is properly
addressed, but is currently a known, honestly-flagged blind spot rather
than something silently absorbed into a worse-fitting global constant.

**What would actually fix this:** the fixed-overhead term likely needs
to stop being a flat constant and become a function of something that
distinguishes "small enough that fixed overhead dominates" from
"large enough that bandwidth dominates" -- candidate variables:
absolute decode time itself (circular, can't use the answer as an
input), total layer count, or total active bytes. Needs at least one
more small/fast real measurement (a third point) before any specific
functional form could be fit rather than guessed.

## 8. Gemma4's parallel dense+MoE MLP and per-layer-type attention -- RESOLVED

`gemma-4-26b-a4b-it-4bit` was this project's single worst-fit
calibration point across every prior refit (+19.6% to +20.1%), never
previously root-caused -- it was just going through the generic
mixer+MLP-per-layer formula like everything else. Reading
`mlx_lm/models/gemma4_text.py` directly found this architecture diverges
from that generic assumption in two real, structural ways:

1. When `enable_moe_block` is set, `DecoderLayer.__call__` runs the
   dense MLP AND the MoE experts on EVERY layer, in parallel, and sums
   them (`h = h1 + h2`) -- it is not an interleaved dense-XOR-MoE split
   like `first_k_dense_replace`-style architectures. The generic
   `moe_layer_mask` path (used everywhere else in `size_estimate.py`)
   treats a layer as either dense or MoE, never both, so for this
   config every one of the 30 layers had its entire dense MLP
   (3*hidden*intermediate_size, ~17.8M params/layer here) silently
   omitted -- not mis-priced, just completely missing.
2. Full-attention layers (5 of 30, per `sliding_window_pattern`) use a
   DIFFERENT `global_head_dim` (512 here, vs. 256 for the other 25
   sliding-attention layers) and, when `attention_k_eq_v` is set (true
   in this config), `num_global_key_value_heads` with NO separate
   `v_proj` at all -- `values = keys` directly, confirmed in
   `Attention.__init__`/`__call__`. A single config-wide head_dim/
   kv_heads undercounted the wider full-attention layers while also
   overcounting a v_proj that these specific layers don't have.

Implemented as a dedicated `_gemma4_estimate()` in `size_estimate.py`
(same pattern as `_nemotron_h_estimate()` -- bypasses the generic
per-layer loop entirely, wired into all 4 downstream estimate functions
plus `count_moe_layers`), rather than special-casing the generic path.
Also handles `num_kv_shared_layers` (trailing layers of a given
attention type that reuse an earlier layer's K/V and own no separate
k_proj/v_proj or KV-cache slot) and per-layer-input gating
(`hidden_size_per_layer_input`, the 2B/4B-variant mechanism) generically,
though no cached config currently exercises either.

Effect: `gemma-4-26b-a4b-it-4bit`'s own error dropped from +20.1% to
+6.7% after a full 9-point refit, and the whole calibration set's mean
error dropped from 7.2% to **5.9%** (max 21.1% -> 20.1% -> 19.8%) -- a
genuine structural fix, confirmed by how it moved the numbers: one
badly-wrong point improved a lot, the rest stayed roughly where they
were, which is what fixing a real bug looks like as opposed to a
refit trading error between points.
