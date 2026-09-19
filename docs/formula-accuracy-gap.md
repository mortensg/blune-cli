# Open accuracy gaps — what still needs research or measurement

Historical record of how the formula got here (every bug found, every
research pass, every real measurement taken) lives in git history and in
`probe_formula.py` / `probe_llamacpp.py`'s own module docstrings, which
carry the current, up-to-date methodology and calibration numbers. This
file only tracks what's still open.

---

## 1. vLLM: two real, unrelated packages -- RESOLVED, both installed and measured

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

**RESOLVED -- both packages installed and directly measured.** Created
an isolated venv (`.venv-vllm-test`, separate from this project's own
working venv) and `pip install`ed both `vllm-metal` and `vllm-mlx` --
both real PyPI packages, installed together with no dependency
conflicts. Ran a real, controlled, single-stream (concurrency=1)
comparison against native `mlx_lm.generate` on the same small model
(`mlx-community/Qwen2.5-0.5B-Instruct-4bit`) on this project's own
reference machine:

    Native mlx_lm.generate: 429.38 tok/s (10-trial mean, std 1.18%)
    vllm-metal (real HTTP server, 1 warmup + 8 timed requests, greedy):
      293.69 tok/s (std 2.4%) -- ratio 0.684
    vllm-mlx (`vllm-mlx bench --max-num-seqs 1`, after discovering and
      controlling for its own large first-run warmup effect -- a
      5-prompt run gave a misleadingly low 52.13 tok/s/ratio 0.121
      before internal state warmed up; two separate 10-prompt runs
      after that gave 300.17 and 319.67 tok/s):
      mean 309.92 tok/s -- ratio 0.7215

**This directly refutes the specific split a research pass proposed**
(vllm-metal ~0.52-0.55, vllm-mlx ~0.90-0.95) -- on this one real small
model, both packages give SIMILAR single-stream ratios, not
dramatically different ones. What DOES look real: this project's
original single vllm-metal point (`gemma-4-26b-a4b-it-4bit`, 26B, ratio
0.554) is meaningfully lower than the new 0.5B point (0.684) for the
SAME package, suggesting MODEL SIZE may be a bigger driver of the ratio
than which package is used -- with only 2 points per backend (well
below this project's own 5-per-family bar), model size and package
choice aren't yet cleanly separable.

Implemented `detect_vllm_backend()` in `probe_vllm.py` (checks which
package, if either, is actually importable) with separate
`VLLM_METAL_SINGLE_STREAM_RATIO` (0.619, mean of the 2 real points) and
`VLLM_MLX_SINGLE_STREAM_RATIO` (0.7215, mean of the 2 real vllm-mlx
runs) -- falling back to the more conservative (lower) ratio when
neither package is installed, since most `blune-cli` users are deciding
whether to install vLLM at all, not calling this with it already
present.

Also confirmed the warmup effect independently for vllm-mlx's own
benchmark tool -- a DIFFERENT warmup mechanism from `probe_mlx.py`'s
(which was already tested and ruled out as a cause for ITS OWN
deficit, see item 4): here it's the serving harness's own first-run
state (prefix cache, scheduler initialization, or similar) rather than
MLX/Metal JIT compilation, but the practical lesson is the same --
single-shot benchmarks of a fresh process can be badly misleading, and
this project's own `vllm-mlx` numbers above were re-measured after
discovering this rather than reported from the first (misleading) run.

Not yet done: `vllm-metal`'s GGUF support (currently narrow per its own
docs -- Q8_0/Q4_0/Q4_1 + F16/F32/BF16 only, explicitly excluding
K-quants/MoE/SSM) was not cross-checked against blune's cached models;
independently re-running a concurrency=1-vs-N comparison (this
project's own vLLM concurrency data is still the single old
`gemma-4-26b` point) also remains open.

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

**RESOLVED (real Level-1 measurement taken):** every prior number in
this section was formula-vs-`probe_mlx.py`, never formula-vs-real. This
project's own established methodology says the probe is a stand-in for
ground truth, not ground truth itself, so the final step was always to
actually download and measure the real repo -- done: 12-trial
`mlx_lm.generate` mean, **78.71 tok/s, std 0.98% relative** (a genuinely
clean, tightly-clustered measurement -- see measurements.json for a
methodological note about a first attempt's cold-disk-cache instability,
resolved by re-measuring once the file was warm). `probe_formula.py`
predicts **79.5 tok/s -- +1.0% error.** The entire originally-reported
+26% to +30% gap that motivated this whole item was a `probe_mlx.py`
proxy artifact from start to finish: the probe's own hybrid-calibrated
estimate for this exact model is 86.9 tok/s, +10.4% over the now-known
real value, and its raw (uncalibrated) deficit is 30.4% -- both
consistent with the hybrid-architecture deficit already characterized
in item 4. The mixed-quantization and metadata bytes-per-token fixes
above were still real, verified, and worth having (they measurably
improved the 9-point in-sample calibration set too, see item 6) -- they
just weren't the reason this *specific* held-out comparison looked as
bad as it did. This real measurement also fed back into refining
`probe_mlx.py`'s own `HYBRID_CALIBRATION_RATIO` (0.63 -> 0.655, now fit
from 3 real points instead of 2 -- see item 4).

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

**Update, a 3rd real point:** downloading `Youssofal/Qwen3.6-35B-A3B-
Abliterated-Heretic-MLX-4bit` for real Level-1 ground truth (see item
3b's resolution) gave a 3rd hybrid data point: real 78.71 tok/s, raw
probe 54.8 tok/s -- a 30% raw deficit, and the 2-point `0.63` ratio
still landed at +10.4% error against it. Refit from all 3 points:
averaging the 3 individual ratios (0.667, 0.602, 0.696) gives **0.655**,
which fits all 3 points better (5.4% mean abs error) than the 2-point
value or a pooled-sum alternative -- `HYBRID_CALIBRATION_RATIO` updated
accordingly. Still below the 5-per-family bar, but a real improvement
in the right direction with each new point.

Also directly tested the **mmap/TLB-locality** hypothesis (a "deep
research" pass's proposed mechanism: heap-fragmented random arrays vs.
mmap-backed contiguous real weight files causing GPU MMU/TLB misses).
Built `probe_mlx.py`'s exact random+quantized model, timed decode, then
saved those exact values to safetensors and reloaded via `mx.load`
(mmap-backed, numerically identical) and re-timed. **Result: the
mmap-reloaded version was 4.9% SLOWER, not faster** -- the wrong
direction for TLB thrashing to be the explanation. Ruled out, same
decisive way the warmup hypothesis was.

Also tested the **subnormal-float-stall** hypothesis (a research pass's
proposed mechanism: `mx.random.normal()`-initialized weights can
dequantize to IEEE 754 subnormal values that trigger slow microcode
paths on Apple Silicon, whereas trained models keep normalized
distributions). Built the same real model twice, once with
`mx.random.normal()` weights (probe_mlx.py's actual default) and once
with every float replaced by a constant `0.05` (subnormals structurally
impossible), quantized identically, timed identically. **Result: 47.3
vs. 47.0 tok/s, a 0.5% difference -- within noise, ruled out.** Third
candidate mechanism eliminated the same decisive way as warmup and
mmap/TLB.

Also checked a specific claim from a later research pass -- that MLX
internally enforces `MLX_MAX_OPS_PER_BUFFER = 50` /
`MLX_MAX_MB_PER_BUFFER = 50` command-buffer-batching thresholds in
`mlx/backend/metal/device.cpp`, with each encoder rollover costing
"~17us" and explaining `BASE_OVERHEAD_SEC` as scaling with layer count.
**These specific constant names do not appear anywhere in the installed
MLX binary** (`strings` on `mlx/core.cpython-312-darwin.so` finds zero
matches for either name, and `mx.metal`'s actual Python API exposes only
cache/memory-limit controls, nothing about command-buffer batching) --
this looks like the same kind of confidently-specific fabrication this
project has caught multiple times before (see item 2's history), not a
verified mechanism. The general idea (MLX batches multiple ops per
Metal command buffer, and encoder transitions have some real cost) is
plausible on its face, but the specific named constants and the "5.4ms
of idle GPU pipeline latency per token" figure built on them are not
independently confirmed and should not be treated as established.

Remaining untested candidate mechanisms: (1) something GatedDeltaNet's
custom Metal kernel (`gated_delta.py`'s `_gated_delta_kernel`) does
differently with random vs. real gating/decay values; (2) actual Metal
command buffer/encoder counts via a real `xctrace` "Metal System Trace"
capture, comparing the synthetic probe against a real downloaded
checkpoint -- the only way to either confirm or debunk the command-
buffer-rollover idea above with real evidence instead of narrowing by
elimination like the three ruled-out hypotheses did.

**The deficit is GatedDeltaNet-specific, not general-SSM -- RESOLVED
detection bug.** The original hybrid detection
(`_analyze(config).n_ssm_layers > 0`) lumped Mamba-2-hybrid
architectures into the same "hybrid" bucket as GatedDeltaNet-hybrids,
applying the same 0.655 ratio to both. Tested directly against two real
Mamba-2-hybrid ground-truth points already in this project's own data:

    granite-4.0-h-tiny-6bit-MLX: real 117.4 tok/s, raw probe 90.0 --
      ratio 0.766. The GatedDeltaNet ratio (0.655) gives +17.0% error;
      the plain dense/MoE ratio (0.82) gives -7.4%.
    NVIDIA-Nemotron-3-Nano-30B-A3B (Nemotron-H, a Mamba-2 hybrid, routed
      through a SEPARATE dedicated size_estimate.py estimator that
      `_analyze()` doesn't understand at all -- meaning the old
      detection never even triggered for it, by accident rather than
      design): real 58.0 tok/s, raw probe 51.0 -- ratio 0.880. Plain
      0.82 gives +7.5% error.

The two Mamba-2-hybrid ratios (0.766, 0.880) average to **0.823** --
essentially identical to the plain `CALIBRATION_RATIO` (0.82), nothing
like the GatedDeltaNet value. This is strong evidence the large
synthetic-probe deficit is a property of GatedDeltaNet's specific
custom Metal kernel implementation, not Mamba-family recurrence in
general. Fixed by gating `HYBRID_CALIBRATION_RATIO` on
`_gated_delta_net_params(c, hidden) is not None` (in addition to
`n_ssm_layers > 0`) rather than the generic SSM-layer count --
Mamba-2-hybrid architectures now correctly fall through to the plain
ratio, landing at -7.4%/+7.5% instead of the previous +17.0%/(accidental
+7.5%-by-luck). This also strengthens the case for candidate mechanism
(1) above: the deficit tracking GatedDeltaNet specifically, not SSMs
generally, points more precisely at `_gated_delta_kernel` itself as the
place to look with a real Metal System Trace.

### 4b. First real Metal System Trace capture -- one fabricated claim definitively refuted, real command-buffer behavior characterized

Finally attempted the `xctrace` "Metal System Trace" capture this
document has been deferring since item 4's first draft. It works, and
`xctrace export` can pull real, structured, per-command-buffer data out
of the resulting `.trace` bundle without needing the Instruments GUI --
worth recording since this wasn't obvious going in. Captured a real
trace of `probe_mlx.py`'s own synthetic decode loop for
`Qwen3.6-35B-A3B-4bit` (30 decode tokens, GatedDeltaNet+MoE hybrid, no
model download needed since the probe builds random weights) and
exported the `metal-application-command-buffer-submissions` table
(5,489 real rows, `xctrace export --xpath ... --output ...`, then
parsed with `xml.etree.ElementTree`, resolving the trace format's
`id`/`ref` cross-references).

**Definitively refutes the specific `MLX_MAX_OPS_PER_BUFFER = 50`
encoder-rollover claim already flagged as likely-fabricated in item 4
above** (that flag was based on the constant not appearing in the
compiled MLX binary; this is now independently confirmed from the
runtime's own real behavior): every single command buffer in the real
capture has `num-encoders` of exactly 0 or 1 -- never anywhere near 50,
and never more than 1. There is no "50 ops then roll to a new encoder"
pattern happening at all for this workload.

**What real command-buffer behavior actually looks like** (5,489 total
submissions over 30 decode tokens -- roughly 183 buffers per token,
i.e. MLX is NOT batching a whole decode step into one or a few command
buffers here, contrary to what a "the whole graph is one buffer"
mental model would suggest):
- Median total buffer-to-buffer duration: 72.6us: p10 1.2us, p90
  19.5ms, p99 81.2ms -- most gaps are tiny, but a real, substantial
  long tail exists (some buffers are separated by tens of
  milliseconds).
- Median encoder-time (GPU work actually being done), for the ~67% of
  buffers that have an encoder at all: 108.9us, p10 21.2us, p90 248.1us
  -- comparatively tight and consistent, i.e. the GPU's own compute
  time per dispatch is NOT where the large variance comes from.
- 1,810 of 5,489 buffers (33%) have `num-encoders = 0` -- likely
  signal-only/barrier submissions rather than genuine compute
  dispatches, consistent with MLX's dependency-tracking machinery
  needing its own synchronization points beyond pure compute encoders.

**Interpretation, still provisional:** the long-tail buffer-to-buffer
gaps (up to tens of ms, vs. a ~100us-scale median encoder time) are
consistent with real, non-trivial CPU-side dispatch/scheduling latency
existing between some command buffers -- but the magnitude is highly
variable, not a fixed per-buffer constant, and this trace alone can't
say whether the long-tail buffers specifically correspond to
GatedDeltaNet's custom kernel or something else (would need per-thread
call-stack correlation within the trace, not attempted here). **Not yet
done: the actual comparison this investigation was for** -- tracing a
REAL downloaded checkpoint's decode loop the same way and diffing
against this synthetic trace, to see whether the num-encoders-per-buffer
pattern or the long-tail-gap distribution differs between random and
real weights. That would need re-downloading a large real checkpoint
(already cleaned up from disk this session) and is the natural next
step, not completed here.

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

### 5b. Systematic width sweep run -- real curve found, but integration into the global formula made things WORSE

Ran the systematic sweep item 5 and the mlx-97-percent-research-prompt.md
brief both called for: chained-layer micro-benchmark (quantized 4-bit,
group_size=64, E=32/k=4/hidden=2048 matching LFM2-8B-A1B, slope read
from both N=4/16 and N=16/32 chain pairs for a stability check -- both
gave the same curve) with `moe_intermediate_size` swept across
256/512/1024/1536/1792/2048/4096/8192. Real, densely-sampled result (not
1-2 anecdotal points):

    w=  256  eta=0.196     w= 1792  eta=0.529 (LFM2-8B-A1B's real width)
    w=  512  eta=0.354     w= 2048  eta=0.594
    w= 1024  eta=0.469     w= 4096  eta=0.680
    w= 1536  eta=0.527     w= 8192  eta=0.715

Fits cleanly to a saturating hyperbolic curve, `eta(w) = eta_dense *
w/(w+Kw)`, with `eta_dense=0.809, Kw=773` (fit error <5% at every
sampled width) -- the same functional FORM a research pass had
independently guessed, though its specific guessed constants
(`eta_dense≈0.85, Kw≈1500-2000`) were off by roughly 2x on `Kw`. Note
this sweep's own `eta=0.529` at LFM2-8B-A1B's exact width is itself
noticeably different from the earlier, cruder single-point estimate
above (0.452) -- likely a real methodology difference (this sweep
didn't include a shared-expert or router term matching LFM2-8B-A1B's
*exact* full layer, only routed experts) rather than either number being
wrong; treat 0.529 as the more careful, reproducible measurement of the
two.

**Integration attempt and result:** implemented this as a bytes-level
correction (routed-expert-FFN bytes divided by `bandwidth * ratio * eta(w)`
instead of the flat `bandwidth * ratio` everything else uses), keeping
`eta_dense`/`Kw` FIXED (not re-fit -- they came from this independent
8-point sweep, not the 9-point calibration set, specifically to avoid
spending more of that set's limited degrees of freedom) and re-fitting
only the existing 3 parameters. Result: **mean error got WORSE, 5.9% ->
15.3%**, and `gemma-4-26b-a4b-it-4bit` (unaffected by this change at all,
since it goes through its own dedicated `_gemma4_estimate` path) jumped
from +6.7% to +40.0% error. Making MoE-heavy points "need more time" via
the eta discount forced the shared `BANDWIDTH_CALIBRATION_RATIO` to
refit upward (0.78 -> 0.89) to compensate, which then overcorrected
every point NOT affected by the discount -- the exact "one shared knob
trades error between points" failure mode this project has repeatedly
flagged as a risk of adding structure without enough independent data
to isolate it. **Not adopted** -- reverted to the flat-bytes formula
(5.9%/19.8%). The `eta(w)` curve itself is kept here as real, verified,
reusable ground truth; what's still needed before it can help is either
more real MoE-family measurements to fit a formula structure that
properly separates "which bytes get which effective bandwidth" without
collapsing back onto one shared ratio, or moving the flat
`MOE_LAYER_OVERHEAD_SEC` term to also depend on width so the two terms
absorb the effect together instead of one uncalibrated knob compensating
for the other.

### 5c. Second integration attempt (additive, not multiplicative) -- also made things WORSE

A later research pass proposed exactly the alternative framing item 5b
called for: instead of dividing routed-expert bytes by a width-dependent
effective bandwidth (multiplicative, touches the shared bandwidth
denominator), replace the flat `MOE_LAYER_OVERHEAD_SEC` with a
width-dependent ADDITIVE dispatch-latency term per MoE layer,
`τ_dispatch(w) = τ_base_moe · (1 + Kw/w)`, using `Kw=772.7` FIXED from
the same independent 8-point sweep (not re-fit, so still only 3 free
parameters total: `RATIO`, `BASE_OVERHEAD_SEC`, `τ_base_moe`). This is a
structurally different mechanism from 5b's attempt -- it doesn't touch
how non-MoE bytes are priced at all, only adds an extra per-MoE-layer
term.

Tested against the current best 9-point set (2.35%/5.0% mean/max):
result was **4.90% mean / 10.19% max -- worse, a third confirmed
negative result.** The two genuinely narrow-expert points did improve
(`Qwen3-Coder-30B-A3B`: -5.0% -> -1.7%; `gemma-4-26b-a4b`: +3.6% ->
+3.5%, about even), but every other point got meaningfully worse
(`Huihui-LFM2.5-1.2B`: -2.1% -> -9.2%; `LFM2-8B-A1B`: +3.3% -> +10.2%;
`granite-4.0-h-tiny`: -1.4% -> -5.4%; `NVIDIA-Nemotron-3-Nano`: +2.0% ->
+4.3%). **Not adopted.**

Three independent attempts now (5b's multiplicative bandwidth discount,
the 4-parameter freely-fit separate-MoE-ratio check in the same section,
and this additive dispatch term) have all failed to net-improve this
9-point set using the real, verified `eta(w)`/`Kw` sweep data. This is
now a reasonably solid conclusion, not just an unlucky first attempt:
whatever the narrow-MoE residual in `Qwen3-Coder-30B-A3B` and
`gemma-4-26b-a4b` actually is, it either isn't cleanly the same
mechanism the synthetic sweep measured, or isolating it needs
meaningfully more real data than 9 points to avoid trading error onto
the other 7. Not worth a fourth attempt without new ground truth.

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
even smaller/faster model. **UPDATE (see item 10): `Huihui-LFM2.5-1.2B`
turned out NOT to be an instance of this after all** -- it had a real,
unrelated, fixable MLP-width bug instead. The reasoning immediately
below about `Josiefied-Qwen3.5-0.8B` (and later `mamba-130m`, item 9)
still stands on its own; only the specific claim that LFM2.5-1.2B was a
second example of the same phenomenon was wrong. **A single global
`BASE_OVERHEAD_SEC` cannot
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

### 7b. Third fast-model point added; layer-count hypothesis tested and found insufficient

Downloaded `mlx-community/SmolLM-135M-4bit` (dense llama-family, 135M
params, 30 layers) as the third point item 7 called for -- real 564.86
tok/s (10-trial mean, std 3.51% relative, noisier than this project's
usual sub-1% since even small jitter is a bigger fraction of a ~1.8ms
per-token budget at this speed).

Tested the most obvious candidate variable from the list above: replace
flat `BASE_OVERHEAD_SEC` with `PER_LAYER_OVERHEAD_SEC * n_layers`, fit
across all 12 real `mlx` measurements. Result is a genuine, real mixed
picture, not a clean win or a clean loss:
- The 9-point core set plus two moderate-speed additions
  (`Youssofal/Qwen3.6-35B-A3B-*`, `Youtu-LLM-2B`) fit well: -6.2% to
  +5.3%, including +1.6% and +3.9% on the two additions.
- The three most extreme points remain badly wrong, and don't even
  agree with each other on direction despite similar layer counts
  (24-30): `Josiefied-Qwen3.5-0.8B` +15.5%, `mamba-130m` -30.3%, this
  new `SmolLM-135M` point +39.4%. A layer-count-only model cannot
  reconcile a model it under-predicts with two others of similar depth
  that it over-predicts -- something else (quantized 4-bit dispatch
  path vs. `mamba-130m`'s unquantized bf16, or MLP-width-specific
  overhead) plausibly differs between them that layer count alone
  doesn't capture.
- Isolated to just the original 9-point core set (dropping the 5
  held-out/extreme additions), the per-layer reformulation gives
  2.64%/6.59% mean/max -- slightly WORSE than the current flat model's
  2.35%/5.0%.

**Not adopted.** It doesn't clearly beat the safest, most-validated
9-point fit, and doesn't fully solve the problem it was meant to solve
either.

**A dequantization-overhead hypothesis was formed here, then directly
tested and refuted -- worth recording precisely to avoid re-proposing
it.** The initial reading of the 3-point pattern (`Josiefied` and
`SmolLM-135M`, both 4-bit, over-predicted; `mamba-130m`, unquantized
bf16, under-predicted) suggested per-group scale/bias dequantization
might be a real missing per-layer cost. This is directly testable: this
project's own cache already had `mlx-community/SmolLM-135M-fp16`, the
SAME architecture as the existing 4-bit point with quantization as the
only real difference. Downloaded and measured it: real 488.79 tok/s
(10-trial mean, std 2.81% relative). Checked against the current
*production* (flat-overhead) formula for a fair, consistent comparison
across all four extreme points:

    Josiefied-Qwen3.5-0.8B (4-bit):      real 339.7   pred 457.6   +34.7%
    mamba-130m (bf16, pure SSM):         real 490.8   pred 388.1   -20.9%
    SmolLM-135M-4bit:                    real 564.9   pred 1502.3  +166.0%
    SmolLM-135M-fp16 (unquantized):      real 488.8   pred 660.5   +35.1%

**The dequantization hypothesis is refuted:** `SmolLM-135M-fp16` has NO
dequantization step at all, yet it's over-predicted in the SAME
direction and roughly the SAME magnitude as its own 4-bit sibling's
fp16-comparable competitor (`Josiefied`, +34.7%) -- if dequant overhead
were the differentiator, the unquantized variant should have looked
like `mamba-130m`, not like the quantized models. It didn't.

**A cleaner, still-unconfirmed pattern emerges instead:** the odd one
out isn't "unquantized" -- it's `mamba-130m` specifically, which is
also the only PURE-SSM (no attention at all, custom Metal scan kernel)
architecture among the four. The three attention-based small/fast
transformers (`Josiefied`, both `SmolLM` variants) are ALL
over-predicted, of varying magnitude (34.7% to 166.0%) but the same
sign; the one non-attention architecture is under-predicted instead.
This suggests the small/fast-model problem may really be two separate
phenomena bundled together -- a general "fixed overhead too high (or
too low) at extreme speed" effect for attention transformers, and a
architecturally distinct effect for pure-SSM decode -- rather than one
effect explainable by a single variable (layer count, quantization
state, or otherwise). Not confirmed: only one pure-SSM fast point
exists. All four extreme points remain in `measurements.json` as real
ground truth for whenever this gets properly resolved.

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

## 9. Classic (non-hybrid) Mamba-1 had zero ground-truth coverage and two real weight-count bugs -- RESOLVED

Every SSM-family calibration point so far (`Qwen3.6-35B-A3B-4bit`,
`LFM2-8B-A1B`, `Huihui-LFM2.5-1.2B`, `granite-4.0-h-tiny`,
`NVIDIA-Nemotron-3-Nano-30B-A3B`, `Josiefied-Qwen3.5-0.8B`) is a
**hybrid** -- attention mixed with GatedDeltaNet/ShortConv/Mamba-2 layer
by layer. Downloaded `mlx-community/mamba-130m-hf-bf16` (real classic
Mamba-1, 130M params, non-hybrid, non-DeepSeek) as the first-ever
pure-SSM ground-truth point, and found two real weight-count bugs
verified against `mlx_lm/models/mamba.py` directly:

1. **Detection gap:** `_analyze()`'s layer_kinds logic only recognizes
   SSM layers via a `layer_types` list, `sliding_window`, or bailing's
   `layer_group_size` field -- a classic Mamba config has NONE of these
   (no `num_attention_heads` field at all, since there's no attention),
   so every layer fell through to the generic "full attention" default
   and got priced as a bare `4*hidden*hidden` block (no real
   heads/kv_heads/head_dim to build an actual formula from). Confirmed
   real: inflated this repo's true ~130M params to 218.8M (+68%). Fixed
   via `_is_pure_ssm_architecture()` (no attention heads + a known SSM
   formula matches -> every layer is that mixer type).
2. **Phantom per-layer MLP:** even after fixing detection, the generic
   per-layer loop still added a dense MLP (`3*hidden*intermediate_size`)
   on top of every SSM layer -- correct for GatedDeltaNet/LFM2 hybrids
   (confirmed each pairs its mixer with a real, separate MLP: read
   `qwen3_next.py`/`lfm2_moe.py` directly), but wrong for classic Mamba:
   `mlx_lm.models.mamba.ResidualBlock.__init__` has ONLY `self.mixer`
   and `self.norm` -- no separate MLP exists at all (the mixer's own
   `in_proj`, expanded to `2*d_inner`, already does what a transformer's
   attention+MLP pair does). This bug happened to be invisible on inputs
   like this one where `intermediate_size` doubles as both Mamba's own
   `d_inner` field AND the generic formula's dense-MLP width, so it
   silently added a second, real-looking but nonexistent block. Fixed
   via `_pure_ssm_has_no_separate_mlp()` -- deliberately narrower than
   detection above, matching ONLY Mamba-1/Mamba-2 (not GatedDeltaNet/
   LFM2, which a synthetic all-GatedDeltaNet test config caught trying
   to wrongly exclude their real MLP too).

Combined effect on `estimate_total_params`: 218.8M -> 167.0M (bug 2
alone) -> 128.4M once a third, more general bug was also fixed (see
below) -- landing within 1.2% of the model's real ~130M name.

**A third, more general bug found alongside these:** `tie_word_embeddings`
was never read anywhere in `size_estimate.py` -- every total-params
estimate assumed a separate `lm_head` matrix even for repos that tie it
to the embedding table (a common practice, not unique to Mamba).
Confirmed real via this repo's own safetensors index: only
`backbone.embeddings.weight` exists, no separate `lm_head` weight at
all, despite `config.json` having no explicit `tie_word_embeddings`
field (classic Mamba ties by hardcoded architectural convention, not a
per-repo config choice). Fixed via `_embedding_multiplier()`: honors an
explicit `tie_word_embeddings` field when present, else checks a small
hardcoded list of known-always-tied `model_type`s, else defaults to
untied (2x) as before -- **this only affects RAM sizing
(`estimate_total_params`/`estimate_bytes`/`fits_in_ram`), not the speed
formula**, which already counted `lm_head` bytes exactly once regardless
of tying.

**Speed-formula result:** even with both weight-count bugs fixed,
`probe_formula.py` still predicts this model at +74.6% error (real 490.8
tok/s over 5 trials, std 0.49%; formula predicts 856.9). This is NOT a
new problem -- it's a second, independent confirmation of item 7's
"small/fast models don't fit the same global fixed-overhead term"
finding (previously seen only on `Josiefied-Qwen3.5-0.8B` -- see item 10
below for why `Huihui-LFM2.5-1.2B` turned out NOT to actually be an
instance of this after all), now reproduced on a completely different
architecture family (classic Mamba, not GatedDeltaNet) and a completely
different quantization state (unquantized bf16, not 4/8-bit) --
strengthening the case that this really is a property of very fast (>300
tok/s) models specifically, not one architecture or quantization scheme.
Not included in the 9-point calibration set for the same reason
`Josiefied-Qwen3.5-0.8B` isn't (see item 7).

## 10. LFM2's `block_auto_adjust_ff_dim`: real MLP width is 50% smaller than declared -- RESOLVED, single most impactful fix found

`Huihui-LFM2.5-1.2B-Instruct-abliterated-8bit` had been this
calibration set's worst-or-near-worst point across every refit in this
entire investigation (-14.3% to -23.3% depending which other fixes were
already in place), and was written off in item 7 and in
`probe_formula.py`'s own docstring as a third instance of the
"small/fast model breaks the global fixed-overhead term" problem. It
was not. Reading `mlx_lm/models/lfm2.py`'s `MLP.__init__` directly found
a real, previously-missed bug: this architecture declares
`intermediate_size`/`block_ff_dim` = 12288 in config.json, but the real
model does NOT use that value directly when `block_auto_adjust_ff_dim`
is set (true in this real cached config) -- it recomputes a LLaMA-style
SwiGLU width from it:

    ff_dim = int(2 * block_ff_dim / 3)               # 12288 -> 8192
    ff_dim = int(block_ffn_dim_multiplier * ff_dim)   # if set (1.0 here, no-op)
    ff_dim = block_multiple_of * ceil(ff_dim / block_multiple_of)  # round up

landing on a REAL matrix width of 8192, not the declared 12288 --
`size_estimate.py`'s generic formula, which just read `intermediate_size`
directly, overcounted this architecture's entire dense-MLP byte budget
by 50%. Confirmed this only affects the DENSE `lfm2` model_type
specifically: `lfm2_moe.py`'s `MLP` class (used by `LFM2-8B-A1B`, a
different real calibration point) takes `intermediate_size` directly
with no such recompute at all, read directly to confirm before scoping
the fix narrowly. Implemented as `_lfm2_dense_mlp_width()` in
`size_estimate.py`, gated on `model_type == "lfm2"`.

Effect, after a full 9-point refit: `Huihui-LFM2.5-1.2B`'s own error
dropped from -14.3% to **-2.2%**, and the whole calibration set's mean
error dropped from 5.9% to **2.7%** (max 19.8% -> **6.3%**) -- every
single point in the 9-point set now sits within 6.3% of real, the
tightest this project's formula has ever been, and the first time it has
landed at or inside the "97% accuracy" (3% error) question that started
the whole `mlx-97-percent-research-prompt.md` investigation (on this
in-sample set -- not yet independently confirmed against new held-out
ground truth beyond what's already in this document).

**The general lesson, worth restating:** a persistently bad-fit point
that already has a plausible-sounding explanation (here, "it's just a
small/fast model, same as two other real ground-truth points") should
still be re-suspected for an actual bug via a direct source read before
being accepted as an inherent modeling limit -- especially when, in
hindsight, the "explanation" was pattern-matching on the wrong shared
property (LFM2.5-1.2B's real decode speed, 176.6 tok/s, was never
actually in the same regime as the two genuinely fast points, 339.7 and
490.8 tok/s, that the pattern was drawn from).

## 11. `gemma-4-26b-a4b-it-4bit`'s recorded ground truth was itself stale -- RESOLVED

Same general lesson as item 10, applied to the MEASUREMENT side instead
of the formula side. `gemma-4-26b-a4b-it-4bit` remained this
calibration set's worst point (+3.6% to +6.7% depending which other
fixes were in place) even after its real structural bug (item 8,
parallel dense+MoE) was fixed. The residual was attributed to the same
width-dependent MoE bandwidth-occupancy effect documented in items 5/5b
(its `moe_intermediate_size=704` is narrow) -- a real, source-verified
mechanism, so a plausible explanation on its face.

Re-measuring it directly (a fresh 10-trial `mlx_lm.generate` run,
prompted by wanting a second real data point to test that explanation)
found the recorded ground truth itself was off: originally 76.7 tok/s
(kept in `measurements.json` with no provenance beyond "thinking mode
enabled" -- likely a single run, or from an earlier, less careful
measurement pass in this project's history), vs. a rigorous, tightly-
clustered new measurement of **79.1 tok/s** (std 0.60, 0.76% relative
over 10 trials). Refitting with the corrected value dropped this
point's own error from +6.3% to **+3.6%**, and the whole 9-point set's
mean from 2.7%/6.3% to **2.35%/5.0%**.

**The lesson, generalized from item 10:** re-suspecting a bad fit for a
FORMULA bug (item 10) is only half of it -- the GROUND TRUTH the formula
is being measured against needs the same scrutiny before being trusted
as a fixed target. A single number sitting in `measurements.json` with
thin provenance ("thinking mode enabled" and nothing else) is not
automatically more trustworthy than a formula prediction; re-measuring
it with this project's own established rigor (10+ trials, `std`/CV
reported) is cheap and should be the default before accepting a
persistent gap as evidence of a missing formula mechanism.

`Qwen3-Coder-30B-A3B-Instruct-4bit` is now the nominal worst point
(-5.0%) and has NOT yet been re-measured with this same scrutiny since
early in this session (it was re-measured once already, for the noise-
floor work in item 4, and that measurement -- 90.5 tok/s, std 0.66,
0.73% relative -- was already rigorous) -- its residual is more likely a
genuine narrow-MoE-width effect (`moe_intermediate_size=768`) than a
stale ground truth, since its own measurement is already known-clean,
but this has not been independently re-verified as thoroughly as
`_gemma4_estimate()`'s or `_lfm2_dense_mlp_width()`'s source-level
audits were.

## 12. First real MLA ground truth -- weight formula confirmed exact, KV-cache assumption found wrong for at least one implementation, +16.8% residual unexplained

Downloaded `mlx-community/Youtu-LLM-2B-mlx-4bit` (real MLA, 2B params,
non-DeepSeek) -- this project's first actual MLA ground-truth
measurement (item D in `mlx-97-percent-research-prompt.md` had flagged
this as a coverage gap; the specific repos a later research pass
suggested for it, `Falcon-H1-Tiny-R-0.6B` and `sarvamai/sarvam-30b`,
were checked and neither actually has `kv_lora_rank`/`qk_rope_head_dim`
fields at all -- another confirmed research fabrication, this repo was
found independently by searching this project's own cached configs).
Real: 165.1 tok/s (10-trial mean, std 1.45, 0.88% relative). Formula
predicts 192.9 -- **+16.8% error**, a real, meaningful residual.

Applied the same source-verification rigor as items 8/10/11 to check
for a weight-count bug first: read `mlx_lm/models/youtu_llm.py`'s
`YoutuLLMAttention.__init__` line by line against this project's
`_mla_weight_params()`. Every term matches exactly --
`q_a_proj`+`q_b_proj` (or a direct `q_proj` when `q_lora_rank` is unset),
`kv_a_proj_with_mqa`+`kv_b_proj`, and `o_proj` all agree with the
formula term-for-term. **Not a weight-formula bug.**

Checked the KV-cache-size assumption next, since MLA's whole point is a
compressed cache, and found something genuinely subtle: `youtu_llm.py`
computes `kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))`
(the FULL per-head decompression) BEFORE calling
`cache.update_and_fetch` -- it caches the decompressed per-head K/V,
not the compressed latent. Compared against
`mlx_lm/models/deepseek_v3.py`'s `DeepseekV3Attention` (read directly,
no DeepSeek model downloaded or run -- reading already-installed
library source code, not testing a DeepSeek model): it caches
`kv_latent`/`k_pe` (the COMPRESSED form) via
`cache.update_and_fetch(kv_latent, k_pe)`, deferring the `kv_b_proj`
up-projection into the attention-score computation itself -- the real
MLA inference-efficiency trick the architecture is known for. **These
are two genuinely different implementations of "MLA" in mlx-lm itself**
-- config.json's `kv_lora_rank`/`qk_rope_head_dim` fields can't
distinguish them, since both declare the same fields. This project's
`kv_elems_per_token_per_attn_layer = kv_lora_rank + qk_rope_head_dim`
assumption is correct for DeepSeek-V3-style MLA but WRONG for
youtu_llm-style MLA (real per-token KV read there is closer to
`num_heads * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)`, no
compression at all).

**This does NOT explain the +16.8% error at `context_length=115`** --
at that short a context, KV-cache bytes for a 2B model are negligible
next to weight bytes regardless of which formula is used (checked: both
the compressed and per-head estimate are under 40MB, vs. ~1.1GB of
active weight bytes). It IS a real, separate finding for this project's
own long-context degradation claims, which previously assumed every MLA
model gets DeepSeek-V3's compression benefit -- untrue for at least this
one architecture. **RESOLVED:** checked `mlx_lm/models/glm4_moe_lite.py`
too (this project's other cached MLA family, GLM) and confirmed it uses
the compressed-cache pattern (`cache.update_and_fetch(kv_latent, k_pe)`,
same as DeepSeek-V3) -- the decompressed variant looks genuinely rare
(1 confirmed case so far), which is why a small, explicit
`_MLA_DECOMPRESSED_CACHE_MODEL_TYPES` set (currently just `{"youtu_llm"}`)
was the right shape for this fix rather than a heuristic: everything not
explicitly listed keeps the compressed-cache assumption, and a real new
"decompressed" architecture just needs one line added once found. Wired
into `_analyze()`'s KV-cache-size branch, verified against the real
Youtu-LLM-2B config (correctly computes 37.7MB at context_length=115,
matching the hand-derived estimate above) and covered by a new
regression test.

**Still open:** the +16.8% short-context error itself remains
unexplained after ruling out both obvious candidates (weight count,
KV-cache size). At 165.1 tok/s this model is close to but still below
the fastest already-well-fit calibration point (`LFM2-8B-A1B`, 192
tok/s) -- plausibly a milder version of the small/fast-model problem
(item 7), plausibly something else specific to this architecture or
checkpoint. Not enough evidence yet to say which; would need either
another real MLA point at a different speed, or the same kind of deep
timing investigation (chained-layer micro-benchmark, or a real Metal
System Trace) already applied to the MoE-width and hybrid-deficit
questions elsewhere in this document.

## 13. Long-context KV-cache scaling -- dense case validated, hybrid case reveals a real formula limitation, one specific research claim debunked

Directly measured real decode-throughput degradation as context grows,
using two already-established real ground-truth models at
`L ∈ {128, 2048, 8192, 16384}` (3-trial means, discarding none --
all runs were clean and low-variance):

    Qwen2.5-Coder-7B-Instruct-4bit (dense): 57.28 -> 54.79 -> 49.40 -> 43.79 tok/s (-23.5% total)
    Qwen3.6-35B-A3B-4bit (hybrid, 10/40 full-attn): 90.05 -> 87.46 -> 81.25 -> 74.26 tok/s (-17.5% total)

A research pass had claimed the hybrid model's degradation rate would
be "exactly 25%" of the dense rate, matching the 10/40 full-attention
layer ratio. **This specific claim is wrong** -- the real ratio is
17.5/23.5 = **74.5%**, roughly 3x the claimed value. The qualitative
direction (hybrid degrades less than dense) is correct, but the
"exactly matches the full-attention layer fraction" mechanism is not:
every layer still does real per-step compute regardless of whether its
KV-cache grows, so a naive "10/40 of the bytes -> 10/40 of the slowdown"
model was never going to hold.

Checked the formula's own predictions against both real curves:
- **Dense case: accurate.** Formula: 56.20/54.70/50.40/45.70 vs. real
  57.28/54.79/49.40/43.79 -- errors of -1.9%/-0.2%/+2.0%/+4.4%, staying
  under 5% even out to 16K context.
- **Hybrid case: a real, growing gap.** Formula: 90.90/89.50/85.30/80.20
  vs. real 90.05/87.46/81.25/74.26 -- errors of +0.9%/+2.3%/+5.0%/+8.0%,
  growing steadily worse with context. The formula predicts a milder
  degradation (90.90->80.20, -11.8%) than reality (-17.5%) shows.

**Root cause, most likely:** this project's whole speed model is
memory-bandwidth-only (`tok/s = bandwidth / bytes_per_token`) -- it
counts KV-cache bytes READ per step but has no term at all for
attention's own compute cost (softmax + weighted sum over L cached
keys), which grows with context independently of memory bandwidth.
At short context this is negligible next to bandwidth-bound weight
streaming, which is why the formula is accurate at `context_length=115`
(this project's own calibration default) and even out to L=2048. By
L=16384, real attention FLOPs in the 10 full-attention layers are
plausibly no longer negligible, and the formula -- having no compute
term to grow with them -- systematically under-predicts the real
slowdown.

**Characterized the residual's shape before considering a fix, given
this project's own track record on adding structure without enough
data** (three separate MoE-width integration attempts all failed net-
improvement, see items 5b/5c). Converting tok/s back to raw per-token
time and subtracting the formula's own prediction:

    L      real time    formula time   residual
    128    11.105ms     11.001ms       +0.104ms
    2048   11.434ms     11.173ms       +0.261ms
    8192   12.308ms     11.723ms       +0.584ms
    16384  13.466ms     12.469ms       +0.997ms

The residual grows roughly with `L` (a 2-point linear fit through the
first and last rows predicts the middle two within 6-24%, not exact but
directionally consistent with an attention-FLOPs mechanism, which
should scale linearly in context length) -- this is real, qualitative
support for the "missing compute roofline" explanation above, not just
a plausible-sounding story.

**Deliberately NOT hard-coded into the formula.** This entire curve
comes from ONE model (`Qwen3.6-35B-A3B-4bit`) at 4 context lengths,
3 trials each -- far short of what would be needed to responsibly fit a
new global parameter (this project's own stated bar, already invoked
against itself three times this session, is 5+ points per architectural
family; here it would be fitting a coefficient from a single point in
an even narrower "family" of one). Adding a FLOPs term calibrated on
this alone would very plausibly repeat the exact failure mode items
5b/5c already hit twice: a new parameter that fits its one motivating
case while quietly making every other point worse. **What would
actually justify implementing it:** long-context curves (the same
4-context-length sweep already run here) on at least 2-3 more hybrid
models with different full-attention-layer counts/head_dims, to see
whether a single FLOPs-based coefficient generalizes across them before
trusting it. Until then, the honest, low-risk fix is a CLI-level
caveat rather than a formula change:

    if context_length > 2048 and model_has_full_attention_layers:
        warn("accuracy beyond L=2048 is not independently validated for "
             "hybrid attention architectures; expect underestimated "
             "slowdown -- see docs/formula-accuracy-gap.md item 13")

Not implemented as an actual CLI change in this pass (the CLI's output
plumbing wasn't touched this session) -- recorded here as the concrete,
scoped, low-risk next step, in contrast to the new formula parameter
that isn't justified by the data in hand yet.
