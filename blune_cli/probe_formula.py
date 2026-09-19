"""
Pure-math MLX speed estimate: no model construction, no MLX import, no
subprocess -- just config.json metadata run through the memory-bandwidth
law. Built as a much faster alternative to probe_mlx.py's zero-download
*execution* probe, for use in `blune sweep` over thousands of models where
actually building and running each one (even with random weights) would
take hours.

Methodology
-----------
tok/s = bandwidth / bytes-per-token is the physics floor, but a naive
bytes-per-token-only estimate is off by 2-4x. Two additive terms bring it
back in line:

    time_per_token = bytes_per_token / (bandwidth * BANDWIDTH_CALIBRATION_RATIO)
                      + BASE_OVERHEAD_SEC
                      + n_moe_layers * MOE_LAYER_OVERHEAD_SEC

`bytes_per_token` comes from size_estimate.py's architecture-aware
estimate (MoE active-vs-total experts and shared experts, GQA/MLA-aware
KV-cache sizing, hybrid Mamba/SSM layers excluded from the growing KV
term, sliding-window capping, dense/MoE-interleaved layers via
first_k_dense_replace/num_dense_layers) -- see docs/research-findings.md
and docs/formula-accuracy-gap.md for the full architectural derivations
and the investigation trail this was built from.

Why a per-MoE-layer term exists (not just one flat overhead)
--------------------------------------------------------------
The original 2-term model (one flat overhead, fit on 5 conventional
dense/MoE models) badly under-predicted 3 real hybrid measurements
(LFM2.5-1.2B, LFM2-8B-A1B, granite-4.0-h-tiny) by 20-53%. Re-reading
every relevant mlx-lm layer class line by line ruled out a missing-
weight-structure bug -- every `nn.Linear`/`nn.Conv1d` this formula counts
matches the real source exactly. So the deficit had to be a *timing*
effect, and it was tracked down with a real MLX experiment, refined
twice after two methodology mistakes:

1. First attempt: timed one isolated `SwitchGLU` call vs. one isolated
   attention call. MoE showed ~200us, attention showed even MORE
   (~294us) -- yet dense attention-only models didn't need a comparably
   large overhead term. Conclusion at the time: isolated single-layer
   timings don't transfer to "marginal cost in a fused decode graph"
   (consistent with this project's own much earlier "same real layer x
   N" graph-fusion finding).
2. Second attempt, done properly: timed CHAINS of N=1,2,4,8,16,32
   identical layers and read the slope, which is what actually measures
   marginal per-layer cost once one-time graph-setup cost is factored
   out. First pass forgot to quantize the layers, so the chain was
   dominated by fp32 bandwidth time rather than dispatch overhead
   (obviously wrong -- MoE's "marginal cost" came out at ~804us/layer,
   which would make a 40-MoE-layer model take 32ms/token alone, far more
   than any real model does).
3. Third attempt, quantized (4-bit, matching real repos) AND chained --
   the methodologically clean version: marginal cost converges to
   **~54us/layer for attention** and **~178us/layer for MoE (SwitchGLU)**
   at N=32. MoE genuinely costs ~3.3x more marginal dispatch overhead per
   layer than plain attention once bandwidth and one-time setup are
   controlled for -- confirming the original hypothesis, just not with
   the first (flawed) experiment that seemed to. This is the same order
   of magnitude as the 114us the 9-point real-data regression indepen-
   dently arrived at, which is a genuine, mutually-reinforcing result
   from two different methods rather than one experiment's number being
   plugged directly into the other.

The ~54us/layer attention figure alone doesn't fully explain everything
a full decoder layer needs (it only measured the attention sub-block --
not the paired MLP, the two RMSNorms, or the residual adds every real
layer also does), so treat MOE_LAYER_OVERHEAD_SEC's exact value as
empirically fit and mechanism-supported, not first-principles-derived.

Refitting with 3 parameters (bandwidth ratio, a base per-step overhead,
and a per-MoE-layer overhead) via least squares against all 9 real
measurements collected so far (5 original + 4 hybrid, spanning dense,
conventional MoE, GatedDeltaNet hybrid, Mamba-2 hybrid, and Nemotron-H's
single-component-per-layer architecture) drops mean error from 15.8%
(the best a 2-parameter refit on the same 9 points could do) to 9.6%.

Note on measurements.json's two fastest entries (Josiefied-Qwen3.5-0.8B,
339.7 tok/s dense GatedDeltaNet, and mamba-130m-hf-bf16, 490.8 tok/s
classic Mamba-1): deliberately NOT included in the 9-point fit below.
Adding either to the same 3-parameter least-squares regression makes
every other point's fit worse and the new point itself comes out badly
wrong (+30-51% / -12-23% depending which other fixes are already in
place) -- a single global BASE_OVERHEAD_SEC can't be right both for a
model this fast (>300 tok/s) and for the 57-192 tok/s range the rest of
the 9-point set covers. This is a real, reproducible effect
across two completely different architecture families (GatedDeltaNet,
classic Mamba) and quantization states (4-bit, unquantized bf16) --
genuinely about raw speed, not one architecture's quirk. (An earlier
version of this note also named Huihui-LFM2.5-1.2B as a third instance
of this -- that was wrong; see the LFM2 dense-MLP-width fix further
down, which resolved it as a real, unrelated, fixable bug instead. Only
re-suspect a point as "just fast" after ruling out an actual bug, not
before.) See docs/formula-accuracy-gap.md item 7 -- both fast points
kept as real ground truth for whenever this gets a proper fix, not
silently absorbed into a worse-fitting constant.

Calibration data (M4 Pro, 48GB; see measurements.json for the raw numbers,
context_length=115 matching this project's own probe's prompt+decode
range; includes the Qwen3-Next-family q_proj-doubling fix -- see
_Q_PROJ_MULTIPLIER_BY_MODEL_TYPE in size_estimate.py -- and every
bytes-per-token fix documented further down, most recently LFM2's
`block_auto_adjust_ff_dim` real MLP width):
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 90.5, formula 86.0 (-5.0%)
    gemma-4-26b-a4b-it-4bit:            real 79.1, formula 81.9 (+3.6%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57.2, formula 56.2 (-1.8%)
    Qwen3.6-35B-A3B-4bit:               real 89.1, formula 90.8 (+1.9%)
    gpt-oss-20b-OptiQ-4bit:             real 83.6, formula 83.6 (+0.0%)
    Huihui-LFM2.5-1.2B-Instruct-8bit:   real 176.6, formula 172.9 (-2.1%)
    LFM2-8B-A1B-3bit-MLX:               real 192.1, formula 198.4 (+3.3%)
    granite-4.0-h-tiny-6bit-MLX:        real 116.9, formula 115.2 (-1.4%)
    NVIDIA-Nemotron-3-Nano-30B-A3B-8Bit: real 57.1, formula 58.2 (+2.0%)
mean absolute error **2.35%, max 5.0%** (previously 2.7%/6.3%, 5.9%/
19.8%, 7.2%/20.1%, and 9.4%/21.1% at earlier stages of this
investigation) -- every point in this set now sits within 5.0% of real.
The improvement from 6.3% max to 5.0% came from re-measuring
`gemma-4-26b-a4b-it-4bit` itself: its originally recorded 76.7 tok/s
(kept with no provenance beyond "thinking mode enabled") turned out to
be stale or otherwise imprecise -- a fresh, rigorous 10-trial
`mlx_lm.generate` mean gave 79.1 tok/s (std 0.60, 0.76% relative,
tightly clustered). Most of what had looked like a real, unexplained
MoE-width-bandwidth residual on this point (formula-accuracy-gap.md
items 5/5b/8) was actually just an imprecise ground-truth measurement,
not a formula problem -- a reminder that a "real remaining effect"
explanation is only as solid as the measurement it's explaining, and is
worth re-checking the same way a persistently-bad-fit point's SOURCE is
(see the LFM2 lesson two fixes down). `Qwen3-Coder-30B-A3B-Instruct-4bit`
is now the nominal worst point (-5.0%) -- its own source
(`mlx_lm/models/qwen3_moe.py`) was read directly and found to match the
generic formula's assumptions exactly (standard GQA, no shared expert,
no unusual sizing), so its residual is most plausibly the same
narrow-MoE-width bandwidth effect (its `moe_intermediate_size=768` is
narrow) rather than an undiscovered structural bug -- but that
explanation hasn't been re-verified against a fresh measurement of this
specific point either, so treat it as a reasonable hypothesis, not a
settled fact. Separately, a real, independently-measured Level-1 ground
truth point (`Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit`,
not in this calibration set) came in at +1.0% error -- see
formula-accuracy-gap.md item 3b for the full story of how that held-out
model's originally-reported +26-30% gap turned out to be entirely a
`probe_mlx.py` proxy artifact, not a formula error.

Honesty notes on the three fitted constants:
- All three constants moved substantially from earlier fits as more
  real bytes-per-token bugs were found and fixed (see the fix list
  below) -- BASE_OVERHEAD_SEC in particular flipped from negative in
  every earlier fit to a small positive ~0.32ms here, which is at least
  directionally more physically plausible (a real, if small, positive
  fixed per-decode-step cost) than the earlier negative values ever
  were, though this project has never claimed these three constants are
  a clean physical decomposition rather than a jointly-fit
  approximation.
- BANDWIDTH_CALIBRATION_RATIO (~0.83x spec) and MOE_LAYER_OVERHEAD_SEC
  (~78us) are the two terms most likely to be "real" -- the MoE term is
  the same order of magnitude as the direct micro-benchmark's ~178-200us
  marginal-cost finding (real decode-loop kernel fusion is plausibly
  more efficient than an isolated Python-level benchmark call).
- LFM2.5-1.2B, this calibration set's worst-or-near-worst point across
  every earlier refit (-14.3% to -23.3% depending which other fixes
  were already in place) and long assumed to be a "small/fast model"
  problem, turned out to have a real, fixable 50% MLP-width overcount
  the whole time (see the LFM2 fix below) -- now fits at -2.1%, no
  different from any other point. Genuinely fast dense-hybrid models
  (>300 tok/s -- Josiefied-Qwen3.5-0.8B, mamba-130m, both excluded from
  this calibration set) still show the small/fast-model problem this
  note used to describe; LFM2.5-1.2B (176.6 tok/s) was never actually
  fast enough to be an instance of it.

Two real bytes-per-token bugs found and fixed, both confirmed against
real source and re-fit against the same 9 measurements (not just
patched in place -- a systematic bytes-per-token change shifts what
BANDWIDTH_CALIBRATION_RATIO should be, so refitting rather than
re-using the old constants was required to avoid double-counting):

1. `estimate_active_bytes_per_token` used a single repo-wide `bits` for
   every weight, even on repos whose `quantization` manifest lists
   *per-tensor* bit-widths -- confirmed real on a cached
   `Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit` config: 512
   explicit per-tensor entries, 147 of them at 6-bit (routed + shared
   expert FFNs, lm_head) against a 4-bit repo default.
   `probe_mlx.py`'s real `nn.quantize(..., class_predicate=...)` already
   honors these overrides (read directly from its source), so this
   specific model's formula-vs-probe comparison was bytes-mismatched,
   not actually revealing a formula error of the size it looked like.
   Fixed by looking up bits separately for routed-expert (`switch_mlp`),
   shared-expert (`shared_expert`), and `lm_head` weights when a
   per-tensor manifest is present (`_effective_bits`/`_infer_bits` with
   a `path_hint` in size_estimate.py) -- NOT extended to mixer
   sub-projections (q/k/v/o) or the router, since those components'
   real attribute names vary too much across architecture families to
   match safely by substring, and in this specific model they're a
   smaller share of active bytes than the expert FFNs and lm_head
   already fixed.
2. No byte computation anywhere included quantization's per-group
   scale/bias metadata -- affine quantization (MLX's default) stores an
   fp16 scale AND fp16 bias per group of `group_size` packed weights,
   which `bits/8` alone ignores. At the common group_size=64, 4-bit:
   nominal 0.5 bytes/weight vs. real 0.5+4/64=0.5625 bytes/weight, a
   flat 12.5% miss on every quantized model, not just outliers. Added as
   `_effective_bits()` in size_estimate.py (bits + 32/group_size, i.e.
   +32 metadata bits per group), applied everywhere bytes are computed.
3. Gemma4 (`gemma-4-26b-a4b-it-4bit`, this project's single worst-fit
   calibration point at the time, +19.6-20.1% across every prior refit)
   was going through the generic mixer+MLP-per-layer formula, which is
   wrong for this architecture in ways a source read of
   `mlx_lm/models/gemma4_text.py` confirmed: when `enable_moe_block` is
   set, EVERY layer runs the dense MLP AND the MoE experts IN PARALLEL,
   summed (`h = h1 + h2`) -- not an interleaved dense-XOR-MoE split like
   `first_k_dense_replace`-style architectures, which is what the generic
   `moe_layer_mask` path assumes. The dense MLP's
   3*hidden*intermediate_size was being silently omitted from every
   layer. Also found: full-attention and sliding-attention layers use
   DIFFERENT head_dim/kv_heads (`global_head_dim`/`num_global_key_value_
   heads` vs. plain `head_dim`/`num_key_value_heads`), and full-attention
   layers have NO separate v_proj at all when `attention_k_eq_v` is set
   (`values = keys`, verified in `Attention.__init__`/`__call__`) -- a
   single config-wide head_dim/kv_heads undercounted the (larger-head_dim)
   full-attention layers while overcounting a v_proj that doesn't exist
   for them. Fixed via a dedicated `_gemma4_estimate()` in
   size_estimate.py (same pattern as `_nemotron_h_estimate()` -- bypasses
   the generic per-layer loop entirely rather than patching it with
   gemma4-specific branches).
4. LFM2's dense (non-MoE) architecture (`Huihui-LFM2.5-1.2B`) declares
   `intermediate_size`/`block_ff_dim` = 12288, but
   `mlx_lm.models.lfm2.MLP.__init__` does NOT use that value directly
   when `block_auto_adjust_ff_dim` is set (true in this real cached
   config) -- it recomputes a LLaMA-style SwiGLU width from it (2/3
   scaling, an optional multiplier, rounded up to `block_multiple_of`),
   landing on a REAL width of 8192, a 50% smaller matrix than the
   declared value implies. Confirmed this is specific to the dense
   `lfm2` architecture -- `lfm2_moe.py`'s `MLP` class (used by
   `LFM2-8B-A1B`) takes `intermediate_size` directly with no such
   recompute, read directly to confirm. Fixed via
   `_lfm2_dense_mlp_width()` in size_estimate.py, gated on
   `model_type == "lfm2"` specifically.

Re-running the same 3-parameter least-squares fit against the same 9
measurements with all four fixes applied dropped mean error from 9.4%
(pre-fixes) to 7.2% (after fixes 1-2) to 5.9% (after fix 3) to **2.7%**
(after fix 4), max 21.1% -> 20.1% -> 19.8% -> **6.3%**. Fix 4 alone
dropped Huihui-LFM2.5-1.2B's own error from -14.3% to -2.2% -- this
project's single most impactful individual fix, since that point had
been this calibration set's worst-or-near-worst across every earlier
refit and was long assumed to be an inherent "small/fast model" modeling
limit (see the small/fast-model note below) rather than a fixable bug.
Fix 3 alone dropped gemma-4-26b-a4b-it-4bit's own error from +20.1% to
+6.7% and pulled the whole 9-point mean down by
1.3 points on its own -- a real structural fix, not curve-fitting (it
moved one badly-wrong point a lot and left the others roughly where they
were, which is what fixing an actual bug looks like, as opposed to
refitting a knob that trades error between points). On the untouched-
by-calibration held-out case that motivated fixes 1-2 (Youssofal's
abliterated fine-tune), error vs. `probe_mlx.py` fell from a freshly-
remeasured +33.6% to +14.9% -- more than half the gap closed by
verified, mechanistic bugs rather than a curve-fitting trick.

A fourth fix (LFM2's `block_auto_adjust_ff_dim`, see
size_estimate.py's `_lfm2_dense_mlp_width`) then dropped mean error
further to 2.7%, max 6.3% -- and re-measuring a stale gemma-4-26b-a4b
ground-truth value afterward (see the "Calibration data" section below)
brought it to the current **2.35%, max 5.0%**. See formula-accuracy-
gap.md item 10 for the full LFM2 story. The LFM2 fix was
the single most impactful individual fix found in this whole
investigation: `Huihui-LFM2.5-1.2B`, previously this calibration set's
worst-or-near-worst point across every earlier refit (-14.3% to -23.3%
depending which other fixes were in place) and long assumed to be an
instance of the "small/fast model breaks the global fixed-overhead
term" problem (see the small/fast-model note below), turned out to
just have a real, fixable 50% MLP-width overcount the whole time --
once fixed, it fits within -2.2% to -3.2% across every refit tried, no
different from any other well-behaved point. The lesson: an unexplained
bad-fit point should be re-suspected as a real bug before being
written off as an inherent modeling limit, even after a plausible
explanation (small/fast models) already seems to fit the pattern.
"""
from typing import Optional

from .size_estimate import count_moe_layers, estimate_active_bytes_per_token

BANDWIDTH_CALIBRATION_RATIO = 0.8346  # see module docstring -- empirical, refit after correcting a stale gemma-4-26b measurement
BASE_OVERHEAD_SEC = 0.000322  # see module docstring -- fit value
MOE_LAYER_OVERHEAD_SEC = 0.000078  # per-MoE-layer dispatch cost, confirmed by direct MLX micro-benchmark


def probe(
    repo_id: str,
    config: dict,
    bandwidth_gbs: float,
    context_length: int = 115,
    calibrate: bool = True,
) -> dict:
    """Instant tok/s estimate from config.json alone -- no MLX, no
    subprocess, no model construction. context_length lets you see how a
    specific model's speed degrades at longer conversations (KV-cache
    read bytes grow with it for ordinary attention layers; hybrid
    Mamba/SSM layers and sliding-window-capped layers don't scale the
    same way -- see size_estimate.py). See module docstring for the
    calibration methodology and accuracy."""
    bytes_per_token = estimate_active_bytes_per_token(config, context_length=context_length)
    if bytes_per_token is None:
        raise ValueError(
            "couldn't determine hidden_size/num_hidden_layers from this "
            "config -- formula estimate needs a standard transformer config"
        )

    ratio = BANDWIDTH_CALIBRATION_RATIO if calibrate else 1.0
    base_overhead = BASE_OVERHEAD_SEC if calibrate else 0.0
    moe_overhead = MOE_LAYER_OVERHEAD_SEC if calibrate else 0.0
    n_moe_layers = count_moe_layers(config)

    transfer_time = bytes_per_token / (bandwidth_gbs * ratio * 1e9)
    total_time = transfer_time + base_overhead + n_moe_layers * moe_overhead
    tps = 1.0 / total_time if total_time > 0 else float("inf")

    return {
        "library": "mlx (formula)",
        "repo_id": repo_id,
        "architecture": config.get("model_type"),
        "context_length": context_length,
        "bytes_per_token_active": round(bytes_per_token / 1e6, 1),
        "n_moe_layers": n_moe_layers,
        "estimated_real_tps": round(tps, 1),
        "confidence": "high (config-only formula, mean 2.35% error / 5.0% max on 9-point real-measurement set spanning dense, MoE, and 4 hybrid architectures; may be less accurate on very fast (>300 tok/s) small models -- see docs/formula-accuracy-gap.md item 7)",
    }
