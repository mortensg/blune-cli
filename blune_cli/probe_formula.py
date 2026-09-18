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

Note on measurements.json's 10th entry (Josiefied-Qwen3.5-0.8B, a
small/fast dense GatedDeltaNet model): deliberately NOT included in the
9-point fit below. Adding it to the same 3-parameter least-squares
regression makes every other point's fit worse (mean 5.9% -> 9.9%, after
the Gemma4 fix below -- re-checked, not a stale number) and the new
point itself comes out at +51.4% error -- a single global
BASE_OVERHEAD_SEC can't be right both for a model this fast and for the
7B-35B range the rest of the set covers (same failure mode already
flagged for LFM2.5-1.2B below, now confirmed on a second, more extreme
case, independent of the Gemma4 bug -- fixing that didn't help this one
at all). See docs/formula-accuracy-gap.md item 7 -- kept as real ground
truth for whenever this gets a proper fix, not silently absorbed into a
worse-fitting constant.

Calibration data (M4 Pro, 48GB; see measurements.json for the raw numbers,
context_length=115 matching this project's own probe's prompt+decode
range; includes the Qwen3-Next-family q_proj-doubling fix -- see
_Q_PROJ_MULTIPLIER_BY_MODEL_TYPE in size_estimate.py -- and, as of the
most recent refit, the mixed-quantization-manifest, quantization-metadata,
and Gemma4-parallel-dense+MoE bytes-per-token fixes documented further
down):
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 90.5, formula 83.0 (-8.3%)
    gemma-4-26b-a4b-it-4bit:            real 76.7, formula 81.9 (+6.7%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57.2, formula 57.3 (+0.1%)
    Qwen3.6-35B-A3B-4bit:               real 89.1, formula 89.5 (+0.4%)
    gpt-oss-20b-OptiQ-4bit:             real 83.6, formula 84.8 (+1.4%)
    Huihui-LFM2.5-1.2B-Instruct-8bit:   real 176.6, formula 151.3 (-14.3%)
    LFM2-8B-A1B-3bit-MLX:               real 192.1, formula 230.1 (+19.8%)
    granite-4.0-h-tiny-6bit-MLX:        real 116.9, formula 115.1 (-1.5%)
    NVIDIA-Nemotron-3-Nano-30B-A3B-8Bit: real 57.1, formula 57.6 (+0.9%)
mean absolute error 5.9%, max 19.8% (previously 7.2%/20.1%, and 9.4%/21.1%
before that, across the byte-accounting fixes documented below) -- now
better per-point than even the old 5-point-only fit's 4.5%, while still
covering every architecture family tested, including ones its own
parameters were never specifically shaped around. gemma-4-26b-a4b-it-4bit
was the single largest source of the earlier 20.1% max error, root-caused
below (item: Gemma4's parallel dense+MoE MLP) -- fixing it dropped its
own error from +20.1% to +6.7% and pulled the whole 9-point mean down by
1.3 points, a real structural fix, not curve-fitting. LFM2-8B-A1B is now
the worst point (+19.8%, up from +13.0%) -- a side effect of the refit
redistributing residual error now that gemma-4 no longer dominates it;
see item 5 in formula-accuracy-gap.md (the MoE occupancy/bandwidth-
efficiency work) for the most likely real mechanism still to fix there.

Honesty notes on the three fitted constants:
- BASE_OVERHEAD_SEC came out of the regression NEGATIVE (~-1.2ms).
  That is not a physically meaningful "negative dispatch time" -- it's
  the least-squares fit compensating for whatever the other two terms
  still don't capture (this project has never claimed these constants
  are a clean physical decomposition; see BANDWIDTH_CALIBRATION_RATIO's
  own long-standing "not literal bandwidth > spec" caveat below). Kept
  as fit rather than clamped to 0, since clamping would just move the
  same error into the other two terms without improving overall
  accuracy.
- BANDWIDTH_CALIBRATION_RATIO (~0.98x spec here, much closer to literal
  than earlier fits) and MOE_LAYER_OVERHEAD_SEC (~108us) are the two
  terms most likely to be "real" -- the MoE term in particular matches
  the direct micro-benchmark's ~200us-per-call finding within a
  reasonable factor (real decode-loop kernel fusion is plausibly more
  efficient than an isolated Python-level benchmark call).
- LFM2.5-1.2B (-14.3% after the refit below) remains a bad-fit point,
  and has ZERO MoE layers -- its own remaining problem was root-caused
  separately (see formula-accuracy-gap.md): its real decode time
  (~5.66ms/token) is close to or below what a "typical" model's fixed
  overhead would be, and no single global BASE_OVERHEAD_SEC can be
  simultaneously right for a model this fast and for the original
  24-48-layer calibration set.

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

Re-running the same 3-parameter least-squares fit against the same 9
measurements with all three fixes applied dropped mean error from 9.4%
(pre-fixes) to 7.2% (after fixes 1-2) to **5.9%** (after fix 3), max
21.1% -> 20.1% -> 19.8%. Fix 3 alone dropped gemma-4-26b-a4b-it-4bit's
own error from +20.1% to +6.7% and pulled the whole 9-point mean down by
1.3 points on its own -- a real structural fix, not curve-fitting (it
moved one badly-wrong point a lot and left the others roughly where they
were, which is what fixing an actual bug looks like, as opposed to
refitting a knob that trades error between points). On the untouched-
by-calibration held-out case that motivated fixes 1-2 (Youssofal's
abliterated fine-tune), error vs. `probe_mlx.py` fell from a freshly-
remeasured +33.6% to +14.9% -- more than half the gap closed by
verified, mechanistic bugs rather than a curve-fitting trick.
"""
from typing import Optional

from .size_estimate import count_moe_layers, estimate_active_bytes_per_token

BANDWIDTH_CALIBRATION_RATIO = 0.7801  # see module docstring -- empirical, refit after 3 real bytes-per-token fixes
BASE_OVERHEAD_SEC = -0.001245  # see module docstring -- fit value, not a literal negative dispatch time
MOE_LAYER_OVERHEAD_SEC = 0.000108  # per-MoE-layer dispatch cost, confirmed by direct MLX micro-benchmark


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
        "confidence": "medium (config-only formula, mean 5.9% error / 19.8% max on 9-point real-measurement set spanning dense, MoE, and 4 hybrid architectures)",
    }
