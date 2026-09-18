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
matches the real source exactly.

An isolated MLX micro-benchmark (instantiating `mlx_lm.models
.switch_layers.SwitchGLU` with random weights, timing a single-token
batch=1 forward pass) showed ~200us of overhead per call, roughly
independent of expert count -- which motivated adding a per-MoE-layer
term, and doing so measurably improved the fit (15.8% -> 9.6% mean error
on all 9 real points). Take that mechanism with a grain of salt, though:
a follow-up benchmark of an ISOLATED standard attention layer showed
*more* isolated overhead (~294us) than the MoE layer did, yet dense
attention-only models (Qwen2.5-Coder-7B, 28 layers) already fit well
without needing any comparably large per-attention-layer term. The
likely explanation is that single-layer isolated benchmarks don't
capture MLX's lazy-eval graph fusion across a real 24-61-layer decode
chain (this project found the same effect from the other direction,
much earlier, in the "same real layer x N" experiment) -- so an
isolated benchmark's absolute overhead number doesn't transfer cleanly
to "marginal cost of one more layer in a fused graph." The per-MoE-layer
term here is best understood as a validated *empirical* correction (it
measurably reduces real prediction error) rather than a proven causal
mechanism.

Follow-up: the methodologically sound way to measure marginal per-layer
overhead in a fused graph is timing chains of N identical layers
(N=1,2,4,8,16) and reading the slope, not timing one layer in isolation.
Done for attention: marginal cost converges to ~232us/layer even at
N=16 (from 384us at N=1) -- meaningfully lower than the naive isolated
number, but nowhere near zero, so graph fusion reduces the isolated
benchmark's one-time setup cost, not the recurring per-layer dispatch
cost. This is consistent in order of magnitude with what a
28-attention-layer dense model like Qwen2.5-Coder-7B actually needs
(roughly 4.5ms of total fitted overhead / 28 layers ~ 160us/layer). The
equivalent chained benchmark for MoE (SwitchGLU) hit a shape-handling
bug in the naive chaining script (exponential blowup, not real MLX
behavior) and was not completed -- the per-MoE-layer term above is
still only validated by the isolated single-call measurement and the
resulting fit improvement, not by a clean chained measurement.

Refitting with 3 parameters (bandwidth ratio, a base per-step overhead,
and a per-MoE-layer overhead) via least squares against all 9 real
measurements collected so far (5 original + 4 hybrid, spanning dense,
conventional MoE, GatedDeltaNet hybrid, Mamba-2 hybrid, and Nemotron-H's
single-component-per-layer architecture) drops mean error from 15.8%
(the best a 2-parameter refit on the same 9 points could do) to 9.6%.

Calibration data (M4 Pro, 48GB; see measurements.json for the raw numbers,
context_length=115 matching this project's own probe's prompt+decode
range):
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 89.8, formula 80.4 (-10.4%)
    gemma-4-26b-a4b-it-4bit:            real 76.7, formula 91.7 (+19.6%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57.2, formula 58.5 (+2.3%)
    Qwen3.6-35B-A3B-4bit:               real 88.3, formula 90.7 (+2.7%)
    gpt-oss-20b-OptiQ-4bit:             real 83.6, formula 90.2 (+7.8%)
    Huihui-LFM2.5-1.2B-Instruct-8bit:   real 176.6, formula 139.4 (-21.0%)
    LFM2-8B-A1B-3bit-MLX:               real 192.1, formula 212.7 (+10.7%)
    granite-4.0-h-tiny-6bit-MLX:        real 116.9, formula 107.0 (-8.5%)
    NVIDIA-Nemotron-3-Nano-30B-A3B-8Bit: real 57.1, formula 55.2 (-3.4%)
mean absolute error 9.6%, max 21.0% -- worse per-point than the old
5-point-only fit's 4.5%, but that fit was simply wrong (not just
imprecise) outside its 5 conventional-architecture calibration set; this
one generalizes to every architecture family tested so far, including
ones its own parameters were never specifically shaped around.

Honesty notes on the three fitted constants:
- BASE_OVERHEAD_SEC came out of the regression NEGATIVE (~-0.76ms).
  That is not a physically meaningful "negative dispatch time" -- it's
  the least-squares fit compensating for whatever the other two terms
  still don't capture (this project has never claimed these constants
  are a clean physical decomposition; see BANDWIDTH_CALIBRATION_RATIO's
  own long-standing "not literal bandwidth > spec" caveat below). Kept
  as fit rather than clamped to 0, since clamping would just move the
  same error into the other two terms without improving overall
  accuracy.
- BANDWIDTH_CALIBRATION_RATIO (~0.99x spec here, much closer to literal
  than earlier fits) and MOE_LAYER_OVERHEAD_SEC (~114us) are the two
  terms most likely to be "real" -- the MoE term in particular matches
  the direct micro-benchmark's ~200us-per-call finding within a
  reasonable factor (real decode-loop kernel fusion is plausibly more
  efficient than an isolated Python-level benchmark call).
- LFM2.5-1.2B (-21.0%) remains the worst-fit point, and has ZERO MoE
  layers -- its own remaining problem was root-caused separately (see
  formula-accuracy-gap.md): its real decode time (~5.66ms/token) is
  close to or below what a "typical" model's fixed overhead would be,
  and no single global BASE_OVERHEAD_SEC can be simultaneously right for
  a model this fast and for the original 24-48-layer calibration set.
  Re-fit all three constants if/when more real measurements (especially
  more small/fast dense-hybrid models) become available.
"""
from typing import Optional

from .size_estimate import count_moe_layers, estimate_active_bytes_per_token

BANDWIDTH_CALIBRATION_RATIO = 0.727  # see module docstring -- empirical, from 9-point regression
BASE_OVERHEAD_SEC = -0.000758  # see module docstring -- fit value, not a literal negative dispatch time
MOE_LAYER_OVERHEAD_SEC = 0.000114  # per-MoE-layer dispatch cost, confirmed by direct MLX micro-benchmark


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
        "confidence": "medium (config-only formula, mean 9.6% error / 21.0% max on 9-point real-measurement set spanning dense, MoE, and 4 hybrid architectures)",
    }
