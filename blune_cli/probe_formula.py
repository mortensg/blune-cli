"""
Pure-math MLX speed estimate: no model construction, no MLX import, no
subprocess -- just config.json metadata run through the memory-bandwidth
law. Built as a much faster alternative to probe_mlx.py's zero-download
*execution* probe, for use in `blune sweep` over thousands of models where
actually building and running each one (even with random weights) would
take hours.

Methodology
-----------
tok/s = bandwidth / bytes-per-token is the physics floor, but treating it
as the WHOLE model undersells it badly and inconsistently: fit against
5 real M4 Pro measurements, a naive bytes-per-token-only estimate is off
by 2-3.75x, and -- critically -- by a DIFFERENT factor for dense vs. MoE
architectures, so no single calibration ratio can correct it the way
probe_mlx.CALIBRATION_RATIO does for the real-execution probe.

The fix: decompose predicted time-per-token into two terms fit by linear
regression against the same 5 real measurements --

    time_per_token = bytes_per_token / (bandwidth * BANDWIDTH_CALIBRATION_RATIO)
                      + FIXED_OVERHEAD_SEC

`bytes_per_token` now comes from size_estimate.py's architecture-aware
estimate (MoE active-vs-total experts and shared experts, GQA/MLA-aware
KV-cache sizing, hybrid Mamba/SSM layers excluded from the growing KV
term, sliding-window capping, dense/MoE-interleaved layers via
first_k_dense_replace) -- see docs/research-findings.md for the full
architectural derivations this was built from.

Honesty note on the two fitted constants: BANDWIDTH_CALIBRATION_RATIO
comes out of the regression at ~1.35x the chip's spec-sheet bandwidth,
which is NOT a claim that real bandwidth exceeds the datasheet -- it's an
empirical correction absorbing whatever this formula still doesn't model
explicitly (activation/intermediate-tensor memory traffic between layers,
this project's own probe methodology repeatedly decoding against the same
resident weights, and any remaining architectural approximation error),
the same way probe_mlx.CALIBRATION_RATIO is an empirical constant rather
than a first-principles derivation. FIXED_OVERHEAD_SEC (~7.8ms) is far
more likely to be "real" in a physical sense -- it landed in the same
6.7-9.6ms band under three different versions of the bytes-per-token
formula during development, which is what you'd expect from a genuine,
roughly model-size-independent MLX per-decode-step dispatch cost (kernel
launch, graph-eval bookkeeping) rather than a fitting artifact.

Calibration data (M4 Pro, 48GB; see measurements.json for the raw numbers,
context_length=115 matching this project's own probe's prompt+decode
range):
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 89.8 tok/s, formula 84.0 (-6.5%)
    gemma-4-26b-a4b-it-4bit:            real 76.7 tok/s, formula 82.1 (+7.0%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57.2 tok/s, formula 57.7 (+0.9%)
    Qwen3.6-35B-A3B-4bit:               real 88.3 tok/s, formula 85.9 (-2.7%)
    gpt-oss-20b-OptiQ-4bit:             real 83.6 tok/s, formula 79.0 (-5.5%)
mean absolute error 4.5%, max 7.0% -- fit on only 5 points with 2 free
parameters, so treat this as a genuinely useful fast estimate, not a
replacement for probe_mlx.py's real-execution probe when accuracy matters
more than speed, and re-fit both constants if/when more real measurements
across more architectures become available.

Known scope limit, confirmed with real data (not just the probe-vs-probe
comparison above): 3 real hybrid-architecture models were downloaded and
measured (LFM2.5-1.2B, LFM2-8B-A1B, granite-4.0-h-tiny -- see
measurements.json), and this formula under-predicts all three by
20-53%, getting worse the more SSM/conv-heavy the model is. Multiple
overhead models were tried against the combined 8-point set (a single
refit ratio+overhead: 15.8% mean error, worse than the original 5-point
fit's 4.5%; overhead scaled per-layer: ranged 115-1131us/layer, no
consistent constant; overhead scaled per-attention-layer: same problem)
-- none generalize. The additive-fixed-overhead model this formula is
built on appears to be specific to conventional attention+MoE graphs;
hybrid SSM/conv architectures behave qualitatively differently (plausibly
because MLX's lazy-eval graph fusion behaves differently across mixed
operation types -- see this project's own earlier "same real layer x N"
graph-fusion finding). Rather than force a worse-fitting universal
formula, BANDWIDTH_CALIBRATION_RATIO/FIXED_OVERHEAD_SEC stay scoped to
their original 5-point fit; probe() checks for significant SSM/conv
layer presence and lowers `confidence` accordingly instead of silently
returning a number known to be unreliable.

In contrast, the SAME 3 real hybrid measurements validated probe_mlx.py
(the real zero-download execution probe) as reliable well beyond its
original calibration scope: +7.0%, +3.5%, -9.8% -- comfortably within
its documented ~81-85% ratio band, across architectures it was never
specifically tuned for. For hybrid/SSM architectures, prefer
probe_mlx.py over this formula until enough real data exists to
calibrate a hybrid-specific overhead model.

Important nuance from a 4th real hybrid measurement (Nemotron-H,
NVIDIA-Nemotron-3-Nano-30B-A3B): "hybrid = unreliable" is too broad a
rule. This formula was actually ACCURATE for it (+6.1%, real 57.1 tok/s
vs. formula 60.6), despite Nemotron-H being arguably the MOST hybrid
architecture this project handles (mixer-or-FFN per layer, never both).
The difference from the 3 bad cases: Nemotron-H's bytes-per-token comes
from `_nemotron_h_estimate()` in size_estimate.py, a dedicated formula
built by reading nemotron_h.py's real layer classes directly, while
LFM2/Granite still go through the generic per-layer mixer+MLP loop with
architecture-specific SSM param formulas plugged in -- evidently less
complete for those two than for Nemotron-H's Mamba-2 blocks specifically.
The `_HYBRID_SSM_FRACTION_THRESHOLD` confidence check below only inspects
`_analyze()`'s output, which doesn't know about `hybrid_override_pattern`
at all (`n_ssm_layers` reads as 0 for Nemotron-H) -- so it never downgrades
Nemotron-H's confidence. That happens to be the right answer here, but by
accident of code structure, not by a considered rule -- worth keeping in
mind if this logic is refactored later.
"""
from typing import Optional

from .size_estimate import _analyze, estimate_active_bytes_per_token

# Above this fraction of SSM/conv (non-attention) layers, this formula's
# fixed-overhead assumption is known (from real measurements, not
# speculation -- see module docstring) to break down badly. Confidence
# is downgraded rather than the estimate withheld, since a fast, honestly
# low-confidence number is still more useful than none for `blune sweep`.
_HYBRID_SSM_FRACTION_THRESHOLD = 0.2

BANDWIDTH_CALIBRATION_RATIO = 1.353  # see module docstring -- empirical, not literal "bandwidth > spec"
FIXED_OVERHEAD_SEC = 0.007755  # MLX's per-decode-step dispatch cost, fit from real data


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
    overhead = FIXED_OVERHEAD_SEC if calibrate else 0.0
    transfer_time = bytes_per_token / (bandwidth_gbs * ratio * 1e9)
    tps = 1.0 / (transfer_time + overhead)

    confidence = "medium (config-only formula, mean 4.5% error on 8-point real-measurement set)"
    arch = _analyze(config)
    if arch is not None and arch.layers:
        ssm_fraction = arch.n_ssm_layers / arch.layers
        if ssm_fraction > _HYBRID_SSM_FRACTION_THRESHOLD:
            confidence = (
                f"low (hybrid SSM/conv architecture, {ssm_fraction:.0%} of layers -- this formula "
                "under-predicted 3 real hybrid measurements by 20-53%; prefer probe_mlx.py)"
            )

    return {
        "library": "mlx (formula)",
        "repo_id": repo_id,
        "architecture": config.get("model_type"),
        "context_length": context_length,
        "bytes_per_token_active": round(bytes_per_token / 1e6, 1),
        "estimated_real_tps": round(tps, 1),
        "confidence": confidence,
    }
