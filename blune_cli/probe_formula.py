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
5 real M4 Pro measurements, the naive bytes-per-token-only estimate is off
by 2-3.75x, and -- critically -- by a DIFFERENT factor for dense vs. MoE
architectures (0.54 vs. 0.27-0.43), so no single calibration ratio can
correct it the way probe_mlx.CALIBRATION_RATIO does.

The fix: decompose predicted time-per-token into two additive terms
instead of one multiplicative ratio --

    time_per_token = bytes_per_token / bandwidth + FIXED_OVERHEAD_SEC

`bytes_per_token / bandwidth` is the genuine memory-bound transfer time.
FIXED_OVERHEAD_SEC is MLX's roughly constant per-step dispatch cost
(kernel launch, graph-eval bookkeeping) -- fit against the same 5 models,
it lands in a narrow 6.7-9.6ms band (mean 7.775ms, stdev 1.16ms) REGARDLESS
of model size, which is exactly what "fixed per-step overhead" predicts
and the earlier multiplicative-ratio model couldn't explain. That's a
proportionally huge fraction of total decode time for MoE models (whose
bytes-per-token is small by design), which is why they showed a much
lower ratio in the naive model.

Calibration data (M4 Pro, 48GB; see measurements.json for the raw numbers):
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 89.8 tok/s, formula 81.9 (-8.8%)
    gemma-4-26b-a4b-it-4bit:            real 76.7 tok/s, formula 88.9 (+15.9%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57.2 tok/s, formula 58.0 (+1.4%)
    Qwen3.6-35B-A3B-4bit:               real 88.3 tok/s, formula 88.4 (+0.1%)
    gpt-oss-20b-OptiQ-4bit:             real 83.6 tok/s, formula 77.4 (-7.4%)
mean absolute error 6.7%, max 15.9% -- fit on only 5 points with 1 free
parameter, so treat this as a genuinely useful fast estimate, not a
replacement for probe_mlx.py's real-execution probe when accuracy matters
more than speed.
"""
from typing import Optional

from .size_estimate import estimate_active_bytes_per_token

FIXED_OVERHEAD_SEC = 0.007775  # MLX's per-decode-step dispatch cost, fit from real data


def probe(repo_id: str, config: dict, bandwidth_gbs: float, calibrate: bool = True) -> dict:
    """Instant tok/s estimate from config.json alone -- no MLX, no
    subprocess, no model construction. See module docstring for the
    calibration methodology and accuracy."""
    bytes_per_token = estimate_active_bytes_per_token(config)
    if bytes_per_token is None:
        raise ValueError(
            "couldn't determine hidden_size/num_hidden_layers from this "
            "config -- formula estimate needs a standard transformer config"
        )

    transfer_time = bytes_per_token / (bandwidth_gbs * 1e9)
    overhead = FIXED_OVERHEAD_SEC if calibrate else 0.0
    tps = 1.0 / (transfer_time + overhead)

    return {
        "library": "mlx (formula)",
        "repo_id": repo_id,
        "architecture": config.get("model_type"),
        "bytes_per_token_active": round(bytes_per_token / 1e6, 1),
        "estimated_real_tps": round(tps, 1),
        "confidence": "medium (config-only formula, mean 6.7% error on 5-point calibration set)",
    }
