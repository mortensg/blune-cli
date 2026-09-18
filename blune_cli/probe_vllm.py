"""
vLLM (via vllm-metal on Apple Silicon) speed probe.

vllm-metal's own architecture doc states plainly: "mlx_lm supplies the
token-wise model layers" -- vLLM on Apple Silicon runs the exact same
mlx-lm model classes we already probe directly, with vLLM's own scheduler,
paged attention and request batching layered on top. So we don't need a
separate zero-download model-building path here: we reuse probe_mlx's raw
number and apply a vLLM-specific correction ratio for the serving-layer
overhead.

Honesty note: that correction ratio is currently based on exactly ONE
real, measured comparison (today, this machine): gemma-4-26b-a4b-it-4bit
ran at 76.7-79 tok/s under plain mlx-lm and 42.5 tok/s under vLLM
(single request, non-batched) -- a ratio of ~0.55. That single point is a
reasonable prior, not a validated constant. vLLM's real strength is
concurrent-request throughput (we separately measured 202 tok/s aggregate
at 24 concurrent requests on the same model), which this single-stream
estimate does not capture -- see `concurrency_note` in the result.
"""
from typing import Optional

from . import probe_mlx

VLLM_SINGLE_STREAM_RATIO = 0.55  # provisional, n=1 real measurement


def probe(repo_id: str, config: dict, calibrate: bool = True, **kwargs) -> dict:
    mlx_result = probe_mlx.probe(repo_id, config, calibrate=calibrate, **kwargs)

    base_tps = mlx_result["estimated_real_tps"]
    vllm_single_stream_tps = base_tps * VLLM_SINGLE_STREAM_RATIO

    return {
        "library": "vllm (metal)",
        "repo_id": repo_id,
        "architecture": mlx_result["architecture"],
        "quantization": mlx_result["quantization"],
        "layers": mlx_result["layers"],
        "estimated_real_tps": round(vllm_single_stream_tps, 1),
        "calibration_ratio": VLLM_SINGLE_STREAM_RATIO,
        "confidence": "low (n=1 real measurement, single-stream only)",
        "concurrency_note": (
            "vLLM's advantage is request batching, not single-stream speed. "
            "At high concurrency (~24 parallel requests) the same class of "
            "model measured ~2.6x this single-stream figure in aggregate "
            "throughput on the reference machine. This probe does not model "
            "concurrency scaling yet."
        ),
    }
