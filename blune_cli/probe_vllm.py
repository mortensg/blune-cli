"""
vLLM-on-Apple-Silicon speed probe.

There are TWO real, unrelated, actively-maintained "vLLM on Apple
Silicon" packages, confirmed by installing and directly measuring both
(not from published docs, which showed >50% inconsistency across their
own tables for the same claim -- see the controlled measurements below):

- `vllm-metal` (vllm-project/vllm-metal on PyPI): the official plugin,
  pairs upstream vLLM's real scheduler with an MLX/Metal execution
  backend. Ships its own `vllm-metal` CLI that starts a real OpenAI-
  compatible HTTP server (no built-in benchmark tool).
- `vllm-mlx` (waybarrios/vllm-mlx on PyPI): an independent, from-scratch
  reimplementation of vLLM's ideas natively in MLX, not running
  upstream vLLM's own scheduler code. Ships its own `vllm-mlx bench`
  CLI subcommand for exactly this kind of measurement.

Both were installed (in an isolated venv, not this project's main one)
and measured directly against a real, controlled single-stream
(concurrency=1) baseline: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`,
native `mlx_lm.generate` mean 429.38 tok/s (10 trials, std 1.18%
relative) on the same M4 Pro machine used throughout this project.

    vllm-metal (real HTTP server, greedy decode, 8 timed requests after
      1 warmup request -- warmup mattered here too, same as
      `probe_mlx.py`'s own well-established warmup sensitivity):
      293.69 tok/s (std 2.4% relative) -- ratio 0.684.
    vllm-mlx (`vllm-mlx bench`, --max-num-seqs 1, 2 separate runs after
      discovering its OWN large first-run warmup effect -- a 5-prompt
      run gave a misleadingly low 52.13 tok/s/ratio 0.121 before the
      benchmark's internal state had warmed up; 10-prompt runs gave
      300.17 and 319.67 tok/s once past that):
      mean 309.92 tok/s -- ratio 0.7215.

**This directly refutes a research pass's specific claim of a stark
0.52-0.55 (vllm-metal) vs. 0.90-0.95 (vllm-mlx) split.** On this one
real, small (0.5B) model, both packages give SIMILAR single-stream
ratios (0.684 vs. 0.7215), not dramatically different ones. What DOES
look real and different is model SIZE: this project's original single
vllm-metal data point (`gemma-4-26b-a4b-it-4bit`, 26B, ratio 0.554) is
meaningfully lower than the new 0.5B point (0.684) for the SAME
package. With only 2 points per package (well below this project's own
5-per-family bar), it's not possible to cleanly separate "package
choice" from "model size" as the dominant variable yet -- both
`VLLM_METAL_SINGLE_STREAM_RATIO` and `VLLM_MLX_SINGLE_STREAM_RATIO`
below are averages across the (size-mismatched) points available,
kept deliberately separate per backend since that's the axis this
project can actually detect from an installed package, even though
model size is the more likely real driver.

`detect_vllm_backend()` checks which package (if either) is actually
importable in the CALLING environment and picks its own ratio
accordingly; if neither is installed (the common case -- most blune-cli
users are using it to decide WHETHER to install vLLM at all, not
because they already have), it falls back to the more conservative
(lower) of the two rather than the more optimistic one.
"""
from typing import Optional

from . import probe_mlx

VLLM_METAL_SINGLE_STREAM_RATIO = 0.619  # see module docstring -- mean of 2 real points (0.5B: 0.684, 26B: 0.554)
VLLM_MLX_SINGLE_STREAM_RATIO = 0.7215  # see module docstring -- mean of 2 real runs, 1 model (0.5B), after ruling out a large first-run warmup artifact


def detect_vllm_backend() -> Optional[str]:
    """Which real vLLM-on-Apple-Silicon package (if either) is importable
    in this process right now. Returns "vllm-metal", "vllm-mlx", or None
    if neither is installed -- both are real, separate PyPI packages
    that can coexist in the same environment (confirmed: installing
    both together produced no dependency conflicts)."""
    try:
        import vllm_metal  # noqa: F401

        return "vllm-metal"
    except ImportError:
        pass
    try:
        import vllm_mlx  # noqa: F401

        return "vllm-mlx"
    except ImportError:
        pass
    return None


def probe(repo_id: str, config: dict, calibrate: bool = True, **kwargs) -> dict:
    mlx_result = probe_mlx.probe(repo_id, config, calibrate=calibrate, **kwargs)
    base_tps = mlx_result["estimated_real_tps"]

    backend = detect_vllm_backend()
    if backend == "vllm-mlx":
        ratio = VLLM_MLX_SINGLE_STREAM_RATIO
    else:
        # "vllm-metal" or (the common case) neither installed -- see
        # module docstring for why the fallback is the more conservative
        # of the two rather than an arbitrary/optimistic pick.
        ratio = VLLM_METAL_SINGLE_STREAM_RATIO
    vllm_single_stream_tps = base_tps * ratio

    return {
        "library": f"vllm ({backend or 'not installed, assumed vllm-metal'})",
        "repo_id": repo_id,
        "architecture": mlx_result["architecture"],
        "quantization": mlx_result["quantization"],
        "layers": mlx_result["layers"],
        "estimated_real_tps": round(vllm_single_stream_tps, 1),
        "calibration_ratio": ratio,
        "confidence": "low (2 real points per backend, model-size effect not yet separable from package choice -- see module docstring)",
        "concurrency_note": (
            "vLLM's advantage is request batching, not single-stream speed. "
            "At high concurrency (~24 parallel requests) the same class of "
            "model measured ~2.6x this single-stream figure in aggregate "
            "throughput on the reference machine. This probe does not model "
            "concurrency scaling yet."
        ),
    }
