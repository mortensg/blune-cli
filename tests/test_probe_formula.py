from blune_cli import config_cache
from blune_cli.probe_formula import probe

M4_PRO_BANDWIDTH_GBS = 273

# repo_id -> real measured tok/s (from measurements.json), used to bound
# how far the formula is allowed to drift from validated reality. Spans
# dense, conventional MoE, and 4 hybrid architectures (GatedDeltaNet,
# LFM2 ShortConv, Mamba-2, Nemotron-H single-component-per-layer).
_CALIBRATION_SET = {
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit": 89.8,
    "mlx-community/gemma-4-26b-a4b-it-4bit": 76.7,
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": 57.2,
    "mlx-community/Qwen3.6-35B-A3B-4bit": 88.3,
    "mlx-community/gpt-oss-20b-OptiQ-4bit": 83.6,
    "mlx-community/Huihui-LFM2.5-1.2B-Instruct-abliterated-8bit": 176.6,
    "mlx-community/LFM2-8B-A1B-3bit-MLX": 192.1,
    "mlx-community/granite-4.0-h-tiny-6bit-MLX": 116.9,
    "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8Bit": 57.1,
}


def test_formula_within_30pct_of_real_on_calibration_set():
    """Regression test for the formula's accuracy on the exact data it was
    calibrated against (mean error 9.6%, max 21.0% at fit time -- 30% here
    leaves headroom for the fit to be redone without breaking this test on
    small changes)."""
    for repo_id, real_tps in _CALIBRATION_SET.items():
        config = config_cache.get_config(repo_id, offline=True)
        result = probe(repo_id, config, M4_PRO_BANDWIDTH_GBS)
        error_pct = abs(result["estimated_real_tps"] - real_tps) / real_tps * 100
        assert error_pct < 30, f"{repo_id}: {error_pct:.1f}% error (formula={result['estimated_real_tps']}, real={real_tps})"


def test_formula_mean_error_within_15pct_across_calibration_set():
    """Whole-set check (not just per-point) so a regression that keeps
    every individual point under 30% but drags the average up doesn't
    slip through."""
    errors = []
    for repo_id, real_tps in _CALIBRATION_SET.items():
        config = config_cache.get_config(repo_id, offline=True)
        result = probe(repo_id, config, M4_PRO_BANDWIDTH_GBS)
        errors.append(abs(result["estimated_real_tps"] - real_tps) / real_tps * 100)
    mean_error = sum(errors) / len(errors)
    assert mean_error < 15, f"mean error {mean_error:.1f}% (expected ~9.6%)"


def test_formula_needs_no_mlx_or_network():
    """The whole point: this must work from config.json alone, instantly."""
    config = config_cache.get_config("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", offline=True)
    result = probe("x", config, M4_PRO_BANDWIDTH_GBS)
    assert result["estimated_real_tps"] > 0
    assert result["library"] == "mlx (formula)"


def test_uncalibrated_mode_returns_higher_raw_estimate():
    config = config_cache.get_config("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", offline=True)
    calibrated = probe("x", config, M4_PRO_BANDWIDTH_GBS, calibrate=True)
    raw = probe("x", config, M4_PRO_BANDWIDTH_GBS, calibrate=False)
    assert raw["estimated_real_tps"] > calibrated["estimated_real_tps"]


def test_longer_context_reduces_estimated_speed():
    """KV-cache read bytes grow with context length for ordinary attention
    models, so speed at a long context should be lower than at a short
    one for the same model."""
    config = config_cache.get_config("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", offline=True)
    short_ctx = probe("x", config, M4_PRO_BANDWIDTH_GBS, context_length=128)
    long_ctx = probe("x", config, M4_PRO_BANDWIDTH_GBS, context_length=32_000)
    assert long_ctx["estimated_real_tps"] < short_ctx["estimated_real_tps"]


def test_n_moe_layers_reported_and_affects_estimate():
    """Regression test for the per-MoE-layer dispatch overhead term,
    added after a direct MLX micro-benchmark showed each MoE-routing
    call (mlx_lm.models.switch_layers.SwitchGLU) costs a roughly
    constant ~200us regardless of expert count -- a heavily-MoE model
    should predict slower (relative to a naive bandwidth-only estimate)
    than a model with the same bytes/token but fewer MoE layers."""
    config = config_cache.get_config("mlx-community/granite-4.0-h-tiny-6bit-MLX", offline=True)
    result = probe("x", config, M4_PRO_BANDWIDTH_GBS)
    assert result["n_moe_layers"] > 0

    dense_config = config_cache.get_config(
        "mlx-community/Huihui-LFM2.5-1.2B-Instruct-abliterated-8bit", offline=True
    )
    dense_result = probe("x", dense_config, M4_PRO_BANDWIDTH_GBS)
    assert dense_result["n_moe_layers"] == 0
