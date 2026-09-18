from blune_cli import config_cache
from blune_cli.probe_formula import probe

M4_PRO_BANDWIDTH_GBS = 273

# repo_id -> real measured tok/s (from measurements.json), used to bound
# how far the formula is allowed to drift from validated reality.
_CALIBRATION_SET = {
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit": 89.8,
    "mlx-community/gemma-4-26b-a4b-it-4bit": 76.7,
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit": 57.2,
    "mlx-community/Qwen3.6-35B-A3B-4bit": 88.3,
    "mlx-community/gpt-oss-20b-OptiQ-4bit": 83.6,
}


def test_formula_within_20pct_of_real_on_calibration_set():
    """Regression test for the formula's accuracy on the exact data it was
    calibrated against. A naive bytes-per-token-only estimate (no fixed
    overhead term) was off by 2-3.75x here, and inconsistently between
    dense and MoE architectures -- this guards against that regressing."""
    for repo_id, real_tps in _CALIBRATION_SET.items():
        config = config_cache.get_config(repo_id, offline=True)
        result = probe(repo_id, config, M4_PRO_BANDWIDTH_GBS)
        error_pct = abs(result["estimated_real_tps"] - real_tps) / real_tps * 100
        assert error_pct < 20, f"{repo_id}: {error_pct:.1f}% error (formula={result['estimated_real_tps']}, real={real_tps})"


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
