from blune_cli.measurements import find_real_measurement, load_measurements


def test_measurements_load():
    data = load_measurements()
    assert len(data) > 0
    for m in data:
        assert "repo_id" in m
        assert "library" in m
        assert "real_decode_tps" in m


def test_find_real_measurement_exact_match():
    m = find_real_measurement(
        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "mlx", machine="m4_pro_48gb"
    )
    assert m is not None
    assert m["real_decode_tps"] == 90.5


def test_find_real_measurement_normalizes_backend_suffix():
    """'vllm' must match a stored 'vllm (metal)' entry -- the parenthetical
    is a display detail, not part of the library identity. Regression test
    for the bug where `blune compare` showed vLLM as "estimated" even
    though a real measurement was recorded."""
    m = find_real_measurement(
        "mlx-community/gpt-oss-20b-OptiQ-4bit", "vllm", machine="m4_pro_48gb"
    )
    assert m is not None
    assert m["real_decode_tps"] == 53.0


def test_find_real_measurement_returns_none_when_absent():
    m = find_real_measurement("nonexistent/repo", "mlx", machine="m4_pro_48gb")
    assert m is None


def test_find_real_measurement_wrong_machine_returns_none():
    m = find_real_measurement(
        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "mlx", machine="m1_8gb"
    )
    assert m is None
