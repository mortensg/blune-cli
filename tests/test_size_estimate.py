from blune_cli.size_estimate import estimate_bytes, fits_in_ram


def test_moe_estimate_prefers_moe_intermediate_size():
    """Regression test: using the dense `intermediate_size` fallback for a
    MoE model's expert MLP (instead of `moe_intermediate_size`) inflated a
    real ~23B model to a wildly wrong 345B estimate, incorrectly flagging
    it as too large for a 48GB machine."""
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "moe_intermediate_size": 512,
        "intermediate_size": 8192,  # present but must NOT be used for MoE
        "vocab_size": 250000,
    }
    est_bytes = estimate_bytes(config)
    assert est_bytes is not None
    est_gb = est_bytes / 1e9
    assert 5 < est_gb < 20, f"expected roughly 10-15GB at 4-bit, got {est_gb:.1f}GB"


def test_dense_model_uses_intermediate_size():
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }
    est_bytes = estimate_bytes(config)
    assert est_bytes is not None
    assert est_bytes > 0


def test_fits_in_ram_true_for_small_model():
    config = {
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 2048,
        "vocab_size": 32000,
    }
    assert fits_in_ram(config, total_ram_gb=48) is True


def test_fits_in_ram_false_for_huge_moe_model():
    config = {
        "hidden_size": 8192,
        "num_hidden_layers": 92,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 160,
        "moe_intermediate_size": 4096,
        "vocab_size": 150000,
    }
    assert fits_in_ram(config, total_ram_gb=48) is False


def test_fits_in_ram_fails_open_when_unestimable():
    assert fits_in_ram({}, total_ram_gb=48) is True
