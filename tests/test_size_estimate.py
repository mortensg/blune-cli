from blune_cli.size_estimate import (
    estimate_active_bytes_per_token,
    estimate_bytes,
    estimate_kv_bytes_per_token,
    estimate_total_params,
    fits_in_ram,
)


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
    assert 10 < est_gb < 25, f"expected roughly 15-20GB at 4-bit, got {est_gb:.1f}GB"


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


def test_three_projections_per_mlp_block():
    """Regression test: SwiGLU/gated MLPs (gate/up/down -- nearly every
    current architecture) have 3 weight matrices per block, not 2. Using
    2 undercounted every MLP-heavy model's params by ~33% -- caught by
    comparing against DeepSeek-V3's real, published ~671B parameter
    count (see docs/research-findings.md)."""
    deepseek_v3_like = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "intermediate_size": 18432,
        "first_k_dense_replace": 3,
        "vocab_size": 129280,
    }
    total = estimate_total_params(deepseek_v3_like)
    total_b = total / 1e9
    assert 620 < total_b < 720, f"expected close to the real 671B, got {total_b:.1f}B"


def test_mla_kv_cache_much_smaller_than_gqa():
    """MLA caches a compressed latent + decoupled RoPE key instead of full
    per-head K/V -- should be tens of times smaller than an equivalent
    GQA model's KV-cache at the same context length."""
    mla_config = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "head_dim": 128,
        "intermediate_size": 18432,
        "vocab_size": 129280,
    }
    gqa_equivalent = {**mla_config}
    del gqa_equivalent["kv_lora_rank"]
    del gqa_equivalent["qk_rope_head_dim"]

    mla_kv = estimate_kv_bytes_per_token(mla_config, context_length=1000)
    gqa_kv = estimate_kv_bytes_per_token(gqa_equivalent, context_length=1000)
    assert mla_kv is not None and gqa_kv is not None
    assert gqa_kv / mla_kv > 20, f"expected >20x reduction, got {gqa_kv / mla_kv:.1f}x"


def test_hybrid_ssm_layers_excluded_from_kv_growth():
    """Mamba/linear-attention layers don't accumulate a growing KV-cache
    -- a model where most layers are SSM should show much less KV growth
    with context than an all-attention model of the same layer count."""
    hybrid_config = {
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    }
    all_attention_config = {**hybrid_config, "layer_types": ["full_attention"] * 40}

    hybrid_kv = estimate_kv_bytes_per_token(hybrid_config, context_length=2000)
    full_kv = estimate_kv_bytes_per_token(all_attention_config, context_length=2000)
    assert hybrid_kv is not None and full_kv is not None
    assert hybrid_kv < full_kv / 3


def test_shared_experts_increase_active_bytes():
    base = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "vocab_size": 32000,
    }
    with_shared = {**base, "n_shared_experts": 2}
    assert estimate_active_bytes_per_token(with_shared) > estimate_active_bytes_per_token(base)


def test_expert_count_field_name_variants():
    """Different converters use different field names for top-k experts
    (num_experts_per_tok, top_k_experts, ...) -- a model using the wrong
    default (1 instead of the real value) silently under-counts active
    bytes. Regression test for a real bug found on gemma-4-26b-a4b-it,
    which uses `top_k_experts` and was defaulting to 1 instead of 8."""
    base = {
        "hidden_size": 2816,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "num_experts": 128,
        "moe_intermediate_size": 704,
        "vocab_size": 32000,
    }
    top_k_1 = {**base, "top_k_experts": 1}
    top_k_8 = {**base, "top_k_experts": 8}
    assert estimate_active_bytes_per_token(top_k_8) > estimate_active_bytes_per_token(top_k_1) * 1.5


def test_sliding_window_caps_kv_growth():
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "sliding_window": 512,
    }
    kv_within_window = estimate_kv_bytes_per_token(config, context_length=512)
    kv_far_beyond_window = estimate_kv_bytes_per_token(config, context_length=50_000)
    assert kv_within_window == kv_far_beyond_window


def test_kv_bytes_grow_with_context_when_no_sliding_window():
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
    }
    short = estimate_kv_bytes_per_token(config, context_length=128)
    long = estimate_kv_bytes_per_token(config, context_length=128_000)
    assert long == short * 1000


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
