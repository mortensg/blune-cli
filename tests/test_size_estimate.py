from blune_cli.size_estimate import (
    _analyze,
    _dsa_indexer_params,
    _gated_delta_net_params,
    _gemma4_estimate,
    _lfm2_conv_params,
    _lfm2_dense_mlp_width,
    _mamba2_ssm_params,
    _mamba_ssm_params,
    _mla_weight_params,
    _nemotron_h_estimate,
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


def test_mla_decompressed_cache_variant_uses_full_per_head_size():
    """Regression test: mlx_lm.models.youtu_llm.YoutuLLMAttention
    decompresses via kv_b_proj BEFORE calling cache.update_and_fetch
    (confirmed reading its source), unlike DeepSeek-V3/GLM4-MoE-Lite
    style MLA which caches the compressed latent and defers the
    up-projection into the attention-score computation. A youtu_llm
    config must NOT get the ~50-100x compression benefit a generic MLA
    config gets -- its real per-token KV-cache is
    num_heads*(qk_nope_head_dim+qk_rope_head_dim+v_head_dim)."""
    heads = 16
    qk_nope, qk_rope, v_head_dim = 128, 64, 128
    youtu_config = {
        "model_type": "youtu_llm",
        "hidden_size": 2048,
        "num_hidden_layers": 32,
        "num_attention_heads": heads,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": qk_rope,
        "qk_nope_head_dim": qk_nope,
        "v_head_dim": v_head_dim,
        "intermediate_size": 6144,
        "vocab_size": 128256,
    }
    generic_mla_config = {**youtu_config, "model_type": "some_other_mla"}

    youtu_kv = estimate_kv_bytes_per_token(youtu_config, context_length=1000)
    generic_kv = estimate_kv_bytes_per_token(generic_mla_config, context_length=1000)
    assert youtu_kv is not None and generic_kv is not None
    assert youtu_kv > generic_kv * 3, (
        f"expected youtu_llm's decompressed cache to be much larger, "
        f"got youtu={youtu_kv} generic={generic_kv}"
    )
    per_token_elems = heads * (qk_nope + qk_rope + v_head_dim)
    num_layers = youtu_config["num_hidden_layers"]
    expected = per_token_elems * 2 * 1000 * num_layers  # 2 bytes/elem, 1000 tokens, all layers
    assert youtu_kv == expected


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


def test_gated_delta_net_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm.models.gated_delta's real
    GatedDeltaNet.__init__ (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
    depthwise conv1d, out_proj), using the real field values from
    Youssofal/Qwen3.6-35B-A3B-*'s config -- not just "some number came
    out", an exact hand-computed expectation."""
    c = {
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    hidden = 2048
    key_dim = 128 * 16
    value_dim = 128 * 32
    conv_dim = key_dim * 2 + value_dim
    expected = (
        hidden * (key_dim * 2 + value_dim)  # in_proj_qkv
        + hidden * value_dim  # in_proj_z
        + hidden * 32  # in_proj_b
        + hidden * 32  # in_proj_a
        + conv_dim * 4  # conv1d
        + value_dim * hidden  # out_proj
    )
    assert _gated_delta_net_params(c, hidden) == expected


def test_gated_delta_net_returns_none_without_required_fields():
    assert _gated_delta_net_params({"hidden_size": 2048}, 2048) is None


def test_mamba_ssm_params_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm.models.mamba's real
    MambaBlock.__init__ (in_proj, depthwise conv1d, x_proj, dt_proj,
    out_proj)."""
    c = {"d_state": 16, "d_conv": 4, "intermediate_size": 4096, "dt_rank": 128}
    hidden = 2048
    expected = (
        hidden * 2 * 4096  # in_proj
        + 4096 * 4  # conv1d
        + 4096 * (128 + 2 * 16)  # x_proj
        + 128 * 4096  # dt_proj
        + 4096 * hidden  # out_proj
    )
    assert _mamba_ssm_params(c, hidden) == expected


def test_hybrid_layer_uses_real_ssm_formula_not_attention_fallback():
    """Regression test: before real SSM param formulas were added, hybrid
    Mamba/linear-attention layers used the generic attention+MLP formula,
    which undercounted them substantially (a real hold-out model was
    over-predicted by 32-39% because of this -- see
    docs/formula-accuracy-gap.md). The SSM-layer weight count should now
    differ from (and, for this real field combination, exceed) the
    generic attention formula's."""
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "layer_types": ["linear_attention"] * 4,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    a = _analyze(config)
    assert a.ssm_weight_params_per_layer != a.attn_weight_params_per_layer
    assert a.ssm_weight_params_per_layer == _gated_delta_net_params(config, 2048)


def test_shared_expert_intermediate_size_counted_even_without_explicit_count():
    """Regression test: Qwen3-Next-family MoE always has one always-on
    shared expert sized by shared_expert_intermediate_size, signaled by
    that field's presence rather than an explicit n_shared_experts count
    -- missing this silently zeroed out a real expert's worth of active
    bytes every decode step (and affected one of this project's own
    5-point real calibration measurements)."""
    base = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "vocab_size": 32000,
    }
    with_shared = {**base, "shared_expert_intermediate_size": 512}
    assert estimate_active_bytes_per_token(with_shared) > estimate_active_bytes_per_token(base)


def test_mamba2_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm's real Mamba2Mixer
    (nemotron_h.py / granitemoehybrid.py), using granite-4.0-h-tiny's
    real field values -- fused single in_proj (not Mamba-1's separate
    x_proj/dt_proj), depthwise conv1d, out_proj."""
    c = {
        "mamba_n_heads": 48,
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_n_groups": 1,
        "mamba_proj_bias": False,
        "mamba_conv_bias": True,
    }
    hidden = 1536
    intermediate = 48 * 64
    conv_dim = intermediate + 2 * 1 * 128
    projection_size = intermediate + conv_dim + 48
    expected = hidden * projection_size + conv_dim * 4 + conv_dim + intermediate * hidden
    assert _mamba2_ssm_params(c, hidden) == expected


def test_mamba2_field_names_dont_collide_with_mamba1_fallback():
    """Regression test: nemotron_h's field names (ssm_state_size,
    conv_kernel) also happen to match Mamba-1's fallback chain in
    _mamba_ssm_params, which would silently produce the wrong (Mamba-1)
    shape for a genuine Mamba-2 layer if checked in the wrong order."""
    c = {
        "mamba_num_heads": 64,
        "mamba_head_dim": 128,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "n_groups": 8,
    }
    assert _mamba2_ssm_params(c, 2688) is not None
    # And the dispatcher must prefer it over the Mamba-1 formula:
    from blune_cli.size_estimate import _ssm_layer_params

    assert _ssm_layer_params(c, 2688, fallback=0.0) == _mamba2_ssm_params(c, 2688)


def test_lfm2_conv_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm's real lfm2.ShortConv --
    NOT a state-space model despite living in a hybrid architecture:
    in_proj (hidden->3*hidden), depthwise conv, out_proj (hidden->hidden)."""
    c = {"conv_L_cache": 3, "conv_bias": False}
    hidden = 2048
    expected = hidden * 3 * hidden + hidden * 3 + hidden * hidden
    assert _lfm2_conv_params(c, hidden) == expected


def test_mla_weight_params_matches_real_deepseek_v3():
    """Exact param count check against mlx_lm's real
    DeepseekV3Attention -- q_a_proj+q_b_proj (when q_lora_rank is set),
    kv_a_proj_with_mqa+kv_b_proj, o_proj. This replaces a generic GQA
    approximation that was never actually correct for MLA's very
    different low-rank weight structure (only the KV-cache term was
    MLA-aware before)."""
    c = {
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }
    hidden, heads = 7168, 128
    expected = (
        hidden * 1536 + 1536 * heads * (128 + 64)  # q_a_proj + q_b_proj
        + hidden * (512 + 64)  # kv_a_proj_with_mqa
        + 512 * heads * (128 + 128)  # kv_b_proj
        + heads * 128 * hidden  # o_proj
    )
    assert _mla_weight_params(c, hidden, heads) == expected


def test_mla_without_q_lora_uses_direct_q_proj():
    c = {"kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}
    hidden, heads = 4096, 32
    expected = (
        hidden * heads * (128 + 64)  # direct q_proj, no q_lora_rank
        + hidden * (512 + 64)
        + 512 * heads * (128 + 128)
        + heads * 128 * hidden
    )
    assert _mla_weight_params(c, hidden, heads) == expected


def test_deepseek_v3_total_params_matches_published_671b():
    """End-to-end check: the real DeepSeek-V3 config, through the full
    pipeline (MLA weights + KV formula, first_k_dense_replace layer
    interleaving, 3x SwiGLU MLP, shared experts), should land close to
    the real, published ~671B parameter count."""
    config = {
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
    total_b = estimate_total_params(config) / 1e9
    assert 655 < total_b < 690, f"expected close to the real 671B, got {total_b:.1f}B"


def test_nemotron_h_single_component_layers_not_double_counted():
    """Regression test for the biggest gap found in this module: a
    Nemotron-H layer is EITHER a Mamba-2 mixer, attention, a plain MLP,
    or a MoE block -- never a mixer PLUS an MLP the way every other
    architecture here works. Forcing it through the generic per-layer
    loop double-counted every MLP-only/MoE-only layer and estimated a
    real config (named 30B-A3B) at 103.1B. The dedicated estimator
    should land close to the real ~30B name instead."""
    config = {
        "hidden_size": 2688,
        "hybrid_override_pattern": "ME" * 20 + "*" * 6 + "E" * 6,  # 52 layers total
        "vocab_size": 131072,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "conv_kernel": 4,
        "n_groups": 8,
        "mamba_proj_bias": False,
        "use_conv_bias": True,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": 3712,
        "intermediate_size": 1856,
    }
    result = _nemotron_h_estimate(config)
    assert result is not None
    total_b = result["total_params"] / 1e9
    assert 15 < total_b < 45, f"expected roughly 30B-ish, got {total_b:.1f}B"


def test_nemotron_h_not_detected_without_hybrid_override_pattern():
    """A standard mixer+MLP architecture must NOT be routed through the
    Nemotron-H single-component path just because it happens to share a
    field name."""
    assert _nemotron_h_estimate({"hidden_size": 2048, "num_hidden_layers": 12}) is None


def _gemma4_config(**overrides):
    config = {
        "model_type": "gemma4",
        "hidden_size": 2816,
        "num_hidden_layers": 6,
        "intermediate_size": 2112,
        "num_attention_heads": 16,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_key_value_heads": 8,
        "num_global_key_value_heads": 2,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "enable_moe_block": True,
        "num_experts": 128,
        "top_k_experts": 8,
        "moe_intermediate_size": 704,
        "vocab_size": 262144,
        "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 1 + ["sliding_attention"],
    }
    config.update(overrides)
    return config


def test_gemma4_runs_dense_mlp_and_moe_in_parallel_not_either_or():
    """Regression test for this project's single worst-fit calibration
    point (gemma-4-26b-a4b-it-4bit, +20.1% error): Gemma4DecoderLayer
    sums a full dense MLP AND the MoE experts on every layer
    (`h = h1 + h2` in mlx_lm.models.gemma4_text), unlike
    first_k_dense_replace-style interleaving where a layer is either
    dense or MoE. Active params must include BOTH per layer, not just
    whichever the generic moe_layer_mask path would have picked."""
    config = _gemma4_config()
    result = _gemma4_estimate(config)
    assert result is not None

    hidden = config["hidden_size"]
    dense_mlp = 3 * hidden * config["intermediate_size"]
    router = hidden * config["num_experts"]
    expert_unit = 3 * hidden * config["moe_intermediate_size"]
    moe_active = router + config["top_k_experts"] * expert_unit

    # Active params per layer must be at least dense_mlp + moe_active
    # (plus attention, plus embedding/lm_head amortized in) -- if the
    # dense MLP were dropped (the bug this guards against), active
    # params would fall short of this floor by a full dense_mlp's worth
    # per layer.
    per_layer_floor = dense_mlp + moe_active
    assert result["active_params"] >= config["num_hidden_layers"] * per_layer_floor


def test_gemma4_full_attention_uses_global_head_dim_and_no_v_proj():
    """Full-attention layers use global_head_dim/num_global_key_value_heads
    and, under attention_k_eq_v, have no separate v_proj at all (values =
    keys) -- verified directly in mlx_lm.models.gemma4_text.Attention.
    A config with only full-attention layers and one with only sliding
    layers must therefore price attention differently."""
    hidden = 2816
    n_experts = 0  # isolate the attention term from the MLP term
    all_full = _gemma4_config(
        num_hidden_layers=1, layer_types=["full_attention"], num_experts=n_experts, enable_moe_block=False
    )
    all_sliding = _gemma4_config(
        num_hidden_layers=1, layer_types=["sliding_attention"], num_experts=n_experts, enable_moe_block=False
    )
    full_result = _gemma4_estimate(all_full)
    sliding_result = _gemma4_estimate(all_sliding)
    assert full_result is not None and sliding_result is not None

    dense_mlp = 3 * hidden * all_full["intermediate_size"]
    full_attn_only = full_result["active_params"] - dense_mlp - hidden * all_full["vocab_size"]
    sliding_attn_only = sliding_result["active_params"] - dense_mlp - hidden * all_sliding["vocab_size"]

    # Full: q_proj+o_proj at global_head_dim=512 (no k_eq_v discount
    # cancels this out) plus k_proj only (no v_proj) at
    # num_global_key_value_heads=2, global_head_dim=512.
    expected_full = 2 * hidden * (16 * 512) + hidden * (2 * 512)
    # Sliding: standard q+o+k+v at head_dim=256, num_key_value_heads=8.
    expected_sliding = 2 * hidden * (16 * 256) + 2 * hidden * (8 * 256)

    assert full_attn_only == expected_full
    assert sliding_attn_only == expected_sliding
    assert full_attn_only != sliding_attn_only


def test_gemma4_not_detected_for_other_model_types():
    """A non-Gemma4 config sharing some field names must not be routed
    through this dedicated estimator."""
    assert _gemma4_estimate({"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 12}) is None


def test_dsa_indexer_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm.models.deepseek_v32's real
    Indexer.__init__ (wq_b, wk, weights_proj) -- glm_moe_dsa.py is a
    thin subclass with no per-layer sharing logic, so mlx-lm 0.31.3
    actually builds this on every MLA layer regardless of what
    config.json's indexer_types says."""
    c = {"q_lora_rank": 2048, "index_n_heads": 32, "index_head_dim": 128}
    hidden = 6144
    expected = (
        2048 * (32 * 128)  # wq_b
        + hidden * 128  # wk
        + hidden * 32  # weights_proj
    )
    assert _dsa_indexer_params(c, hidden) == expected


def test_bailing_linear_attention_matches_real_mlx_lm_layer():
    """Exact param count check against mlx_lm.models.bailing_moe_linear's
    real LinearAttention.__init__ -- a FUSED query_key_value projection
    (not separate q/k/v matrices, which an earlier research pass
    guessed wrong), dense output, and g_proj gate. The class hardcodes
    its own KV head count to equal num_attention_heads internally,
    ignoring config's num_key_value_heads (that field only applies to
    this architecture's separate, standard-GQA global Attention class)."""
    from blune_cli.size_estimate import _bailing_linear_attn_params

    c = {"num_attention_heads": 32, "num_key_value_heads": 4}
    hidden = 4096
    head_dim = hidden // 32
    expected = (
        hidden * (32 + 2 * 32) * head_dim  # query_key_value (kv forced = heads)
        + 32 * head_dim * hidden  # dense
        + hidden * 32 * head_dim  # g_proj
    )
    assert _bailing_linear_attn_params(c, hidden) == expected


def test_bailing_is_global_layer_pattern_matches_real_source():
    """Exact per-layer pattern check against
    mlx_lm.models.bailing_moe_linear.DecoderLayer's real is_global
    formula -- every layer_group_size-th layer plus any trailing
    remainder layers are global (standard attention); the rest are
    LinearAttention (no growing KV-cache)."""
    from blune_cli.size_estimate import _bailing_is_global_layers

    result = _bailing_is_global_layers({"layer_group_size": 8}, layers=32)
    expected = [(i + 1) % 8 == 0 or i >= (32 // 8) * 8 for i in range(32)]
    assert result == expected
    assert sum(result) == 4  # layers 7, 15, 23, 31 (0-indexed)


def test_bailing_moe_linear_layers_excluded_from_kv_growth():
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 9216,
        "vocab_size": 157184,
        "layer_group_size": 8,
    }
    a = _analyze(config)
    assert a.n_ssm_layers == 28
    assert a.layer_kinds.count("full") == 4


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


def _mamba1_config(**overrides):
    # Real field names/values from mlx-community/mamba-130m-hf-bf16.
    config = {
        "model_type": "mamba",
        "hidden_size": 768,
        "num_hidden_layers": 24,
        "intermediate_size": 1536,
        "state_size": 16,
        "conv_kernel": 4,
        "time_step_rank": 48,
        "vocab_size": 50280,
    }
    config.update(overrides)
    return config


def test_pure_mamba1_layers_detected_as_ssm_not_generic_attention():
    """Regression test: a config with NO num_attention_heads field (a
    classic, non-hybrid Mamba-1 architecture) has no layer_types,
    sliding_window, or layer_group_size field either, so the generic
    layer_kinds detection fell through to 'full' (ordinary attention)
    for every layer -- then priced each one as a generic 4*hidden*hidden
    block, since there were no heads/kv_heads/head_dim to build a real
    attention formula from either. A real mamba-130m-hf-bf16 config was
    inflated from ~130M real params to 218.8M (+68%) by this."""
    a = _analyze(_mamba1_config())
    assert a is not None
    assert set(a.layer_kinds) == {"ssm"}
    assert a.is_pure_ssm is True


def test_pure_mamba1_layer_has_no_separate_mlp():
    """Regression test for a second bug found alongside the detection
    gap: mlx_lm.models.mamba.ResidualBlock has ONLY a mixer and a norm,
    no separate feed-forward block at all (unlike GatedDeltaNet/LFM2
    hybrid layers, which each pair their mixer with its own MLP). Even
    after fixing SSM-layer detection, the generic per-layer loop was
    still adding a phantom dense MLP on top of every Mamba layer."""
    config = _mamba1_config()
    total = estimate_total_params(config)
    # Real ~130M-param model (with tied embeddings, see below): a
    # spurious per-layer dense MLP (using intermediate_size=1536, which
    # coincidentally IS also Mamba's own d_inner field) would add
    # roughly another 85M on top -- assert well below that ceiling.
    assert 100_000_000 < total < 180_000_000, f"expected ~130M-ish, got {total}"


def test_hybrid_ssm_still_gets_its_own_mlp():
    """The no-separate-MLP rule must NOT apply to hybrid architectures
    where the same layer legitimately has both a mixer AND its own MLP
    (GatedDeltaNet, LFM2 ShortConv) -- only to a genuinely pure,
    single-mixer-type architecture like classic Mamba-1."""
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "intermediate_size": 8192,
        "vocab_size": 32000,
    }
    a = _analyze(config)
    assert a is not None
    assert a.is_pure_ssm is False


def test_tied_embeddings_not_double_counted():
    """Regression test: config.json's tie_word_embeddings was never read
    anywhere -- every total-params estimate assumed a separate lm_head
    matrix even for repos that tie it to the embedding table. Confirmed
    on a real mlx-community/mamba-130m-hf-bf16 safetensors index (only
    `backbone.embeddings.weight`, no separate lm_head weight at all,
    despite config.json having no explicit tie_word_embeddings field --
    classic Mamba always ties by architectural convention)."""
    tied_total = estimate_total_params(_mamba1_config())
    untied_total = estimate_total_params(_mamba1_config(tie_word_embeddings=False))
    hidden, vocab = 768, 50280
    assert untied_total - tied_total == hidden * vocab


def test_lfm2_auto_adjust_ff_dim_uses_real_recomputed_width():
    """Regression test for this project's single most impactful fix:
    real cached Huihui-LFM2.5-1.2B-Instruct-abliterated-8bit config
    declares intermediate_size/block_ff_dim=12288, but
    mlx_lm.models.lfm2.MLP.__init__ does NOT use that directly when
    block_auto_adjust_ff_dim is set -- it recomputes a LLaMA-style
    SwiGLU width (2/3 scaling, optional multiplier, rounded up to
    block_multiple_of), landing on a real width of 8192. Using 12288
    directly overcounted this architecture's dense MLP bytes by 50%,
    and was long mistaken for an inherent 'small/fast model' modeling
    limit rather than a fixable bug (see probe_formula.py's docstring)."""
    config = {
        "model_type": "lfm2",
        "block_ff_dim": 12288,
        "block_multiple_of": 256,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": True,
    }
    assert _lfm2_dense_mlp_width(config) == 8192


def test_lfm2_moe_variant_not_affected_by_auto_adjust():
    """lfm2_moe.py's MLP class takes intermediate_size directly with no
    such recompute (confirmed reading its source) -- this must only
    apply to the dense `lfm2` model_type, not `lfm2_moe`, even if the
    latter somehow also set block_auto_adjust_ff_dim."""
    config = {
        "model_type": "lfm2_moe",
        "block_ff_dim": 12288,
        "block_multiple_of": 256,
        "block_auto_adjust_ff_dim": True,
    }
    assert _lfm2_dense_mlp_width(config) is None


def test_lfm2_dense_mlp_width_none_without_auto_adjust_flag():
    """No recompute should happen for an lfm2 config that doesn't set
    block_auto_adjust_ff_dim -- the declared width is already real."""
    config = {"model_type": "lfm2", "block_ff_dim": 8192, "intermediate_size": 8192}
    assert _lfm2_dense_mlp_width(config) is None
