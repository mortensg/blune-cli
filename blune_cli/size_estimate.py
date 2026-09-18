"""Architecture-agnostic parameter/byte estimates from config.json alone.

Used by two callers with different accuracy needs:
  - `fits_in_ram` / `estimate_bytes`: a cheap safety pre-filter before
    `blune sweep` spawns a real probe, so an absurdly large model doesn't
    get attempted and OOM-kill the machine. Approximate is fine here.
  - `probe_formula.py`: the *only* thing standing between this estimate
    and a reported tok/s number, so this module is where the real
    architectural correctness work lives.

Methodology and field-name catalog below come from a research pass into
what a truly general "will this run, how fast" model needs (MoE routing,
shared experts, GQA/MLA, hybrid Mamba/attention layers, sliding-window
attention, per-layer mixed quantization, dense/MoE layer interleaving via
first_k_dense_replace) -- see docs/research-findings.md for the full
derivations this was built from. `probe_mlx.py`'s real model construction
remains the ground truth when architectural detail matters more than
speed; this module is the fast, config-only approximation.

SSM/hybrid layer weight params: two real layer implementations are
supported, matched against mlx-lm's actual source
(site-packages/mlx_lm/models/{gated_delta,mamba}.py) rather than
guessed -- Qwen3.5/3.6/3-Next's GatedDeltaNet (linear_num_value_heads
etc. fields) and classic Mamba/Mamba2/Jamba-style SSM blocks
(mamba_d_state/d_state etc. fields). An architecture using neither
naming convention falls back to the generic attention+MLP formula for
that layer's weight count, which is known to be wrong for genuine SSM
layers -- see docs/formula-accuracy-gap.md for how wrong (a hybrid
model using GatedDeltaNet, before this was added, was over-predicted
by 32-39% because its SSM layers were undercounted as much smaller
attention layers).
"""
from dataclasses import dataclass
from typing import Optional

_DTYPE_BITS = {
    "float32": 32,
    "fp32": 32,
    "float16": 16,
    "fp16": 16,
    "bfloat16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
}

# layer_types strings (as seen across real configs) that mean "this layer
# has no growing KV-cache" -- Mamba/SSM/linear-attention/conv variants.
_SSM_LAYER_TYPES = {"linear_attention", "mamba", "mamba2", "ssm", "recurrent", "conv"}
_SLIDING_LAYER_TYPES = {"sliding_attention", "local_attention"}

# KV-cache entries are conventionally stored at fp16/bf16 regardless of
# the model's own weight quantization (MLX/llama.cpp/vLLM's default
# behavior) -- a separately configurable KV-cache dtype (e.g. vLLM's FP8
# cache) is an engine-level serving choice out of scope for this local,
# single-request estimate.
_KV_CACHE_BYTES_PER_ELEM = 2

# Per-architecture-family quirks that don't generalize from config.json
# field presence alone -- each entry here is verified against mlx-lm's
# real source, not inferred from a heuristic. A plausible-looking generic
# rule was tried and rejected for this specific case: the config field
# `attn_output_gate: true` looked like it might signal q_proj doubling,
# but Gemma4 also sets it without doubling its own q_proj -- using it
# generically would have silently broken Gemma4's already-validated
# attention param count to fix Qwen3-Next's. Keyed by model_type instead.
# mlx_lm.models.qwen3_next.Qwen3NextAttention.q_proj outputs
# num_attention_heads*head_dim*2 (packs query + an output gate vector
# into one GEMM); qwen3_5.py and qwen3_5_moe.py both import this same
# Attention class for their "full attention" layers.
_Q_PROJ_MULTIPLIER_BY_MODEL_TYPE = {
    "qwen3_next": 2.0,
    "qwen3_5": 2.0,
    "qwen3_5_moe": 2.0,
}


def _infer_bits(config: dict, path_hint: str = "") -> int:
    """Bits-per-weight: prefer an explicit quantization config (honoring
    a per-path override for e.g. lm_head/embed_tokens when path_hint is
    given -- MLX/GGUF repos commonly quantize those at a different bit
    width than the rest of the model), then fall back to the declared
    torch dtype for unquantized (fp16/bf16) repos -- NOT a blind 4-bit
    default, which silently treated every unquantized bf16/fp16 model as
    if it were 4-bit quantized."""
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict):
        if path_hint:
            for key, override in quant.items():
                if path_hint in key and isinstance(override, dict) and "bits" in override:
                    return override["bits"]
        if "bits" in quant:
            return quant["bits"]

    c = config.get("text_config", config)
    dtype = config.get("torch_dtype") or config.get("dtype") or c.get("torch_dtype") or c.get("dtype")
    if dtype:
        bits = _DTYPE_BITS.get(str(dtype).lower())
        if bits:
            return bits

    return 4  # last resort: most curated repos are 4-bit MLX conversions


def _gated_delta_net_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for Qwen3.5/3.6/3-Next's GatedDeltaNet linear-
    attention layer, matching mlx_lm.models.gated_delta.GatedDeltaNet's
    actual __init__ (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
    depthwise conv1d, out_proj) -- not a generic approximation."""
    num_v_heads = c.get("linear_num_value_heads")
    num_k_heads = c.get("linear_num_key_heads")
    head_k_dim = c.get("linear_key_head_dim")
    head_v_dim = c.get("linear_value_head_dim")
    conv_kernel = c.get("linear_conv_kernel_dim")
    if not all([num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_kernel]):
        return None

    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim

    in_proj_qkv = hidden * (key_dim * 2 + value_dim)
    in_proj_z = hidden * value_dim
    in_proj_b = hidden * num_v_heads
    in_proj_a = hidden * num_v_heads
    conv1d = conv_dim * conv_kernel  # depthwise, no bias
    out_proj = value_dim * hidden

    return in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + conv1d + out_proj


def _mamba_ssm_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for a classic Mamba/Mamba2 SSM block (Jamba-style
    hybrid architectures), matching mlx_lm.models.mamba.MambaBlock's
    actual __init__ (in_proj, depthwise conv1d, x_proj, dt_proj,
    out_proj)."""
    d_state = c.get("mamba_d_state") or c.get("d_state") or c.get("ssm_state_size") or c.get("state_size")
    d_conv = c.get("mamba_d_conv") or c.get("d_conv") or c.get("conv_kernel") or c.get("conv_kernel_size")
    d_inner = c.get("mamba_d_inner") or c.get("intermediate_size")
    if not d_inner:
        expand = c.get("mamba_expand") or c.get("expand") or c.get("expand_factor")
        d_inner = expand * hidden if expand else None
    dt_rank = c.get("dt_rank") or c.get("mamba_dt_rank") or c.get("time_step_rank")
    if not dt_rank and hidden:
        dt_rank = -(-hidden // 16)  # ceil(hidden/16), Mamba's own default when unspecified
    if not all([d_state, d_conv, d_inner]):
        return None

    in_proj = hidden * 2 * d_inner
    conv1d = d_inner * d_conv  # depthwise
    x_proj = d_inner * (dt_rank + 2 * d_state)
    dt_proj = dt_rank * d_inner
    out_proj = d_inner * hidden

    return in_proj + conv1d + x_proj + dt_proj + out_proj


def _mamba2_ssm_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for a Mamba-2 SSM block (NVIDIA Nemotron-H, IBM
    Granite hybrid), matching mlx_lm.models.{nemotron_h,granitemoehybrid}
    's real Mamba2Mixer.__init__: a single fused in_proj (not Mamba-1's
    separate x_proj/dt_proj), depthwise conv1d, out_proj. Checked BEFORE
    the classic Mamba-1 formula in the dispatcher, since nemotron_h's
    field names (ssm_state_size, conv_kernel) would otherwise also match
    Mamba-1's fallback chain and silently use the wrong (Mamba-1) shape."""
    n_heads = c.get("mamba_num_heads") or c.get("mamba_n_heads")
    head_dim = c.get("mamba_head_dim") or c.get("mamba_d_head")
    d_state = c.get("ssm_state_size") or c.get("mamba_d_state")
    d_conv = c.get("conv_kernel") or c.get("mamba_d_conv")
    if not all([n_heads, head_dim, d_state, d_conv]):
        return None
    n_groups = c.get("n_groups") or c.get("mamba_n_groups") or 1

    proj_bias = bool(c.get("mamba_proj_bias"))
    conv_bias = bool(c.get("use_conv_bias", c.get("mamba_conv_bias")))

    intermediate = n_heads * head_dim
    conv_dim = intermediate + 2 * n_groups * d_state
    projection_size = intermediate + conv_dim + n_heads

    in_proj = hidden * projection_size + (projection_size if proj_bias else 0)
    conv1d = conv_dim * d_conv + (conv_dim if conv_bias else 0)  # depthwise
    out_proj = intermediate * hidden + (hidden if proj_bias else 0)

    return in_proj + conv1d + out_proj


def _lfm2_conv_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for LFM2's ShortConv block, matching
    mlx_lm.models.lfm2.ShortConv's real __init__ -- NOT a state-space
    model at all despite living in a hybrid architecture: in_proj
    (hidden -> 3*hidden), a depthwise conv over `hidden` channels with
    kernel size conv_L_cache, and out_proj (hidden -> hidden)."""
    conv_kernel = c.get("conv_L_cache")
    if not conv_kernel:
        return None
    bias = bool(c.get("conv_bias"))

    in_proj = hidden * 3 * hidden + (3 * hidden if bias else 0)
    conv = hidden * conv_kernel + (hidden if bias else 0)  # depthwise
    out_proj = hidden * hidden + (hidden if bias else 0)

    return in_proj + conv + out_proj


def _bailing_linear_attn_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for InclusionAI/Ring's Lightning Attention layer
    (bailing_moe_linear's real LinearAttention class), matching
    mlx_lm.models.bailing_moe_linear.LinearAttention.__init__ exactly:
    a FUSED query_key_value projection (not separate q/k/v matrices --
    an earlier research pass guessed wrong here), a `dense` output
    projection, and a `g_proj` gate. The class hardcodes its own KV head
    count to always equal num_attention_heads internally (ignoring
    config's num_key_value_heads, which only applies to this
    architecture's separate, standard-GQA `Attention` class used on
    "global" layers -- see _bailing_is_global_layers)."""
    heads = c.get("num_attention_heads")
    if not heads:
        return None
    head_dim = hidden // heads

    qkv = hidden * (heads + 2 * heads) * head_dim  # kv_heads forced equal to heads
    dense = heads * head_dim * hidden
    g_proj = hidden * heads * head_dim

    return qkv + dense + g_proj


def _bailing_is_global_layers(c: dict, layers: int) -> Optional[list]:
    """Per-layer True/False for whether a bailing_moe_linear layer uses
    standard (global) Attention vs. LinearAttention, matching
    mlx_lm.models.bailing_moe_linear.DecoderLayer's exact formula:
    every layer_group_size-th layer, plus any trailing remainder
    layers, is global. Not a layer_types list or a simple modulo --
    verified against the real source rather than guessed."""
    group_size = c.get("layer_group_size")
    if not group_size:
        return None
    cutoff = (layers // group_size) * group_size
    return [(i + 1) % group_size == 0 or i >= cutoff for i in range(layers)]


def _ssm_layer_params(c: dict, hidden: int, fallback: float) -> float:
    """Best-available SSM/hybrid-layer weight param count: a real layer
    implementation's exact formula if the config has the fields for one,
    else `fallback` (the generic attention+MLP estimate, known
    inaccurate for genuine SSM/conv layers -- see module docstring)."""
    return (
        _gated_delta_net_params(c, hidden)
        or _mamba2_ssm_params(c, hidden)
        or _lfm2_conv_params(c, hidden)
        or _mamba_ssm_params(c, hidden)
        or (c.get("layer_group_size") and _bailing_linear_attn_params(c, hidden))
        or fallback
    )


def _mla_weight_params(c: dict, hidden: int, heads: int) -> Optional[float]:
    """Exact param count for DeepSeek-V3-style Multi-Head Latent Attention
    WEIGHTS (distinct from its KV-cache size, handled separately), matching
    mlx_lm.models.deepseek_v3.DeepseekV3Attention's real __init__:
    q_a_proj+q_a_layernorm+q_b_proj when q_lora_rank is set (else a direct
    q_proj), kv_a_proj_with_mqa+kv_a_layernorm+kv_b_proj, and o_proj.
    Replaces the generic GQA weight formula for MLA layers -- that
    approximation was never actually checked against real MLA weight
    structure before (only the KV-cache term was MLA-aware)."""
    kv_lora = c.get("kv_lora_rank")
    qk_nope = c.get("qk_nope_head_dim")
    qk_rope = c.get("qk_rope_head_dim")
    v_head_dim = c.get("v_head_dim")
    if not all([kv_lora, qk_nope, qk_rope, v_head_dim, heads]):
        return None

    q_lora = c.get("q_lora_rank")
    if q_lora:
        q_params = hidden * q_lora + q_lora * heads * (qk_nope + qk_rope)  # q_a_proj + q_b_proj
    else:
        q_params = hidden * heads * (qk_nope + qk_rope)  # direct q_proj

    kv_a_proj = hidden * (kv_lora + qk_rope)
    kv_b_proj = kv_lora * heads * (qk_nope + v_head_dim)
    o_proj = heads * v_head_dim * hidden

    return q_params + kv_a_proj + kv_b_proj + o_proj


def _nemotron_h_component(char: str, c: dict, hidden: int) -> tuple:
    """(total_params, active_params, kv_elems_per_token) for ONE
    Nemotron-H single-component layer, matching mlx_lm.models.nemotron_h
    exactly: 'M' Mamba-2 mixer, '*' attention, '-' a plain 2-matrix
    up/down MLP (ReLU2 activation, NOT SwiGLU -- no gate_proj, verified
    against NemotronHMLP directly), 'E' a MoE block (router weight matrix
    + SwitchMLP routed experts, also 2-matrix per expert + optional
    always-on shared expert sized by moe_shared_expert_intermediate_size,
    computed on the pre-latent-projection residual + optional
    fc1/fc2 latent-projection wrap for Nemotron-3-Super's LatentMoE).
    Unlike every other architecture this module handles, a Nemotron-H
    layer is EITHER a mixer OR an FFN/MoE block, never both."""
    if char == "M":
        p = _mamba2_ssm_params(c, hidden) or 0.0
        return p, p, 0.0
    if char == "*":
        heads = c.get("num_attention_heads")
        kv_heads = c.get("num_key_value_heads") or heads
        head_dim = c.get("head_dim") or (hidden // heads if heads else None)
        if heads and kv_heads and head_dim:
            p = 2 * hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim)
            kv = 2 * kv_heads * head_dim
        else:
            p = 4 * hidden * hidden
            kv = 2 * hidden
        return p, p, kv
    if char == "-":
        inter = c.get("intermediate_size") or 4 * hidden
        p = 2 * hidden * inter  # up_proj + down_proj only -- no gate_proj
        return p, p, 0.0
    if char == "E":
        n_experts = c.get("n_routed_experts") or 0
        experts_per_tok = c.get("num_experts_per_tok") or 1
        moe_inter = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        moe_latent = c.get("moe_latent_size")
        expert_input_dim = moe_latent or hidden

        router = hidden * n_experts
        latent_wrap = (hidden * moe_latent + moe_latent * hidden) if moe_latent else 0.0
        expert_unit = 2 * expert_input_dim * moe_inter  # SwitchMLP: fc1 + fc2, no gate

        n_shared = c.get("n_shared_experts")
        shared_inter = c.get("moe_shared_expert_intermediate_size")
        # Shared expert runs on the pre-latent residual (hidden), not the
        # latent-projected dim, even when moe_latent_size is set.
        shared = 2 * hidden * shared_inter if (n_shared is not None and shared_inter) else 0.0

        total = router + latent_wrap + n_experts * expert_unit + shared
        active = router + latent_wrap + experts_per_tok * expert_unit + shared
        return total, active, 0.0
    return 0.0, 0.0, 0.0  # unrecognized pattern char: fail open, not crash


def _nemotron_h_estimate(config: dict) -> Optional[dict]:
    """Dedicated estimator for Nemotron-H-style single-component-per-layer
    architectures -- bypasses the generic mixer+MLP-per-layer loop
    entirely, since forcing this architecture through it double-counts
    every MLP-only/MoE-only layer (it has no separate mixer at all: a
    real Nemotron-H config previously estimated at 103.1B for a model
    named "30B" because of this). Detected via hybrid_override_pattern,
    a single character per layer ('M'/'*'/'-'/'E')."""
    c = config.get("text_config", config)
    pattern = c.get("hybrid_override_pattern")
    if not isinstance(pattern, str) or not pattern:
        return None
    hidden = c.get("hidden_size")
    if not hidden:
        return None
    vocab = c.get("vocab_size") or config.get("vocab_size") or 0

    total_params = active_params = kv_elems_per_token = 0.0
    n_full_attn_layers = 0
    for ch in pattern:
        t, a, kv = _nemotron_h_component(ch, c, hidden)
        total_params += t
        active_params += a
        if kv:
            kv_elems_per_token = kv
            n_full_attn_layers += 1

    total_params += 2 * vocab * hidden
    active_params += vocab * hidden

    return {
        "total_params": total_params,
        "active_params": active_params,
        "kv_elems_per_token": kv_elems_per_token,
        "n_full_attn_layers": n_full_attn_layers,
    }


def _dsa_indexer_params(c: dict, hidden: int) -> Optional[float]:
    """Exact param count for GLM's Dynamic Sparse Attention indexer
    (glm_moe_dsa), matching mlx_lm.models.deepseek_v32.Indexer's real
    __init__ (wq_b, wk, weights_proj). glm_moe_dsa.py is a thin subclass
    of deepseek_v32.py's Model with NO per-layer sharing logic --
    despite config.json's indexer_types marking most layers "shared",
    mlx-lm 0.31.3 actually instantiates a full Indexer on every MLA
    layer regardless, so this is added unconditionally per MLA layer,
    not gated on indexer_types' full/shared split."""
    q_lora = c.get("q_lora_rank")
    index_heads = c.get("index_n_heads")
    index_head_dim = c.get("index_head_dim")
    if not all([q_lora, index_heads, index_head_dim]):
        return None

    wq_b = q_lora * (index_heads * index_head_dim)
    wk = hidden * index_head_dim
    weights_proj = hidden * index_heads

    return wq_b + wk + weights_proj


@dataclass
class ArchProfile:
    hidden: int
    layers: int
    vocab: int
    attn_weight_params_per_layer: float
    ssm_weight_params_per_layer: float
    is_mla: bool
    kv_elems_per_token_per_attn_layer: float  # MLA latent size, or 2*kv_heads*head_dim
    layer_kinds: list  # per-layer: "full", "sliding", or "ssm"
    n_full_attn_layers: int  # KV cache grows unbounded with context
    n_sliding_attn_layers: int  # KV cache capped at sliding_window
    n_ssm_layers: int  # no growing KV cache at all (Mamba/SSM/linear-attention)
    sliding_window: Optional[int]
    n_experts: int
    experts_per_tok: int
    n_shared_experts: int
    moe_layer_mask: list  # per-layer bool: True if that layer's MLP is MoE
    mlp_dense_params: float  # per dense-MLP-layer params
    mlp_moe_active_params: float  # per MoE-layer active params (routed + shared, excl. router)
    mlp_moe_total_params: float  # per MoE-layer total params (all routed + shared)
    router_params_per_moe_layer: float


def _analyze(config: dict) -> Optional[ArchProfile]:
    c = config.get("text_config", config)

    hidden = c.get("hidden_size")
    layers = c.get("num_hidden_layers")
    if not hidden or not layers:
        return None

    vocab = c.get("vocab_size") or config.get("vocab_size") or 0

    # --- attention: MLA vs. GQA/MHA, and per-token KV-cache footprint ---
    heads = c.get("num_attention_heads")
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or (hidden // heads if heads else None)

    kv_lora_rank = c.get("kv_lora_rank")
    qk_rope_head_dim = c.get("qk_rope_head_dim")
    is_mla = bool(kv_lora_rank and qk_rope_head_dim)

    if heads and kv_heads and head_dim:
        q_multiplier = _Q_PROJ_MULTIPLIER_BY_MODEL_TYPE.get(
            config.get("model_type") or c.get("model_type"), 1.0
        )
        q_proj = hidden * (heads * head_dim) * q_multiplier
        o_proj = hidden * (heads * head_dim)  # o_proj reads the (non-doubled) attention output
        kv_proj = 2 * hidden * (kv_heads * head_dim)
        attn_weight_params_per_layer = q_proj + o_proj + kv_proj
    else:
        attn_weight_params_per_layer = 4 * hidden * hidden

    if is_mla:
        # The generic GQA formula above was never actually right for MLA's
        # very different low-rank projection structure (q_a/q_b, kv_a/kv_b)
        # -- only the KV-cache side was MLA-aware before. Use the real
        # formula when we have the fields for it.
        mla_params = _mla_weight_params(c, hidden, heads)
        if mla_params is not None:
            attn_weight_params_per_layer = mla_params
        indexer_params = _dsa_indexer_params(c, hidden)
        if indexer_params is not None:
            attn_weight_params_per_layer += indexer_params

    if is_mla:
        # MLA caches only the compressed latent + decoupled RoPE key per
        # token per layer, not full per-head K/V -- typically 50-100x
        # smaller than standard MHA/GQA cache. This is THE reason MLA
        # exists; approximating it as GQA was a real accuracy gap.
        kv_elems_per_token_per_attn_layer = kv_lora_rank + qk_rope_head_dim
    elif kv_heads and head_dim:
        kv_elems_per_token_per_attn_layer = 2 * kv_heads * head_dim
    else:
        kv_elems_per_token_per_attn_layer = 2 * hidden  # fallback

    # --- layer_types: SSM/Mamba layers (no growing KV-cache at all) vs.
    # sliding-window layers (KV capped at sliding_window) vs. full attention
    sliding_window = c.get("sliding_window")
    layer_types = c.get("layer_types")
    bailing_is_global = _bailing_is_global_layers(c, layers)
    if isinstance(layer_types, list) and len(layer_types) == layers:
        layer_kinds = [
            "ssm" if t in _SSM_LAYER_TYPES else "sliding" if t in _SLIDING_LAYER_TYPES else "full"
            for t in layer_types
        ]
    elif bailing_is_global is not None:
        # bailing_moe_linear: LinearAttention layers have no growing
        # KV-cache (a fixed recurrent state, like GatedDeltaNet/Mamba);
        # only the "global" layers use standard attention.
        layer_kinds = ["full" if g else "ssm" for g in bailing_is_global]
    elif sliding_window:
        # No per-layer info to say which layers are sliding vs. full -- if
        # sliding_window is declared at all, the common case (Mistral-style)
        # is that every layer uses it uniformly.
        layer_kinds = ["sliding"] * layers
    else:
        layer_kinds = ["full"] * layers

    n_ssm_layers = layer_kinds.count("ssm")
    n_sliding_attn_layers = layer_kinds.count("sliding")
    n_full_attn_layers = layer_kinds.count("full")

    ssm_weight_params_per_layer = (
        _ssm_layer_params(c, hidden, fallback=attn_weight_params_per_layer) if n_ssm_layers else 0.0
    )

    # --- MoE: experts, shared experts, router, and dense/MoE interleaving ---
    n_experts = (
        c.get("num_local_experts")
        or c.get("num_experts")
        or c.get("n_routed_experts")
        or c.get("moe_num_experts")
        or 0
    )
    experts_per_tok = (
        c.get("num_experts_per_tok")
        or c.get("num_activated_experts")
        or c.get("top_k_experts")
        or c.get("moe_top_k")
        or c.get("moe_k")
        or (1 if n_experts else 1)
    )
    n_shared_experts = (
        c.get("n_shared_experts") or c.get("n_shared_expert") or c.get("num_shared_experts") or 0
    )
    # Qwen3-Next-family MoE (mlx_lm's Qwen3NextSparseMoeBlock) always has
    # exactly one always-on shared expert with its OWN intermediate size,
    # signaled by one of these fields rather than a count field -- missing
    # this entirely undercounted every layer's active bytes by a full
    # extra expert's worth of compute. Field name varies by converter
    # (Qwen3-Next/GLM use shared_expert_intermediate_size, Nemotron-H uses
    # moe_shared_expert_intermediate_size, Granite uses shared_intermediate_size).
    shared_expert_inter = (
        c.get("shared_expert_intermediate_size")
        or c.get("moe_shared_expert_intermediate_size")
        or c.get("shared_intermediate_size")
    )
    if not n_shared_experts and shared_expert_inter:
        n_shared_experts = 1

    # 3 projections per MLP block (gate/up/down), matching the SwiGLU/
    # gated-MLP structure nearly every current architecture uses (Llama,
    # Qwen, Mistral, DeepSeek, Gemma, ...) -- using 2 here undercounted
    # every MLP-heavy (i.e. almost every) model's params by ~33%.
    if n_experts:
        moe_inter = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        routed_expert_params = 3 * hidden * moe_inter
        shared_expert_params = 3 * hidden * (shared_expert_inter or moe_inter)
        mlp_moe_active_params = routed_expert_params * experts_per_tok + shared_expert_params * n_shared_experts
        mlp_moe_total_params = routed_expert_params * n_experts + shared_expert_params * n_shared_experts
        router_params_per_moe_layer = hidden * n_experts + (hidden if n_shared_experts else 0)
    else:
        mlp_moe_active_params = 0.0
        mlp_moe_total_params = 0.0
        router_params_per_moe_layer = 0.0

    dense_inter = c.get("intermediate_size") or 4 * hidden
    mlp_dense_params = 3 * hidden * dense_inter

    # DeepSeek-style: first_k_dense_replace layers (and every moe_layer_freq-th
    # layer thereafter) use the dense MLP instead of MoE experts. LFM2-MoE
    # uses a differently-named but equivalent field (num_dense_layers).
    first_k_dense = c.get("first_k_dense_replace") or c.get("num_dense_layers") or 0
    moe_freq = c.get("moe_layer_freq", 1) or 1
    if n_experts:
        moe_layer_mask = [
            (i >= first_k_dense) and ((i - first_k_dense) % moe_freq == 0) for i in range(layers)
        ]
    else:
        moe_layer_mask = [False] * layers

    return ArchProfile(
        hidden=hidden,
        layers=layers,
        vocab=vocab,
        attn_weight_params_per_layer=attn_weight_params_per_layer,
        ssm_weight_params_per_layer=ssm_weight_params_per_layer,
        is_mla=is_mla,
        kv_elems_per_token_per_attn_layer=kv_elems_per_token_per_attn_layer,
        layer_kinds=layer_kinds,
        n_full_attn_layers=n_full_attn_layers,
        n_sliding_attn_layers=n_sliding_attn_layers,
        n_ssm_layers=n_ssm_layers,
        sliding_window=sliding_window,
        n_experts=n_experts,
        experts_per_tok=experts_per_tok,
        n_shared_experts=n_shared_experts,
        moe_layer_mask=moe_layer_mask,
        mlp_dense_params=mlp_dense_params,
        mlp_moe_active_params=mlp_moe_active_params,
        mlp_moe_total_params=mlp_moe_total_params,
        router_params_per_moe_layer=router_params_per_moe_layer,
    )


def estimate_total_params(config: dict) -> Optional[int]:
    """Total resident parameters (used for RAM sizing) -- includes ALL
    routed + shared experts (they all sit in memory even though only a
    few are active per token), router weights, and correctly splits
    dense-MLP vs. MoE-MLP layers where the config declares interleaving
    (first_k_dense_replace)."""
    nemotron = _nemotron_h_estimate(config)
    if nemotron is not None:
        return int(nemotron["total_params"])

    a = _analyze(config)
    if a is None:
        return None

    total = 0.0
    moe_mask = a.moe_layer_mask or [False] * a.layers
    for i in range(a.layers):
        mixer = a.ssm_weight_params_per_layer if a.layer_kinds[i] == "ssm" else a.attn_weight_params_per_layer
        mlp = (a.mlp_moe_total_params + a.router_params_per_moe_layer) if moe_mask[i] else a.mlp_dense_params
        total += mixer + mlp

    total += 2 * a.vocab * a.hidden  # embedding + lm_head, worst case untied
    return int(total)


def estimate_bytes(config: dict) -> Optional[int]:
    """Estimate total in-memory weight bytes at this repo's declared
    quantization (or inferred dtype if unquantized)."""
    a = _analyze(config)
    if a is None:
        return None

    bits = _infer_bits(config)
    embed_bits = _infer_bits(config, path_hint="embed_tokens")
    lm_head_bits = _infer_bits(config, path_hint="lm_head")

    weight_params = estimate_total_params(config) - 2 * a.vocab * a.hidden
    total_bytes = weight_params * bits / 8
    total_bytes += a.vocab * a.hidden * embed_bits / 8
    total_bytes += a.vocab * a.hidden * lm_head_bits / 8
    return int(total_bytes)


def estimate_active_bytes_per_token(config: dict, context_length: int = 128) -> Optional[int]:
    """Bytes read from memory to produce ONE decode token at the given
    context length -- the quantity the bandwidth law
    (tok/s = bandwidth / bytes-per-token) needs. Two parts:

      1. Active weight bytes: for MoE, only the routed experts actually
         selected per token (experts_per_tok) plus always-on shared
         experts -- not the full expert bank. Dense/MoE-interleaved
         layers (first_k_dense_replace) are split correctly.
      2. KV-cache read bytes: grows with context_length for ordinary
         attention layers, is capped at sliding_window for sliding-window
         layers, uses the MLA-compressed latent size instead of full
         per-head K/V when the architecture is MLA, and is EXCLUDED for
         Mamba/SSM/linear-attention layers (their state doesn't grow
         with context -- that's the point of those layers).

    context_length=128 as the default matches the short-context regime
    this project's own MLX calibration measurements were taken in (see
    probe_formula.py) -- pass a larger value to see how a specific model
    degrades at long context.
    """
    nemotron = _nemotron_h_estimate(config)
    if nemotron is not None:
        bits = _infer_bits(config)
        active_weight_bytes = nemotron["active_params"] * bits / 8
        kv_bytes_per_layer = nemotron["kv_elems_per_token"] * _KV_CACHE_BYTES_PER_ELEM
        kv_bytes = nemotron["n_full_attn_layers"] * kv_bytes_per_layer * context_length
        return int(active_weight_bytes + kv_bytes)

    a = _analyze(config)
    if a is None:
        return None

    bits = _infer_bits(config)

    # 1. active weight bytes
    active_params = 0.0
    moe_mask = a.moe_layer_mask or [False] * a.layers
    for i in range(a.layers):
        mixer = a.ssm_weight_params_per_layer if a.layer_kinds[i] == "ssm" else a.attn_weight_params_per_layer
        mlp = (a.mlp_moe_active_params + a.router_params_per_moe_layer) if moe_mask[i] else a.mlp_dense_params
        active_params += mixer + mlp
    active_params += a.vocab * a.hidden  # lm_head projection, read every decode step
    active_weight_bytes = active_params * bits / 8

    kv_bytes = _kv_bytes(a, context_length)
    return int(active_weight_bytes + kv_bytes)


def _kv_bytes(a: ArchProfile, context_length: int) -> int:
    """Bytes read from the KV-cache during ONE decode step with
    `context_length` tokens already cached -- which is numerically the
    same quantity as "total bytes the KV-cache occupies at that context
    length" (every cached token's K/V is read exactly once per step).
    SSM/Mamba layers contribute nothing here (see ArchProfile.n_ssm_layers);
    sliding-window layers are capped at sliding_window regardless of how
    long the real conversation has gotten."""
    kv_bytes_per_layer = a.kv_elems_per_token_per_attn_layer * _KV_CACHE_BYTES_PER_ELEM
    sliding_context = min(context_length, a.sliding_window) if a.sliding_window else context_length
    total = a.n_full_attn_layers * kv_bytes_per_layer * context_length
    total += a.n_sliding_attn_layers * kv_bytes_per_layer * sliding_context
    return int(total)


def estimate_kv_bytes_per_token(config: dict, context_length: int = 128) -> Optional[int]:
    """Just the KV-cache-read component of estimate_active_bytes_per_token,
    exposed separately since it's the term that changes with context
    length (weight bytes don't). Also equals the KV-cache's total memory
    footprint at that context length -- see _kv_bytes."""
    nemotron = _nemotron_h_estimate(config)
    if nemotron is not None:
        kv_bytes_per_layer = nemotron["kv_elems_per_token"] * _KV_CACHE_BYTES_PER_ELEM
        return int(nemotron["n_full_attn_layers"] * kv_bytes_per_layer * context_length)

    a = _analyze(config)
    if a is None:
        return None
    return _kv_bytes(a, context_length)


def count_moe_layers(config: dict) -> int:
    """Number of layers whose MLP is a MoE block -- used by
    probe_formula.py's per-MoE-layer dispatch-overhead term. A direct
    MLX micro-benchmark (isolating mlx_lm.models.switch_layers.SwitchGLU
    with random weights) found each MoE-routing call costs a roughly
    constant ~200us of dispatch overhead regardless of expert count,
    distinct from the model-wide fixed overhead the original formula
    used -- a flat per-decode-step constant systematically under-counted
    models where most or all layers are MoE."""
    nemotron = _nemotron_h_estimate(config)
    if nemotron is not None:
        c = config.get("text_config", config)
        pattern = c.get("hybrid_override_pattern", "")
        return sum(1 for ch in pattern if ch == "E")

    a = _analyze(config)
    if a is None or not a.moe_layer_mask:
        return 0
    return sum(a.moe_layer_mask)


def fits_in_ram(
    config: dict,
    total_ram_gb: float,
    context_length: int = 4096,
    safety_margin: float = 0.75,
) -> bool:
    """True if we're not confident the model is too big -- i.e. fail open
    when we can't estimate, since an occasional real OOM in a subprocess is
    recoverable (the sweep just marks it FAILED and moves on), but skipping
    every model we can't parse would defeat the point of the sweep.

    safety_margin defaults to 0.75, matching macOS's own default
    iogpu.wired_mem_limit clamp (a single process can lock roughly 75% of
    unified memory into GPU address space unless the operator raises it
    via `sudo sysctl iogpu.wired_mem_limit=...`) -- not a made-up number.

    Includes a KV-cache estimate at context_length (default 4096, a more
    realistic "will a real conversation fit" figure than the tiny context
    the speed formula defaults to), since weight size alone understates
    real memory needs, especially for long-context sessions."""
    est_bytes = estimate_bytes(config)
    if est_bytes is None:
        return True
    kv_bytes = estimate_kv_bytes_per_token(config, context_length) or 0
    total_needed = est_bytes + kv_bytes
    budget_bytes = total_ram_gb * (1024**3) * safety_margin
    return total_needed <= budget_bytes
