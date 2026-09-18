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

Known simplification: weight-parameter counts (used for RAM sizing) use
one generic per-layer attention+MLP formula for every layer, including
hybrid Mamba/SSM/linear-attention layers, which have a materially
different weight structure (in_proj/conv1d/x_proj/dt_proj/out_proj) that
config.json field names aren't consistent enough across converters to
parse generically yet. Where hybrid layers DO get modeled precisely is
the part that matters most for decode speed: they're excluded from the
growing per-token KV-cache read term, since that's the actual point of
those layers (see estimate_kv_bytes_per_token).
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
# has no growing KV-cache" -- Mamba/SSM/linear-attention variants.
_SSM_LAYER_TYPES = {"linear_attention", "mamba", "mamba2", "ssm", "recurrent"}
_SLIDING_LAYER_TYPES = {"sliding_attention", "local_attention"}

# KV-cache entries are conventionally stored at fp16/bf16 regardless of
# the model's own weight quantization (MLX/llama.cpp/vLLM's default
# behavior) -- a separately configurable KV-cache dtype (e.g. vLLM's FP8
# cache) is an engine-level serving choice out of scope for this local,
# single-request estimate.
_KV_CACHE_BYTES_PER_ELEM = 2


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


@dataclass
class ArchProfile:
    hidden: int
    layers: int
    vocab: int
    attn_weight_params_per_layer: float
    is_mla: bool
    kv_elems_per_token_per_attn_layer: float  # MLA latent size, or 2*kv_heads*head_dim
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
        attn_weight_params_per_layer = 2 * hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim)
    else:
        attn_weight_params_per_layer = 4 * hidden * hidden

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
    if isinstance(layer_types, list) and len(layer_types) == layers:
        n_ssm_layers = sum(1 for t in layer_types if t in _SSM_LAYER_TYPES)
        n_sliding_attn_layers = sum(1 for t in layer_types if t in _SLIDING_LAYER_TYPES)
        n_full_attn_layers = layers - n_ssm_layers - n_sliding_attn_layers
    else:
        n_ssm_layers = 0
        # No per-layer info to say which layers are sliding vs. full -- if
        # sliding_window is declared at all, the common case (Mistral-style)
        # is that every layer uses it uniformly.
        if sliding_window:
            n_sliding_attn_layers = layers
            n_full_attn_layers = 0
        else:
            n_sliding_attn_layers = 0
            n_full_attn_layers = layers

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
    n_shared_experts = c.get("n_shared_experts") or c.get("n_shared_expert") or 0

    # 3 projections per MLP block (gate/up/down), matching the SwiGLU/
    # gated-MLP structure nearly every current architecture uses (Llama,
    # Qwen, Mistral, DeepSeek, Gemma, ...) -- using 2 here undercounted
    # every MLP-heavy (i.e. almost every) model's params by ~33%.
    if n_experts:
        moe_inter = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        expert_params = 3 * hidden * moe_inter
        mlp_moe_active_params = expert_params * (experts_per_tok + n_shared_experts)
        mlp_moe_total_params = expert_params * (n_experts + n_shared_experts)
        router_params_per_moe_layer = hidden * n_experts
    else:
        mlp_moe_active_params = 0.0
        mlp_moe_total_params = 0.0
        router_params_per_moe_layer = 0.0

    dense_inter = c.get("intermediate_size") or 4 * hidden
    mlp_dense_params = 3 * hidden * dense_inter

    # DeepSeek-style: first_k_dense_replace layers (and every moe_layer_freq-th
    # layer thereafter) use the dense MLP instead of MoE experts.
    first_k_dense = c.get("first_k_dense_replace", 0) or 0
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
        is_mla=is_mla,
        kv_elems_per_token_per_attn_layer=kv_elems_per_token_per_attn_layer,
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
    a = _analyze(config)
    if a is None:
        return None

    total = 0.0
    for is_moe_layer in a.moe_layer_mask or [False] * a.layers:
        mlp = (a.mlp_moe_total_params + a.router_params_per_moe_layer) if is_moe_layer else a.mlp_dense_params
        total += a.attn_weight_params_per_layer + mlp

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
    a = _analyze(config)
    if a is None:
        return None

    bits = _infer_bits(config)

    # 1. active weight bytes
    active_params = 0.0
    for is_moe_layer in a.moe_layer_mask or [False] * a.layers:
        mlp = (a.mlp_moe_active_params + a.router_params_per_moe_layer) if is_moe_layer else a.mlp_dense_params
        active_params += a.attn_weight_params_per_layer + mlp
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
    a = _analyze(config)
    if a is None:
        return None
    return _kv_bytes(a, context_length)


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
