"""Rough, architecture-agnostic parameter-count estimate from config.json
alone -- used only as a cheap safety pre-filter before `blune sweep` spawns
a real probe, so an absurdly large model (hundreds of billions of params)
doesn't get attempted and OOM-kill the machine. Not meant to be precise;
`probe_mlx.py`'s actual model construction is the real source of truth."""
from typing import Optional


def estimate_total_params(config: dict) -> Optional[int]:
    c = config.get("text_config", config)

    hidden = c.get("hidden_size")
    layers = c.get("num_hidden_layers")
    if not hidden or not layers:
        return None

    vocab = c.get("vocab_size") or config.get("vocab_size") or 0

    heads = c.get("num_attention_heads")
    kv_heads = c.get("num_key_value_heads") or heads
    head_dim = c.get("head_dim") or (hidden // heads if heads else None)
    if heads and kv_heads and head_dim:
        # GQA-aware: q_proj + o_proj scale with `heads`, k_proj + v_proj with `kv_heads`
        attn_params_per_layer = 2 * hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim)
    else:
        attn_params_per_layer = 4 * hidden * hidden  # fallback: assume vanilla MHA

    n_experts = (
        c.get("num_local_experts")
        or c.get("num_experts")
        or c.get("n_routed_experts")
        or c.get("moe_num_experts")
    )
    # MoE models size their per-expert MLP with a dedicated (usually much
    # smaller) field -- falling back to the dense `intermediate_size` here
    # was the bug that made this wildly overestimate MoE models.
    if n_experts:
        inter = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        mlp_params_per_layer = 2 * hidden * inter * n_experts
    else:
        inter = c.get("intermediate_size") or 4 * hidden
        mlp_params_per_layer = 2 * hidden * inter

    total = layers * (attn_params_per_layer + mlp_params_per_layer)
    total += 2 * vocab * hidden  # embedding + lm_head, worst case untied
    return int(total)


def estimate_bytes(config: dict) -> Optional[int]:
    """Estimate total in-memory weight bytes at this repo's declared
    quantization (defaults to 4-bit if unspecified, matching probe_mlx's
    own default)."""
    params = estimate_total_params(config)
    if params is None:
        return None
    quant = config.get("quantization") or config.get("quantization_config") or {}
    bits = quant.get("bits", 4) if isinstance(quant, dict) else 4
    return int(params * bits / 8)


def fits_in_ram(config: dict, total_ram_gb: float, safety_margin: float = 0.7) -> bool:
    """True if we're not confident the model is too big -- i.e. fail open
    when we can't estimate, since an occasional real OOM in a subprocess is
    recoverable (the sweep just marks it FAILED and moves on), but skipping
    every model we can't parse would defeat the point of the sweep."""
    est_bytes = estimate_bytes(config)
    if est_bytes is None:
        return True
    budget_bytes = total_ram_gb * (1024**3) * safety_margin
    return est_bytes <= budget_bytes
