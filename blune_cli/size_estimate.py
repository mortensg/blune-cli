"""Rough, architecture-agnostic parameter-count estimate from config.json
alone -- used only as a cheap safety pre-filter before `blune sweep` spawns
a real probe, so an absurdly large model (hundreds of billions of params)
doesn't get attempted and OOM-kill the machine. Not meant to be precise;
`probe_mlx.py`'s actual model construction is the real source of truth."""
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


def _infer_bits(config: dict) -> int:
    """Bits-per-weight: prefer an explicit quantization config, then fall
    back to the declared torch dtype for unquantized (fp16/bf16) repos --
    NOT a blind 4-bit default, which silently treated every unquantized
    bf16/fp16 model (2-4x the real bytes-per-weight) as if it were 4-bit
    quantized and made e.g. a "-bf16" and a "-4bit" repo of the same model
    report identical speed."""
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict) and "bits" in quant:
        return quant["bits"]

    c = config.get("text_config", config)
    dtype = config.get("torch_dtype") or config.get("dtype") or c.get("torch_dtype") or c.get("dtype")
    if dtype:
        bits = _DTYPE_BITS.get(str(dtype).lower())
        if bits:
            return bits

    return 4  # last resort: most curated repos are 4-bit MLX conversions


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
    return int(params * _infer_bits(config) / 8)


def estimate_active_bytes_per_token(config: dict) -> Optional[int]:
    """Estimate bytes read from memory to produce ONE decode token -- the
    quantity the bandwidth law (tok/s = bandwidth / bytes-per-token) needs.
    Unlike estimate_total_params (used for the RAM-fit safety check), this
    counts only ACTIVE params per token: for MoE models that's
    num_experts_per_tok experts, not all of them. See probe_formula.py for
    how this is turned into a tok/s estimate."""
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
        attn_params_per_layer = 2 * hidden * (heads * head_dim) + 2 * hidden * (kv_heads * head_dim)
    else:
        attn_params_per_layer = 4 * hidden * hidden

    n_experts = (
        c.get("num_local_experts")
        or c.get("num_experts")
        or c.get("n_routed_experts")
        or c.get("moe_num_experts")
    )
    experts_per_tok = c.get("num_experts_per_tok") or c.get("num_activated_experts") or 1
    if n_experts:
        inter = c.get("moe_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        mlp_active_params_per_layer = 2 * hidden * inter * experts_per_tok
    else:
        inter = c.get("intermediate_size") or 4 * hidden
        mlp_active_params_per_layer = 2 * hidden * inter

    active_params = layers * (attn_params_per_layer + mlp_active_params_per_layer)
    active_params += vocab * hidden  # lm_head projection, read every decode step

    return int(active_params * _infer_bits(config) / 8)


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
