"""
MLX speed probe: estimate real tokens/sec for an MLX model on Apple Silicon
without downloading any weight data.

Method: build the model's real architecture class (from mlx-lm's own model
registry) with randomly-initialized weights, quantize it to match the repo's
declared quantization scheme, and run a real prefill+decode loop. Speed
depends on weight shape/dtype/quantization format, not on weight values, so
this gives a genuine estimate from config.json alone.

Validated on an M4 Pro (48GB) against real full-model measurements:
    Qwen3-Coder-30B-A3B-Instruct-4bit:  real 86-90 tok/s, this probe ~70-73 tok/s (~81%)
    gemma-4-26b-a4b-it-4bit:            real ~77 tok/s,   this probe ~64 tok/s   (~82%)
    Qwen2.5-Coder-7B-Instruct-4bit:     real 57 tok/s,    this probe ~48 tok/s   (~85%)

The ~81-85% ratio held consistently across three different architectures
(dense + two MoE variants) in repeated runs -- treat CALIBRATION_RATIO below
as a real, if provisional, correction factor, not a magic constant.
"""
from typing import Optional

from .size_estimate import _infer_bits

CALIBRATION_RATIO = 0.82  # probe_tok_s / real_tok_s, averaged across validation runs


def probe(
    repo_id: str,
    config: dict,
    prompt_len: int = 40,
    decode_tokens: int = 150,
    calibrate: bool = True,
) -> dict:
    """Run the zero-download MLX probe. Requires mlx and mlx-lm to be
    installed -- raises ImportError with a clear message if not (this
    library should stay optional so the CLI works on non-Apple-Silicon
    machines for the other library probes)."""
    try:
        import time

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import _get_classes
    except ImportError as e:
        raise ImportError(
            "MLX probe requires 'mlx' and 'mlx-lm' (Apple Silicon only). "
            f"Install with: pip install mlx mlx-lm  ({e})"
        )

    if "quantization_config" not in config and "quantization" in config:
        config["quantization_config"] = config["quantization"]

    Model, ModelArgs = _get_classes(config)
    args = ModelArgs.from_dict(config)
    model = Model(args)

    quantization = config.get("quantization")
    default_group_size = (quantization or {}).get("group_size", 64)

    def class_predicate(p, m):
        # Some architectures (e.g. AI21 Jamba's Mamba/SSM projections) have
        # weight dims not divisible by the group size -- nn.quantize hard
        # errors on those rather than skipping them. Quantizing everything
        # blindly isn't even the right behavior anyway: a real conversion
        # would leave such a layer unquantized too, so skipping it here is
        # both a crash fix and a more accurate match to reality.
        if not hasattr(m, "to_quantized"):
            return False
        override = quantization.get(p) if quantization else None
        if override is False:
            return False
        group_size = default_group_size
        if isinstance(override, dict):
            group_size = override.get("group_size", default_group_size)
        weight = getattr(m, "weight", None)
        if weight is not None and weight.shape[-1] % group_size != 0:
            return False
        return override if override is not None else True

    if quantization is not None:
        nn.quantize(
            model,
            group_size=default_group_size,
            bits=quantization.get("bits", 4),
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )
        quant_desc = f"{quantization.get('bits', 4)}-bit (repo's own scheme)"
    else:
        # No quantization field: this repo could be a genuine 4-bit MLX
        # conversion that just omitted the field, OR a real unquantized
        # bf16/fp16 upload -- blindly assuming 4-bit misrepresented the
        # latter case as ~4x faster/smaller than it really is. Defer to
        # the config's own declared dtype (same logic size_estimate.py's
        # RAM pre-filter uses) instead of a fixed guess.
        inferred_bits = _infer_bits(config)
        if inferred_bits >= 16:
            from mlx.utils import tree_map

            dtype = mx.bfloat16 if inferred_bits == 16 else mx.float32
            model.update(tree_map(lambda x: x.astype(dtype), model.parameters()))
            quant_desc = f"{inferred_bits}-bit (unquantized, per config's declared dtype)"
        else:
            nn.quantize(model, group_size=64, bits=4, class_predicate=class_predicate)
            quant_desc = "4-bit (assumed; no quantization info in config)"

    mx.eval(model.parameters())

    vocab_size = config.get("vocab_size") or config.get("text_config", {}).get(
        "vocab_size", 32000
    )
    input_ids = mx.random.randint(0, vocab_size, (1, prompt_len))
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    prefill_tps = prompt_len / prefill_time

    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)
    for _ in range(5):  # warmup, excluded from timing
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)

    t0 = time.perf_counter()
    for _ in range(decode_tokens):
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)
    decode_time = time.perf_counter() - t0
    raw_decode_tps = decode_tokens / decode_time

    calibrated_tps = (
        raw_decode_tps / CALIBRATION_RATIO if calibrate else raw_decode_tps
    )

    n_layers = config.get("num_hidden_layers") or config.get(
        "text_config", {}
    ).get("num_hidden_layers")

    return {
        "library": "mlx",
        "repo_id": repo_id,
        "architecture": config.get("model_type"),
        "quantization": quant_desc,
        "layers": n_layers,
        "prefill_tps": round(prefill_tps, 1),
        "raw_decode_tps": round(raw_decode_tps, 1),
        "estimated_real_tps": round(calibrated_tps, 1),
        "calibration_ratio": CALIBRATION_RATIO if calibrate else None,
    }
