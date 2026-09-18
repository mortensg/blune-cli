"""
llama.cpp / GGUF speed probe.

Unlike safetensors+config.json, a GGUF file doesn't cleanly separate
"architecture" from "weights" -- but GGUF's own header format lists every
tensor's name, shape and quantization type BEFORE any tensor data, so we can
still get exact bytes-per-token without downloading weight data: fetch only
the header via an HTTP range request, sum the byte size of the tensors that
matter for one decode step (all of them, for a dense model; only the active
experts' share for MoE, approximated from Q_ffn count if declared).

Honesty note: we do NOT have a real, measured llama.cpp calibration ratio
the way we do for MLX (validated three times against real generations
today). GGUF_CALIBRATION_RATIO below is a placeholder seeded from the MLX
ratio plus llama.cpp's generally-lower Metal efficiency (per our own
earlier research: MLX beats llama.cpp by 20-87% on Apple Silicon). Treat
estimated_real_tps from this module as lower-confidence than the MLX probe
until someone runs `blune bench --contribute-llamacpp` (not yet built) to
replace it with a real number.
"""
import struct
from typing import Optional

import requests

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_CALIBRATION_RATIO = 0.60  # provisional -- see module docstring

_GGUF_TYPE_SIZES = {
    0: 4,   # F32
    1: 2,   # F16
    2: 1,   # Q4_0 unused directly; block types handled via block size below
}

# Approximate bits-per-weight for common GGUF quant types (block-quantized
# formats don't map to a clean "bytes per element" the way MLX's affine
# quant does, so these are the well-known nominal bits-per-weight figures).
_GGML_TYPE_BITS = {
    0: 32, 1: 16, 2: 4.5, 3: 4.5, 6: 5.5, 7: 5.5, 8: 8.5, 9: 8.5,
    10: 5.0, 11: 6.5625, 12: 4.5, 13: 4.5, 14: 5.5, 15: 6.5625,
    16: 3.4375, 17: 3.5, 18: 2.5,
}


def _read_gguf_string(buf: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    s = buf[offset : offset + length].decode("utf-8", errors="replace")
    return s, offset + length


def _read_gguf_value(buf: bytes, offset: int, vtype: int):
    """Skip/read a single GGUF metadata value; returns (value_or_None, new_offset).
    Type IDs per the GGUF spec (ggml_type is a SEPARATE enum, used only for
    tensor dtypes, not metadata values -- these are the metadata value types):
    0=UINT8 1=INT8 2=UINT16 3=INT16 4=UINT32 5=INT32 6=FLOAT32 7=BOOL
    8=STRING 9=ARRAY 10=UINT64 11=INT64 12=FLOAT64"""
    simple_types = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
        10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
    }
    if vtype == 8:  # string
        return _read_gguf_string(buf, offset)
    if vtype == 9:  # array
        (item_type,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", buf, offset)
        offset += 8
        for _ in range(count):
            _, offset = _read_gguf_value(buf, offset, item_type)
        return None, offset
    fmt, size = simple_types[vtype]
    (val,) = struct.unpack_from(fmt, buf, offset)
    return val, offset + size


def _try_parse_header(buf: bytes) -> dict:
    """Parse metadata + tensor info from a header buffer. Raises
    struct.error (via unpack_from) if buf is truncated mid-header, which the
    caller uses as the signal to retry with a bigger fetch."""
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", buf, 0)
    if magic != GGUF_MAGIC:
        raise ValueError("Not a GGUF file (bad magic)")

    offset = 24
    metadata = {}
    for _ in range(n_kv):
        key, offset = _read_gguf_string(buf, offset)
        (vtype,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        val, offset = _read_gguf_value(buf, offset, vtype)
        metadata[key] = val

    tensors = []
    for _ in range(n_tensors):
        name, offset = _read_gguf_string(buf, offset)
        (n_dims,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        dims = struct.unpack_from(f"<{n_dims}Q", buf, offset)
        offset += 8 * n_dims
        (ggml_type,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        offset += 8  # tensor data offset (unused -- we only need sizes)
        n_elements = 1
        for d in dims:
            n_elements *= d
        tensors.append({"name": name, "dims": dims, "ggml_type": ggml_type, "n_elements": n_elements})

    return {"metadata": metadata, "tensors": tensors}


def fetch_gguf_header(url: str, initial_fetch_bytes: int = 4_000_000) -> dict:
    """Fetch enough of a GGUF file's front to parse its metadata + tensor
    info sections. GGUF embeds the full tokenizer vocabulary as metadata,
    which can be several MB for large-vocab models, so we grow the fetch
    (doubling, up to a sane cap) whenever the buffer runs out mid-parse
    rather than guessing one fixed size for every model."""
    fetch_bytes = initial_fetch_bytes
    max_fetch_bytes = 128_000_000
    last_error = None

    while fetch_bytes <= max_fetch_bytes:
        r = requests.get(url, headers={"Range": f"bytes=0-{fetch_bytes - 1}"})
        r.raise_for_status()
        buf = r.content
        try:
            return _try_parse_header(buf)
        except struct.error as e:
            last_error = e
            fetch_bytes *= 4
            continue

    raise ValueError(
        f"GGUF header still incomplete after fetching {fetch_bytes // 4} bytes "
        f"(last error: {last_error})"
    )


def probe(gguf_url: str, machine_bandwidth_gbs: float, calibrate: bool = True) -> dict:
    """Estimate tokens/sec for a GGUF model given the machine's real memory
    bandwidth, using only the GGUF header (no weight data downloaded)."""
    header = fetch_gguf_header(gguf_url)
    meta = header["metadata"]

    arch = meta.get("general.architecture", "unknown")
    n_layers = meta.get(f"{arch}.block_count")
    n_experts = meta.get(f"{arch}.expert_count", 0) or 0
    n_experts_used = meta.get(f"{arch}.expert_used_count", 0) or 0

    # Sum bytes for tensors that are read on EVERY decode step. For MoE
    # expert tensors we approximate active bytes as (used/total) of the
    # expert weight bytes, since GGUF fuses all experts into one tensor.
    total_bytes = 0
    expert_bytes = 0
    for t in header["tensors"]:
        bits = _GGML_TYPE_BITS.get(t["ggml_type"], 4.5)
        tensor_bytes = t["n_elements"] * bits / 8
        if "ffn_gate_exps" in t["name"] or "ffn_up_exps" in t["name"] or "ffn_down_exps" in t["name"]:
            expert_bytes += tensor_bytes
        else:
            total_bytes += tensor_bytes

    if n_experts and n_experts_used:
        total_bytes += expert_bytes * (n_experts_used / n_experts)
    else:
        total_bytes += expert_bytes

    raw_tps = machine_bandwidth_gbs * 1e9 / total_bytes if total_bytes else 0
    calibrated_tps = raw_tps * GGUF_CALIBRATION_RATIO if calibrate else raw_tps

    return {
        "library": "llama.cpp",
        "gguf_url": gguf_url,
        "architecture": arch,
        "layers": n_layers,
        "bytes_per_token_active": round(total_bytes / 1e6, 1),
        "raw_theoretical_tps": round(raw_tps, 1),
        "estimated_real_tps": round(calibrated_tps, 1),
        "calibration_ratio": GGUF_CALIBRATION_RATIO if calibrate else None,
        "confidence": "low (no real llama.cpp calibration data yet -- see module docstring)",
    }
