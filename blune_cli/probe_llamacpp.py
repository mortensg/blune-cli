"""
llama.cpp / GGUF speed probe.

Unlike safetensors+config.json, a GGUF file doesn't cleanly separate
"architecture" from "weights" -- but GGUF's own header format lists every
tensor's name, shape and quantization type BEFORE any tensor data, so we can
still get exact bytes-per-token without downloading weight data: fetch only
the header via an HTTP range request, sum the byte size of the tensors that
matter for one decode step (all of them, for a dense model; only the active
experts' share for MoE -- confirmed mechanistically correct, not just
convenient, by reading llama.cpp's real Metal MoE kernel: `mul_mv_id`
(ggml/src/ggml-metal/kernels/mul_mv.metal) reads the selected expert index
per token and pointer-offsets directly into the fused per-layer expert
tensor by that expert's own stride, touching only the selected experts'
bytes during single-token decode).

Methodology (mirrors probe_formula.py's MLX methodology exactly)
------------------------------------------------------------------
tok/s = bandwidth / bytes-per-token, corrected by two constants fit via
linear regression against 5 real measurements (Apple M4 Pro, 273 GB/s,
`llama-bench`, Qwen2.5-0.5B-Instruct-GGUF across 4 quant levels + one
Qwen2.5-7B-Instruct-GGUF Q4_K_M point to check size-dependence):

    time_per_token = bytes_per_token / (bandwidth * BANDWIDTH_CALIBRATION_RATIO)
                      + FIXED_OVERHEAD_SEC

| Model | Real tok/s | Formula | Error |
|---|---|---|---|
| Qwen2.5-0.5B Q4_0   | 307.56 | 327.59 | +6.5% |
| Qwen2.5-0.5B Q4_K_M | 269.81 | 297.90 | +10.4% |
| Qwen2.5-0.5B Q6_K   | 256.85 | 242.26 | -5.7% |
| Qwen2.5-0.5B Q8_0   | 256.10 | 235.24 | -8.1% |
| Qwen2.5-7B Q4_K_M   | 49.07  | 49.15  | +0.2% |
mean absolute error 6.2%, max 10.4% -- fit on only 5 points spanning one
architecture family (Qwen2, dense) at one size regime (0.5B/7B), so
treat this as a first real anchor, not a fully validated formula. This
still massively improves on the previous flat `GGUF_CALIBRATION_RATIO =
0.60` guess, which gave 17.1% mean / 28.7% max error on these same 5
points (it wasn't derived from any real llama.cpp measurement at all --
see git history). BANDWIDTH_CALIBRATION_RATIO (~0.755, i.e. real
bandwidth utilization closer to 100% of spec than MLX's own ~0.73-1.35x
range across different formula versions) and FIXED_OVERHEAD_SEC
(~1.0ms) are both far smaller-magnitude corrections than MLX needed,
consistent with llama.cpp being a compiled C++ binary with a single
Metal command buffer per token (confirmed: `ggml-metal.cpp` defaults
`n_cb=1`) rather than a Python-orchestrated MLX graph.

Bug fixed alongside this recalibration: `_GGML_TYPE_BITS` had a real
ID-mapping error (not a precision issue) -- comparing enum IDs against
ggml's own block-struct byte layout (ggml/src/ggml-common.h) found
Q2_K and Q3_K were priced at roughly 2x their real bytes (the table had
been built by listing nominal bits-per-weight values in name order and
assigning them sequentially to enum IDs, but the real enum has a gap at
IDs 4-5 for retired types, shifting every K-quant/IQ value onto the
wrong ID). Fixed using the verified struct-derived table; several
quant types in real use (MXFP4 -- used by real gpt-oss GGUF releases,
IQ-series, TQ1_0/TQ2_0, BF16) were also simply missing before and fell
through to a default guess.

Known gaps, honestly: only one architecture family (Qwen2 dense) and
one machine have been measured; no real MoE GGUF data point exists yet
despite the MoE byte-accounting itself being verified correct at the
kernel level (see above) -- the *speed* impact of MoE-specific Metal
dispatch overhead (analogous to MLX's measured ~178us/MoE-layer) is
still unknown for llama.cpp. See docs/formula-accuracy-gap.md for the
prioritized list of what a follow-up real-measurement session should
test next.
"""
import struct
from typing import Optional

import requests

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
BANDWIDTH_CALIBRATION_RATIO = 0.755  # see module docstring -- fit from 5 real measurements
FIXED_OVERHEAD_SEC = 0.0010  # per-decode-step dispatch cost, fit from real data

_GGUF_TYPE_SIZES = {
    0: 4,   # F32
    1: 2,   # F16
    2: 1,   # Q4_0 unused directly; block types handled via block size below
}

# Exact bits-per-weight per ggml_type enum ID, derived from each block
# struct's real byte layout (bytes_per_block * 8 / elements_per_block),
# not the "well-known" nominal figures -- verified directly against
# ggml/src/ggml-common.h's block struct definitions and ggml/include
# /ggml.h's enum ggml_type. The previous version of this table listed
# nominal bpw values in {Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q8_1,Q2_K,Q3_K,Q4_K,
# Q5_K,Q6_K,Q8_K,IQ2_XXS,IQ2_XS,IQ3_XXS} order and assigned them
# SEQUENTIALLY to enum IDs {0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18}
# -- but the real enum has a gap at IDs 4-5 (retired Q4_2/Q4_3), which
# silently shifted every K-quant/IQ value onto the wrong ID. Real-world
# impact: any Q2_K or Q3_K tensor was priced at ~2x its actual bytes.
_GGML_TYPE_BITS = {
    0: 32.0,       # F32
    1: 16.0,       # F16
    2: 4.5,        # Q4_0: d(half) + qs[16], 32-elem block
    3: 5.0,        # Q4_1: d,m(2xhalf) + qs[16]
    6: 5.5,        # Q5_0: d(half) + qh[4] + qs[16]
    7: 6.0,        # Q5_1: d,m(2xhalf) + qh[4] + qs[16]
    8: 8.5,        # Q8_0: d(half) + qs[32]
    9: 9.0,        # Q8_1: activation-only, not a stored weight tensor
    10: 2.625,     # Q2_K: 256-elem superblock, 84 bytes/block
    11: 3.4375,    # Q3_K
    12: 4.5,       # Q4_K
    13: 5.5,       # Q5_K
    14: 6.5625,    # Q6_K
    15: 9.125,     # Q8_K: activation-only intermediate, not a stored weight tensor
    16: 2.0625,    # IQ2_XXS
    17: 2.3125,    # IQ2_XS
    18: 3.0625,    # IQ3_XXS
    19: 1.5625,    # IQ1_S
    20: 4.5,       # IQ4_NL: 32-elem block
    21: 3.4375,    # IQ3_S
    22: 2.5625,    # IQ2_S
    23: 4.25,      # IQ4_XS
    29: 1.75,      # IQ1_M
    30: 16.0,      # BF16
    34: 1.6875,    # TQ1_0 (ternary)
    35: 2.0625,    # TQ2_0 (ternary)
    39: 4.25,       # MXFP4 (used by real gpt-oss GGUF releases)
    40: 4.5,       # NVFP4
    42: 2.25,      # Q2_0
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

    ratio = BANDWIDTH_CALIBRATION_RATIO if calibrate else 1.0
    overhead = FIXED_OVERHEAD_SEC if calibrate else 0.0
    if total_bytes:
        total_time = total_bytes / (machine_bandwidth_gbs * ratio * 1e9) + overhead
        calibrated_tps = 1.0 / total_time
    else:
        calibrated_tps = 0.0

    return {
        "library": "llama.cpp",
        "gguf_url": gguf_url,
        "architecture": arch,
        "layers": n_layers,
        "bytes_per_token_active": round(total_bytes / 1e6, 1),
        "raw_theoretical_tps": round(raw_tps, 1),
        "estimated_real_tps": round(calibrated_tps, 1),
        "confidence": "medium (config-only formula, mean 6.2% error / 10.4% max on 5-point real-measurement set -- one architecture family, see module docstring)",
    }
