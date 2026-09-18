import struct

from blune_cli.probe_llamacpp import _GGML_TYPE_BITS, GGUF_MAGIC, _try_parse_header


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _build_minimal_gguf(vocab_size: int = 0) -> bytes:
    """Build a tiny valid GGUF header buffer by hand: magic, version,
    n_tensors, n_kv, then one string kv (architecture), one uint32 kv
    (block_count), optionally a big string-array kv to simulate a
    large-vocab tokenizer payload, then one tensor entry."""
    kvs = []

    kvs.append((_gguf_string("general.architecture"), struct.pack("<I", 8) + _gguf_string("llama")))
    kvs.append((_gguf_string("llama.block_count"), struct.pack("<I", 4) + struct.pack("<I", 32)))

    if vocab_size:
        # array of strings, type 9 (ARRAY) containing item_type 8 (STRING)
        arr_body = struct.pack("<I", 8) + struct.pack("<Q", vocab_size)
        arr_body += b"".join(_gguf_string("tok") for _ in range(vocab_size))
        kvs.append((_gguf_string("tokenizer.ggml.tokens"), struct.pack("<I", 9) + arr_body))

    n_kv = len(kvs)
    kv_bytes = b"".join(k + v for k, v in kvs)

    tensor_name = _gguf_string("token_embd.weight")
    n_dims = struct.pack("<I", 2)
    dims = struct.pack("<2Q", 100, 64)
    ggml_type = struct.pack("<I", 0)  # F32
    data_offset = struct.pack("<Q", 0)
    tensor_bytes = tensor_name + n_dims + dims + ggml_type + data_offset
    n_tensors = 1

    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, n_tensors, n_kv) + kv_bytes + tensor_bytes
    return header


def test_parses_minimal_header():
    buf = _build_minimal_gguf()
    result = _try_parse_header(buf)
    assert result["metadata"]["general.architecture"] == "llama"
    assert result["metadata"]["llama.block_count"] == 32
    assert len(result["tensors"]) == 1
    assert result["tensors"][0]["name"] == "token_embd.weight"
    assert result["tensors"][0]["n_elements"] == 100 * 64


def test_metadata_value_type_ids_match_gguf_spec():
    """Regression test for the bug where metadata value type IDs were
    confused with the separate ggml_type tensor-dtype enum. UINT32 (id 4)
    must decode as a 4-byte unsigned int, not misparsed against a
    different-width format."""
    buf = _build_minimal_gguf()
    result = _try_parse_header(buf)
    assert isinstance(result["metadata"]["llama.block_count"], int)
    assert result["metadata"]["llama.block_count"] == 32


def test_truncated_buffer_raises_struct_error():
    """This is the signal fetch_gguf_header's retry loop depends on --
    a short buffer must raise struct.error, not silently misparse."""
    import struct as _struct

    buf = _build_minimal_gguf(vocab_size=500)
    with __import__("pytest").raises(_struct.error):
        _try_parse_header(buf[: len(buf) // 2])


def test_large_vocab_header_parses_once_fully_fetched():
    buf = _build_minimal_gguf(vocab_size=500)
    result = _try_parse_header(buf)
    assert result["metadata"]["general.architecture"] == "llama"
    assert len(result["tensors"]) == 1


def test_ggml_type_bits_match_verified_block_struct_layout():
    """Regression test for a real ID-mapping bug: the previous table
    listed nominal bits-per-weight values in name order and assigned
    them sequentially to enum IDs {0,1,2,3,6,7,8,9,10,11,12,...}, but
    the real ggml_type enum has a gap at IDs 4-5 (retired Q4_2/Q4_3),
    which silently shifted every K-quant/IQ value onto the wrong ID --
    Q2_K and Q3_K were priced at roughly 2x their real bytes. Values
    below are bytes_per_block*8/elements_per_block, derived from each
    block struct's real field list in ggml/src/ggml-common.h."""
    assert _GGML_TYPE_BITS[2] == 4.5  # Q4_0: d(half) + qs[16], 32-elem block
    assert _GGML_TYPE_BITS[8] == 8.5  # Q8_0: d(half) + qs[32]
    assert _GGML_TYPE_BITS[10] == 2.625  # Q2_K: was wrongly 5.0 (~90% too high)
    assert _GGML_TYPE_BITS[11] == 3.4375  # Q3_K: was wrongly 6.5625 (~91% too high)
    assert _GGML_TYPE_BITS[12] == 4.5  # Q4_K
    assert _GGML_TYPE_BITS[13] == 5.5  # Q5_K: was wrongly 4.5
    assert _GGML_TYPE_BITS[14] == 6.5625  # Q6_K: was wrongly 5.5
    # Previously missing entirely (fell through to a 4.5 default guess):
    assert _GGML_TYPE_BITS[39] == 4.25  # MXFP4 -- used by real gpt-oss GGUF releases
    assert _GGML_TYPE_BITS[30] == 16.0  # BF16
