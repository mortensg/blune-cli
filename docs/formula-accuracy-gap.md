# Open accuracy gaps — what still needs research or measurement

Historical record of how the formula got here (every bug found, every
research pass, every real measurement taken) lives in git history and in
`probe_formula.py` / `probe_llamacpp.py`'s own module docstrings, which
carry the current, up-to-date methodology and calibration numbers. This
file only tracks what's still open.

---

## 1. vLLM: two real, unrelated packages, only one (mis-)calibrated

`probe_vllm.py`'s `VLLM_SINGLE_STREAM_RATIO = 0.55` is based on exactly
ONE real comparison (`gemma-4-26b-a4b-it-4bit`, MLX vs. vLLM), and
doesn't run its own probe at all -- it calls the MLX probe and
multiplies by this ratio.

Research found there are **two separate, real, unrelated "vLLM on Apple
Silicon" packages**:
- `vllm-project/vllm-metal` (official org) -- wraps vLLM's actual
  CUDA-lineage scheduler/paged-block-manager around mlx-lm's own layer
  classes, with a custom Metal attention kernel replacing CUDA
  PagedAttention. This is almost certainly what the existing 0.55 ratio
  was measured against.
- `waybarrios/vllm-mlx` (independent) -- a from-scratch reimplementation
  of vLLM's *ideas* (continuous batching, paged KV-cache) built natively
  in MLX, not running vLLM's own scheduler code. Its own published
  benchmarks suggest single-stream (concurrency=1) speed much closer to
  native mlx-lm than the 0.55 ratio implies.

**One ratio cannot represent both packages.** `probe_vllm.py` needs
either to detect/ask which one a user has installed, or to ship two
ratios.

**What to research/measure:**
1. Install both `vllm-metal` and `vllm-mlx` (both real, actively
   released pip packages) and run a real, controlled, single-stream
   (concurrency=1) comparison against plain mlx-lm on the same small
   model (e.g. one of the Qwen3-0.6B / Llama-3.2-1B models `vllm-mlx`'s
   own docs already benchmark) on the same machine. This is the
   highest-value single measurement in this whole list -- it directly
   resolves whether one or two ratios are needed and gives real numbers
   for whichever is chosen.
2. Independently re-run any concurrency=1-vs-N comparison rather than
   trusting published numbers at face value -- `vllm-mlx`'s own docs
   showed >50% different single-stream tok/s for the identical
   model/chip in two different tables in the same file, so single-run
   third-party benchmarks in this space are known to be noisy.
3. Confirm whether `vllm-metal`'s GGUF support (currently narrow: only
   Q8_0/Q4_0/Q4_1 + F16/F32/BF16, explicitly excluding K-quants/MoE/SSM
   per its own docs) matters for any model blune's cache already covers.

## 2. `bailing_moe_linear`'s real layer structure is still unverified

A research pass described this architecture using fields
(`linear_key_dim`, `linear_value_dim`) that don't exist in the real
cached config (which has `head_dim`/`group_norm_size` instead) --
left unimplemented rather than guess. Only 13 models in the curated
cache use this architecture, so low priority, but the real fix is the
same recipe used successfully elsewhere in this project: read
`mlx_lm/models/bailing_moe_linear.py`'s actual mixer class directly
(locally, if installed, or from `github.com/ml-explore/mlx-lm` at the
pinned version) rather than re-guessing from a description.

## 3. Qwen3-Next's doubled `q_proj` -- and other per-architecture-family quirks

`Qwen3NextAttention.q_proj` outputs `num_attention_heads * head_dim * 2`
(double the standard size) in mlx-lm's real implementation, for reasons
not visible in config.json. A plausible-looking generic fix (using the
`attn_output_gate` config field as a signal) was tested and rejected
after confirming Gemma4 also sets that field without doubling its
`q_proj` -- using it generically would have broken an already-validated
architecture to fix this one.

This class of quirk (a real per-architecture-family exception, not
inferable from config field presence alone) will keep recurring as more
architectures are added. Options, not yet decided:
- A small `model_type -> known quirks` lookup table, hardcoded per
  architecture as they're discovered (simple, but doesn't scale
  automatically).
- Live with the current formula's error on Qwen3-Next specifically
  (documented, not silently wrong) until enough real measurements exist
  to know whether it's worth fixing at all.

## 4. `probe_mlx.py`'s own ~15-19% synthetic-probe deficit

The zero-download real-execution probe (`probe_mlx.py`) runs at only
~81-85% of real generation speed when compared against genuine
downloaded-weight generations -- a gap absorbed today by
`CALIBRATION_RATIO`, never root-caused. Candidate mechanisms (unverified
speculation, not confirmed):
- Random/uninitialized weights don't go through the same quantization
  code path as a real checkpoint load, so the probe may not be hitting
  MLX's fused `gather_qmm`/`qmv` kernels the way real inference does.
- Python-allocated random arrays vs. `mmap`-loaded real weight files may
  have different memory locality/TLB behavior.
- Insufficient warmup iterations before timing (currently 5) leaving
  some JIT/pipeline-compilation cost inside the measured window.

This is a distinct, substantial investigation into MLX runtime behavior
(would need controlled experiments comparing probe execution against
real generation on the *same* model, not just comparing to a formula) --
not a config-formula change, and not started yet.

## 5. LFM2-8B-A1B / granite-4.0-h-tiny's remaining ~2-2.4x active-bytes gap

Every weight matrix in both architectures' real mlx-lm source has been
verified to match this project's byte-counting exactly (no missing
layer-structure bug). Yet a naive bytes/bandwidth-only estimate (zero
overhead) implies both models transfer ~2-2.4x more bytes per token
than the formula currently counts. The per-MoE-layer overhead term
added later (see `probe_formula.py`'s docstring) empirically improves
the fit for these two models but was derived from a *general* 9-point
regression plus a generic MLX chained-layer benchmark -- it was never
confirmed as the specific, complete causal explanation for these two
architectures' particular gap.

**What to measure:** a chained-layer micro-benchmark using these two
models' *exact* expert configuration (LFM2-8B-A1B: 32 experts/4 active,
`moe_intermediate_size=1792`; granite-4.0-h-tiny: 64 experts/6 active)
at N=1,2,4,8,16,32, quantized to match the real repos' bit-width,
mirroring the methodology already used successfully for the generic
attention-vs-MoE comparison in `probe_formula.py`'s docstring -- to
confirm whether the remaining gap is fully explained by per-MoE-layer
dispatch overhead at this specific scale, or whether something else
(e.g. the `argpartition`/gating computation itself, or MLX's `do_sort`
threshold behavior at very small `top_k` counts) is still missing.
