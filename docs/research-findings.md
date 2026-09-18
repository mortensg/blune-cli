# Research findings: general inference speed/feasibility model

Paste the deep-research output into the matching sections below. Keep the
headings as they are (even if a section ends up empty) -- this file's
structure mirrors the deliverable format requested in
`deep-research-prompt.md`, and I'll read it section by section to update
`probe_formula.py`, `size_estimate.py`, and `probe_mlx.py` /
`probe_llamacpp.py` / `probe_vllm.py` accordingly.

Where the research gives a formula or coefficient, paste it as-is
(don't pre-simplify) -- variable names and units matter for wiring it
into the actual code correctly. Where it cites a source (paper, repo,
benchmark), keep the link/citation so the number can be re-checked later.

---

## 0. Prior art survey

What already exists, what it gets right/wrong, and what gaps remain
unaddressed.

<!-- paste here -->


---

## 1. Physical foundation

The base bandwidth/compute roofline law, and the derivation (or citation)
of the compute-bound/memory-bound crossover point.

<!-- paste here -->


---

## 2. Workload regimes

### 2a. Prefill (compute-bound)

<!-- paste here -->

### 2b. Single-stream decode (memory-bandwidth-bound)

<!-- paste here -->

### 2c. Concurrent/batched decode throughput

<!-- paste here -->

### 2d. KV-cache growth with context length

<!-- paste here -->

---

## 3. Architecture feature corrections

For each feature: how it changes bytes/FLOPs per token, and which
config.json field(s) signal it (list every field-name variant found
across architecture families, not just one).

### 3a. Dense transformer (baseline)

<!-- paste here -->

### 3b. MoE -- total vs. active experts

<!-- paste here -->

### 3c. Router / gating layer

<!-- paste here -->

### 3d. Shared experts (always-active)

<!-- paste here -->

### 3e. GQA / MQA

<!-- paste here -->

### 3f. Multi-latent attention (MLA)

<!-- paste here -->

### 3g. Sliding-window attention

<!-- paste here -->

### 3h. Hybrid attention/Mamba/SSM/linear-attention layers

<!-- paste here -->

### 3i. Mixed/per-layer quantization

<!-- paste here -->

### 3j. Multi-token prediction (MTP) heads

<!-- paste here -->

### 3k. Speculative decoding (draft model + acceptance rate)

<!-- paste here -->

### 3l. Vision/multimodal components

<!-- paste here -->

### 3m. Anything else the research surfaced that wasn't in the original list

<!-- paste here -->

---

## 4. Engine-specific efficiency factors

### 4a. MLX / mlx-lm

<!-- paste here -->

### 4b. llama.cpp (GGUF)

<!-- paste here -->

### 4c. vLLM

<!-- paste here -->

### 4d. Others found worth noting (TensorRT-LLM, ExLlamaV2, transformers/optimum, ...)

<!-- paste here -->

---

## 5. Hardware-side inputs

Sustained vs. peak bandwidth/compute, unified memory vs. VRAM+RAM split,
CPU-only path, usable-vs-installed memory.

<!-- paste here -->


---

## 6. Feasibility formula

The "does it fit" check: weight bytes + KV-cache bytes + activation
bytes + safety margin vs. usable memory, with default assumptions for
each term.

<!-- paste here -->


---

## 7. Proposed modular model structure

The layered structure itself: shared physical core + pluggable
per-feature correction terms + pluggable per-engine efficiency profile.

<!-- paste here -->


---

## 8. Accuracy / confidence assessment

Which terms are backed by real measured data vs. reasoned-but-uncalibrated,
and what calibration dataset (models x hardware x engines) would be
needed to validate each piece.

<!-- paste here -->


---

## 9. Known remaining gaps

What's still not modeled after all of the above, and why.

<!-- paste here -->


---

## 10. Raw notes / anything that didn't fit above

<!-- paste here -->
