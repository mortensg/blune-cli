# Research brief: a general-purpose LLM inference speed/feasibility model

## Goal

Design a **modular analytical model** (not necessarily one closed-form
equation) that predicts, from (a) a machine's hardware specs and (b) a
model's `config.json`/metadata alone -- no downloaded weights, no test
run -- both:

1. **Feasibility**: will this model fit and run on this machine at all
   (memory budget, including KV-cache and activation overhead, not just
   weight size)?
2. **Speed**: an estimated tokens/sec, for a clearly specified workload
   regime (see "Workload regimes" below -- do not conflate these).

Target machines: consumer/prosumer desktops and laptops running macOS
(Apple Silicon unified memory), Windows, or Linux, with either a discrete
GPU, an integrated GPU, or CPU-only. Out of scope: mobile/NPU-only
devices, unless the research finds this is essentially free to include.

Target inference engines to cover (survey their actual runtime behavior,
don't assume they're interchangeable): **MLX / mlx-lm** (Apple Silicon),
**llama.cpp** (GGUF), **vLLM**, and note where **TensorRT-LLM**,
**ExLlamaV2**, and Hugging Face **transformers**/**optimum** differ
enough to matter.

## Step 0: survey existing work before designing anything new

Before proposing a new model, find and summarize what already exists:

- Academic "roofline model for LLM inference" papers (arithmetic
  intensity, compute-bound vs. memory-bound crossover analysis)
- NVIDIA's and AMD's own inference performance/roofline analysis
  tooling and whitepapers
- vLLM's and llama.cpp's own internal cost/scheduling models (if
  documented) and published benchmark methodology
- Public VRAM/hardware calculators (e.g. `llm-vram-calculator`,
  `quantprobe`, Hugging Face's `optimum-benchmark`) -- what do they get
  right, what do they get wrong or omit (note: several of these omit
  Apple Silicon's "Pro" tier chips entirely and mishandle unified memory
  vs. discrete VRAM+RAM)
- MLPerf Inference results and methodology, as a source of real
  calibration data across hardware/model combinations
- Any existing attempt at a *general, engine-agnostic* formula (most
  prior art is engine-specific or hardware-vendor-specific -- note this
  gap explicitly if found)

## Step 1: the physical foundation

Establish the base law clearly, with its precise boundary conditions:

    decode_tok/s = memory_bandwidth / bytes_read_per_decode_step

State explicitly why this holds for single-stream autoregressive decode
(memory-bandwidth-bound) and why it does **not** hold for prefill
(compute/FLOPs-bound) or for high-concurrency batched serving
(compute-bound again, because batching amortizes weight reads across
many requests). Derive (or find published derivations of) the
crossover point -- the batch size / context length at which a workload
shifts from memory-bound to compute-bound -- since a correct model needs
different formulas on either side of that line.

## Step 2: workload regimes to model separately

Do not produce a single "tokens/sec" -- produce (at minimum):

1. **Prefill** tok/s as a function of prompt length (compute-bound;
   depends on FLOPs/token and the hardware's realized (not peak) compute
   throughput at the relevant precision)
2. **Single-stream decode** tok/s (memory-bandwidth-bound, as above)
3. **Concurrent/batched decode** throughput as a function of batch size
   or concurrent request count, including where it saturates
4. How KV-cache read bytes **grow with context length** during decode --
   this is currently omitted from most naive models but becomes a large
   fraction of per-token bytes at long context; needs its own term,
   parameterized by current context length, not just a constant

## Step 3: model architecture features the formula must account for

For each, state (a) how it changes bytes-read-per-token and/or
FLOPs-per-token, and (b) how to detect it from `config.json` metadata
alone (exact field names vary by architecture family -- catalog the
variants):

- Dense transformer (baseline case)
- **MoE**: total experts vs. active experts per token (`num_experts`,
  `num_experts_per_tok` and their many naming variants across HF configs)
- **Router/gating layer** itself (small, but should be included for
  completeness rather than assumed negligible without checking)
- **Shared experts** (always-active experts in addition to routed ones,
  e.g. DeepSeek-style architectures) -- distinct from the routed-expert
  active fraction
- **GQA / MQA** (grouped/multi-query attention) -- fewer KV heads than
  query heads, reduces both weight bytes and KV-cache bytes
- **Multi-latent attention (MLA)** (DeepSeek-style) -- a fundamentally
  different low-rank KV compression scheme with very different
  KV-cache-per-token bytes than standard GQA; needs its own formula
  branch, not a GQA approximation
- **Sliding-window attention** (Mistral-style) -- KV-cache is capped at a
  window size regardless of context length, changing the long-context
  KV-growth term
- **Hybrid architectures mixing standard attention with
  Mamba/SSM/linear-attention layers** (e.g. Jamba, Qwen's hybrid
  variants) -- these layers have an entirely different parameter
  structure and do NOT accumulate a standard KV-cache; a uniform
  "treat every layer as standard attention" approximation is wrong here
  and needs a distinct per-layer-type accounting
- **Mixed/per-layer quantization** -- some repos quantize most layers at
  N bits but override specific layers (commonly `lm_head`,
  `embed_tokens`) at a different bit-width; a single global
  bits-per-weight assumption undercounts these
- **Multi-token prediction (MTP) heads** -- extra parameters/layers used
  only in specific decoding modes; clarify when they're in the hot path
  and when they're not
- **Speculative decoding** (draft model + acceptance rate) -- changes
  *effective* tok/s multiplicatively based on acceptance rate; note this
  is a serving-strategy effect layered on top of the base model speed,
  not a property of the target model alone
- Vision/multimodal components (vision tower, image token processing) --
  scope note: how much this affects *text decode* speed specifically
  (usually the vision tower's cost is front-loaded into prefill, not
  decode) vs. prefill

## Step 4: engine-specific efficiency factors

The same model does not run at the same speed on different engines on
identical hardware. For each target engine, characterize:

- Its KV-cache memory layout (contiguous vs. paged/vLLM-style) and the
  resulting memory-efficiency difference
- Its batching strategy (static vs. continuous batching) and how that
  shifts the memory-bound/compute-bound crossover point
- Quantization kernel efficiency at a given bit-width (a GGUF Q4_0 kernel
  and MLX's affine 4-bit kernel are not equally fast on comparable
  hardware -- quantify this gap if published benchmarks allow it)
- Fixed per-decode-step dispatch/overhead cost (kernel launch, graph
  construction) -- note: informally measured for MLX on Apple Silicon at
  roughly 6.7-9.6ms per step across 5 real models on an M4 Pro (mean
  ~7.8ms, appears roughly constant regardless of model size); check
  whether comparable fixed-overhead terms exist and have been measured
  or published for llama.cpp and vLLM
- Whether/how it supports splitting a model across GPU + CPU RAM
  (llama.cpp's `--n-gpu-layers` and similar), which introduces a
  PCIe-bandwidth-bound term for the offloaded layers distinct from the
  GPU-resident layers' bandwidth term

## Step 5: hardware-side inputs needed

Beyond peak bandwidth/compute from a spec sheet, identify what's needed
for a realistic (not optimistic) estimate:

- **Sustained vs. peak** memory bandwidth and compute -- laptops/Macs can
  thermal-throttle under sustained load; note any published methodology
  for measuring or estimating sustained-vs-peak ratios per device class
- Unified memory (Apple Silicon) vs. VRAM+system-RAM split (Windows/Linux
  discrete GPU) -- and, for split systems, PCIe bandwidth between them
- CPU-only fallback path: relevant CPU features (AVX-512, AMX, etc.) and
  realistic CPU memory bandwidth, since not every target machine has a
  usable GPU
- Memory actually available for the model (total RAM/VRAM minus OS and
  other running processes' overhead) vs. total installed

## Step 6: feasibility (not just speed) formula

Separately from the speed estimate, define a feasibility check:

    fits = (weight_bytes + kv_cache_bytes_at_target_context
            + activation_bytes + safety_margin) <= usable_memory

Specify reasonable default assumptions/ranges for activation memory and
KV-cache sizing (as a function of context length, batch size, and the
architecture's actual KV-cache-per-token bytes from Step 3), and a
sensible safety margin, with justification.

## Deliverable format requested

1. A summary of existing prior art (Step 0), with an explicit list of
   what gaps remain unaddressed by current public tools
2. The proposed model as a **layered/modular structure**: one shared
   physical core (bandwidth/compute roofline + workload regime
   selection) plus pluggable correction terms per architecture feature
   (Step 3) and a pluggable per-engine efficiency profile (Step 4) --
   explicitly not a single flat equation, since the goal is to add new
   architectures/engines without redesigning the core
3. For each proposed term/correction factor: what `config.json` (or
   engine-specific metadata) field(s) it reads, and a worked numeric
   example
4. An honest accuracy/confidence assessment: which terms are backed by
   real measured data vs. which are reasoned-but-uncalibrated, and what
   minimum calibration dataset (which models, which hardware, which
   engines) would be needed to validate each piece
5. Explicit statement of what is still *not* modeled after all of the
   above, and why (diminishing returns, lack of public data, etc.)

## Context already established (do not re-derive)

- `tok/s = bandwidth / bytes_per_token` validated as the base decode-time
  law on Apple Silicon (M4 Pro, 273GB/s) across 5 real MLX model
  measurements
- A naive single-ratio calibration (probe-estimate/real-speed) varied
  2-3.75x between dense and MoE architectures and was abandoned in favor
  of an additive fixed-overhead-per-step term, which fit the same 5
  points to ~6.7% mean error -- suggesting the fixed-dispatch-overhead
  effect in Step 4 is real and worth prioritizing in the research
- A real bug was found and fixed where MoE models' active-expert MLP
  size was estimated using the dense `intermediate_size` field instead
  of the (usually much smaller) `moe_intermediate_size` field, causing a
  345B-parameter overestimate of a real ~23-35B model -- flagging this
  as a concrete example of the kind of field-naming inconsistency across
  HF configs the research should catalog per architecture family
