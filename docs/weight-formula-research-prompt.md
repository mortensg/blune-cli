# Research brief: exact per-architecture weight-parameter formulas from mlx-lm's source (online)

## Where to find the source

GitHub repo: **https://github.com/ml-explore/mlx-lm**

Pin to the tag matching this project's installed version so nothing has
drifted: **`v0.31.3`** — browse it at
`https://github.com/ml-explore/mlx-lm/tree/v0.31.3/mlx_lm/models`
(if that exact tag doesn't exist, use the closest release ≥ that version
and note which one you used).

Each architecture's file is at:
`https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/<model_type>.py`

Some architectures share an implementation file with a different name
than their `model_type` string (e.g. Qwen3.5's hybrid layers are defined
in `qwen3_next.py`, not `qwen3_5.py` — `qwen3_5.py` just imports them).
Check the top of each file's `import` section for `from .X import Y as Z`
lines before concluding a file has no relevant layer.

## The exact extraction recipe (do this per architecture)

1. Open `mlx_lm/models/<file>.py` at the pinned tag.
2. Find the attention/mixer class (commonly named `Attention`,
   `<Name>Attention`, `GatedDeltaNet`, `MambaBlock`, etc.) and the MLP
   class (`MLP`, `<Name>MLP`, or a MoE wrapper like `SparseMoeBlock`).
3. In each class's `__init__`, list **every** `nn.Linear(...)` and
   `nn.Conv1d(...)` call. For each one, write down:
   - the exact **input and output dimension**, as an expression in
     config fields (e.g. `num_attention_heads * head_dim`, NOT a
     hardcoded number)
   - whether `bias=True` (adds `+ out_dim` params if so)
   - for `Conv1d`: kernel size and `groups=` (if `groups == out_channels`,
     it's depthwise: params = `channels * kernel_size`, NOT
     `in_channels * out_channels * kernel_size`)
4. If the MLP is a MoE wrapper, also record:
   - the router/gate layer's shape (usually `hidden -> num_experts`)
   - whether there's an always-on **shared expert** and what field sizes
     it (some architectures use a generic `n_shared_experts` count with
     the same size as routed experts; others — like Qwen3-Next — use a
     SEPARATE `shared_expert_intermediate_size` field for a single
     always-on expert; note which pattern applies)
5. Note the **KV-cache shape**: what exactly gets stored in `cache[...]`
   between decode steps, and its size per token. For ordinary attention
   this is `2 * num_key_value_heads * head_dim`; for MLA-style or
   linear-attention layers it's something else entirely (a compressed
   latent, a fixed-size recurrent state, etc.) — say what.
6. Note **any config field this class reads that isn't obviously a
   dimension** (a boolean flag, a scaling factor) if it changes which
   `nn.Linear` gets created or its size — that's exactly the kind of
   architecture-specific quirk that breaks a naive formula (example
   already found: `Qwen3NextAttention.q_proj` outputs
   `num_attention_heads * head_dim * 2`, double the standard size, for
   no reason visible in config.json alone).
7. Cross-check against `TextModelArgs`/`ModelArgs` in the same file (or
   imported from elsewhere) to get the **exact config.json field names**
   each dimension corresponds to — these vary across architectures (see
   the field-name catalog below for known variants already found).

## Priority order — by how many models in our own cache use each one

(2856 cached configs, 130 distinct `model_type` values found; this list
is the ones worth spending time on, not all 130.)

**Already reasonably covered (standard GQA dense) — low priority, but
spot-check at least one for hidden quirks:**
`llama` (508 models), `qwen2` (406), `qwen3` (266), `mistral` (149)

**MoE variants — need router/shared-expert structure specifically:**
`qwen3_moe` (93), `falcon_h1` (78, also hybrid), `granitemoehybrid` (68,
also hybrid), `qwen3_5_moe` (44), `glm4_moe` (29), `minimax_m2` (24),
`deepseek_v4` (23), `mixtral` (23), `glm4_moe_lite` (15),
`glm_moe_dsa` (14), `bailing_moe_linear` (13), `gpt_oss` (13)

**Hybrid/SSM — highest expected payoff, each is likely a DIFFERENT SSM
implementation, not interchangeable with the GatedDeltaNet/Mamba formulas
already implemented:**
`falcon_h1` (78, TII's own hybrid design), `lfm2` (76, Liquid AI's LFM2
architecture), `granitemoehybrid` (68, IBM Granite), `nemotron_h` (40,
NVIDIA's Mamba-2 hybrid), `qwen3_5` (41), `qwen3_5_moe` (44),
`qwen3_next` (16), `qwen3_5_mtp` (18, note: `_mtp` suggests multi-token
prediction — check if it's actually a distinct mixer or just a training
variant of qwen3_5)

**Other architecture-specific quirks worth checking (in the spirit of
the Qwen3-Next q_proj-doubling find):**
`gemma2` / `gemma3` / `gemma3_text` / `gemma4` / `gemma4_assistant`
(~240 models combined, multiple generations with different
sliding-window/attention patterns), `cohere` / `cohere2` (41, Command-R
has known logit-scaling and tied-embedding quirks), `glm4` (17),
`deepseek_v3` (18, should match the MLA formula already implemented --
worth verifying it actually does)

## Field-name catalog already discovered (don't re-derive these)

- MoE total-expert-count field: `num_local_experts` | `num_experts` |
  `n_routed_experts` | `moe_num_experts`
- MoE active-experts-per-token field: `num_experts_per_tok` |
  `num_activated_experts` | `top_k_experts` | `moe_top_k` | `moe_k`
- MoE expert intermediate size: `moe_intermediate_size` (NOT the dense
  `intermediate_size` -- using the dense one overestimated a real ~23B
  model as 345B)
- Shared experts: `n_shared_experts` (a count, same size as routed
  experts) OR `shared_expert_intermediate_size` (signals exactly ONE
  always-on shared expert with its own separate size -- Qwen3-Next
  pattern, doesn't use a count field at all)
- MLA: `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`,
  `qk_nope_head_dim`, `v_head_dim`
- Dense/MoE layer interleaving: `first_k_dense_replace`,
  `moe_layer_freq`
- Per-layer type list: `layer_types` (array of strings per layer --
  seen values: `"full_attention"`, `"sliding_attention"`,
  `"linear_attention"`) or a pattern integer like
  `full_attention_interval` (layer is non-linear when
  `(layer_idx + 1) % full_attention_interval == 0`)
- GatedDeltaNet (Qwen3.5/3.6/3-Next linear attention):
  `linear_num_value_heads`, `linear_num_key_heads`,
  `linear_key_head_dim`, `linear_value_head_dim`,
  `linear_conv_kernel_dim`
- Classic Mamba/SSM: `mamba_d_state`/`d_state`/`ssm_state_size`/
  `state_size`, `mamba_d_conv`/`d_conv`/`conv_kernel`/`conv_kernel_size`,
  `mamba_expand`/`expand`/`expand_factor` (or `intermediate_size`
  directly), `dt_rank`/`mamba_dt_rank`/`time_step_rank`

## Deliverable format

For each architecture, in this exact shape (so it can be turned directly
into code, the way the GatedDeltaNet/Mamba findings were):

```
### <model_type>
Source: mlx_lm/models/<file>.py @ v0.31.3, class <ClassName>

Mixer layer:
  <layer_name>: nn.Linear(<in_dim_expr>, <out_dim_expr>, bias=<bool>)
  ... (every Linear/Conv1d)

MLP / MoE:
  ... (same, plus router/shared-expert notes if MoE)

KV-cache per token: <expression, or "fixed-size state: <expr>, doesn't grow with context">

Config field names confirmed: <list>

Quirks found (if any): <e.g. "q_proj doubled for no config-visible reason">
```

Skip an architecture (note it as skipped) if its file is unusually
complex (e.g. multimodal vision+text) and note why, rather than guessing.
