# blune-cli

**Know which model runs best, how fast, in which library, on the machine
you're actually sitting at — before you download anything.**

```bash
$ blune
 _     _                      ____ _     ___
| |__ | |_   _ _ __   ___    / ___| |   |_ _|
| '_ \| | | | | '_ \ / _ \  | |   | |    | |
| |_) | | |_| | | | |  __/  | |___| |___ | |
|_.__/|_|\__,_|_| |_|\___|   \____|_____|___|

╭──────────────────────────────────────────────╮
│ Your machine: Apple M4 Pro                    │
│ RAM: 48 GB   |   Bandwidth: 273 GB/s          │
╰──────────────────────────────────────────────╯

What do you want to do?
  1. Find the fastest model for my machine (search + rank)
  2. Test a specific model
  3. Compare one model across libraries
  4. Just show my hardware info
```

## Why this exists

Every public "will this model run on my machine" calculator we checked has
the same two problems on Apple Silicon:

1. **They don't know your chip.** Most GPU databases are built for discrete
   Nvidia/AMD cards and either omit Apple Silicon's "Pro" tier entirely, or
   auto-detect it as a fictional split VRAM+RAM system that doesn't match
   how unified memory actually works. We measured one popular tool predict
   **46 tok/s** for a model that really runs at **86-90 tok/s** on the exact
   same machine, because it guessed the wrong hardware bandwidth.
2. **They estimate speed from the model's total size, not from what a real
   generation loop actually does.** A naive `bandwidth ÷ model size`
   calculation ignores quantization format, MoE active-parameter counts,
   and per-library serving overhead -- all things that change the real
   number by 2x or more.

This project takes a different approach: rather than a formula and a lookup
table, it downloads a model's **config.json only** (a few KB), builds the
model's *real* architecture class with randomly-initialized weights (speed
depends on weight shape/dtype/quantization, not on the values), and runs an
actual prefill+decode loop through it. That's a genuine, validated
measurement of your genuine hardware running the model's genuine compute
graph -- just without needing the multi-gigabyte weight files.

## Validated accuracy (MLX / Apple Silicon)

Measured on an Apple M4 Pro, 48GB, macOS 26.5:

| Model | Real (full download) | This tool (config only) | Ratio |
|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct-4bit | 86.8-89.8 tok/s | ~70-73 tok/s | ~81% |
| gemma-4-26b-a4b-it-4bit | ~76.7 tok/s | ~64 tok/s | ~82% |
| Qwen2.5-Coder-7B-Instruct-4bit | 57.2 tok/s | ~48 tok/s | ~85% |

The ~81-85% ratio held consistently across a dense model and two different
MoE architectures, across repeated runs. `probe_mlx.py` applies a `0.82`
calibration factor by default so the reported `estimated_real_tps` lands
close to the true number, not just the raw internal figure.

**vLLM support remains provisional; llama.cpp (GGUF) now has a first real
calibration** -- see the docstrings
in `probe_llamacpp.py` and `probe_vllm.py` for exactly what's measured vs.
guessed. Contributions with real measurements are very welcome (see below).

## Install

```bash
pip install -e ".[mlx]"   # Apple Silicon, full MLX probing
pip install -e .          # everything except the MLX probe (llama.cpp/vLLM
                           # probes work without mlx installed)
```

## Usage

Interactive (recommended first run):

```bash
blune
```

Or scriptable, for automation / CI:

```bash
blune hw                                        # show detected hardware
blune search --limit 20 --library mlx           # find + rank candidates
blune test mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --library mlx
blune compare mlx-community/gpt-oss-20b-OptiQ-4bit   # mlx vs vllm
blune gguf https://huggingface.co/.../model.gguf     # llama.cpp estimate

blune sync-configs                    # bulk-cache config.json for every
                                       # mlx-community model (2800+, one
                                       # HTTP call each, run once ever)
blune sweep --library mlx             # probe EVERY cached model, one at a
                                       # time, printing a permanent result
                                       # line as each one finishes:
                                       #   [12/2856] repo/name - 83.6 tok/s (estimated)
                                       #   [13/2856] repo/name - kører....
                                       # skips models estimated too large
                                       # for this machine's RAM; each probe
                                       # runs in its own subprocess so one
                                       # crash/hang can't kill the sweep
blune sweep --formula                 # instant config-only math estimate
                                       # instead of actually running each
                                       # model -- sweeps the whole cache
                                       # (thousands of models) in under a
                                       # second; ~9.6% avg error, 21% max
                                       # across dense/MoE/hybrid architectures
                                       # (see probe_formula.py)
blune sweep --compare                 # run BOTH the real probe and the
                                       # formula for each model, printing
                                       # both plus the delta, so you can
                                       # see where the fast estimate is
                                       # (and isn't) trustworthy
blune sweep --context 32000           # assume a long-context conversation
                                       # when estimating speed -- KV-cache
                                       # read grows with context for most
                                       # architectures (MLA and sliding-
                                       # window/hybrid-SSM models degrade
                                       # far less; see size_estimate.py)
```

Add `--offline` to any command to use only the local config cache -- zero
network calls, once a model's config has been fetched once.

## How the community config cache works

`configs/<org>/<repo>/config.json` in this repo is a **curated, git-tracked
cache**. Looking up a model checks, in order:

1. This repo's own `configs/` cache (instant, no network, no Hugging Face
   load at all)
2. Your local `~/.cache/huggingface` (via `huggingface_hub`'s normal
   caching)
3. A live fetch from the Hub -- which then gets written into (1), so the
   next person who pulls this repo never needs to fetch it again.

If you probe a model that isn't in `configs/` yet, please open a PR adding
the resulting `config.json` -- that's the whole contribution. A model
you've *actually measured* (not estimated) is even more valuable: add it to
`blune_cli/measurements.json` with the exact repo ID, library, and machine
identifier, and the tool will prefer your real number over any estimate for
that exact combination from then on.

## Architecture

```
blune_cli/
  hardware.py        auto-detect chip, RAM, real bandwidth (no user input)
  hardware_data.py   static chip database -- Apple Silicon "Pro" tier
                      explicitly included; this is the gap that started
                      the whole project
  probe_mlx.py        MLX: zero-download probe, validated (see table above)
  probe_llamacpp.py   llama.cpp: GGUF-header-only probe (bandwidth+overhead,
                      calibrated against 5 real measurements)
  probe_vllm.py       vLLM: reuses the MLX probe + a measured correction
                      ratio, since vllm-metal runs mlx-lm's own model
                      classes under the hood
  config_cache.py     curated-cache -> local-cache -> live-fetch lookup
  measurements.py     real, measured (not estimated) tok/s dataset
  search.py           Hugging Face Hub discovery
  ui.py               rich-based terminal rendering
  cli.py              interactive wizard + scriptable subcommands
```

## Known limitations (read before trusting a number)

- The MLX probe's random-weight approach measures the compute/memory
  pattern accurately, but MoE routing with random router weights may not
  perfectly replicate a trained router's token-to-token expert-reuse
  behavior. The ~81-85% calibration ratio absorbs this empirically; it is
  not a from-first-principles derivation.
- The llama.cpp probe now uses a bandwidth-ratio + fixed-overhead model
  fit against 5 real `llama-bench` measurements (mean 6.2% error, max
  10.4%) -- a big improvement on the old flat `0.60` guess (17.1% mean
  error on the same points), but all 5 points are one architecture
  family (Qwen2, dense) at two sizes on one machine, so treat it as a
  first real anchor, not a fully validated formula across GGUF's many
  quant types and MoE models. See `probe_llamacpp.py`'s docstring.
- The vLLM probe's ratio (`0.55`) comes from **one** real comparison on one
  machine. It also only estimates single-stream throughput; vLLM's actual
  strength is concurrent-request batching (we separately measured ~2.6x
  single-stream throughput at 24 concurrent requests on the same model --
  this tool doesn't model that scaling curve yet).
- Windows/Linux discrete-GPU detection uses `nvidia-smi` for VRAM and name,
  but does not query real memory bandwidth the way `hardware.py` does for
  Apple Silicon -- it falls back to a small static table by GPU name match.
- `probe_formula.py` (the instant, config-only estimate used by
  `blune sweep --formula`) models MoE active/shared experts, MLA,
  hybrid Mamba/SSM layers, sliding-window attention, and dense/MoE layer
  interleaving -- but its per-layer *weight-parameter* count still uses
  one generic attention+MLP formula for every layer, including Mamba/SSM
  ones, which have a materially different (and not yet generically
  parseable from config.json) weight structure. `blune sweep --compare`
  will show you exactly how far the formula drifts from the real probe on
  a specific model -- treat a large delta as a sign the model's
  architecture isn't well-represented by the formula yet, not as the
  formula being simply "wrong." See `docs/research-findings.md` for the
  full methodology and `size_estimate.py`'s docstring for the precise
  list of what is and isn't modeled.

## License

MIT.
