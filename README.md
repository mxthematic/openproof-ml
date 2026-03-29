# openproof-ml

Training pipeline for on-device Lean 4 tactic prediction models.

Takes a goal state, outputs a tactic. Small enough to run locally (~1-1.5GB quantized), fast enough for real-time proof search (<300ms/tactic on Apple Silicon).

## Overview

This repo trains a step-level tactic model for [OpenProof](https://github.com/markm39/openproof). The model plugs into OpenProof's best-first search via ollama -- zero code changes needed on the inference side.

**Training pipeline:**
1. **SFT** on 1.2M (state, tactic) pairs from Mathlib4, Lean Workbook, and Goedel-Pset
2. **Expert iteration** -- self-play proof search discovers new proofs, generating fresh training data
3. **DAPO RL** -- reinforcement learning with per-tactic Lean compiler feedback

**Base models compared:**
- Qwen3.5-2B (March 2026, hybrid GDN attention -- first application to theorem proving)
- Qwen3-1.7B (proven base, used by Kimina-Prover-RL-1.7B at 76.6% MiniF2F)

## Quick start (fresh GPU instance)

Everything from zero to trained model in one shot:

```bash
git clone https://github.com/markm39/openproof-ml.git
cd openproof-ml
make all    # installs deps, downloads data from HuggingFace, trains
```

Or step by step:

```bash
# 1. Install Python deps
make setup

# 2. Download pre-extracted training data from HuggingFace (~350K pairs)
make get-data

# 3. Train SFT (needs GPU)
make train-sft CONFIG=configs/sft_qwen35_2b.yaml

# 4. Evaluate on MiniF2F
make eval CONFIG=configs/eval_minif2f.yaml

# 5. Export to GGUF + ollama
make export CONFIG=configs/export.yaml
```

### Re-extracting data from scratch (optional)

If you want to re-extract training data from source (requires Lean + Pantograph):

```bash
make setup-all          # installs Lean, Mathlib, Pantograph
make download-data      # downloads raw datasets
make extract            # extracts (state, tactic) pairs via Pantograph
```

### Prerequisites

- Python 3.10+
- CUDA GPU (A100-80GB recommended, ~$0.78/hr on Thunder Compute)
- ~50GB disk (datasets + Mathlib cache + checkpoints)
- Internet access for initial downloads

Everything else (Lean, elan, Mathlib, Pantograph) is installed automatically by `make setup-all`.

## Project structure

```
configs/          YAML configs for each experiment
scripts/          Data download, extraction, export scripts
src/openproof_ml/
  data/           Dataset loading, prompt formatting
  model/          Model wrappers
  training/       SFT, expert iteration, DAPO trainers
  eval/           MiniF2F evaluation harness
  search/         Pantograph client + best-first search (Python)
  utils/          Config loading, logging
tests/            Unit tests
paper/            Paper (LaTeX)
lean/             Lean project (created by make setup-lean)
vendor/           Pantograph REPL (built by make setup-lean)
```

## Prompt format

The model uses the BFS-Prover-V2 format:

```
{goal_state}:::
```

Input is the raw Lean goal state (Pantograph `target.pp` format). Output is a single tactic. No chat template.

## Integration with OpenProof

The trained model is served via ollama and consumed by OpenProof's `OllamaProposer`:

```
openproof-ml (training) --> GGUF --> ollama --> openproof (inference)
```

## Training cost

| Stage | GPU Hours | Cost (A100 @ $0.78/hr) |
|-------|----------|------------------------|
| SFT (x2 bases) | 16 | $12 |
| Expert iteration (3 rounds) | 200 | $156 |
| DAPO RL | 24 | $19 |
| Eval | 10 | $8 |
| **Total** | **~250** | **~$195** |

## License

Apache 2.0
