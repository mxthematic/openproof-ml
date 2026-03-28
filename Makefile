.PHONY: setup setup-lean setup-all download-data extract train-sft train-expert-iter train-dapo eval export test lint format clean

CONFIG ?= configs/sft_qwen35_2b.yaml
LEAN_VERSION ?= v4.28.0
MATHLIB_VERSION ?= v4.28.0

ELAN_BIN = $(HOME)/.elan/bin
LEAN_DIR = lean
PANTOGRAPH_DIR = vendor/Pantograph

# ── Full pipeline (one command on a fresh instance) ──────────────────

all: setup-all download-data extract train-sft
	@echo "Pipeline complete. Run 'make eval' to benchmark."

# ── Setup ────────────────────────────────────────────────────────────

setup:
	pip install -e ".[dev]"
	pre-commit install || true

setup-lean: setup-lean-toolchain setup-lean-mathlib setup-lean-pantograph
	@echo "=== Lean + Pantograph setup complete ==="

setup-lean-toolchain:
	@echo "=== Installing elan (Lean toolchain manager) ==="
	@if [ ! -f $(ELAN_BIN)/lean ]; then \
		curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:$(LEAN_VERSION); \
	else \
		echo "  elan already installed"; \
	fi

setup-lean-mathlib:
	@echo "=== Setting up Lean project with Mathlib ==="
	bash scripts/setup_lean_project.sh $(LEAN_VERSION) $(MATHLIB_VERSION) $(LEAN_DIR)
	cd $(LEAN_DIR) && $(ELAN_BIN)/lake update
	@echo "=== Downloading Mathlib cache ==="
	cd $(LEAN_DIR) && $(ELAN_BIN)/lake exe cache get || echo "  Cache download failed. Continuing..."

setup-lean-pantograph:
	@echo "=== Building Pantograph for Lean $(LEAN_VERSION) ==="
	@# Clean slate: remove any previous build
	rm -rf $(PANTOGRAPH_DIR)
	git clone --depth 1 --branch dev https://github.com/leanprover/Pantograph.git $(PANTOGRAPH_DIR)
	@# Apply 4.28 compatibility: override Library.lean and set toolchain
	cp scripts/pantograph-overrides/Library.lean $(PANTOGRAPH_DIR)/Pantograph/Library.lean
	echo "leanprover/lean4:$(LEAN_VERSION)" > $(PANTOGRAPH_DIR)/lean-toolchain
	cd $(PANTOGRAPH_DIR) && $(ELAN_BIN)/lake update && $(ELAN_BIN)/lake build repl
	@echo "  Pantograph REPL built at $(PANTOGRAPH_DIR)/.lake/build/bin/repl"

setup-all: setup setup-lean

# ── Data ─────────────────────────────────────────────────────────────

download-data:
	bash scripts/download_data.sh

extract:
	python scripts/extract_tactics.py \
		--input data/raw \
		--output data/processed/train.jsonl \
		--val-output data/processed/val.jsonl \
		--val-split 0.05 \
		--pantograph $(PANTOGRAPH_DIR)/.lake/build/bin/repl \
		--lean-project $(LEAN_DIR)

extract-fast:
	python scripts/extract_tactics.py \
		--input data/raw \
		--output data/processed/train.jsonl \
		--val-output data/processed/val.jsonl \
		--val-split 0.05 \
		--skip-phase2

# ── Training ─────────────────────────────────────────────────────────

train-sft:
	python -m openproof_ml.training.sft --config $(CONFIG)

train-expert-iter:
	python -m openproof_ml.training.expert_iteration --config $(CONFIG)

train-dapo:
	python -m openproof_ml.training.dapo --config $(CONFIG)

# ── Evaluation ───────────────────────────────────────────────────────

eval:
	python -m openproof_ml.eval.minif2f --config $(CONFIG)

# ── Export ────────────────────────────────────────────────────────────

export:
	python scripts/export_gguf.py --config $(CONFIG)

# ── Dev ──────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

clean:
	rm -rf checkpoints/ outputs/ wandb/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
