#!/usr/bin/env python3
"""Download pre-extracted training data from HuggingFace.

This is the fast path -- skips Lean/Pantograph setup entirely.
Data was extracted using Pantograph's frontend.process with invocations.

Usage:
    python scripts/download_processed.py
"""

import os
from datasets import load_dataset

HF_DATASET = "markm39/openproof-tactic-pairs"

def main():
    os.makedirs("data/processed", exist_ok=True)

    print(f"Downloading {HF_DATASET} from HuggingFace...")
    ds = load_dataset(HF_DATASET)

    ds["train"].to_json("data/processed/train.jsonl")
    ds["validation"].to_json("data/processed/val.jsonl")

    print(f"Downloaded {len(ds['train'])} train, {len(ds['validation'])} val pairs")
    print("Saved to data/processed/train.jsonl and data/processed/val.jsonl")


if __name__ == "__main__":
    main()
