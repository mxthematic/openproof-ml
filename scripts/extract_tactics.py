#!/usr/bin/env python3
"""Extract (goal_state, tactic) pairs from proof datasets.

Phase 1: Pre-traced datasets (LeanDojo, Lean Workbook) -- instant
Phase 2: LeanDojo-v2 batch tracing of whole-proof datasets (Goedel)

Usage:
    python scripts/extract_tactics.py \
        --input data/raw \
        --output data/processed/train.jsonl \
        --val-output data/processed/val.jsonl \
        --val-split 0.05
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

from openproof_ml.data.formatting import BANNED_TACTICS, format_training_example

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Pre-traced extraction (no Lean needed)
# ---------------------------------------------------------------------------


def extract_leandojo(input_dir: Path) -> list[dict]:
    """Extract from LeanDojo/tasksource format (pre-traced)."""
    pairs = []
    path = input_dir / "leandojo" / "train.jsonl"
    if not path.exists():
        logger.warning(f"LeanDojo data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            traced = ex.get("traced_tactics", [])
            if traced:
                for step in traced:
                    if isinstance(step, dict):
                        state = step.get("state_before", "")
                        tactic = step.get("tactic", "")
                        if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                            pairs.append(format_training_example(state.strip(), tactic.strip()))
                continue

            state = ex.get("state_before") or ex.get("state", "")
            tactic = ex.get("tactic") or ex.get("action", "")
            if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"LeanDojo: extracted {len(pairs)} pairs")
    return pairs


def extract_lean_workbook(input_dir: Path) -> list[dict]:
    """Extract from Lean Workbook format (pre-traced fields)."""
    pairs = []
    path = input_dir / "lean_workbook" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Lean Workbook data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            state = ex.get("state_before") or ex.get("tactic_state", "")
            tactic = ex.get("tactic") or ex.get("action", "")
            if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"Lean Workbook: extracted {len(pairs)} pairs (pre-traced)")
    return pairs


# ---------------------------------------------------------------------------
# Phase 2: LeanDojo-v2 batch tracing
# ---------------------------------------------------------------------------


def build_goedel_lean_project(input_dir: Path, project_dir: Path) -> int:
    """Write all Goedel whole proofs into a single Lean project for batch tracing.

    Each proof becomes a separate .lean file in the project. LeanDojo traces
    the entire project at once, which is much faster than tracing individually.

    Returns the number of proof files written.
    """
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        return 0

    project_dir.mkdir(parents=True, exist_ok=True)

    # Write lean-toolchain and lakefile
    (project_dir / "lean-toolchain").write_text("leanprover/lean4:v4.28.0\n")
    (project_dir / "lakefile.toml").write_text(
        'name = "goedel-proofs"\nversion = "0.1.0"\n\n'
        '[[require]]\nname = "mathlib"\n'
        'git = "https://github.com/leanprover-community/mathlib4.git"\n'
        'rev = "v4.28.0"\n'
    )

    proofs_dir = project_dir / "Proofs"
    proofs_dir.mkdir(exist_ok=True)

    count = 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            full_proof = ex.get("full_proof", "")
            if not full_proof or ":= by" not in full_proof:
                continue

            # Write each proof as a separate file
            proof_file = proofs_dir / f"P{count:05d}.lean"
            proof_file.write_text(full_proof)
            count += 1

    logger.info(f"Wrote {count} proof files to {proofs_dir}")
    return count


def trace_goedel_with_leandojo(project_dir: Path) -> list[dict]:
    """Trace the Goedel proof project with LeanDojo-v2 and extract pairs."""
    try:
        from lean_dojo_v2.database import DynamicDatabase
    except ImportError:
        logger.warning(
            "lean-dojo-v2 not installed. Skipping Goedel tracing. "
            "Install with: pip install lean-dojo-v2"
        )
        return []

    pairs = []
    db = DynamicDatabase()

    logger.info(f"Tracing project at {project_dir} with LeanDojo-v2...")
    try:
        traced_repo = db.trace_repository(
            url=str(project_dir.resolve()),
            commit=None,
            build_deps=False,
        )

        for traced_file in traced_repo.traced_files:
            for theorem in traced_file.traced_theorems:
                for tt in theorem.get_traced_tactics(atomic_only=False):
                    state = tt.state_before
                    tactic = tt.tactic
                    if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                        pairs.append(format_training_example(state.strip(), tactic.strip()))

        logger.info(f"LeanDojo tracing: extracted {len(pairs)} pairs")
    except Exception as e:
        logger.error(f"LeanDojo tracing failed: {e}")
        logger.info("Falling back to Phase 1 data only")

    return pairs


def extract_goedel_phase2(input_dir: Path) -> list[dict]:
    """Full Phase 2: build project, trace with LeanDojo, extract pairs."""
    project_dir = Path("data/goedel_lean_project")

    # Build the Lean project from whole proofs
    count = build_goedel_lean_project(input_dir, project_dir)
    if count == 0:
        logger.info("Goedel-Pset: no whole proofs to trace")
        return []

    # Trace with LeanDojo
    return trace_goedel_with_leandojo(project_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicates."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["prompt"], p["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    logger.info(f"Dedup: {len(pairs)} -> {len(unique)} ({len(pairs) - len(unique)} removed)")
    return unique


def main():
    parser = argparse.ArgumentParser(description="Extract tactic training pairs")
    parser.add_argument("--input", required=True, help="Raw data directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--val-output", help="Validation JSONL path")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-phase2", action="store_true", help="Skip LeanDojo tracing")
    # Legacy flags (ignored)
    parser.add_argument("--lean-project", help="(ignored)")
    parser.add_argument("--pantograph", help="(ignored)")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Phase 1: Pre-traced datasets
    logger.info("=== Phase 1: Pre-traced extraction ===")
    all_pairs = []
    all_pairs.extend(extract_leandojo(input_dir))
    all_pairs.extend(extract_lean_workbook(input_dir))
    logger.info(f"Phase 1 total: {len(all_pairs)} pairs")

    # Phase 2: LeanDojo tracing of whole-proof datasets
    if not args.skip_phase2:
        logger.info("=== Phase 2: LeanDojo-v2 tracing ===")
        all_pairs.extend(extract_goedel_phase2(input_dir))
    else:
        logger.info("=== Phase 2: Skipped (--skip-phase2) ===")

    logger.info(f"Total raw pairs: {len(all_pairs)}")
    all_pairs = deduplicate(all_pairs)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)

    val_size = int(len(all_pairs) * args.val_split) if args.val_output else 0
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Wrote {len(train_pairs)} training pairs to {output_path}")

    if args.val_output and val_pairs:
        val_path = Path(args.val_output)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_path, "w") as f:
            for p in val_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Wrote {len(val_pairs)} validation pairs to {val_path}")

    logger.info("=== Extraction complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
