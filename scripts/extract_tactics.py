#!/usr/bin/env python3
"""Extract (goal_state, tactic) pairs from proof datasets.

Phase 1: Pre-traced datasets (LeanDojo, Lean Workbook) -- instant
Phase 2: Pantograph frontend.process with invocations -- kernel-level extraction

Usage:
    python scripts/extract_tactics.py \
        --input data/raw \
        --output data/processed/train.jsonl \
        --val-output data/processed/val.jsonl \
        --val-split 0.05 \
        --pantograph vendor/Pantograph/.lake/build/bin/repl \
        --lean-project lean
"""

import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
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
# Phase 2: Pantograph frontend.process with invocations
# ---------------------------------------------------------------------------


class PantographFrontend:
    """Pantograph REPL client using frontend.process for kernel-level extraction.

    Sends entire Lean files to Pantograph and gets back (goalBefore, tactic,
    goalAfter) triples extracted by Lean's own elaborator. No regex parsing.
    """

    def __init__(self, repl_path: str, lean_project_path: str):
        self.repl_path = str(Path(repl_path).resolve())
        self.lean_project_path = str(Path(lean_project_path).resolve())
        self.process = None
        self.lean_path = None

    def start(self):
        """Spawn the Pantograph REPL with Mathlib loaded."""
        self.lean_path = self._resolve_lean_path()

        self.process = subprocess.Popen(
            [self.repl_path, "Mathlib"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "LEAN_PATH": self.lean_path},
            cwd=self.lean_project_path,
        )

        ready = self.process.stdout.readline().decode().strip()
        if not ready.startswith("ready"):
            raise RuntimeError(f"Pantograph did not send ready: {ready}")
        logger.info("Pantograph REPL ready")

    def _resolve_lean_path(self) -> str:
        lake = str(Path.home() / ".elan" / "bin" / "lake")
        result = subprocess.run(
            [lake, "env", "sh", "-c", "echo $LEAN_PATH"],
            cwd=self.lean_project_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _send(self, cmd: str, payload: dict) -> dict:
        msg = json.dumps({"cmd": cmd, "payload": payload})
        self.process.stdin.write(f"{msg}\n".encode())
        self.process.stdin.flush()
        response = self.process.stdout.readline().decode().strip()
        return json.loads(response)

    def extract_invocations(self, lean_source: str) -> list[dict]:
        """Send a Lean file to frontend.process and get tactic invocations.

        Returns list of {"goalBefore": ..., "tactic": ..., "goalAfter": ...}
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            invocations_path = f.name

        try:
            response = self._send("frontend.process", {
                "file": lean_source,
                "readHeader": True,
                "inheritEnv": False,
                "newConstants": True,
                "invocations": invocations_path,
            })

            # Check for errors
            if "error" in response:
                return []

            # Read invocations file
            try:
                with open(invocations_path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return []

            invocations = []
            for unit in data.get("units", []):
                for inv in unit.get("invocations", []):
                    invocations.append(inv)
            return invocations
        finally:
            try:
                os.unlink(invocations_path)
            except OSError:
                pass

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


def extract_goedel_pantograph(input_dir: Path, pantograph: PantographFrontend) -> list[dict]:
    """Extract (state, tactic) pairs from Goedel whole proofs via Pantograph.

    Uses frontend.process with invocations to get kernel-level tactic states.
    Each proof is sent as a complete Lean file -- Pantograph handles parsing.
    """
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    traced = 0
    failed = 0

    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            full_proof = ex.get("full_proof", "")
            if not full_proof or ":= by" not in full_proof:
                continue

            invocations = pantograph.extract_invocations(full_proof)

            if invocations:
                for inv in invocations:
                    goal_before = inv.get("goalBefore", "")
                    tactic = inv.get("tactic", "")
                    if goal_before and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                        pairs.append(format_training_example(goal_before.strip(), tactic.strip()))
                traced += 1
            else:
                failed += 1

            if (i + 1) % 5000 == 0:
                logger.info(
                    f"  Goedel progress: {i+1} processed, "
                    f"{traced} traced, {failed} failed, {len(pairs)} pairs"
                )

    logger.info(
        f"Goedel-Pset (Pantograph): {traced} traced, {failed} failed, {len(pairs)} pairs"
    )
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def deduplicate(pairs: list[dict]) -> list[dict]:
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
    parser.add_argument("--pantograph", help="Path to Pantograph REPL binary")
    parser.add_argument("--lean-project", help="Path to Lean project with Mathlib")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Pantograph tracing")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Phase 1: Pre-traced datasets
    logger.info("=== Phase 1: Pre-traced extraction ===")
    all_pairs = []
    all_pairs.extend(extract_leandojo(input_dir))
    all_pairs.extend(extract_lean_workbook(input_dir))
    logger.info(f"Phase 1 total: {len(all_pairs)} pairs")

    # Phase 2: Pantograph kernel-level extraction
    if not args.skip_phase2 and args.pantograph and args.lean_project:
        logger.info("=== Phase 2: Pantograph frontend.process ===")
        pg = PantographFrontend(args.pantograph, args.lean_project)
        try:
            pg.start()
            all_pairs.extend(extract_goedel_pantograph(input_dir, pg))
        except Exception as e:
            logger.error(f"Pantograph extraction failed: {e}")
        finally:
            pg.close()
    elif not args.skip_phase2:
        logger.info("=== Phase 2: Skipped (no --pantograph/--lean-project) ===")
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
