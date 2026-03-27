#!/usr/bin/env python3
"""Extract (goal_state, tactic) pairs from proof datasets.

Two modes:
  1. Direct extraction: datasets with pre-traced (state_before, tactic) fields
  2. Pantograph replay: datasets with whole proofs, replayed step-by-step

Usage:
    python scripts/extract_tactics.py \
        --input data/raw \
        --output data/processed/train.jsonl \
        --val-output data/processed/val.jsonl \
        --val-split 0.05 \
        --lean-project lean \
        --pantograph vendor/Pantograph/.lake/build/bin/repl
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

from openproof_ml.data.formatting import BANNED_TACTICS, format_training_example

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pantograph replay for whole-proof datasets
# ---------------------------------------------------------------------------


def replay_proof_through_pantograph(
    pantograph_client,
    statement: str,
    tactics: list[str],
) -> list[dict]:
    """Replay a proof through Pantograph, extracting (state, tactic) at each step.

    Args:
        pantograph_client: Running PantographClient instance.
        statement: The theorem type expression (forall ...).
        tactics: List of tactic strings from the proof.

    Returns:
        List of {"prompt": ..., "completion": ...} training examples.
    """
    pairs = []

    state_id = pantograph_client.start_goal(statement)
    if state_id is None:
        return pairs

    allocated = [state_id]
    current_state_id = state_id

    for tactic in tactics:
        tactic = tactic.strip()
        if not tactic or tactic.lower() in BANNED_TACTICS:
            continue

        # Get the current goal state description
        # We use try_tactic to both get the state and advance
        result = pantograph_client.try_tactic(current_state_id, 0, tactic)

        if result.success and result.new_state_id is not None:
            # Record the goal state BEFORE this tactic was applied
            # For the first tactic, this is the theorem statement itself
            # For subsequent tactics, it's the remaining goal from the previous step
            goal_text = result.remaining_goals[0] if result.remaining_goals else ""

            # The state we want is what was shown BEFORE the tactic,
            # which we can get from the parent's goal description
            # Since Pantograph doesn't directly expose "current goals" without
            # applying a tactic, we track goals from the previous result
            pairs.append(format_training_example(statement if not pairs else pairs[-1].get("_next_goal", statement), tactic))

            # Store the next goal for the following tactic
            if result.remaining_goals:
                pairs[-1]["_next_goal"] = result.remaining_goals[0]

            allocated.append(result.new_state_id)
            current_state_id = result.new_state_id

            # Proof complete
            if not result.remaining_goals:
                break
        else:
            # Tactic failed -- skip rest of proof
            break

    # Clean up the internal _next_goal field
    for p in pairs:
        p.pop("_next_goal", None)

    # Clean up Pantograph states
    for sid in allocated:
        try:
            pantograph_client.delete_goal(sid)
        except Exception:
            pass

    return pairs


def parse_proof_tactics(proof_text: str) -> list[str]:
    """Parse individual tactics from a whole-proof string.

    Handles:
      - by tactic1\n  tactic2\n  tactic3
      - by { tactic1; tactic2; tactic3 }
      - Nested have/suffices blocks (flattened)
    """
    # Strip leading "by" if present
    text = proof_text.strip()
    if text.startswith("by"):
        text = text[2:].strip()

    # Strip surrounding braces
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()

    # Split on newlines, filter empty lines and comments
    tactics = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("/-"):
            continue
        # Handle semicolon-separated tactics
        for part in line.split(";"):
            part = part.strip()
            if part:
                tactics.append(part)

    return tactics


# ---------------------------------------------------------------------------
# Dataset-specific extractors
# ---------------------------------------------------------------------------


def extract_leandojo(input_dir: Path) -> list[dict]:
    """Extract from LeanDojo format (pre-traced state/tactic pairs)."""
    pairs = []
    path = input_dir / "leandojo" / "train.jsonl"
    if not path.exists():
        logger.warning(f"LeanDojo data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            state = ex.get("state_before") or ex.get("state") or ex.get("tactic_state", "")
            tactic = ex.get("tactic") or ex.get("action", "")

            if not state or not tactic:
                continue
            if tactic.lower().strip() in BANNED_TACTICS:
                continue

            pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"LeanDojo: extracted {len(pairs)} pairs")
    return pairs


def extract_lean_workbook(input_dir: Path) -> list[dict]:
    """Extract from Lean Workbook format."""
    pairs = []
    path = input_dir / "lean_workbook" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Lean Workbook data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # Try pre-traced fields first
            state = ex.get("state_before") or ex.get("tactic_state", "")
            tactic = ex.get("tactic") or ex.get("action", "")

            if state and tactic:
                if tactic.lower().strip() not in BANNED_TACTICS:
                    pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"Lean Workbook: extracted {len(pairs)} pairs (pre-traced)")
    return pairs


def extract_goedel_pset_direct(input_dir: Path) -> list[dict]:
    """Extract pre-traced pairs from Goedel-Pset (no Pantograph needed)."""
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if "traced_tactics" in ex:
                for step in ex["traced_tactics"]:
                    state = step.get("state_before", "")
                    tactic = step.get("tactic", "")
                    if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                        pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"Goedel-Pset (pre-traced): extracted {len(pairs)} pairs")
    return pairs


def extract_goedel_pset_replay(
    input_dir: Path, pantograph_client
) -> list[dict]:
    """Extract pairs from Goedel-Pset whole proofs via Pantograph replay."""
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    replayed = 0
    failed = 0

    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip if already has traced tactics (handled by direct extractor)
            if "traced_tactics" in ex:
                continue

            # Need both statement and proof
            statement = ex.get("formal_statement") or ex.get("statement", "")
            proof = ex.get("proof", "")
            if not statement or not proof:
                continue

            tactics = parse_proof_tactics(proof)
            if not tactics:
                continue

            # Extract type expression from statement
            type_expr = extract_type_from_statement(statement)
            if not type_expr:
                continue

            try:
                new_pairs = replay_proof_through_pantograph(
                    pantograph_client, type_expr, tactics
                )
                pairs.extend(new_pairs)
                replayed += 1
            except Exception as e:
                failed += 1
                if failed <= 10:
                    logger.debug(f"Replay failed for proof {i}: {e}")

            if (i + 1) % 10000 == 0:
                logger.info(
                    f"  Goedel-Pset replay progress: {i+1} processed, "
                    f"{replayed} replayed, {failed} failed, {len(pairs)} pairs"
                )

    logger.info(
        f"Goedel-Pset (replay): {replayed} proofs replayed, "
        f"{failed} failed, {len(pairs)} pairs extracted"
    )
    return pairs


def extract_type_from_statement(statement: str) -> str | None:
    """Extract a type expression from a Lean theorem statement.

    Converts: theorem foo (x : Nat) : x + 0 = x := by ...
    To: forall (x : Nat), x + 0 = x
    """
    # Find theorem/lemma keyword and strip it + name
    match = re.match(r"(?:theorem|lemma|def)\s+\S+\s*(.*)", statement, re.DOTALL)
    if not match:
        return None

    rest = match.group(1).strip()

    # Find ":= by" and take everything before it
    by_idx = rest.rfind(":= by")
    if by_idx == -1:
        by_idx = rest.rfind(":=")
    if by_idx == -1:
        return None

    signature = rest[:by_idx].strip()

    # Find the last top-level ":" separating params from conclusion
    depth = 0
    colon_pos = None
    for i in range(len(signature) - 1, -1, -1):
        c = signature[i]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            depth -= 1
        elif c == ":" and depth == 0:
            if i + 1 < len(signature) and signature[i + 1] == "=":
                continue
            colon_pos = i
            break

    if colon_pos is None:
        return None

    params = signature[:colon_pos].strip()
    conclusion = signature[colon_pos + 1 :].strip()

    if not params:
        return conclusion
    return f"forall {params}, {conclusion}"


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
    parser.add_argument("--lean-project", help="Path to Lean project (for Pantograph replay)")
    parser.add_argument("--pantograph", help="Path to Pantograph REPL binary")
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Phase 1: Direct extraction (no Pantograph needed)
    logger.info("=== Phase 1: Direct extraction ===")
    all_pairs = []
    all_pairs.extend(extract_leandojo(input_dir))
    all_pairs.extend(extract_lean_workbook(input_dir))
    all_pairs.extend(extract_goedel_pset_direct(input_dir))
    logger.info(f"Phase 1 total: {len(all_pairs)} pairs")

    # Phase 2: Pantograph replay for whole-proof datasets
    if args.lean_project and args.pantograph:
        logger.info("=== Phase 2: Pantograph replay ===")
        from openproof_ml.search.pantograph_client import PantographClient

        lean_path = Path(args.lean_project)
        try:
            client = PantographClient(lean_path, repl_path=args.pantograph)
            client.start()
            logger.info("Pantograph started, replaying whole proofs...")

            replay_pairs = extract_goedel_pset_replay(input_dir, client)
            all_pairs.extend(replay_pairs)

            client.close()
        except Exception as e:
            logger.error(f"Pantograph replay failed: {e}")
            logger.info("Continuing with Phase 1 data only")
    else:
        logger.info("=== Phase 2: Skipped (no --lean-project / --pantograph) ===")
        logger.info("  To enable Pantograph replay: make setup-lean, then re-run with flags")

    logger.info(f"Total raw pairs: {len(all_pairs)}")

    # Deduplicate
    all_pairs = deduplicate(all_pairs)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)

    val_size = int(len(all_pairs) * args.val_split) if args.val_output else 0
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    # Write output
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
