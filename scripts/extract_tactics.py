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

    For each tactic, we record the goal state BEFORE the tactic is applied,
    paired with the tactic itself. This matches what the model sees at
    inference time: given a goal, predict the next tactic.

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

    # The goal BEFORE the first tactic is the theorem statement itself.
    # After each tactic, remaining_goals gives us the state for the next tactic.
    current_goal = statement

    for tactic in tactics:
        tactic = tactic.strip()
        if not tactic or tactic.lower() in BANNED_TACTICS:
            continue

        result = pantograph_client.try_tactic(current_state_id, 0, tactic)

        if result.success and result.new_state_id is not None:
            # Record: (goal before this tactic, this tactic)
            pairs.append(format_training_example(current_goal, tactic))

            allocated.append(result.new_state_id)
            current_state_id = result.new_state_id

            # Update current_goal for the next tactic
            if result.remaining_goals:
                current_goal = result.remaining_goals[0]
            else:
                # Proof complete
                break
        else:
            # Tactic failed -- stop replaying
            break

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

    # Split on newlines, filter empty lines and comments.
    # Do NOT split on semicolons -- in Lean 4, `<;>` and `;` are tactic
    # combinators (apply to all goals), not statement separators.
    tactics = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("/-"):
            continue
        tactics.append(line)

    return tactics


# ---------------------------------------------------------------------------
# Dataset-specific extractors
# ---------------------------------------------------------------------------


def extract_leandojo(input_dir: Path) -> list[dict]:
    """Extract from LeanDojo/tasksource format.

    The tasksource/leandojo dataset has nested traced_tactics:
    {
      "traced_tactics": [
        {"state_before": "...", "tactic": "...", "state_after": "..."},
        ...
      ]
    }
    """
    pairs = []
    path = input_dir / "leandojo" / "train.jsonl"
    if not path.exists():
        logger.warning(f"LeanDojo data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)

            # Handle nested traced_tactics (tasksource/leandojo format)
            traced = ex.get("traced_tactics", [])
            if traced:
                for step in traced:
                    if isinstance(step, dict):
                        state = step.get("state_before", "")
                        tactic = step.get("tactic", "")
                    elif isinstance(step, str):
                        continue
                    else:
                        continue

                    if not state or not tactic:
                        continue
                    if tactic.lower().strip() in BANNED_TACTICS:
                        continue
                    pairs.append(format_training_example(state.strip(), tactic.strip()))
                continue

            # Fallback: top-level fields
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
    """Extract what we can from Goedel data without Pantograph.

    Goedel-LM/Lean-workbook-proofs has {problem_id, full_proof} where
    full_proof is a complete Lean file. We can't extract step-level pairs
    without Pantograph replay, but we log what's available.
    """
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    whole_proofs = 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # Check for pre-traced tactics (unlikely but handle it)
            if "traced_tactics" in ex:
                for step in ex["traced_tactics"]:
                    if isinstance(step, dict):
                        state = step.get("state_before", "")
                        tactic = step.get("tactic", "")
                        if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                            pairs.append(format_training_example(state.strip(), tactic.strip()))
            elif "full_proof" in ex:
                whole_proofs += 1

    if whole_proofs > 0 and not pairs:
        logger.info(
            f"Goedel-Pset: {whole_proofs} whole proofs found but need Pantograph replay "
            f"to extract step-level pairs. Run with --lean-project and --pantograph flags."
        )
    else:
        logger.info(f"Goedel-Pset (pre-traced): extracted {len(pairs)} pairs")
    return pairs


def split_full_proof(full_proof: str) -> tuple[str, str] | None:
    """Split a full_proof field into (theorem_statement, proof_body).

    full_proof is an entire Lean file as a string:
        import Mathlib\nimport Aesop\nset_option ...\nopen ...\n
        theorem foo (x : Nat) : x = x := by\n  rfl

    We find the last `:= by` (the proof start), walk backwards to find
    `theorem`/`lemma`, and split there.

    Returns (statement_without_by, proof_tactics_text) or None.
    """
    # Find the LAST ":= by" (in case imports have weird content)
    by_idx = full_proof.rfind(":= by")
    if by_idx == -1:
        return None

    # Walk backwards to find "theorem" or "lemma" or "def"
    before_by = full_proof[:by_idx]
    best_kw_idx = -1
    for keyword in ["theorem ", "lemma ", "def "]:
        kw_idx = before_by.rfind(keyword)
        if kw_idx > best_kw_idx:
            best_kw_idx = kw_idx

    if best_kw_idx == -1:
        return None

    # statement = from "theorem" to just before ":= by"
    statement = full_proof[best_kw_idx:by_idx].strip()
    # proof_body = everything after ":= by"
    proof_body = full_proof[by_idx + len(":= by"):].strip()

    return (statement, proof_body)


def extract_goedel_pset_replay(
    input_dir: Path, pantograph_client
) -> list[dict]:
    """Extract pairs from Goedel-Pset whole proofs via Pantograph replay.

    The Goedel-LM/Lean-workbook-proofs dataset has {problem_id, full_proof}
    where full_proof is an entire Lean file. We parse out the theorem
    statement and proof tactics, then replay through Pantograph.
    """
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    replayed = 0
    failed = 0
    skipped = 0

    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)

            # Skip if already has traced tactics (handled by direct extractor)
            if "traced_tactics" in ex:
                continue

            # Parse full_proof field (Goedel-LM/Lean-workbook-proofs format)
            full_proof = ex.get("full_proof", "")
            if not full_proof:
                # Try older field names as fallback
                statement = ex.get("formal_statement") or ex.get("statement", "")
                proof = ex.get("proof", "")
                if not statement or not proof:
                    skipped += 1
                    continue
            else:
                parsed = split_full_proof(full_proof)
                if parsed is None:
                    skipped += 1
                    continue
                statement, proof = parsed

            tactics = parse_proof_tactics(proof)
            if not tactics:
                skipped += 1
                continue

            # Extract type expression from statement
            type_expr = extract_type_from_statement(statement)
            if not type_expr:
                skipped += 1
                continue

            try:
                new_pairs = replay_proof_through_pantograph(
                    pantograph_client, type_expr, tactics
                )
                if new_pairs:
                    pairs.extend(new_pairs)
                    replayed += 1
                else:
                    failed += 1
                    if failed <= 5:
                        logger.debug(
                            f"Replay got 0 pairs for proof {i}: "
                            f"type_expr={type_expr[:80]}... tactics={tactics[:2]}"
                        )
            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.debug(f"Replay exception for proof {i}: {e}")

            if (i + 1) % 5000 == 0:
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

    Handles both formats:
        theorem foo (x : Nat) : x + 0 = x := by ...
        theorem foo (x : Nat) : x + 0 = x           (no := by)
    Converts to: forall (x : Nat), x + 0 = x
    """
    # Find theorem/lemma keyword and strip it + name
    match = re.match(r"(?:theorem|lemma|def)\s+\S+\s*(.*)", statement, re.DOTALL)
    if not match:
        return None

    rest = match.group(1).strip()

    # Strip trailing ":= by ..." or ":= ..." if present
    by_idx = rest.rfind(":= by")
    if by_idx != -1:
        rest = rest[:by_idx].strip()
    else:
        eq_idx = rest.rfind(":=")
        if eq_idx != -1:
            rest = rest[:eq_idx].strip()

    signature = rest

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
