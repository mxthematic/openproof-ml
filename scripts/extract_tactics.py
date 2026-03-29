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
import time
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

    def _send(self, cmd: str, payload: dict, timeout: float = 30.0) -> dict:
        import select

        msg = json.dumps({"cmd": cmd, "payload": payload})
        self.process.stdin.write(f"{msg}\n".encode())
        self.process.stdin.flush()

        # Wait for response with timeout (prevents hanging on stuck proofs)
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            # Kill and mark as dead so caller restarts
            self.process.kill()
            self.process.wait()
            self.process = None
            raise TimeoutError(f"Pantograph timed out after {timeout}s")

        response = self.process.stdout.readline().decode().strip()
        if not response:
            raise RuntimeError("Pantograph returned empty response (process died)")
        return json.loads(response)

    def extract_invocations(self, lean_source: str) -> list[dict]:
        """Send a Lean file to frontend.process and get tactic invocations.

        Uses inheritEnv=True so Mathlib (already loaded at REPL startup) is
        reused. Import lines are stripped since the env already has them.

        Returns list of {"goalBefore": ..., "tactic": ..., "goalAfter": ...}
        """
        # Strip import/set_option/open lines -- env already has Mathlib loaded
        lines = lean_source.split("\n")
        body_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("set_option ") or stripped.startswith("open "):
                continue
            body_lines.append(line)
        body = "\n".join(body_lines)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            invocations_path = f.name

        try:
            response = self._send("frontend.process", {
                "file": body,
                "readHeader": False,
                "inheritEnv": True,
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

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


def _worker_extract_chunk(args: tuple) -> list[dict]:
    """Worker function: process a chunk of proofs with its own Pantograph instance.

    Each worker spawns its own Pantograph REPL (~18s startup), then processes
    its assigned proofs independently. Automatically restarts on crash.
    """
    worker_id, proofs, repl_path, lean_project = args
    pairs = []
    traced = 0
    failed = 0
    restarts = 0

    pg = PantographFrontend(repl_path, lean_project)
    pg.start()
    logger.info(f"  Worker {worker_id}: started, processing {len(proofs)} proofs")

    for local_idx, (global_idx, full_proof) in enumerate(proofs):
        if not pg.is_alive():
            pg.close()
            pg = PantographFrontend(repl_path, lean_project)
            pg.start()
            restarts += 1

        try:
            t0 = time.monotonic()
            invocations = pg.extract_invocations(full_proof)
            elapsed = time.monotonic() - t0

            if invocations:
                n_before = len(pairs)
                for inv in invocations:
                    goal_before = inv.get("goalBefore", "")
                    tactic = inv.get("tactic", "")
                    if goal_before and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                        pairs.append(format_training_example(goal_before.strip(), tactic.strip()))
                n_new = len(pairs) - n_before
                traced += 1
                if local_idx < 3 or (local_idx + 1) % 200 == 0:
                    logger.info(
                        f"  W{worker_id} [{global_idx}] OK {elapsed:.1f}s "
                        f"+{n_new} pairs (total: {len(pairs)})"
                    )
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if local_idx < 3 or (local_idx + 1) % 200 == 0:
                logger.info(f"  W{worker_id} [{global_idx}] ERR: {e}")

        if (local_idx + 1) % 500 == 0:
            logger.info(
                f"  W{worker_id}: {local_idx+1}/{len(proofs)} "
                f"traced={traced} failed={failed} pairs={len(pairs)}"
            )

    pg.close()
    logger.info(
        f"  Worker {worker_id} done: {traced} traced, {failed} failed, "
        f"{restarts} restarts, {len(pairs)} pairs"
    )
    return pairs


def extract_goedel_pantograph(
    input_dir: Path, repl_path: str, lean_project: str, num_workers: int = 1,
) -> list[dict]:
    """Extract (state, tactic) pairs from Goedel whole proofs via Pantograph.

    Uses frontend.process with invocations to get kernel-level tactic states.
    Spawns num_workers parallel Pantograph instances for throughput.
    """
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return []

    # Pre-read all proofs into memory
    logger.info(f"Loading proofs from {path}...")
    proofs = []
    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            full_proof = ex.get("full_proof", "")
            if full_proof and ":= by" in full_proof:
                proofs.append((i, full_proof))
    logger.info(f"Loaded {len(proofs)} proofs to process")

    if not proofs:
        return []

    if num_workers <= 1:
        # Single-worker path (no multiprocessing overhead)
        return _worker_extract_chunk((0, proofs, repl_path, lean_project))

    # Shard proofs across workers
    from multiprocessing import Pool

    chunk_size = (len(proofs) + num_workers - 1) // num_workers
    chunks = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, len(proofs))
        if start < len(proofs):
            chunks.append((w, proofs[start:end], repl_path, lean_project))

    logger.info(f"Spawning {len(chunks)} workers ({chunk_size} proofs each)")

    all_pairs = []
    with Pool(processes=len(chunks)) as pool:
        results = pool.map(_worker_extract_chunk, chunks)
        for worker_pairs in results:
            all_pairs.extend(worker_pairs)

    logger.info(f"Goedel-Pset (parallel): {len(all_pairs)} pairs from {len(proofs)} proofs")
    return all_pairs


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
    parser.add_argument("--workers", type=int, default=1, help="Parallel Pantograph workers for Phase 2")
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
        logger.info(f"=== Phase 2: Pantograph frontend.process ({args.workers} workers) ===")
        all_pairs.extend(
            extract_goedel_pantograph(input_dir, args.pantograph, args.lean_project, num_workers=args.workers)
        )
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
