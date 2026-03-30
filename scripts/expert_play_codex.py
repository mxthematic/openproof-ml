"""Run expert play using Codex CLI + Pantograph verification.

This path uses the local Codex ChatGPT login rather than an API key:
  codex exec -> tactic JSON -> Pantograph verification -> saved JSONL

Usage:
    python scripts/expert_play_codex.py --config configs/codex_expert_play.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from openproof_ml.data.formatting import format_prompt
from openproof_ml.search.best_first import best_first_search
from openproof_ml.search.codex_cli import CodexCLIProposer
from openproof_ml.search.pantograph_client import PantographClient
from openproof_ml.utils.config import load_config

logger = logging.getLogger(__name__)


def load_problems(path: str | Path, max_problems: int | None = None, seed: int = 42) -> list[dict]:
    """Load theorem statements from a JSONL file or directory of JSONL files."""
    problems = []
    problems_path = Path(path)

    files = [problems_path] if problems_path.is_file() else sorted(problems_path.glob("*.jsonl"))
    for jsonl_file in files:
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line.strip())
                expr = data.get("type_expr") or data.get("goal_state") or data.get("statement")
                if expr:
                    problems.append({"type_expr": expr, "name": data.get("name", "")})

    if max_problems is not None and len(problems) > max_problems:
        random.Random(seed).shuffle(problems)
        problems = problems[:max_problems]

    logger.info("Loaded %d problems from %s", len(problems), path)
    return problems


def retrace_positive_pairs(
    pantograph: PantographClient,
    type_expr: str,
    tactics: list[str],
) -> list[dict]:
    """Re-run a solved proof to capture verified (prompt, completion) pairs."""
    state_id = pantograph.start_goal(type_expr)
    if state_id is None:
        return []

    pairs = []
    current_goals = [type_expr]

    try:
        for tactic in tactics:
            goal_text = current_goals[0] if current_goals else type_expr
            result = pantograph.try_tactic(state_id, 0, tactic)
            if not result.success or result.new_state_id is None:
                break

            pairs.append({"prompt": format_prompt(goal_text), "completion": tactic})
            pantograph.delete_goal(state_id)
            state_id = result.new_state_id
            current_goals = result.remaining_goals
    finally:
        pantograph.delete_goal(state_id)

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    codex_cfg = cfg["codex"]
    search_cfg = cfg["search"]
    data_cfg = cfg["data"]
    panto_cfg = cfg["pantograph"]

    run_dir = Path(data_cfg["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problems(
        data_cfg["problems_path"],
        max_problems=data_cfg.get("max_problems"),
        seed=data_cfg.get("seed", 42),
    )

    proposer = CodexCLIProposer(
        codex_bin=codex_cfg.get("codex_bin", "codex"),
        workdir=codex_cfg.get("workdir", "/tmp/openproof-codex-worker"),
        model=codex_cfg.get("model"),
        reasoning_effort=codex_cfg.get("reasoning_effort", "minimal"),
        verbosity=codex_cfg.get("verbosity", "low"),
        timeout=codex_cfg.get("timeout", 120),
        sandbox=codex_cfg.get("sandbox", "read-only"),
    )
    proposer.ensure_login()

    pantograph = PantographClient(
        panto_cfg.get("lean_project_path", "lean"),
        panto_cfg.get("repl_path"),
    )
    pantograph.start()
    logger.info("Pantograph ready")

    results_path = run_dir / "results.jsonl"
    positives_path = run_dir / "positives.jsonl"
    solved = 0
    total_pairs = 0

    try:
        for i, problem in enumerate(problems, start=1):
            if not pantograph.is_alive():
                pantograph.close()
                pantograph.start()

            type_expr = problem["type_expr"]
            result = best_first_search(
                pantograph,
                lambda goal_text: proposer.propose(goal_text, max_candidates=search_cfg.get("beam_width", 5)),
                type_expr,
                beam_width=search_cfg.get("beam_width", 5),
                max_expansions=search_cfg.get("max_expansions", 40),
                timeout=search_cfg.get("timeout", 60),
                max_depth=search_cfg.get("max_depth", 12),
                length_penalty=search_cfg.get("length_penalty", 0.05),
            )

            record = {
                "name": problem.get("name", f"problem_{i}"),
                "type_expr": type_expr,
                "solved": result.solved,
                "tactics": result.tactics,
                "remaining_goals": result.remaining_goals,
                "expansions": result.expansions,
                "elapsed": result.elapsed,
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if result.solved:
                solved += 1
                pairs = retrace_positive_pairs(pantograph, type_expr, result.tactics)
                total_pairs += len(pairs)
                with open(positives_path, "a") as f:
                    for pair in pairs:
                        f.write(json.dumps(pair) + "\n")

            logger.info(
                "Progress %d/%d solved=%d pairs=%d last_solved=%s expansions=%d elapsed=%.2fs",
                i,
                len(problems),
                solved,
                total_pairs,
                result.solved,
                result.expansions,
                result.elapsed,
            )

        summary = {
            "total_problems": len(problems),
            "solved": solved,
            "solve_rate": solved / len(problems) if problems else 0.0,
            "positive_pairs": total_pairs,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Finished. Run dir: %s", run_dir)
    finally:
        pantograph.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
