"""Python client for the Pantograph REPL.

Mirrors the Rust implementation in openproof's pantograph.rs.
Used during expert iteration and evaluation for fast tactic verification.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TacticResult:
    """Result of applying a tactic via Pantograph."""

    success: bool
    remaining_goals: list[str]
    new_state_id: int | None
    error: str | None


class PantographClient:
    """Client for the Pantograph REPL process.

    Keeps the Lean/Mathlib environment loaded in memory (~18s startup),
    then tests tactics in ~3ms each.
    """

    def __init__(self, lean_project_path: str | Path, repl_path: str | None = None):
        self.lean_project_path = Path(lean_project_path)
        self.repl_path = repl_path or self._find_repl()
        self.process: subprocess.Popen | None = None

    def _find_repl(self) -> str:
        """Find the Pantograph REPL binary."""
        # Check vendor location (sibling to lean project)
        vendor = self.lean_project_path / "../vendor/Pantograph/.lake/build/bin/repl"
        if vendor.exists():
            return str(vendor.resolve())

        # Check PATH
        import shutil

        path = shutil.which("pantograph-repl")
        if path:
            return path

        raise FileNotFoundError(
            "Pantograph REPL not found. "
            "Build it: cd vendor/Pantograph && lake build repl"
        )

    def _find_lake(self) -> str:
        """Find the lake binary."""
        # Check elan install location
        elan_lake = Path.home() / ".elan" / "bin" / "lake"
        if elan_lake.exists():
            return str(elan_lake)
        # Check PATH
        import shutil
        path = shutil.which("lake")
        if path:
            return path
        raise FileNotFoundError("lake not found. Install elan: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh")

    def _resolve_lean_path(self) -> str:
        """Get LEAN_PATH from the Lean project."""
        lake = self._find_lake()
        result = subprocess.run(
            [lake, "env", "sh", "-c", "echo $LEAN_PATH"],
            cwd=self.lean_project_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def start(self):
        """Spawn the Pantograph REPL and wait for Mathlib to load."""
        lean_path = self._resolve_lean_path()

        self.process = subprocess.Popen(
            [self.repl_path, "Mathlib"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "LEAN_PATH": lean_path},
            cwd=str(self.lean_project_path),
        )

        # Wait for "ready." signal (~18s for Mathlib import)
        ready_line = self.process.stdout.readline().decode().strip()
        if not ready_line.startswith("ready"):
            raise RuntimeError(f"Pantograph did not send ready signal: {ready_line}")

    def _send(self, cmd: str, payload: dict) -> dict:
        """Send a command and read the JSON response."""
        msg = json.dumps({"cmd": cmd, "payload": payload})
        self.process.stdin.write(f"{msg}\n".encode())
        self.process.stdin.flush()

        response_line = self.process.stdout.readline().decode().strip()
        return json.loads(response_line)

    def is_alive(self) -> bool:
        """Check if the REPL process is still running."""
        return self.process is not None and self.process.poll() is None

    def start_goal(self, expr: str) -> int | None:
        """Start a proof goal from a type expression. Returns state_id."""
        response = self._send("goal.start", {"expr": expr})
        return response.get("stateId")

    def try_tactic(self, state_id: int, goal_id: int, tactic: str) -> TacticResult:
        """Apply a tactic to a goal state."""
        response = self._send(
            "goal.tactic",
            {"stateId": state_id, "goalId": goal_id, "tactic": tactic},
        )

        # Check for parse errors
        if parse_error := response.get("parseError"):
            return TacticResult(
                success=False, remaining_goals=[], new_state_id=None, error=parse_error
            )

        # Check for tactic errors
        if tactic_errors := response.get("tacticErrors"):
            if tactic_errors:
                return TacticResult(
                    success=False,
                    remaining_goals=[],
                    new_state_id=None,
                    error="; ".join(tactic_errors),
                )

        # Check for error messages
        messages = response.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        if errors:
            return TacticResult(
                success=False,
                remaining_goals=[],
                new_state_id=None,
                error=errors[0].get("data", "unknown error"),
            )

        new_state_id = response.get("nextStateId") or response.get("stateId")

        # Extract goals in target.pp format (matching Rust pantograph.rs)
        goals = []
        for g in response.get("goals", []):
            if isinstance(g, dict):
                pp = g.get("target", {}).get("pp")
                if pp:
                    goals.append(pp)
            elif isinstance(g, str):
                goals.append(g)

        return TacticResult(
            success=True,
            remaining_goals=goals,
            new_state_id=new_state_id,
            error=None,
        )

    def delete_goal(self, state_id: int):
        """Delete a goal state to free memory."""
        self._send("goal.delete", {"stateId": state_id})

    def close(self):
        """Kill the REPL process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
