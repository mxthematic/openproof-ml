"""Codex CLI proposer backed by ChatGPT OAuth login.

Uses `codex exec` in non-interactive mode to generate tactic candidates
without requiring an OpenAI API key. This is intended for low-volume,
high-quality expert-play data generation rather than dense search.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..data.formatting import parse_tactic

logger = logging.getLogger(__name__)


DEFAULT_INSTRUCTIONS = """You are a Lean 4 tactic suggestion engine.
Return only the final JSON object requested by the user.
Do not run shell commands or inspect the filesystem.
Do not add explanations, markdown, or comments.
Prefer short, directly executable next-step tactics.
Never use sorry, admit, or native_decide.
"""


class CodexCLIProposer:
    """Generate tactic candidates through `codex exec`."""

    def __init__(
        self,
        codex_bin: str = "codex",
        workdir: str | Path = "/tmp/openproof-codex-worker",
        model: str | None = None,
        reasoning_effort: str = "minimal",
        verbosity: str = "low",
        timeout: int = 120,
        sandbox: str = "read-only",
    ):
        self.codex_bin = codex_bin
        self.workdir = Path(workdir)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.timeout = timeout
        self.sandbox = sandbox

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.instructions_file = self.workdir / "instructions.txt"
        if not self.instructions_file.exists():
            self.instructions_file.write_text(DEFAULT_INSTRUCTIONS)

    def ensure_login(self):
        """Fail fast if Codex is not logged in."""
        result = subprocess.run(
            [self.codex_bin, "login", "status"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Codex CLI is not logged in. Run `codex login` with ChatGPT auth first."
            )

    @staticmethod
    def build_prompt(goal_text: str, max_candidates: int) -> str:
        """Build a minimal prompt for next-tactic generation."""
        return f"""Return a JSON object with a single key `tactics`.

Task:
- Suggest up to {max_candidates} Lean 4 tactics for the current goal state.
- Order tactics from most promising to least promising.
- Each array element must be a single next-step tactic string.
- If no reasonable tactic is available, return an empty array.

Current goal state:
```lean
{goal_text}
```
"""

    @staticmethod
    def parse_tactics_payload(payload: str, max_candidates: int) -> list[str]:
        """Parse Codex JSON output into a filtered tactic list."""
        data = json.loads(payload)
        tactics = []
        for raw in data.get("tactics", [])[:max_candidates]:
            if not isinstance(raw, str):
                continue
            tactic = parse_tactic(raw)
            if tactic and tactic not in tactics:
                tactics.append(tactic)
        return tactics

    def _schema_file(self, max_candidates: int) -> Path:
        schema_path = self.workdir / f"tactics_schema_{max_candidates}.json"
        if schema_path.exists():
            return schema_path

        schema = {
            "type": "object",
            "properties": {
                "tactics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max_candidates,
                }
            },
            "required": ["tactics"],
            "additionalProperties": False,
        }
        schema_path.write_text(json.dumps(schema))
        return schema_path

    def propose(self, goal_text: str, max_candidates: int = 5) -> list[str]:
        """Generate tactic candidates for a single goal state."""
        schema_path = self._schema_file(max_candidates)
        prompt = self.build_prompt(goal_text, max_candidates=max_candidates)

        with NamedTemporaryFile("w+", suffix=".json", delete=False, dir=self.workdir) as f:
            output_path = Path(f.name)

        cmd = [
            self.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            self.sandbox,
            "-C",
            str(self.workdir),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-c",
            f'model_instructions_file="{self.instructions_file}"',
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            f'model_verbosity="{self.verbosity}"',
            "-c",
            'approval_policy="never"',
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "codex exec failed")

            payload = output_path.read_text().strip()
            if not payload:
                return []
            return self.parse_tactics_payload(payload, max_candidates=max_candidates)
        finally:
            output_path.unlink(missing_ok=True)
