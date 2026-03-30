"""Tests for the Codex CLI tactic proposer."""

from openproof_ml.search.codex_cli import CodexCLIProposer


def test_build_prompt_includes_goal_and_candidate_budget():
    prompt = CodexCLIProposer.build_prompt("n : Nat\n|- n = n", max_candidates=3)
    assert "up to 3 Lean 4 tactics" in prompt
    assert "n : Nat\n|- n = n" in prompt


def test_parse_tactics_payload_filters_duplicates_and_banned_tactics():
    payload = """
    {
      "tactics": ["simp", "simp", "sorry", "omega\\n-- comment", "native_decide"]
    }
    """
    assert CodexCLIProposer.parse_tactics_payload(payload, max_candidates=5) == ["simp", "omega"]
