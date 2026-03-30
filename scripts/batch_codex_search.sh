#!/usr/bin/env bash
set -euo pipefail

# Batch Codex tactic search on Goedel-Pset statements.
# Uses `openproof run` which handles session creation, Pantograph, everything.
#
# Usage:
#   bash scripts/batch_codex_search.sh data/raw/goedel_pset/statements.jsonl 4 100
#
# Prerequisites:
#   - openproof built and on PATH
#   - Logged in (openproof login)

INPUT="${1:?Usage: $0 <statements.jsonl> <parallel> [limit]}"
PARALLEL="${2:-4}"
LIMIT="${3:-100}"

export OPENPROOF_TACTIC_PROPOSER=codex
export OPENPROOF_TACTIC_MODEL="${OPENPROOF_TACTIC_MODEL:-gpt-5.4}"

EXPORT_DIR="$HOME/.openproof/expert-data"
POSITIVES="$EXPORT_DIR/positives.jsonl"

echo "=== Batch Codex Tactic Search ==="
echo "Input: $INPUT"
echo "Workers: $PARALLEL"
echo "Limit: $LIMIT"
echo "Model: $OPENPROOF_TACTIC_MODEL"
echo ""

# Extract theorem statements from JSONL
# Each line becomes a problem for `openproof run`
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

python3 -c "
import json, sys, os
limit = int('$LIMIT')
tmpdir = '$TMPDIR'
count = 0
with open('$INPUT') as f:
    for line in f:
        if count >= limit:
            break
        ex = json.loads(line)
        full_proof = ex.get('full_proof', '')
        statement = ex.get('formal_statement', '') or ex.get('statement', '')

        # Build a lean file with sorry
        if full_proof and ':= by' in full_proof:
            idx = full_proof.rfind(':= by')
            lean = full_proof[:idx] + ':= by\n  sorry\n'
        elif statement:
            lean = f'import Mathlib\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n{statement}'
            if ':= by' not in lean:
                continue
            idx = lean.rfind(':= by')
            lean = lean[:idx] + ':= by\n  sorry\n'
        else:
            continue

        # Write to temp file
        path = os.path.join(tmpdir, f'p{count:06d}.lean')
        with open(path, 'w') as out:
            out.write(lean)
        count += 1

print(f'Prepared {count} problems')
"

INITIAL_PAIRS=0
if [ -f "$POSITIVES" ]; then
    INITIAL_PAIRS=$(wc -l < "$POSITIVES")
fi
echo "Starting with $INITIAL_PAIRS existing pairs"

# Run openproof run on each lean file content in parallel
# openproof run takes a problem description and creates a session
run_one() {
    local lean_file="$1"
    local content
    content=$(cat "$lean_file")

    # Use openproof run with the theorem as the problem
    timeout 180 openproof run "$content" --label "expert_batch" 2>/dev/null || true
}
export -f run_one

find "$TMPDIR" -name "*.lean" | sort | \
    xargs -P "$PARALLEL" -I {} bash -c 'run_one "$@"' _ {}

FINAL_PAIRS=0
if [ -f "$POSITIVES" ]; then
    FINAL_PAIRS=$(wc -l < "$POSITIVES")
fi
NEW_PAIRS=$((FINAL_PAIRS - INITIAL_PAIRS))

echo ""
echo "=== Done ==="
echo "New verified pairs: $NEW_PAIRS"
echo "Total pairs: $FINAL_PAIRS"
echo "Export: $POSITIVES"
