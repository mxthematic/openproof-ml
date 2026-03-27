#!/usr/bin/env bash
set -euo pipefail

# Create a minimal Lean project with Mathlib dependency.
# Usage: ./scripts/setup_lean_project.sh <lean_version> <mathlib_version> <output_dir>

LEAN_VERSION="${1:-v4.28.0}"
MATHLIB_VERSION="${2:-v4.28.0}"
LEAN_DIR="${3:-lean}"

mkdir -p "$LEAN_DIR"

echo "leanprover/lean4:${LEAN_VERSION}" > "$LEAN_DIR/lean-toolchain"

cat > "$LEAN_DIR/lakefile.toml" << EOF
[package]
name = "openproof-ml-lean"
version = "0.1.0"

[[require]]
name = "mathlib"
git = "https://github.com/leanprover-community/mathlib4.git"
rev = "${MATHLIB_VERSION}"
EOF

echo "Created $LEAN_DIR/lean-toolchain and $LEAN_DIR/lakefile.toml"
