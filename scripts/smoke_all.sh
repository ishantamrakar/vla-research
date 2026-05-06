#!/usr/bin/env bash
# Run all smoke tests in sequence. Exit 0 = env is healthy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] PushT Diffusion Policy ==="
bash "$SCRIPT_DIR/smoke_pusht.sh"

echo "=== [2/3] LIBERO-Spatial ACT ==="
bash "$SCRIPT_DIR/smoke_libero.sh"

echo "=== [3/3] ManiSkill 3 GPU-parallel ==="
uv run python "$SCRIPT_DIR/smoke_maniskill.py"

echo ""
echo "All smoke tests passed."
