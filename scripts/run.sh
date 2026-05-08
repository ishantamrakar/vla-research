#!/usr/bin/env bash
# run.sh — run a wrapped training command and auto-verify the result.
#
# Usage:
#   ./scripts/run.sh python scripts/lerobot_train_wrapped.py [args...]
#   ./scripts/run.sh python scripts/my_experiment.py --output_dir outputs/exp1
#
# After the command finishes, this script extracts --output_dir from the
# arguments (if present) and runs verify_run.py automatically.
# If no --output_dir is found, it calls last_run.py --verify instead.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

# Run the command
uv run "$@"
EXIT_CODE=$?

echo ""
echo "─── Verifying run ───────────────────────────────────"

# Extract --output_dir value from the argument list
OUTPUT_DIR=""
for i in "$@"; do
    if [[ "$i" == --output_dir=* ]]; then
        OUTPUT_DIR="${i#--output_dir=}"
        break
    fi
done
# Also handle --output_dir <value> (space-separated)
PREV=""
for i in "$@"; do
    if [[ "$PREV" == "--output_dir" ]]; then
        OUTPUT_DIR="$i"
        break
    fi
    PREV="$i"
done

if [[ -n "$OUTPUT_DIR" ]]; then
    uv run python scripts/verify_run.py "$OUTPUT_DIR"
else
    uv run python scripts/last_run.py --verify
fi
