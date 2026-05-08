#!/usr/bin/env python
"""
last_run.py — find and display the most recent summary.json under outputs/.

Usage:
    uv run python scripts/last_run.py            # show last run
    uv run python scripts/last_run.py --verify   # show + run verify_run.py

Tip: pipe into verify_run.py manually:
    uv run python scripts/verify_run.py $(uv run python scripts/last_run.py --path)
"""

import json
import subprocess
import sys
from pathlib import Path


def find_latest(root: Path) -> Path | None:
    summaries = sorted(root.glob("**/summary.json"), key=lambda p: p.stat().st_mtime)
    return summaries[-1] if summaries else None


def main() -> None:
    verify   = "--verify" in sys.argv
    path_only = "--path" in sys.argv

    root = Path("outputs")
    if not root.exists():
        print("No outputs/ directory found.", file=sys.stderr)
        sys.exit(1)

    latest = find_latest(root)
    if latest is None:
        print("No summary.json found under outputs/.", file=sys.stderr)
        sys.exit(1)

    if path_only:
        print(latest.parent)
        return

    try:
        data = json.loads(latest.read_text())
    except Exception as exc:
        print(f"Could not read {latest}: {exc}", file=sys.stderr)
        sys.exit(2)

    # Pretty-print key fields
    print(f"run_id       : {data.get('run_id', '?')}")
    print(f"script       : {data.get('script', '?')}")
    print(f"status       : {data.get('status', '?')}")
    print(f"started_at   : {data.get('started_at', '?')}")
    print(f"runtime      : {data.get('runtime_seconds', '?')}s")
    print(f"steps        : {data.get('steps_completed', '?')} / {data.get('steps_requested', '?')}")
    initial = data.get("initial_loss")
    final   = data.get("final_loss")
    if initial is not None and final is not None:
        print(f"loss         : {initial:.6g} → {final:.6g}")
    vram = data.get("vram_peak_mb")
    if vram:
        total = data.get("preflight", {}).get("gpu_total_vram_mb")
        print(f"VRAM peak    : {vram} MB" + (f" / {total} MB" if total else ""))
    anomalies = data.get("anomalies", [])
    if anomalies:
        print(f"anomalies    : {len(anomalies)}")
    print(f"summary.json : {latest}")

    if verify:
        print()
        result = subprocess.run(
            [sys.executable, "scripts/verify_run.py", str(latest.parent)],
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
