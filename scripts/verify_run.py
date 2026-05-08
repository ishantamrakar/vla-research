#!/usr/bin/env python
"""
verify_run.py — read a run's summary.json and enforce the pass/fail protocol.

Usage:
    uv run python scripts/verify_run.py <output_dir>
    uv run python scripts/verify_run.py <output_dir>/summary.json

Exit codes:
    0  — verified (warnings printed to stderr if any)
    1  — one or more hard failures
    2  — summary.json missing or unparseable
"""

import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_summary(target: Path) -> dict:
    path = target / "summary.json" if target.is_dir() else target
    if not path.exists():
        print(f"FAIL  summary.json not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"FAIL  summary.json unparseable: {exc}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _bad_float(v) -> bool:
    if v is None:
        return True
    try:
        f = float(v)
        return math.isnan(f) or math.isinf(f)
    except (TypeError, ValueError):
        return True


def verify(s: dict) -> tuple[list[str], list[str]]:
    """Return (failures, warnings). failures → exit 1; warnings → exit 0 + stderr."""
    failures: list[str] = []
    warnings: list[str] = []

    # ── Hard failures ──────────────────────────────────────────────────────

    status = s.get("status")
    if status != "success":
        failures.append(f"status='{status}' (expected 'success')")

    initial = s.get("initial_loss")
    final   = s.get("final_loss")

    # Only apply loss checks when the run recorded loss values
    if initial is not None or final is not None:
        if _bad_float(final):
            failures.append(f"final_loss invalid: {final!r}")
        elif _bad_float(initial):
            failures.append(f"initial_loss invalid: {initial!r}")
        elif float(final) >= float(initial):
            failures.append(
                f"loss did not decrease: initial={initial:.6g}  final={final:.6g}"
            )

    for anomaly in s.get("anomalies", []):
        msg = anomaly.get("message", str(anomaly))
        failures.append(f"anomaly recorded: {msg}")

    steps_req  = s.get("steps_requested")
    steps_done = s.get("steps_completed", 0)
    if steps_req is not None and steps_done < steps_req:
        failures.append(
            f"steps incomplete: completed={steps_done}  requested={steps_req}"
        )

    runtime = s.get("runtime_seconds")
    if runtime is not None and float(runtime) < 1.0:
        failures.append(f"runtime suspiciously short: {runtime:.3f}s")

    pf = s.get("preflight", {})
    dev_req = pf.get("device_requested", "")
    dev_act = pf.get("device_actual", "")
    if "cuda" in str(dev_req) and dev_act != "cuda":
        failures.append(
            f"device mismatch: requested='{dev_req}'  actual='{dev_act}'"
        )

    # ── Warnings (exit 0, but flag to stderr) ──────────────────────────────

    vram_peak  = s.get("vram_peak_mb")
    gpu_total  = pf.get("gpu_total_vram_mb")
    if vram_peak and gpu_total:
        ratio = float(vram_peak) / float(gpu_total)
        if ratio >= 0.95:
            warnings.append(
                f"VRAM near capacity: {vram_peak} MB / {gpu_total} MB ({ratio:.0%})"
            )

    # Propagate any warnings written by the run itself
    for w in s.get("warnings", []):
        warnings.append(w)

    return failures, warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: verify_run.py <output_dir_or_summary_json>", file=sys.stderr)
        sys.exit(1)

    summary = load_summary(Path(sys.argv[1]))
    run_id  = summary.get("run_id", "unknown")

    failures, warnings = verify(summary)

    for w in warnings:
        print(f"WARN  {w}", file=sys.stderr)

    if failures:
        print(f"\nVERIFICATION FAILED  run_id={run_id}", file=sys.stderr)
        for f in failures:
            print(f"  ✗  {f}", file=sys.stderr)
        sys.exit(1)

    print(f"VERIFICATION PASSED  run_id={run_id}")
    if warnings:
        print(f"  {len(warnings)} warning(s) — see stderr")


if __name__ == "__main__":
    main()
