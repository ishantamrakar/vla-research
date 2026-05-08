#!/usr/bin/env python
"""
lerobot_train_wrapped.py — Phase 2 bridge: wraps lerobot-train and synthesizes summary.json.

This script is a temporary compatibility shim. LeRobot does not use RunContext
internally, so we parse its log output after the fact and produce a conformant
summary.json. All new scripts from Phase 3.2 onward must use RunContext directly
inside Python entry points instead.

Usage:
    uv run python scripts/lerobot_train_wrapped.py \\
        --dataset.repo_id=lerobot/pusht \\
        --policy.type=diffusion \\
        --policy.device=cuda \\
        --policy.push_to_hub=false \\
        --output_dir=outputs/smoke/pusht \\
        --batch_size=64 \\
        --steps=500 \\
        --wandb.enable=false \\
        --save_checkpoint=false

Then verify:
    uv run python scripts/verify_run.py outputs/smoke/pusht
"""

import json
import math
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
SCHEMA_VERSION = "1.0"

# LeRobot step log format:
#   INFO ... step:200 smpl:13K ep:103 epch:0.50 loss:0.345 grdn:5.503 ...
_STEP_RE = re.compile(r"step:(\d+)\b.*?\bloss:([\d.eE+\-]+)")

_KEY_PACKAGES = [
    "lerobot", "torch", "torchvision", "diffusers", "gymnasium",
]


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _preflight(device: str) -> dict:
    info = {
        "device_requested": device,
        "device_actual": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_vram_mb": None,
        "hostname": socket.gethostname(),
        "packages": {p: _pkg_version(p) for p in _KEY_PACKAGES},
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_actual"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_vram_mb"] = (
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
    except Exception:
        pass
    return info


def _vram_peak_mb() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except Exception:
        pass
    return None


def _atomic_write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str))
    tmp.replace(path)


def _parse_cli(args: list[str]) -> tuple[dict, Path, int | None, str]:
    """Extract config fields, output_dir, steps, and device from CLI args."""
    raw = " ".join(args)

    def grab(pattern: str) -> str | None:
        m = re.search(pattern, raw)
        return m.group(1) if m else None

    output_dir = Path(grab(r"--output_dir[= ](\S+)") or f"outputs/train/run_{uuid.uuid4().hex[:8]}")
    steps_str  = grab(r"--steps[= ](\d+)")
    steps      = int(steps_str) if steps_str else None
    device     = grab(r"--policy\.device[= ](\S+)") or "cuda"

    config = {}
    for flag, key in [
        (r"--policy\.type[= ](\S+)",       "policy"),
        (r"--dataset\.repo_id[= ](\S+)",   "dataset"),
        (r"--batch_size[= ](\d+)",          "batch_size"),
        (r"--policy\.device[= ](\S+)",      "device"),
    ]:
        val = grab(flag)
        if val is not None:
            config[key] = val

    return config, output_dir, steps, device


def main() -> None:
    train_args = sys.argv[1:]
    if not train_args:
        print(
            "Usage: lerobot_train_wrapped.py [lerobot-train args...]\n"
            "Example: uv run python scripts/lerobot_train_wrapped.py "
            "--dataset.repo_id=lerobot/pusht --policy.type=diffusion ...",
            file=sys.stderr,
        )
        sys.exit(1)

    config, output_dir, steps_requested, device = _parse_cli(train_args)
    summary_path = output_dir / "summary.json"

    run_id = (
        f"lerobot_{output_dir.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    pf = _preflight(device)
    started_iso  = datetime.now(timezone.utc).isoformat()
    started_mono = time.monotonic()

    # Write status=running immediately — if we crash before the process finishes,
    # the artifact is still there and verify_run.py will report status=running (fails).
    _atomic_write(summary_path, {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "script": "lerobot-train (wrapped)",
        "status": "running",
        "started_at": started_iso,
        "finished_at": None,
        "runtime_seconds": None,
        "steps_requested": steps_requested,
        "steps_completed": 0,
        "initial_loss": None,
        "final_loss": None,
        "loss_history": [],
        "vram_peak_mb": None,
        "preflight": pf,
        "config": config,
        "tags": [],
        "anomalies": [],
        "warnings": [],
        "output_dir": str(output_dir),
    })

    # ── Run lerobot-train, streaming output and collecting loss log ─────────
    cmd = ["lerobot-train"] + train_args
    print(f"+ {shlex.join(cmd)}", flush=True)

    loss_history:  list[dict] = []
    initial_loss:  float | None = None
    final_loss:    float | None = None
    steps_done:    int = 0
    exit_code:     int = 1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = _STEP_RE.search(line)
            if m:
                step      = int(m.group(1))
                loss_val  = float(m.group(2))
                if math.isnan(loss_val) or math.isinf(loss_val):
                    continue
                if initial_loss is None:
                    initial_loss = loss_val
                final_loss = loss_val
                steps_done = step
                loss_history.append({"step": step, "loss": loss_val})
        proc.wait()
        exit_code = proc.returncode
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(f"ERROR launching lerobot-train: {exc}", file=sys.stderr)
        exit_code = 1

    # If the process exited cleanly, steps_done == steps_requested
    if exit_code == 0 and steps_requested is not None:
        steps_done = steps_requested

    runtime       = round(time.monotonic() - started_mono, 3)
    finished_iso  = datetime.now(timezone.utc).isoformat()
    status        = "success" if exit_code == 0 else "failed"

    _atomic_write(summary_path, {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "script": "lerobot-train (wrapped)",
        "status": status,
        "started_at": started_iso,
        "finished_at": finished_iso,
        "runtime_seconds": runtime,
        "steps_requested": steps_requested,
        "steps_completed": steps_done,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "vram_peak_mb": _vram_peak_mb(),
        "preflight": pf,
        "config": config,
        "tags": [],
        "anomalies": [],
        "warnings": [],
        "output_dir": str(output_dir),
    })

    print(f"\nsummary.json written → {summary_path}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
