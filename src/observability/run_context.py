"""
RunContext — structured logging and summary.json protocol for all training,
evaluation, and experiment runs in vla-research.

Every script entry point should wrap its main body:

    from src.observability import RunContext

    with RunContext(
        run_id="smoke_pusht",
        script=__file__,
        config={"policy": "diffusion", "dataset": "lerobot/pusht"},
        output_dir=Path("outputs/smoke/pusht"),
        steps_requested=500,
        device="cuda",
    ) as run:
        for step in range(500):
            loss = train_step(batch)
            run.log_step(step, loss=loss)

After the block, verify with:
    uv run python scripts/verify_run.py outputs/smoke/pusht
"""

import faulthandler
import json
import logging
import platform
import socket
import sys
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

_KEY_PACKAGES = [
    "lerobot", "torch", "torchvision", "torchaudio",
    "diffusers", "gymnasium", "mani_skill", "accelerate",
]


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _collect_preflight(device: str) -> dict:
    """Snapshot environment at run start: versions, GPU info, hostname."""
    info: dict[str, Any] = {
        "device_requested": device,
        "device_actual": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_vram_mb": None,
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {pkg: _package_version(pkg) for pkg in _KEY_PACKAGES},
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_actual"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_vram_mb"] = props.total_memory // (1024 * 1024)
    except Exception as exc:
        log.warning("preflight: torch not importable — %s", exc)
    return info


def _atomic_write(path: Path, doc: dict) -> None:
    """Write JSON atomically via a .tmp rename so readers never see partial data."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Top-level import-failure guard
# ---------------------------------------------------------------------------
# If something goes wrong before RunContext.__enter__ is ever called (e.g. a
# missing dependency at module level in a script), this module-level hook
# writes an emergency summary so verify_run.py still finds an artifact.

_emergency_output_dir: Path | None = None


def _emergency_handler(exc_type, exc_val, exc_tb) -> None:
    if _emergency_output_dir is not None:
        try:
            path = _emergency_output_dir / "summary.json"
            _atomic_write(path, {
                "schema_version": SCHEMA_VERSION,
                "run_id": "emergency",
                "status": "failed",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": exc_type.__name__ if exc_type else "unknown",
                    "message": str(exc_val),
                },
            })
        except Exception:
            pass
    sys.__excepthook__(exc_type, exc_val, exc_tb)


sys.excepthook = _emergency_handler


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

class RunContext:
    """Context manager that bookends a training / eval run with structured logging.

    On ``__enter__``:
    - enables faulthandler (segfault tracebacks)
    - routes Python warnings into logging
    - writes summary.json with status='running' immediately
    - snapshots preflight info (torch version, CUDA, GPU, packages)

    On ``__exit__``:
    - overwrites summary.json with final status, metrics, and timing
    - never suppresses exceptions — the caller sees them normally

    The summary.json is always written, even if the run crashes.
    """

    def __init__(
        self,
        run_id: str,
        script: str | Path,
        config: dict[str, Any],
        output_dir: Path | str,
        *,
        steps_requested: int | None = None,
        device: str = "cuda",
        tags: list[str] | None = None,
    ) -> None:
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.script = str(script)
        self.config = config
        self.output_dir = Path(output_dir)
        self.steps_requested = steps_requested
        self.device = device
        self.tags = tags or []

        self._summary_path: Path = self.output_dir / "summary.json"
        self._started_at_mono: float = 0.0
        self._started_at_iso: str = ""

        # Metrics accumulated during the run
        self._loss_history: list[dict[str, Any]] = []
        self._anomalies: list[dict[str, Any]] = []
        self._warnings: list[str] = []
        self._steps_completed: int = 0
        self._initial_loss: float | None = None
        self._final_loss: float | None = None
        self._preflight: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunContext":
        global _emergency_output_dir
        _emergency_output_dir = self.output_dir

        # Ensure faulthandler writes a traceback on segfault / hard crash
        faulthandler.enable()

        # Route warnings (e.g. DeprecationWarning, UserWarning) into logging
        logging.captureWarnings(True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started_at_mono = time.monotonic()
        self._started_at_iso = datetime.now(timezone.utc).isoformat()

        self._preflight = _collect_preflight(self.device)

        # Write status=running immediately so a crash leaves a parseable artifact
        self._write_summary(status="running", finished_at=None, runtime_seconds=None)

        log.info(
            "RunContext started  run_id=%s  output_dir=%s",
            self.run_id, self.output_dir,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = round(time.monotonic() - self._started_at_mono, 3)
        finished_iso = datetime.now(timezone.utc).isoformat()

        # Capture peak VRAM at the last possible moment
        with suppress(Exception):
            import torch
            if torch.cuda.is_available():
                self._preflight["vram_peak_mb"] = (
                    torch.cuda.max_memory_allocated() // (1024 * 1024)
                )

        status = "success" if exc_type is None else "failed"
        error = (
            {"type": exc_type.__name__, "message": str(exc_val)}
            if exc_type is not None
            else None
        )

        if exc_type is not None:
            log.error(
                "RunContext exiting with error  run_id=%s  %s: %s",
                self.run_id, exc_type.__name__, exc_val,
            )

        self._write_summary(
            status=status,
            finished_at=finished_iso,
            runtime_seconds=elapsed,
            error=error,
        )

        log.info(
            "RunContext finished  run_id=%s  status=%s  runtime=%.1fs",
            self.run_id, status, elapsed,
        )
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Logging helpers called inside the with-block
    # ------------------------------------------------------------------

    def log_step(self, step: int, *, loss: float | None = None, **metrics: Any) -> None:
        """Record per-step metrics. Call once per training step."""
        self._steps_completed = step + 1
        entry: dict[str, Any] = {"step": step}
        if loss is not None:
            entry["loss"] = loss
            if self._initial_loss is None:
                self._initial_loss = loss
            self._final_loss = loss
        entry.update(metrics)
        self._loss_history.append(entry)

    def log_anomaly(self, message: str, **details: Any) -> None:
        """Record an anomaly event (gradient explosion, NaN loss, etc.).

        Any recorded anomaly causes verify_run.py to fail, so only call this
        for conditions that genuinely invalidate the run.
        """
        event: dict[str, Any] = {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event.update(details)
        self._anomalies.append(event)
        log.warning("ANOMALY  run_id=%s  %s  details=%s", self.run_id, message, details)

    def log_warning(self, message: str) -> None:
        """Record a soft warning (high VRAM, slow steps, version drift)."""
        self._warnings.append(message)
        log.warning("RUN-WARN  run_id=%s  %s", self.run_id, message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_summary(
        self,
        *,
        status: str,
        finished_at: str | None,
        runtime_seconds: float | None,
        error: dict | None = None,
    ) -> None:
        vram_peak = self._preflight.pop("vram_peak_mb", None)
        doc: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "script": self.script,
            "status": status,
            "started_at": self._started_at_iso,
            "finished_at": finished_at,
            "runtime_seconds": runtime_seconds,
            "steps_requested": self.steps_requested,
            "steps_completed": self._steps_completed,
            "initial_loss": self._initial_loss,
            "final_loss": self._final_loss,
            "loss_history": self._loss_history,
            "vram_peak_mb": vram_peak,
            "preflight": self._preflight,
            "config": self.config,
            "tags": self.tags,
            "anomalies": self._anomalies,
            "warnings": self._warnings,
            "output_dir": str(self.output_dir),
        }
        if error is not None:
            doc["error"] = error
        _atomic_write(self._summary_path, doc)
