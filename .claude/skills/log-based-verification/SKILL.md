---
name: log-based-verification
description: >
  Verification and logging protocol for all ML training, evaluation, and
  experiment work in this repository. Invoke when: (1) a training, eval, or
  experiment run has just finished or crashed; (2) the user asks to RUN or
  KICK OFF any training or experiment command; (3) you are about to write a
  new Python script or add a function/class to the codebase; (4) the user
  asks "did that work", "did the run succeed", "check the results", "verify
  the training", "what happened", or any variant. Counter-acts Claude Code's
  tendency to paraphrase terminal output — the structured summary.json
  artifact is the only valid source of truth. Trigger any time new code is
  written, any time a training command is suggested or run, or any time the
  user wants to know if something worked, even if they don't use the word
  "verify".
---

# log-based-verification

## The one rule

**Never declare a run succeeded by reading terminal output.** Always read the
structured artifact and run the verification script. The terminal is for
human progress monitoring only — it is not a source of truth.

```
# The only valid verification sequence:
uv run python scripts/verify_run.py <output_dir>
```

If `verify_run.py` exits 0, the run passed. If it exits 1 or 2, it failed.
There is no other definition of "the run worked."

---

## When the user asks you to run a training command

Always use the wrapper instead of calling `lerobot-train` directly. The
wrapper is a thin pass-through that synthesizes `summary.json` so the run
is verifiable — without it there is no artifact to check.

```bash
# Always use this form:
uv run python scripts/lerobot_train_wrapped.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/smoke/pusht \
    --batch_size=64 \
    --steps=500 \
    --wandb.enable=false \
    --save_checkpoint=false

# Not this:
uv run lerobot-train ...
```

All arguments pass through unchanged. After the wrapper finishes, run:
```
uv run python scripts/verify_run.py <output_dir>
```

Or use `scripts/run.sh` which chains both steps automatically:
```bash
./scripts/run.sh python scripts/lerobot_train_wrapped.py [args...]
```

---

## After any training or eval run (including crashes)

RunContext writes `summary.json` with `status="running"` the moment a run
starts, then overwrites it on exit — whether clean, exception, or OOM crash.
A run that died before finishing still has a `summary.json`; read it.

1. **Locate the summary.json.** It lives in the `--output_dir` passed to the
   command. Default paths:
   - `outputs/smoke/pusht/summary.json`
   - `outputs/smoke/libero_spatial/summary.json`
   - `outputs/train/<job_name>/summary.json`

2. **Read it.** Use the Read tool on `<output_dir>/summary.json`. Do not
   summarize from memory — read the file, even if you saw the terminal output.

3. **Run verify_run.py:**
   ```
   uv run python scripts/verify_run.py <output_dir>
   ```

4. **Report the result** with specifics from the artifact:
   - `run_id`, `status`, `runtime_seconds`
   - `initial_loss` → `final_loss` (if training)
   - `steps_completed` / `steps_requested`
   - `error.type` and `error.message` if the run failed
   - Any failures or warnings printed by verify_run.py

5. **If verify_run.py exits 1**, surface the exact failure lines verbatim.
   Never soften "loss did not decrease" into "the loss may need more steps."

6. **If the run crashed mid-flight** (OOM, segfault, KeyboardInterrupt), the
   summary.json will have `status="failed"` (or `"running"` if the crash
   happened before `__exit__` ran). Either way, verify_run.py will exit 1 and
   tell you exactly which checks failed — read its output.

### The anti-pattern — never do this

> "I can see from the output that loss went from 0.34 to 0.08 over 500 steps,
> so the training run succeeded."

This is wrong even if the numbers look right. Terminal output is ephemeral.
Read the file.

---

## When writing any new Python script or adding code

Every Python entry point in this repo must wrap its main body in `RunContext`.
A script without `RunContext` produces no verifiable artifact and violates the
logging protocol. This applies to all new scripts regardless of size or scope.

```python
from pathlib import Path
from src.observability import RunContext

def main():
    output_dir = Path("outputs/my_experiment")

    with RunContext(
        run_id="my_experiment_001",
        script=__file__,
        config={"policy": "act", "dataset": "lerobot/libero_goal_image"},
        output_dir=output_dir,
        steps_requested=1000,
        device="cuda",
    ) as run:
        for step in range(1000):
            loss = train_one_step(step)
            run.log_step(step, loss=loss)

            if loss != loss:  # NaN check
                run.log_anomaly("NaN loss detected", step=step)
                raise RuntimeError(f"NaN loss at step {step}")

if __name__ == "__main__":
    main()
```

After writing any script, remind the user to run it via:
```
uv run python scripts/my_experiment.py
uv run python scripts/verify_run.py outputs/my_experiment
```

### What RunContext does automatically
- Enables `faulthandler` (segfault tracebacks) and `logging.captureWarnings`
- Writes `summary.json` with `status="running"` immediately on entry
- Records preflight: torch version, CUDA availability, GPU name and VRAM,
  hostname, and key package versions
- Overwrites `summary.json` with final status, metrics, and timing on exit
- Handles exceptions: sets `status="failed"`, records `error.type` / `error.message`
- Writes an emergency summary if the script crashes before `__enter__`

### Logging metrics and anomalies

```python
# Per-step metrics
run.log_step(step, loss=loss, grad_norm=grad_norm, lr=current_lr)

# Anomalies — cause verify_run.py to exit 1
if grad_norm > 1000:
    run.log_anomaly("gradient explosion", step=step, grad_norm=float(grad_norm))

# Warnings — exit 0 but printed to stderr
if vram_used / vram_total > 0.98:
    run.log_warning("VRAM near capacity")
```

---

## summary.json schema

See `references/healthy_summary.json` and `references/failed_summary.json`
for complete examples. Key fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | str | Always `"1.0"` |
| `run_id` | str | Unique identifier for this run |
| `status` | str | `"running"` / `"success"` / `"failed"` |
| `steps_requested` | int\|null | Target number of steps |
| `steps_completed` | int | Steps actually finished |
| `initial_loss` | float\|null | Loss at first logged step |
| `final_loss` | float\|null | Loss at last logged step |
| `loss_history` | list | `[{"step": N, "loss": F}, ...]` |
| `vram_peak_mb` | int\|null | Peak GPU memory during run |
| `preflight` | dict | torch version, CUDA, GPU, packages, hostname |
| `anomalies` | list | Events that invalidate the run |
| `warnings` | list | Soft flags (VRAM, slow steps, version drift) |
| `error` | dict\|null | `{type, message}` if run raised an exception |

---

## verify_run.py — hard failures (exit 1)

| Check | Condition |
|---|---|
| Missing/unparseable artifact | `summary.json` not found or invalid JSON → exit 2 |
| Bad status | `status != "success"` |
| Loss invalid | `final_loss` is NaN, Infinity, or missing (when loss was recorded) |
| No improvement | `final_loss >= initial_loss` |
| Anomaly recorded | Any entry in `anomalies` |
| Steps incomplete | `steps_completed < steps_requested` |
| Suspiciously short | `runtime_seconds < 1.0` |
| Device mismatch | `device_requested=cuda` but `device_actual=cpu` |

## verify_run.py — warnings (exit 0, printed to stderr)

| Check | Condition |
|---|---|
| VRAM near capacity | `vram_peak_mb / gpu_total_vram_mb >= 0.95` |
| Propagated run warnings | Any entry in `summary.json["warnings"]` |

---

## Answering "did that run work?" / "what happened?"

When the user asks any variant of "did it work", "what happened", "did the
run succeed", "check the results", or "was that correct":

1. Do **not** answer from memory or from terminal output you saw earlier.
2. Run `uv run python scripts/last_run.py --verify` if you don't know the
   output dir, or `uv run python scripts/verify_run.py <output_dir>` if you do.
3. Read the summary.json to extract key numbers.
4. Report: `run_id`, `status`, loss trajectory, steps completed, any failures
   or warnings from verify_run.py. If the run crashed, report `error.type`
   and `error.message` from the artifact, not from what you remember seeing.

---

## Quick reference

```bash
# Suggest and run a training job, then auto-verify
./scripts/run.sh python scripts/lerobot_train_wrapped.py \
    --dataset.repo_id=lerobot/pusht --policy.type=diffusion \
    --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/smoke/pusht --batch_size=64 --steps=500 \
    --wandb.enable=false --save_checkpoint=false

# Verify a specific run
uv run python scripts/verify_run.py outputs/smoke/pusht

# Show and verify the most recent run
uv run python scripts/last_run.py --verify

# Show the most recent run summary without verifying
uv run python scripts/last_run.py
```
