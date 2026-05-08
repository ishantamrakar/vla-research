# VLA Research Template - Gemini CLI Guide

This file provides context and instructions for AI coding assistants (like Gemini CLI) working in this repository. It is a companion to `CLAUDE.md`.

## Core Project Overview
- **Goal**: Research vehicle for vision-language-action (VLA) and imitation learning experiments (single-arm manipulation).
- **Workflow**: Primarily fine-tuning from public checkpoints.
- **Stack**: Python 3.12, `uv` for env management, PyTorch (CUDA 12.6), LeRobot (editable install), LIBERO, ManiSkill 3, Hydra, W&B.
- **Constraints**: Main lab workstation has 10GB VRAM (RTX 3080). Keep batch sizes small (<= 16) for fine-tuning.

## CRITICAL: Log-Based Verification Protocol

This repository uses a strict "log-based verification" protocol. **You must never determine if a run succeeded by reading terminal output.** The terminal is ephemeral; the structured artifact is the source of truth.

1. **The Source of Truth**: Every valid run produces a `summary.json` artifact in its output directory.
2. **How to Verify**: Always use the verification scripts to determine success.
   - `uv run python scripts/verify_run.py <output_dir>`
   - `uv run python scripts/last_run.py --verify` (to verify the most recent run)
   - If the script exits `0`, it passed. If `1` or `2`, it failed.

### Running Training/Eval Commands
Do not run `lerobot-train` directly. Use the wrappers that generate the `summary.json` artifact:
```bash
# Good: Generates summary.json and allows verification
uv run python scripts/lerobot_train_wrapped.py [args...]

# Better: Runs the wrapper and automatically verifies the output
./scripts/run.sh python scripts/lerobot_train_wrapped.py [args...]
```

### Writing New Python Scripts
Every new Python entry point script **must** wrap its execution in `RunContext`. This ensures the script participates in the verifiable run protocol.

```python
from pathlib import Path
from src.observability import RunContext

def main():
    output_dir = Path("outputs/my_experiment")
    
    with RunContext(
        run_id="my_experiment_001",
        script=__file__,
        config={"key": "value"},
        output_dir=output_dir,
        steps_requested=1000,
        device="cuda",
    ) as run:
        for step in range(1000):
            # ... do work ...
            loss = 0.5 # example
            run.log_step(step, loss=loss)
            
            if loss != loss: # NaN check
                run.log_anomaly("NaN loss", step=step)
                raise RuntimeError("NaN loss")

if __name__ == "__main__":
    main()
```

## Agent Directives
- Read this file and `CLAUDE.md` to understand current phases and constraints.
- When asked "did it work?" or to run a job, always refer to the `summary.json` and use `verify_run.py`. Never hallucinate success based on terminal loss numbers.
- Do not commit data, datasets, or checkpoints to the repository.
- Ensure any modifications or additions align with the structured logging and verifiable run requirements.