# vla-research

A personal research template for vision-language-action (VLA) and imitation
learning experiments on single-arm manipulation in simulation. The default
workflow is **fine-tuning from a public checkpoint**, not training from scratch.

Designed to make new ideas — alternative action heads, planners, memory
modules, test-time compute — trivial to drop into a working pipeline and
compare against a known baseline. If you are starting a VLA project and want
a clean, opinionated starting point, clone this.

---

## Features

- **Multi-policy support** — ACT, Diffusion Policy, SmolVLA out of the box;
  adding a new policy is one config file
- **Multi-env support** — LIBERO (primary), ManiSkill 3, PushT; adding a new
  env is one config file
- **Hydra config** — every experiment is fully reproducible from its CLI
  override string
- **GPU utilization sweep** — `sweep.py` probes VRAM at increasing batch sizes
  and caches the result; `train.py` reads it automatically
- **bfloat16 AMP** — mixed-precision training with no overflow risk
- **Checkpointing + resume** — numbered checkpoints + `checkpoint_latest.pt`;
  interrupted runs resume from where they stopped
- **W&B integration** — loss, eval reward, success rate, and VRAM logged
- **Eval harness** — vectorized rollout across all LIBERO tasks via LeRobot's
  `eval_policy_all`; optional video recording
- **Observability** — every run writes `summary.json` (status, loss history,
  VRAM peak, errors); `verify_run.py` checks it

---

## Hardware target

Developed and tested on an RTX 3080 (10 GB VRAM). Long training runs use
UMD's Zaratan HPC cluster (A100s) via the Apptainer container in
`apptainer/`.

---

## Prerequisites

### System

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- CUDA 12.6 driver (cu126 wheels; driver ≥ 525 required)
- (HPC only) Apptainer ≥ 1.1

### Sibling repositories

Clone these **next to** this repo (same parent directory):

```bash
# LeRobot — installed editable so you can read and patch their code
git clone https://github.com/huggingface/lerobot.git ../lerobot

# LIBERO — sim environment
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
```

---

## Installation

```bash
git clone <this-repo> vla-research
cd vla-research
uv sync
```

`uv sync` reads `pyproject.toml`, pulls the cu126 PyTorch wheels, and installs
LeRobot editable from `../lerobot`. Everything lands in `.venv/`.

**One-time LIBERO setup** — create `~/.libero/config.yaml` to suppress the
interactive prompt on first import:

```bash
mkdir -p ~/.libero
python - <<'EOF'
from pathlib import Path
import shutil, libero
src = Path(libero.__file__).parent / "default_config.yaml"
dst = Path.home() / ".libero/config.yaml"
if not dst.exists():
    shutil.copy(src, dst)
    print("LIBERO config written to", dst)
else:
    print("Already exists:", dst)
EOF
```

---

## Quickstart

Verify the stack with a 500-step smoke test (no W&B, no eval):

```bash
# PushT + Diffusion Policy
uv run python scripts/finetune.py \
  policy=diffusion env=pusht \
  train.steps=500 wandb.enable=false experiment_name=smoke_pusht

# LIBERO Spatial + ACT
uv run python scripts/finetune.py \
  policy=act env=libero_spatial \
  train.steps=500 wandb.enable=false experiment_name=smoke_libero
```

Both should complete without error and show steadily falling loss.

---

## Training

### Step 1 — find the optimal batch size (once per policy × dataset)

```bash
uv run python scripts/sweep.py policy=act env=libero_spatial
```

Result is cached in `outputs/sweep_cache.json`. Skip this step if the cache
already has an entry for your combination.

### Step 2 — run training

```bash
uv run python scripts/train.py \
  policy=act \
  env=libero_spatial \
  train.steps=50000 \
  train.checkpoint_interval=5000 \
  eval.eval_freq=25000 \
  eval.n_episodes=5 \
  experiment_name=act_libero_spatial_50k \
  wandb.enable=true
```

Training resumes automatically if
`outputs/train/<experiment_name>/checkpoints/checkpoint_latest.pt` exists —
interrupted runs restart with the same command.

### Key config knobs

| Override | Default | Notes |
|---|---|---|
| `train.steps` | 500 | Total gradient steps |
| `train.checkpoint_interval` | 5000 | Steps between numbered checkpoints |
| `train.render_eval_videos` | false | Set true to record MP4s during eval |
| `eval.eval_freq` | 500 | Steps between eval calls; 0 = disabled |
| `eval.n_episodes` | 5 | Episodes per task during eval |
| `optimizer.lr` | 1e-4 | Not scaled with batch size (AdamW) |
| `wandb.enable` | true | Set false for smoke tests |

---

## Adding a new policy

1. Confirm the policy is supported by LeRobot
   (`lerobot.policies.make_policy_config`).
2. Create `configs/policy/<name>.yaml`:
   ```yaml
   type: "<lerobot_policy_type>"
   # any policy-specific kwargs
   ```
3. Pass `policy=<name>` on the CLI. Done.

The training loop is policy-agnostic — it calls `make_policy_config`,
`make_policy`, and `make_pre_post_processors` from LeRobot, which dispatch
on `type`.

---

## Adding a new environment

1. Confirm the env is supported by LeRobot (`lerobot.envs.EnvConfig`).
2. Create `configs/env/<name>.yaml` with `# @package _global_` at the top:
   ```yaml
   # @package _global_
   env:
     type: "<lerobot_env_type>"
     task: "<task_name>"
     # env-specific kwargs (e.g. camera_name_mapping for LIBERO)

   dataset:
     repo_id: "hf_org/dataset_name"
   ```
3. Pass `env=<name>` on the CLI. Done.

The `# @package _global_` directive is required — it lets a single file set
both `env.*` and `dataset.*` in the Hydra config tree.

---

## Running on HPC (UMIACS Nexus-CFAR)

See [`apptainer/`](apptainer/) for the container definition and
[`slurm/`](slurm/) for job scripts.

```bash
# 1. Build container on your local Linux machine (once, ~15 min)
apptainer build --fakeroot apptainer/vla.sif apptainer/vla.def

# 2. Transfer to Nexus
rsync -av --exclude='.venv' --exclude='outputs' --exclude='wandb' \
  . <user>@nexus.umiacs.umd.edu:~/vla-research/
scp apptainer/vla.sif <user>@nexus.umiacs.umd.edu:~/vla-research/apptainer/
# Also transfer the sibling lerobot clone if not already there:
rsync -av ~/Documents/lerobot/ <user>@nexus.umiacs.umd.edu:~/lerobot/

# 3. On the Nexus login node — add W&B key (once)
echo 'export WANDB_API_KEY=your_key_here' >> ~/.bashrc && source ~/.bashrc

# 4. Run the batch-size sweep (once per GPU type, ~1-2 hr)
sbatch slurm/sweep.slurm

# 5. Submit training
sbatch slurm/train.slurm

# Monitor
squeue --user $USER
```

The SLURM scripts default to L40S (48 GB) on the scavenger partition.
Outputs land on scratch (`/fs/nexus-scratch/<user>/vla-outputs/`) and are
rsynced back to the repo on job exit. Training resumes automatically from
`checkpoint_latest.pt` if the job is preempted — just resubmit.

---

## Project structure

```
vla-research/
├── configs/
│   ├── train.yaml          # top-level Hydra config
│   ├── env/                # one file per environment
│   │   ├── libero_spatial.yaml
│   │   └── pusht.yaml
│   └── policy/             # one file per policy
│       ├── act.yaml
│       ├── diffusion.yaml
│       └── smolvla.yaml
├── scripts/
│   ├── finetune.py         # lightweight training (smoke tests, quick runs)
│   ├── sweep.py            # GPU batch-size sweep → outputs/sweep_cache.json
│   └── train.py            # production training: AMP, checkpointing, resume
├── src/
│   └── observability/      # RunContext → summary.json
├── apptainer/              # container definition for HPC
├── slurm/                  # SLURM job scripts for Zaratan
└── outputs/
    ├── sweep_cache.json    # cached optimal batch sizes per policy×dataset
    └── train/
        └── <experiment>/
            ├── checkpoints/
            ├── summary.json
            └── eval/
```

---

## Verified smoke tests

| Policy | Env | Steps | Notes |
|---|---|---|---|
| Diffusion | PushT | 500 | loss 0.345 → 0.077, ~10 steps/s |
| ACT | LIBERO Spatial | 500 | loss 6.67 → 3.02 |
| SmolVLA | LIBERO Spatial | 500 | loss 2.13 → 0.93, peak VRAM 4.6 GB |
| ACT | LIBERO Spatial | 10 000 | loss 76 → 0.24, bs=64, bfloat16 AMP |

---

## License

MIT
