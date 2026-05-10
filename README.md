# vla-research

A minimal research scaffold for vision-language-action (VLA) and imitation learning experiments on single-arm manipulation in simulation.

**The baseline is one command.** Everything else in this repo exists to make it easy to swap in a new idea — a different action head, a memory module, a planner — and compare it against that baseline.

---

## How it works

[LeRobot](https://github.com/huggingface/lerobot) provides the policies, datasets, training loop, and eval harness. This repo provides:

- **Configs** — one YAML per environment, one per policy; mix and match
- **HPC setup** — Apptainer container + SLURM scripts for GPU clusters
- **Custom training hook** — `scripts/finetune.py` for when you need to inject research code into the training loop
- **`src/`** — where new modules live (planners, memory, novel action heads)

---

## Setup

**Prerequisites**

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- CUDA 12.6+ driver

**Clone the sibling repos** (must be next to this one):

```bash
git clone https://github.com/huggingface/lerobot.git ../lerobot
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
```

**Install:**

```bash
git clone https://github.com/ishan-tamrakar/vla-research.git
cd vla-research
uv sync
```

Everything lands in `.venv/`. LeRobot is installed editable from `../lerobot`.

---

## Baseline training

ACT on LIBERO Spatial — the default baseline:

```bash
uv run lerobot-train \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --policy.type=act \
  --env.type=libero \
  --env.task=libero_spatial \
  --output_dir=outputs/train/act_libero_spatial \
  --job_name=act_libero_spatial \
  --wandb.enable=true \
  --policy.push_to_hub=false
```

This uses ACT's built-in training preset: `lr=1e-5`, `batch_size=8`, `steps=100k`. No config files needed — `lerobot-train` handles everything.

---

## Running on a GPU cluster (SLURM + Apptainer)

**One-time setup:**

```bash
# 1. Build the container on a Linux machine with root (≈15 min)
apptainer build --fakeroot apptainer/vla.sif apptainer/vla.def

# 2. Copy cluster.env.example and fill in your paths
cp slurm/cluster.env.example slurm/cluster.env
# edit SCRATCH, REPO, LEROBOT

# 3. Transfer everything to the cluster
rsync -av --exclude='.venv' --exclude='outputs' --exclude='wandb' \
  . <user>@nexus.umiacs.umd.edu:$REPO/
scp apptainer/vla.sif <user>@nexus.umiacs.umd.edu:$REPO/apptainer/
rsync -av ~/Documents/lerobot/ <user>@nexus.umiacs.umd.edu:$LEROBOT/
```

**Submit a training job:**

```bash
sbatch slurm/train.slurm
squeue --user $USER
```

Edit the variables at the top of `slurm/train.slurm` to change steps, eval frequency, or experiment name. Outputs land in `$SCRATCH/vla-outputs/`.

**Resume a preempted job** — uncomment the `--resume` block at the bottom of `train.slurm` and resubmit.

---

## Smoke tests

Verify the stack locally (no W&B, no eval, 500 steps):

```bash
# PushT + Diffusion Policy
uv run lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --steps=500 \
  --wandb.enable=false \
  --save_checkpoint=false

# LIBERO Spatial + ACT
uv run lerobot-train \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=500 \
  --wandb.enable=false \
  --save_checkpoint=false
```

Both should complete without error and show steadily falling loss.

---

## Adding a new environment or policy

**New policy** — create `configs/policy/<name>.yaml`:
```yaml
type: "<lerobot_policy_type>"
```
Pass `--policy.type=<name>` on the CLI.

**New environment** — create `configs/env/<name>.yaml`:
```yaml
# env-specific overrides as needed
```
Pass `--env.type=<name> --env.task=<task>` on the CLI.

If the policy or env isn't in LeRobot yet, implement it in `src/` and register it with the LeRobot plugin system.

---

## Custom training loop

When you need to inject research code (a new loss term, a custom action head, mid-loop logging), use `scripts/finetune.py` instead of `lerobot-train`:

```bash
uv run python scripts/finetune.py \
  policy=act \
  env=libero_spatial \
  train.steps=50000 \
  experiment_name=my_experiment \
  wandb.enable=true
```

This wraps LeRobot's policy/dataset/eval stack with a Hydra config and `RunContext` (writes `outputs/.../summary.json` after every run). Extend the training loop in `finetune.py` to add your custom logic.

---

## Project structure

```
vla-research/
├── configs/
│   ├── train.yaml              # base Hydra config for finetune.py
│   ├── env/                    # one file per environment
│   │   ├── libero_spatial.yaml
│   │   └── pusht.yaml
│   └── policy/                 # one file per policy
│       ├── act.yaml
│       ├── diffusion.yaml
│       └── smolvla.yaml
├── scripts/
│   ├── finetune.py             # custom training loop (Hydra + RunContext)
│   ├── train.py                # longer runs with AMP + checkpointing
│   ├── sweep.py                # throughput-optimal batch size finder
│   └── verify_run.py           # check summary.json after a run
├── src/
│   └── observability/          # RunContext → summary.json
├── apptainer/
│   └── vla.def                 # container definition for HPC
└── slurm/
    ├── train.slurm             # SLURM job script
    └── cluster.env.example     # cluster path template
```

---

## Verified baselines

| Policy | Environment | Steps | Result |
|---|---|---|---|
| Diffusion Policy | PushT | 500 | loss 0.345 → 0.077 |
| ACT | LIBERO Spatial | 500 | loss 6.67 → 3.02 |
| SmolVLA | LIBERO Spatial | 500 | loss 2.13 → 0.93, peak VRAM 4.6 GB |

Hardware: RTX 3080 (10 GB). Cluster runs on L40S via SLURM + Apptainer.

---

## License

MIT
