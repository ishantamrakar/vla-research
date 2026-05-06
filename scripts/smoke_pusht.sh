#!/usr/bin/env bash
# Phase 1 smoke test: PushT Diffusion Policy, 500 steps.
# Pass: loss drops clearly (expected ~0.34 → ~0.08 by step 400).
set -euo pipefail

uv run lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/smoke/pusht \
  --job_name=smoke_pusht \
  --batch_size=64 \
  --steps=500 \
  --wandb.enable=false \
  --save_checkpoint=false
