#!/usr/bin/env bash
# Phase 2 smoke test: LIBERO-Spatial ACT, 500 steps.
# Pass: loss drops clearly (expected ~6.7 → ~3.0 by step 400; still in LR warmup).
set -euo pipefail

export MUJOCO_GL=egl  # required for headless rendering (HPC or no display)

uv run lerobot-train \
  --dataset.repo_id=lerobot/libero_spatial_image \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/smoke/libero_spatial \
  --job_name=smoke_libero_spatial \
  --batch_size=8 \
  --steps=500 \
  --wandb.enable=false \
  --save_checkpoint=false
