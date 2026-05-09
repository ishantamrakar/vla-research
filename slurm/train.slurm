#!/bin/bash
# =============================================================================
# VLA training job — UMIACS Nexus-CFAR (SLURM + Apptainer)
#
# Default: L40S on scavenger (48GB VRAM, preemptable but checkpointing handles it).
# Alternative: rtxa5000 with --qos=medium (24GB, 12hr time limit, non-preemptable).
#
# Submit:  sbatch slurm/train.slurm
# Monitor: squeue --user itamraka
#          tail -f /fs/nexus-scratch/itamraka/vla-outputs/train/$EXPERIMENT/train.log
# Resume:  just resubmit — train.py picks up from checkpoint_latest.pt automatically
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --job-name=vla-train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/%j-train.out
#SBATCH --error=slurm/logs/%j-train.err
#
# --- Alternative: rtxa5000 (non-preemptable, 12hr limit) --------------------
# #SBATCH --partition=default          # verify with `sinfo`
# #SBATCH --qos=medium
# #SBATCH --gres=gpu:rtxa5000:1
# #SBATCH --time=12:00:00

# ---- experiment config ------------------------------------------------------
EXPERIMENT="act_libero_spatial_50k"
POLICY="act"
ENV="libero_spatial"
STEPS=50000
CHECKPOINT_INTERVAL=5000
EVAL_FREQ=25000
N_EPISODES=5
RENDER_VIDEOS="false"
WANDB_ENABLE="true"

# ---- paths ------------------------------------------------------------------
SCRATCH=/fs/nexus-scratch/itamraka
REPO=$HOME/vla-research
SIF=$REPO/apptainer/vla.sif
OUTPUTS=$SCRATCH/vla-outputs

mkdir -p $OUTPUTS/train slurm/logs

# ---- W&B auth --------------------------------------------------------------
# ~/.netrc is automatically bind-mounted by Apptainer (home dir bind).
# No extra config needed if you have wandb credentials in ~/.netrc.
# Alternatively set WANDB_API_KEY in ~/.bashrc.

# ---- sync outputs back to repo on exit -------------------------------------
cleanup() {
    echo "Syncing outputs from scratch to repo..."
    rsync -a $OUTPUTS/train/$EXPERIMENT/ $REPO/outputs/train/$EXPERIMENT/ 2>/dev/null || true
}
trap cleanup EXIT

# ---- run --------------------------------------------------------------------
apptainer exec \
    --nv \
    --bind $REPO:/workspace \
    --bind $HOME/lerobot:/lerobot \
    --bind $SCRATCH:/scratch \
    --env HF_HOME=$SCRATCH/hf_cache \
    --env HF_DATASETS_CACHE=$SCRATCH/hf_cache \
    --env TMPDIR=$SCRATCH/tmp \
    --env UV_CACHE_DIR=$SCRATCH/uv_cache \
    --env MUJOCO_GL=osmesa \
    $SIF bash -c "
        set -e
        cd /workspace
        uv sync --quiet
        uv pip install --editable /lerobot --quiet

        uv run python scripts/train.py \
            policy=$POLICY \
            env=$ENV \
            train.steps=$STEPS \
            train.checkpoint_interval=$CHECKPOINT_INTERVAL \
            train.render_eval_videos=$RENDER_VIDEOS \
            eval.eval_freq=$EVAL_FREQ \
            eval.n_episodes=$N_EPISODES \
            experiment_name=$EXPERIMENT \
            output_dir=$OUTPUTS/train/$EXPERIMENT \
            wandb.enable=$WANDB_ENABLE
    "
