#!/bin/bash
# =============================================================================
# VLA training job
# Reads cluster config from slurm/cluster.env (copy from cluster.env.example).
#
# Submit:  sbatch slurm/train.slurm
# Monitor: squeue --user $USER
# Resume:  just resubmit — train.py picks up from checkpoint_latest.pt
# =============================================================================

# ---- load cluster config ----------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$SCRIPT_DIR/cluster.env" ]; then
    echo "ERROR: slurm/cluster.env not found."
    echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your values."
    exit 1
fi
source "$SCRIPT_DIR/cluster.env"

# ---- SLURM directives (use cluster.env values) ------------------------------
#SBATCH --job-name=vla-train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/tmp/vla-train-%j.out
#SBATCH --error=/tmp/vla-train-%j.err
# Note: SLURM directives above are defaults. The actual values come from
# cluster.env via the srun call inside, but #SBATCH lines are parsed before
# the script runs. After setting up cluster.env, regenerate directives with:
#   bash slurm/gen_headers.sh  (or just edit the #SBATCH lines above manually)

# ---- experiment config — edit before submitting ----------------------------
EXPERIMENT="act_libero_spatial_50k"
POLICY="act"
ENV="libero_spatial"
STEPS=50000
CHECKPOINT_INTERVAL=5000
EVAL_FREQ=25000
N_EPISODES=5
RENDER_VIDEOS="false"
WANDB_ENABLE="true"

# ---- derived paths ----------------------------------------------------------
SIF=$REPO/apptainer/vla.sif
OUTPUTS=$SCRATCH/vla-outputs
LOGDIR=$REPO/slurm/logs

mkdir -p $OUTPUTS/train $LOGDIR

# ---- W&B auth ---------------------------------------------------------------
# ~/.netrc is auto-mounted by Apptainer (home dir bind). No extra config needed.
# If your cluster does not mount $HOME, set WANDB_API_KEY in cluster.env.
WANDB_ENV_FLAG=""
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ENV_FLAG="--env WANDB_API_KEY=$WANDB_API_KEY"
fi

# ---- run inside container ---------------------------------------------------
# Scratch is identity-mounted (same path inside and outside the container).
# $HOME (/nfshomes/<user>) is auto-mounted by Apptainer — covers .netrc, .gitconfig.
apptainer exec \
    --nv \
    --bind $SCRATCH:$SCRATCH \
    $WANDB_ENV_FLAG \
    --env HF_HOME=$SCRATCH/hf_cache \
    --env HF_DATASETS_CACHE=$SCRATCH/hf_cache \
    --env TMPDIR=$SCRATCH/tmp \
    --env UV_CACHE_DIR=$SCRATCH/uv_cache \
    --env MUJOCO_GL=osmesa \
    $SIF bash -c "
        set -e
        cd $REPO
        uv sync --quiet
        uv pip install --editable $LEROBOT --quiet

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
    " 2>&1 | tee $LOGDIR/${SLURM_JOB_ID}-train.log
