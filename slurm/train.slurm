#!/bin/bash
# =============================================================================
# VLA training job
# Submit from repo root: sbatch slurm/train.slurm
# Resume: just resubmit — train.py picks up from checkpoint_latest.pt
# =============================================================================
#SBATCH --job-name=vla-train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ---- load cluster config ----------------------------------------------------
CLUSTER_ENV="${SLURM_SUBMIT_DIR}/slurm/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: slurm/cluster.env not found."
    echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your values."
    exit 1
fi
source "$CLUSTER_ENV"

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
WANDB_ENV_FLAG=""
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ENV_FLAG="--env WANDB_API_KEY=$WANDB_API_KEY"
fi

# ---- run inside container ---------------------------------------------------
apptainer exec \
    --nv \
    --bind $SCRATCH:$SCRATCH \
    $WANDB_ENV_FLAG \
    --env PATH=/opt/uv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
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
