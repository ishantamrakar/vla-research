#!/bin/bash
# =============================================================================
# VLA training job — uses lerobot-train directly (ACT built-in hyperparameters)
# Submit from repo root: sbatch slurm/train.slurm
# Resume: resubmit with --resume flag (uncomment below)
# =============================================================================
#SBATCH --job-name=vla-train
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/train-%j.out
#SBATCH --error=slurm/logs/train-%j.err

# ---- load cluster config ----------------------------------------------------
CLUSTER_ENV="${SLURM_SUBMIT_DIR}/slurm/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: slurm/cluster.env not found."
    echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your values."
    exit 1
fi
source "$CLUSTER_ENV"

# ---- experiment config — edit before submitting ----------------------------
EXPERIMENT="act_libero_spatial_100k"
STEPS=100000
SAVE_FREQ=10000
EVAL_FREQ=50000
N_EPISODES=10
WANDB_ENABLE="true"

# ---- derived paths ----------------------------------------------------------
SIF=$REPO/apptainer/vla.sif
OUTPUTS=$SCRATCH/vla-outputs

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
    --env PATH=/opt/uv:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env PYTHONPATH=$REPO \
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

        uv run lerobot-train \
            --dataset.repo_id=lerobot/libero_spatial_image \
            --policy.type=act \
            --env.type=libero \
            --env.task=libero_spatial \
            --output_dir=$OUTPUTS/train/$EXPERIMENT \
            --job_name=$EXPERIMENT \
            --steps=$STEPS \
            --save_freq=$SAVE_FREQ \
            --eval_freq=$EVAL_FREQ \
            --eval.n_episodes=$N_EPISODES \
            --wandb.enable=$WANDB_ENABLE \
            --policy.push_to_hub=false
    "
    # To resume a previous run, replace the lerobot-train call above with:
    # uv run lerobot-train \
    #     --resume \
    #     --config_path=$OUTPUTS/train/$EXPERIMENT/checkpoints/last/pretrained_model/train_config.json
