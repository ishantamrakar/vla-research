#!/bin/bash
# =============================================================================
# VLA eval job — runs lerobot-eval on a saved checkpoint.
# Submit from repo root: sbatch slurm/eval.slurm
#
# Outputs (on $SCRATCH):
#   eval results + 10 rendered videos → $OUTPUTS/eval/<EXPERIMENT>_<CHECKPOINT>/
# =============================================================================
#SBATCH --job-name=vla-eval
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm/logs/eval-%j.out
#SBATCH --error=slurm/logs/eval-%j.err

# ---- load cluster config ----------------------------------------------------
CLUSTER_ENV="${SLURM_SUBMIT_DIR}/slurm/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: slurm/cluster.env not found."
    exit 1
fi
source "$CLUSTER_ENV"

# ---- eval config — edit before submitting -----------------------------------
EXPERIMENT="act_libero_spatial_100k"
CHECKPOINT="last"          # "last" or a step number e.g. "000100000"
N_EPISODES=50
WANDB_ENABLE="false"

# ---- derived paths ----------------------------------------------------------
SIF=$REPO/apptainer/vla.sif
OUTPUTS=$SCRATCH/vla-outputs
POLICY_PATH=$OUTPUTS/train/$EXPERIMENT/checkpoints/$CHECKPOINT/pretrained_model
EVAL_OUTPUT=$REPO/outputs/eval/${EXPERIMENT}_${CHECKPOINT}

if [ ! -d "$POLICY_PATH" ]; then
    echo "ERROR: checkpoint not found at $POLICY_PATH"
    echo "Check EXPERIMENT and CHECKPOINT variables."
    exit 1
fi
echo "Evaluating: $POLICY_PATH"
echo "Output:     $EVAL_OUTPUT"

# ---- W&B auth ---------------------------------------------------------------
WANDB_ENV_FLAG=""
if [ -n "$WANDB_API_KEY" ] && [ "$WANDB_ENABLE" = "true" ]; then
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
    --env TMPDIR=$SCRATCH/tmp \
    --env UV_CACHE_DIR=$SCRATCH/uv_cache \
    --env MUJOCO_GL=osmesa \
    --env LIBERO_CONFIG_PATH=$SCRATCH/.libero \
    $SIF bash -c "
        set -e
        cd $REPO
        uv sync --quiet
        uv pip install --editable $LEROBOT --quiet
        uv run python scripts/setup_libero.py
        uv run lerobot-eval \
            --policy.path=$POLICY_PATH \
            --env.type=libero \
            --env.task=libero_spatial \
            --env.camera_name_mapping='{\"robot0_eye_in_hand_image\": \"wrist_image\"}' \
            --eval.n_episodes=$N_EPISODES \
            --policy.device=cuda \
            --output_dir=$EVAL_OUTPUT
    "

echo "Done. Results and videos in: $EVAL_OUTPUT"
