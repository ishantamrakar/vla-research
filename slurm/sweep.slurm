#!/bin/bash
# =============================================================================
# GPU batch-size sweep — run once before train.slurm on a new GPU type.
# Submit from repo root: sbatch slurm/sweep.slurm
# =============================================================================
#SBATCH --job-name=vla-sweep
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/slurm-%j.out 
#SBATCH --error=slurm/logs/slurm-%j.err

# ---- load cluster config ----------------------------------------------------
CLUSTER_ENV="${SLURM_SUBMIT_DIR}/slurm/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: slurm/cluster.env not found."
    echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your values."
    exit 1
fi
source "$CLUSTER_ENV"

POLICY="act"
ENV="libero_spatial"
SIF=$REPO/apptainer/vla.sif

# Clear stale sweep cache entry (tuned for a different GPU on local dev machine)
python3 -c "
import json, pathlib
cache = pathlib.Path('$REPO/outputs/sweep_cache.json')
if cache.exists():
    data = json.loads(cache.read_text())
    key = '${POLICY}|lerobot/libero_spatial_image'
    if key in data:
        del data[key]
        cache.write_text(json.dumps(data, indent=2))
        print(f'Cleared stale cache entry: {key}')
" 2>/dev/null || true

apptainer exec \
    --nv \
    --bind $SCRATCH:$SCRATCH \
    --env PATH=/opt/uv:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
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
        uv run python scripts/sweep.py policy=$POLICY env=$ENV
    "
