#!/bin/bash
# =============================================================================
# GPU batch-size sweep — run once before train.slurm on a new GPU type.
# Clears the cached batch size for this policy+dataset so it re-probes
# for the current hardware (local dev result is not valid on a different GPU).
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$SCRIPT_DIR/cluster.env" ]; then
    echo "ERROR: slurm/cluster.env not found."
    echo "Copy slurm/cluster.env.example to slurm/cluster.env and fill in your values."
    exit 1
fi
source "$SCRIPT_DIR/cluster.env"

#SBATCH --job-name=vla-sweep
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/vla-sweep-%j.out
#SBATCH --error=/tmp/vla-sweep-%j.err

POLICY="act"
ENV="libero_spatial"
SIF=$REPO/apptainer/vla.sif
LOGDIR=$REPO/slurm/logs
mkdir -p $LOGDIR

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
    " 2>&1 | tee $LOGDIR/${SLURM_JOB_ID}-sweep.log
