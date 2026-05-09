#!/bin/bash
# =============================================================================
# GPU batch-size sweep — run once before train.slurm on a new GPU type.
# Clears the existing sweep cache entry so the correct batch size is found
# for this hardware (cache from local dev may be tuned for a different GPU).
# =============================================================================

#SBATCH --job-name=vla-sweep
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%j-sweep.out
#SBATCH --error=slurm/logs/%j-sweep.err

POLICY="act"
ENV="libero_spatial"

SCRATCH=/fs/nexus-scratch/itamraka
REPO=$HOME/vla-research
SIF=$REPO/apptainer/vla.sif

mkdir -p slurm/logs

# Clear stale cache entry from local dev machine (different GPU, different optimal bs).
# sweep.py will regenerate it for this hardware.
python3 -c "
import json, pathlib
cache = pathlib.Path('outputs/sweep_cache.json')
if cache.exists():
    data = json.loads(cache.read_text())
    data.pop('${POLICY}|lerobot/libero_spatial_image', None)
    cache.write_text(json.dumps(data, indent=2))
    print('Cleared stale sweep cache entry.')
" 2>/dev/null || true

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
        uv run python scripts/sweep.py policy=$POLICY env=$ENV
    "
