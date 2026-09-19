#!/bin/bash
# =============================================================================
# VLA inference latency profile — p50/p90/p99 per step, fp16 vs fp32,
# with and without preprocessing. Submit from repo root:
#
#     sbatch slurm/latency.slurm
#
# Outputs: outputs/latency/<policy>_<env>.{md,json}
#
# WHAT THIS NUMBER IS FOR. Two theses need it before either is worth building:
# a latency-evaluation chapter that injects measured delay into LIBERO
# rollouts, and a gaze-timing chapter premised on policies being slower than
# human eye-hand coordination. The deliverable is the TAIL against one 50ms
# control step, not a mean.
#
# HARDWARE CAVEAT, and it matters for how the result is read. This runs on an
# L40S (48GB). The lab workstation is an RTX 3080 (10GB) and is what the
# thesis is actually about. An L40S is substantially faster, so these numbers
# are a LOWER BOUND on 3080 latency:
#   * if p99 MISSES the control step here, it certainly misses on a 3080 --
#     the latency chapter is alive, conclusively;
#   * if p99 FITS here, that is NOT the kill, because the 3080 may still miss.
#     Re-run on the workstation before declaring the chapter dead.
# Run `scripts/profile_latency.py` directly on the 3080 for the real figure;
# it needs no SLURM and no container.
#
# Short job: this is minutes of work, so it is worth requesting a small time
# limit to get scheduled quickly on `scavenger`. Preemption is survivable
# here (unlike training) -- just resubmit, there is no checkpoint to lose.
#
# CHOOSING THE GPU. The --gres line below is only a DEFAULT. Override it at
# submit time, because #SBATCH directives cannot read shell variables:
#
#     sbatch --gres=gpu:rtxa4000:1 slurm/latency.slurm
#
# This matters practically and scientifically. Practically: Nexus scavenger
# often has every l40s busy while dozens of rtxa4000/gtx1080ti nodes sit idle,
# so pinning l40s can mean waiting hours for a job that runs in minutes.
# Scientifically: this experiment is ABOUT the GPU, so a spread across
# generations is a stronger result than any single card. `slurm/sweep_latency.sh`
# submits one job per GPU type for exactly that reason.
# =============================================================================
#SBATCH --job-name=vla-latency
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm/logs/latency-%j.out
#SBATCH --error=slurm/logs/latency-%j.err

# ---- load cluster config ----------------------------------------------------
CLUSTER_ENV="${SLURM_SUBMIT_DIR}/slurm/cluster.env"
if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: slurm/cluster.env not found."
    exit 1
fi
source "$CLUSTER_ENV"

# ---- profile config — edit before submitting --------------------------------
POLICY="${POLICY:-smolvla}"
ENV_NAME="${ENV_NAME:-libero_spatial}"
STEPS="${STEPS:-500}"
WARMUP="${WARMUP:-20}"

# Results are keyed by GPU so a sweep does not overwrite itself. SLURM does not
# expose the GPU model directly, so read it from the device rather than from
# the --gres string, which may say "l40s" while the node hands you something
# else entirely.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
           | tr -d " " | tr "[:upper:]" "[:lower:]" | sed "s|nvidia||g")

# ---- derived paths ----------------------------------------------------------
SIF=$REPO/apptainer/vla.sif
OUT_DIR=$REPO/outputs/latency/$GPU_NAME

echo "Profiling: policy=$POLICY env=$ENV_NAME steps=$STEPS gpu=$GPU_NAME"
echo "Output:    $OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---- run inside container ---------------------------------------------------
apptainer exec \
    --nv \
    --bind $SCRATCH:$SCRATCH \
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
        uv run python scripts/profile_latency.py \
            policy=$POLICY \
            env=$ENV_NAME \
            steps=$STEPS \
            warmup=$WARMUP \
            out_dir=$OUT_DIR
    "

echo "Done. Report in: $OUT_DIR"
