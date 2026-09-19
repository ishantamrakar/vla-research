#!/usr/bin/env bash
# Submit one latency job per GPU generation. Run from repo root:
#
#     bash slurm/sweep_latency.sh                 # default spread
#     bash slurm/sweep_latency.sh rtxa4000 l40s   # just these
#
# WHY A SWEEP RATHER THAN ONE NUMBER. Experiment 1 asks whether VLA inference
# latency is large enough to matter. That has no single answer -- it depends
# entirely on the deployment GPU, and the thesis's own target (an RTX 3080) is
# neither the fastest nor the slowest thing a robot might carry. A curve from
# a 2017 card to a current datacentre part says where the control deadline is
# missed and where it is not, which is a far stronger claim than one figure
# from whichever node happened to be free.
#
# It is also just faster. Nexus scavenger frequently has every l40s busy while
# dozens of rtxa4000 and gtx1080ti nodes sit idle, so a sweep starts returning
# results immediately instead of waiting in one queue.
#
# Cost is near zero: each job is minutes of compute on a preemptible
# partition, and they queue independently.
set -euo pipefail

cd "$(dirname "$0")/.."

# Default spread, chosen to span generations rather than to be exhaustive.
# Checked against `sinfo -p scavenger` on 2026-09-19:
#   gtx1080ti  Pascal, 2017     -- 16 nodes idle, the slow end
#   rtxa4000   Ampere, 16GB     -- 31 nodes idle, closest available to a 3080
#   rtx2080ti  Turing, 2018     -- several idle
#   rtxa5000   Ampere, 24GB     -- busy but plentiful
#   l40s       Ada, 48GB        -- the original target; often fully allocated
# a100/h100/h200/b300 exist here too and are worth adding once the pipeline is
# known good -- they are in heaviest demand, so they queue longest.
GPUS=("${@:-}")
if [ -z "${GPUS[0]:-}" ]; then
    GPUS=(rtxa4000 gtx1080ti rtx2080ti rtxa5000 l40s)
fi

mkdir -p slurm/logs

echo "Submitting latency jobs for: ${GPUS[*]}"
echo
for gpu in "${GPUS[@]}"; do
    # --gres on the command line overrides the #SBATCH default in the script;
    # --job-name makes squeue readable when several are pending at once.
    jid=$(sbatch --parsable \
        --gres="gpu:${gpu}:1" \
        --job-name="lat-${gpu}" \
        slurm/latency.slurm)
    echo "  ${gpu}: job ${jid}"
done

echo
echo "Watch:    squeue --user \$USER"
echo "Results:  outputs/latency/<gpu>/smolvla_libero_spatial.md"
echo
echo "Note: results are keyed by the GPU the job actually landed on (read from"
echo "nvidia-smi), not by the requested type, so nothing overwrites anything"
echo "even if SLURM hands you a different card than requested."
