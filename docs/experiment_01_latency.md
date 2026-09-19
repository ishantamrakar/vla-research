# Experiment 1 — VLA inference latency

Part of the Aria thesis feasibility sequence (see
`aria/experiments/README.md` for the full index). Lives here rather than in
the `aria` repo because it needs lerobot, CUDA torch, the Apptainer image and
the SLURM setup that already work in this repo — and because experiment 5
(LIBERO delay injection) will build directly on its output.

## Question

Per-step inference latency for SmolVLA: **p50/p90/p99** over 500 steps, fp16
and fp32, with and without preprocessing.

## Kill criterion

**If p99 is under one LIBERO control step for every model you can run, the
latency-evaluation chapter is dead** — there is nothing to inject, and the
harness for experiment 5 is not worth building.

One LIBERO control step is **50 ms** (20 Hz controller).

## Run it

On the cluster:

```bash
sbatch slurm/latency.slurm
```

On the lab workstation (no SLURM, no container needed):

```bash
uv run python scripts/profile_latency.py policy=smolvla env=libero_spatial
```

Output lands in `outputs/latency/<policy>_<env>.{md,json}`.

## The hardware caveat, which decides how the result is read

The cluster GPU is an **L40S (48 GB)**. The thesis is about the **RTX 3080
(10 GB)** lab workstation. An L40S is substantially faster, so cluster numbers
are a **lower bound** on 3080 latency:

- p99 **misses** the control step on the L40S → it certainly misses on a 3080.
  The chapter is alive, conclusively, and you can proceed to experiment 5.
- p99 **fits** on the L40S → **this is not the kill.** The 3080 may still miss.
  Re-run on the workstation before declaring anything dead.

This asymmetry is why the job script says so in its header: the cheap cluster
run can only *confirm* the chapter, never *kill* it.

## Measurement decisions, and what breaks without each

These are the parts that quietly produce wrong numbers if skipped.

**CUDA is asynchronous.** A bare `perf_counter()` around a forward pass times
the kernel *launch*, not its execution, and reports impossibly low latency.
Every timed region ends with `torch.cuda.synchronize()`.

**Warmup is discarded but reported.** cuDNN autotuning, lazy module init and
memory-pool growth make the first calls several times slower. 20 warmup steps
are dropped from the statistics — but kept in the report, because a policy
taking seconds on its first call has a real deployment problem that a
steady-state distribution hides entirely.

**Preprocessing is timed separately, not excluded.** A robot does not receive
pre-normalised tensors for free; resize, normalise and tokenise happen every
step. Timing only the forward pass measures something no deployment ever sees.
Both are measured and the report breaks out preprocessing's share — if it
dominates, the fix is a faster input pipeline, not a smaller model.

**`policy.reset()` every step.** This is the subtle one. Chunked policies
(ACT, SmolVLA) queue an action chunk and return one action per call, so only
every Nth call actually runs the network. Without a reset you measure the
*amortised* cost, which is far lower and is not what a controller facing a
deadline experiences. Resetting forces every call to do full work — the worst
case, which is the number experiment 5's action-hold wrapper needs.

**fp16 casts inputs too.** Only floating tensors; integer tensors are token
ids and indices and must stay integral.

## Reading p99 honestly

p99 of 500 samples is the 5th-worst observation — a real but coarse estimate.
Do not read three significant figures into it. 500 is enough to separate "fits
in a control step" from "does not", which is the only question being asked.

## Status

**Harness written, not yet run.** No numbers exist until a job completes.

OpenVLA is not wired up. The plan lists it as optional ("if Python 3.12 is
handy"); this repo is on 3.12, so it is feasible, but SmolVLA answers the kill
question alone and adding a second model before the first has run is scope
that buys nothing.
