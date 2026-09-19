# VLA Research Template

## Goal
A research vehicle for vision-language-action and imitation learning experiments
on single-arm manipulation. The default workflow is **fine-tuning from a public
checkpoint**, not training from scratch. Sim-only for now; real-robot deferred.

The repo exists to make new research ideas (planners, action heads, memory,
event-camera inputs, MCTS-style test-time compute) trivial to drop into a
working pipeline and compare against a known baseline.

## Stack (locked decisions — change only with explicit discussion)
- **Python**: 3.12 (upgraded from 3.11 — latest LeRobot requires >=3.12 as of v0.5.0)
- **Env manager**: uv (not conda, not pip-tools)
- **DL framework**: PyTorch with CUDA cu126 (upgraded from cu124 — cu124 index tops out at torch 2.6.0; LeRobot>=0.5 requires torch>=2.7; driver 580.x supports up to CUDA 13.0 so cu126 is safe)
- **Policy library**: LeRobot (HuggingFace), installed editable from a sibling
  clone so we can read and modify their code
- **Sim**:
  - Primary: LIBERO (well-supported benchmark, single-arm Franka)
  - Secondary: ManiSkill 3 (GPU-parallel, broader task coverage)
  - Sanity-check toy: PushT (LeRobot built-in)
- **Config**: Hydra
- **Experiment tracking**: Weights & Biases
- **Containers**: Apptainer for Zaratan deployment (built later)

## Hardware

**Policy: HPC is where runs happen. The 3080 and the MacBook are prototyping
machines only.** Anything whose numbers go in a paper — training, evaluation,
benchmarks, latency profiles — runs on Nexus. Local hardware is for making
code work before it gets there: shape errors, config bugs, does-it-import,
does-one-step-run. A local result is a smoke test, not a finding.

Why the rule, rather than "use whatever is free": local and cluster runs
disagree in ways that are invisible in the output. Different GPU, different
VRAM ceiling forcing a different batch size, different CUDA wheel. A number
produced on the 3080 and a number produced on the L40S are not comparable,
and mixing them across a results table is the kind of error nobody catches
by reading. One machine owns the numbers.

- **UMIACS Nexus-CFAR (UMD HPC) — PRIMARY, all real runs.** SLURM + Apptainer.
  Container: `apptainer/vla.def` → `vla.sif`. Job scripts: `slurm/train.slurm`,
  `slurm/eval.slurm`, `slurm/latency.slurm`.
  Scratch path: `/fs/nexus-scratch/$USER`. GPU: L40S (48 GB).
  (Switched from Zaratan — no free student tier.)
- **Lab workstation (prototyping)**:
  - GPU: NVIDIA GeForce RTX 3080 (10GB VRAM)
  - Driver: 580.126.09 (supports CUDA up to 13.0)
  - System CUDA toolkit: 12.0
  - PyTorch wheel: cu124
  - OS: Ubuntu 24.04.4 LTS (Noble)
  - Repo location: ~/Documents/vla-research
  - LeRobot sibling clone: ~/Documents/lerobot
  - Use for: short smoke runs, debugging a policy that crashes, checking a
    config change loads. Not for results.
- **Personal M3 MacBook (prototyping)**: scaffolding, documentation, analysis
  of already-computed results, writing job scripts. No CUDA, so no sim and no
  training — a run that needs a GPU is a run that goes to Nexus.

**The one deliberate exception: hardware-specific measurements.** Experiment 1
(inference latency) is *about* a particular GPU, so an L40S number is not a
substitute for a 3080 number — it is a lower bound. Where a measurement's
whole point is the hardware it ran on, run it on both and report both. See
`docs/experiment_01_latency.md`.

## VRAM budget reality (10GB)
- Diffusion Policy / ACT / SmolVLA: fine for fine-tuning, batch ≤ 16
- Pi-0: requires 4-bit quantization + LoRA; may not fit even then
- GR00T N1.7+ scale: Zaratan only
- Default fine-tuning batch size: 8 (verify by running once and checking
  `nvidia-smi` for headroom)

## Conventions
- **One change at a time.** No new component lands without a smoke-test run
  that exercises it end-to-end.
- All experiments run via `scripts/` entry points driven by Hydra configs.
- Never commit data, datasets, or checkpoints. They live outside the repo.
- `CLAUDE.md` is updated whenever a non-obvious decision is made.

## Smoke tests
- **Phase 1 target**: ✅ `uv run lerobot-train --dataset.repo_id=lerobot/pusht --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --batch_size=64 --steps=500 --wandb.enable=false --save_checkpoint=false` — loss dropped 0.345 → 0.077 over 500 steps at ~10 steps/s on RTX 3080.
- **Phase 2 target**: ✅ `uv run lerobot-train --dataset.repo_id=lerobot/libero_spatial_image --policy.type=act --policy.device=cuda --policy.push_to_hub=false --batch_size=8 --steps=500 --wandb.enable=false --save_checkpoint=false` — loss dropped 6.671 → 3.019 over 500 steps (ACT, lr warmup). ManiSkill 3 v3.0.1 GPU-parallel smoke test passed: 4×PickCube-v1 envs stepping on SAPIEN/PhysX.
- **Phase 4 target**: ✅ `uv run python scripts/finetune.py policy=act
  env=libero_spatial train.steps=500` runs end-to-end and logs to W&B.
  (DiffusionPolicy is PushT-only; its dual-camera reshape fails on LIBERO eval.)
- **Phase 5 target**: ✅ `uv run python scripts/finetune.py policy=smolvla
  env=libero_spatial train.steps=500` — loss 2.13 → 0.93, VRAM peak 4.6GB / 9.9GB,
  eval harness exercised at step 499. Requires `num2words` package (already in uv.lock).

## Phase tracker
- [x] Phase 0: Repo init, CLAUDE.md, .gitignore, first commit
- [x] Phase 1: uv env + LeRobot install + reproduce PushT Diffusion Policy
- [x] Phase 2: LIBERO + ManiSkill 3 smoke tests
- [x] Phase 3: Hydra refactor into template structure
- [x] Phase 4: Eval harness + W&B integration
- [x] Phase 5: Add SmolVLA as second policy
- [x] Phase 6: Apptainer + UMIACS Nexus-CFAR SLURM scripts (switched from Zaratan — no free student tier; Nexus-CFAR uses SLURM + Apptainer same as Zaratan would have)
- [ ] Phase 7+: research-specific scaffolding (planners/, memory/, etc.)
- [ ] Future: multi-GPU distributed training (PyTorch DDP) — requires gradient sync, adjusted LR scaling, SLURM multi-node or --ntasks-per-node; unlock once single-GPU baseline is solid

## Working with Claude Code on this repo
- Start each session by reading this file and the current phase status.
- Do not leap ahead phases. If asked to do Phase N, do only Phase N.
- After any meaningful change, run the smoke test and report the output.
- If a decision is made about stack, conventions, or hardware, append it here.
- **Cluster training** uses `lerobot-train` directly via `slurm/train.slurm`. Do NOT use `scripts/train.py` or `scripts/sweep.py` for cluster runs — they were a premature reimplementation with wrong hyperparameters (lr 1e-4 vs ACT preset 1e-5; batch_size 512 from flawed VRAM sweep vs ACT's tested 8).
- **Local dev / custom loop** uses `scripts/finetune.py` with Hydra + RunContext. Use when injecting custom research code into the training loop.
- `sweep.py` is retained for throughput-optimal batch size discovery on custom (non-preset) policies. Not needed for stock `lerobot-train` runs.
- W&B logging is built into `lerobot-train` (`--wandb.enable=true`). Also available in `finetune.py` via `wandb.enable=true` Hydra key.