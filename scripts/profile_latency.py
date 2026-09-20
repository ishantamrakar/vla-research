#!/usr/bin/env python
"""
Per-step inference latency for a VLA policy: p50/p90/p99, fp16 and fp32,
with and without preprocessing.

Why this exists. Two separate theses need this number before either is worth
building:
  * a latency-evaluation chapter, which injects measured delay into LIBERO
    rollouts -- if inference already fits inside a control step there is
    nothing to inject and the chapter is dead;
  * a gaze-timing chapter, whose whole premise is that a policy cannot react
    within the window a human eye-hand system does.

So the deliverable is not a mean. It is the TAIL, against one control step.
A policy whose p50 is comfortable and whose p99 blows three control steps is
exactly the interesting case, and a mean hides it completely.

    Usage:
        uv run python scripts/profile_latency.py policy=smolvla env=libero_spatial
        uv run python scripts/profile_latency.py policy=smolvla steps=500 dtypes=[fp16,fp32]

Measurement notes, each of which changes the number if skipped:

  * **CUDA is asynchronous.** A bare time.perf_counter() around a forward pass
    measures kernel LAUNCH, not execution, and reports impossibly low latency.
    Every timed region here ends with torch.cuda.synchronize().
  * **The first steps are not representative.** cuDNN autotuning, lazy module
    init and memory-pool growth make early calls several times slower. We run
    --warmup steps and discard them; they are reported separately rather than
    silently dropped, because an enormous warmup is itself a finding for a
    system that must start responding immediately.
  * **Preprocessing is part of the loop.** A robot does not get pre-normalised
    tensors for free; image resize/normalise and tokenisation happen per step.
    Timing the bare forward pass measures something no deployment ever sees,
    so we time both and report the difference.
  * **fp16 is measured with `torch.autocast`, not `policy.half()`.** Hard-
    casting the weights fails: the preprocessor holds fp32 normalisation
    buffers and re-emits fp32 every step, so a cast applied to the batch is
    undone before the weights see it ("mat1 and mat2 must have the same
    dtype, but got Float and Half" -- cluster job 7584256). The cast has to
    sit downstream of preprocessing, which autocast does by construction.
    It is also safer: SmolVLA wraps a VLM whose LayerNorms want fp32, and
    half()-ing everything can produce NaNs rather than an error, which
    yields plausible timings from a broken forward pass.
  * **Results are written after every condition, not at the end.** That same
    job completed fp32 and then died in fp16, losing the fp32 numbers with
    it. A later failure should cost only the conditions that did not run.
  * **p99 of 500 samples is the 5th-worst observation.** It is a real but
    coarse estimate; do not read three significant figures into it. 500 is the
    plan's number and is enough to separate "fits in a control step" from
    "does not", which is the only question being asked.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# LIBERO runs its controller at 20Hz, so one control step is 50ms. This is the
# bar every latency number here is measured against: a p99 above it means the
# policy misses its deadline one step in a hundred.
#
# It is a constant of the benchmark rather than of our config, but it is
# recorded here (not buried in a comparison) so that changing benchmark is a
# one-line edit instead of a hunt.
LIBERO_CONTROL_HZ = 20.0
LIBERO_CONTROL_STEP_MS = 1000.0 / LIBERO_CONTROL_HZ


@dataclass
class LatencyStats:
    """Timing distribution for one (policy, dtype, preprocessing) condition."""

    label: str
    dtype: str
    with_preprocessing: bool
    n: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    warmup_ms: list[float] = field(default_factory=list)

    @property
    def control_steps_p99(self) -> float:
        return self.p99_ms / LIBERO_CONTROL_STEP_MS

    @property
    def fits_in_control_step(self) -> bool:
        """The kill criterion: is even the tail inside one control step?"""
        return self.p99_ms < LIBERO_CONTROL_STEP_MS


def summarize(label: str, dtype: str, with_preprocessing: bool,
              samples_ms: list[float], warmup_ms: list[float]) -> LatencyStats:
    a = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        label=label,
        dtype=dtype,
        with_preprocessing=with_preprocessing,
        n=a.size,
        # numpy's default linear interpolation between order statistics; for
        # n=500 the p99 sits between the 5th and 6th worst sample.
        p50_ms=float(np.percentile(a, 50)),
        p90_ms=float(np.percentile(a, 90)),
        p99_ms=float(np.percentile(a, 99)),
        mean_ms=float(a.mean()),
        std_ms=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        min_ms=float(a.min()),
        max_ms=float(a.max()),
        warmup_ms=[float(x) for x in warmup_ms],
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_policy(
    policy,
    batch: dict,
    preprocessor,
    steps: int,
    warmup: int,
    with_preprocessing: bool,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[list[float], list[float]]:
    """Time `steps` select_action calls, after `warmup` discarded ones.

    Returns (samples_ms, warmup_ms). Warmup is returned rather than dropped
    because a policy that takes seconds on its first call has a real deployment
    problem that a steady-state distribution would hide.

    `autocast_dtype` runs the forward pass under `torch.autocast`. That is
    used instead of hard-casting the policy's weights (see main()), so the
    reduced-precision numbers reflect how fp16 is actually deployed.
    """
    samples: list[float] = []
    warmups: list[float] = []

    # When not timing preprocessing, run it ONCE up front and reuse the result,
    # rather than timing the forward pass on an unprocessed batch (which would
    # either crash or silently measure the wrong thing).
    static_batch = batch if with_preprocessing else preprocessor(batch)

    for i in range(warmup + steps):
        _sync()
        t0 = time.perf_counter()

        step_batch = preprocessor(batch) if with_preprocessing else static_batch
        with torch.inference_mode():
            if autocast_dtype is not None:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    policy.select_action(step_batch)
            else:
                policy.select_action(step_batch)

        _sync()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if i < warmup:
            warmups.append(dt_ms)
        else:
            samples.append(dt_ms)

        # select_action is stateful for chunked policies (ACT, SmolVLA queue a
        # chunk and return one action per call, so only every Nth call runs the
        # network). Resetting each step forces every call to do the full work,
        # which is the number the delay-injection harness needs -- the worst
        # case a controller can hit, not the amortised one.
        if hasattr(policy, "reset"):
            policy.reset()

    return samples, warmups


def format_report(all_stats: list[LatencyStats], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# VLA inference latency")
    lines.append("")
    lines.append(f"- GPU: {meta.get('gpu', 'unknown')}")
    lines.append(f"- torch: {meta.get('torch')} / CUDA {meta.get('cuda')}")
    lines.append(f"- policy: {meta.get('policy')}  env: {meta.get('env')}")
    lines.append(f"- steps per condition: {meta.get('steps')} (warmup {meta.get('warmup')})")
    lines.append(
        f"- control step: **{LIBERO_CONTROL_STEP_MS:.1f} ms** "
        f"(LIBERO @ {LIBERO_CONTROL_HZ:.0f} Hz)"
    )
    lines.append("")
    lines.append("| dtype | preproc | p50 ms | p90 ms | p99 ms | mean | max | p99 / control step | fits? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in all_stats:
        lines.append(
            f"| {s.dtype} | {'yes' if s.with_preprocessing else 'no'} | "
            f"{s.p50_ms:.2f} | {s.p90_ms:.2f} | **{s.p99_ms:.2f}** | "
            f"{s.mean_ms:.2f} | {s.max_ms:.2f} | "
            f"{s.control_steps_p99:.2f}x | "
            f"{'**YES**' if s.fits_in_control_step else 'no'} |"
        )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    all_fit = all(s.fits_in_control_step for s in all_stats)
    if all_fit:
        lines.append(
            "**Every condition fits inside one control step at p99.** "
            "This is the kill condition for the latency-evaluation chapter: "
            "there is nothing to inject, because real inference never misses a "
            "deadline on this hardware. Note the bar moves with the hardware -- "
            "a slower GPU or a larger policy may still miss."
        )
    else:
        worst = max(all_stats, key=lambda s: s.p99_ms)
        lines.append(
            f"**Latency is real.** Worst condition ({worst.dtype}, "
            f"preproc={'yes' if worst.with_preprocessing else 'no'}) has p99 "
            f"{worst.p99_ms:.1f} ms = {worst.control_steps_p99:.1f} control "
            "steps. Delay injection has something to inject; use this "
            "distribution as the source for the action-hold wrapper."
        )
    lines.append("")

    # Preprocessing cost is worth calling out separately: if it dominates, the
    # engineering fix is a faster input pipeline, not a smaller model.
    pre = {(s.dtype): s for s in all_stats if s.with_preprocessing}
    nopre = {(s.dtype): s for s in all_stats if not s.with_preprocessing}
    shared = sorted(set(pre) & set(nopre))
    if shared:
        lines.append("## Preprocessing cost")
        lines.append("")
        lines.append("| dtype | p50 forward | p50 with preproc | preproc share |")
        lines.append("|---|---|---|---|")
        for d in shared:
            a, b = nopre[d].p50_ms, pre[d].p50_ms
            share = (b - a) / b * 100 if b > 0 else 0.0
            lines.append(f"| {d} | {a:.2f} ms | {b:.2f} ms | {share:.0f}% |")
        lines.append("")

    warm = [s for s in all_stats if s.warmup_ms]
    if warm:
        lines.append("## Warmup (discarded from the statistics)")
        lines.append("")
        for s in warm:
            first = s.warmup_ms[0]
            lines.append(
                f"- {s.dtype}/preproc={'yes' if s.with_preprocessing else 'no'}: "
                f"first call {first:.0f} ms "
                f"({first / max(s.p50_ms, 1e-9):.0f}x steady-state p50)"
            )
        lines.append("")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../configs", config_name="latency")
def main(cfg: DictConfig) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors

    steps = int(cfg.get("steps", 500))
    warmup = int(cfg.get("warmup", 20))
    dtypes = list(cfg.get("dtypes", ["fp32", "fp16"]))
    out_dir = Path(cfg.get("out_dir", "outputs/latency"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. This profiles GPU inference latency; running it on "
            "CPU would produce a number unrelated to any deployment. Submit via "
            "slurm/latency.slurm, or run on the lab workstation."
        )

    device = "cuda"
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "policy": str(cfg.policy.type),
        "env": str(cfg.env.type),
        "steps": steps,
        "warmup": warmup,
        "control_step_ms": LIBERO_CONTROL_STEP_MS,
    }
    log.info("profiling on %s", meta["gpu"])

    # One real batch from the dataset the policy was trained for. A synthetic
    # tensor of the right shape would time the same kernels, but a real sample
    # also exercises the preprocessing path (resize, normalise, tokenise) with
    # realistic content, which is half of what we are measuring.
    ds = LeRobotDataset(cfg.dataset.repo_id, episodes=[0])
    sample = ds[0]
    batch = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }

    stem = f"{cfg.policy.type}_{cfg.env.type}"

    def save(stats: list[LatencyStats]) -> None:
        """Write the report after every condition, not once at the end.

        Job 7584256 completed fp32 and then died in fp16, and because the
        report was written only after the whole loop, the fp32 numbers were
        lost with it. Saving incrementally means a later failure costs only
        the conditions that did not run.
        """
        if not stats:
            return
        (out_dir / f"{stem}.md").write_text(format_report(stats, meta))
        (out_dir / f"{stem}.json").write_text(
            json.dumps(
                {"meta": meta, "stats": [asdict(x) for x in stats]}, indent=2
            )
        )

    all_stats: list[LatencyStats] = []
    for dtype_name in dtypes:
        log.info("=== %s ===", dtype_name)

        policy_cfg = make_policy_config(cfg.policy.type, **{
            k: v for k, v in OmegaConf.to_container(cfg.policy).items()
            if k != "type"
        })
        policy = make_policy(policy_cfg, ds_meta=ds.meta)
        policy.to(device)
        policy.eval()
        preprocessor, _ = make_pre_post_processors(
            policy_cfg, dataset_stats=ds.meta.stats
        )

        # fp16 is done with autocast, NOT by half()-ing the weights.
        #
        # Two reasons, both learned from a failed cluster run (job 7584256,
        # "mat1 and mat2 must have the same dtype, but got Float and Half").
        # First, the preprocessor holds fp32 normalisation buffers and re-emits
        # fp32 every step, so any cast applied to the batch beforehand is undone
        # before the weights ever see it -- the cast has to sit downstream of
        # preprocessing, which autocast does by construction.
        # Second, SmolVLA wraps a VLM whose LayerNorms want fp32; half()-ing
        # everything can yield NaNs instead of an error, which is worse than a
        # crash because it produces plausible timings from a broken forward
        # pass. Autocast picks precision per op and is also how fp16 is actually
        # deployed, so it is the more honest thing to measure.
        autocast_dtype = torch.float16 if dtype_name == "fp16" else None

        for with_pre in (False, True):
            samples, warmups = time_policy(
                policy, batch, preprocessor, steps, warmup, with_pre,
                autocast_dtype=autocast_dtype,
            )
            st = summarize(cfg.policy.type, dtype_name, with_pre, samples, warmups)
            all_stats.append(st)
            log.info(
                "%s preproc=%s: p50 %.2f p90 %.2f p99 %.2f ms (%.2f control steps)",
                dtype_name, with_pre, st.p50_ms, st.p90_ms, st.p99_ms,
                st.control_steps_p99,
            )

        save(all_stats)
        del policy
        torch.cuda.empty_cache()

    if not all_stats:
        raise SystemExit("no condition completed -- see the traceback above")

    print(format_report(all_stats, meta))
    log.info("wrote %s", out_dir / f"{stem}.md")


if __name__ == "__main__":
    main()
