#!/usr/bin/env python
"""
GPU batch-size sweep: find the throughput-optimal batch size (max samples/sec)
that also stays within VRAM headroom. Caches the result for train.py.

Usage:
    uv run python scripts/sweep.py policy=act env=libero_spatial
"""
import copy
import gc
import json
import logging
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy_config, make_policy, make_pre_post_processors

log = logging.getLogger(__name__)

SWEEP_CANDIDATES = [8, 16, 32, 64, 96, 128, 256, 512, 640, 786, 896, 1024]
# Warm-up + timed steps; first SWEEP_WARMUP steps are discarded, then SWEEP_TIMED are measured.
SWEEP_WARMUP = 2
SWEEP_TIMED = 5
VRAM_HEADROOM_GB = 1.5
SWEEP_CACHE = Path("outputs/sweep_cache.json")


def _gb(n: int) -> float:
    return n / 1024**3


def _cache_key(policy_type: str, repo_id: str) -> str:
    return f"{policy_type}|{repo_id}"


def load_cache(policy_type: str, repo_id: str) -> int | None:
    if not SWEEP_CACHE.exists():
        return None
    try:
        return json.loads(SWEEP_CACHE.read_text()).get(_cache_key(policy_type, repo_id))
    except Exception:
        return None


def save_cache(policy_type: str, repo_id: str, batch_size: int):
    data = {}
    if SWEEP_CACHE.exists():
        try:
            data = json.loads(SWEEP_CACHE.read_text())
        except Exception:
            pass
    data[_cache_key(policy_type, repo_id)] = batch_size
    SWEEP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_CACHE.write_text(json.dumps(data, indent=2))
    log.info(f"Cached batch_size={batch_size} → {SWEEP_CACHE}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device = torch.device(cfg.device)

    dataset_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
    repo_id = dataset_kwargs["repo_id"]

    policy_kwargs = OmegaConf.to_container(cfg.policy, resolve=True)
    policy_type = policy_kwargs.pop("type")
    policy_cfg = make_policy_config(policy_type, **policy_kwargs)

    cached = load_cache(policy_type, repo_id)
    if cached is not None:
        log.info(f"Already cached: policy={policy_type}, dataset={repo_id} → batch_size={cached}")
        log.info("Delete outputs/sweep_cache.json to re-run the sweep.")
        return

    log.info("Loading dataset...")
    dataset = LeRobotDataset(**dataset_kwargs)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, dataset.meta)
    if delta_timestamps:
        dataset_kwargs["delta_timestamps"] = delta_timestamps
        dataset = LeRobotDataset(**dataset_kwargs)

    preprocessor, _ = make_pre_post_processors(policy_cfg, dataset_stats=dataset.meta.stats)

    gpu_total = torch.cuda.get_device_properties(device).total_memory
    max_allowed = gpu_total - VRAM_HEADROOM_GB * 1024**3
    pin = "cuda" in cfg.device
    camera_keys = dataset.meta.camera_keys
    grad_clip = cfg.train.grad_clip_norm

    log.info(f"=== GPU batch-size sweep: {policy_type} + {repo_id} ===")
    log.info(f"GPU total: {_gb(gpu_total):.2f} GB   headroom: {VRAM_HEADROOM_GB} GB   "
             f"limit: {_gb(max_allowed):.2f} GB")
    log.info(f"Selecting by throughput (samples/sec), not by max VRAM fit.")

    best_bs = SWEEP_CANDIDATES[0]
    best_throughput = 0.0
    policy_init = copy.deepcopy(
        make_policy(policy_cfg, ds_meta=dataset.meta).state_dict()
    )

    for bs in SWEEP_CANDIDATES:
        log.info(f"  Trying batch_size={bs} ...")
        torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        torch.cuda.empty_cache()

        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        policy.load_state_dict(copy.deepcopy(policy_init))
        policy.to(device).train()

        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda")

        dl = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=pin,
            persistent_workers=cfg.train.num_workers > 0,
            prefetch_factor=2 if cfg.train.num_workers > 0 else None,
        )
        dl_iter = iter(dl)

        def _run_step():
            try:
                batch = next(dl_iter)
            except StopIteration:
                batch = next(iter(dl))
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            for k in camera_keys:
                if k in batch and batch[k].dtype == torch.uint8:
                    batch[k] = batch[k].float() / 255.0
            batch = preprocessor(batch)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = policy.forward(batch)
                loss = out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        try:
            # warm-up (not timed)
            for _ in range(SWEEP_WARMUP):
                _run_step()
            torch.cuda.synchronize(device)

            # timed steps
            t0 = time.perf_counter()
            for _ in range(SWEEP_TIMED):
                _run_step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            peak = torch.cuda.max_memory_allocated(device)
            throughput = bs * SWEEP_TIMED / elapsed  # samples/sec
            log.info(f"    batch_size={bs}: peak VRAM={_gb(peak):.2f}GB  "
                     f"throughput={throughput:.1f} samples/sec  "
                     f"({elapsed/SWEEP_TIMED:.2f}s/step)")

            del policy, optimizer, scaler, dl, dl_iter
            gc.collect()
            torch.cuda.empty_cache()

            if peak > max_allowed:
                log.info(f"    Over VRAM limit — stopping.")
                break

            if throughput > best_throughput:
                best_throughput = throughput
                best_bs = bs
            else:
                # throughput is declining — stop searching larger batches
                log.info(f"    Throughput dropped vs batch_size={best_bs} "
                         f"({best_throughput:.1f} samples/sec) — stopping.")
                break

        except torch.cuda.OutOfMemoryError:
            log.info(f"    OOM at batch_size={bs} — stopping.")
            try:
                del policy, optimizer, scaler, dl, dl_iter
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            break

    log.info(f"=== Optimal batch_size: {best_bs} "
             f"({best_throughput:.1f} samples/sec) ===")
    save_cache(policy_type, repo_id, best_bs)


if __name__ == "__main__":
    main()
