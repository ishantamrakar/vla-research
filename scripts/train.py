#!/usr/bin/env python
"""
Long training run with AMP, checkpointing, and optional eval/video recording.
Reads optimal batch size from outputs/sweep_cache.json (run sweep.py first).

Usage:
    uv run python scripts/train.py policy=act env=libero_spatial \
        train.steps=50000 experiment_name=act_libero_spatial_50k wandb.enable=true
"""
import json
import logging
import shutil

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs import make_env, make_env_pre_post_processors, EnvConfig
from lerobot.policies import make_policy_config, make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

from src.observability import RunContext

log = logging.getLogger(__name__)

SWEEP_CACHE = Path("outputs/sweep_cache.json")


def _gb(n: int) -> float:
    return n / 1024**3


def _load_optimal_batch_size(policy_type: str, repo_id: str) -> int | None:
    if not SWEEP_CACHE.exists():
        return None
    try:
        return json.loads(SWEEP_CACHE.read_text()).get(f"{policy_type}|{repo_id}")
    except Exception:
        return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir)
    device = torch.device(cfg.device)

    with RunContext(
        run_id=f"run_{cfg.experiment_name}",
        script=__file__,
        config=OmegaConf.to_container(cfg, resolve=True),
        output_dir=output_dir,
        steps_requested=cfg.train.steps,
        device=cfg.device,
    ) as run:

        use_wandb = cfg.get("wandb", {}).get("enable", False)
        if use_wandb:
            wandb.init(
                project=cfg.wandb.get("project", "vla-research"),
                entity=cfg.wandb.get("entity", None),
                name=cfg.wandb.get("name", cfg.experiment_name),
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=output_dir,
            )

        # ---- dataset & policy config ---------------------------------------
        log.info("Loading dataset...")
        dataset_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
        repo_id = dataset_kwargs["repo_id"]

        policy_kwargs = OmegaConf.to_container(cfg.policy, resolve=True)
        policy_type = policy_kwargs.pop("type")
        policy_cfg = make_policy_config(policy_type, **policy_kwargs)

        dataset = LeRobotDataset(**dataset_kwargs)
        delta_timestamps = resolve_delta_timestamps(policy_cfg, dataset.meta)
        if delta_timestamps:
            dataset_kwargs["delta_timestamps"] = delta_timestamps
            dataset = LeRobotDataset(**dataset_kwargs)

        # ---- env -----------------------------------------------------------
        log.info("Setting up environment...")
        env_kwargs = OmegaConf.to_container(cfg.get("env", {}), resolve=True)
        # Store env config so we can recreate environments fresh for each eval
        # (LIBERO's OffScreenRenderEnv.close() deletes self.env, corrupting reuse).
        eval_env_cfg = None
        env_preprocessor = env_postprocessor = None

        if env_kwargs:
            env_type = env_kwargs.pop("type")
            eval_env_cfg = EnvConfig.get_choice_class(env_type)(**env_kwargs)
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(eval_env_cfg, policy_cfg)

        # ---- preprocessors & policy ----------------------------------------
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg, dataset_stats=dataset.meta.stats
        )
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        policy.to(device)

        # ---- batch size & LR -----------------------------------------------
        cached_bs = _load_optimal_batch_size(policy_type, repo_id)
        if cached_bs is not None:
            batch_size = cached_bs
            log.info(f"Using cached batch_size={batch_size} (from sweep.py)")
        else:
            batch_size = cfg.train.batch_size
            log.warning(
                f"No sweep cache found for {policy_type}|{repo_id}. "
                f"Using cfg.train.batch_size={batch_size}. "
                "Run sweep.py first to maximize GPU utilization."
            )

        # Adam's adaptive second moment absorbs batch-size variance, so the
        # linear scaling rule (from SGD theory) doesn't apply here. Use the
        # base LR directly regardless of batch size.
        scaled_lr = cfg.optimizer.lr
        log.info(f"batch_size={batch_size}  lr={scaled_lr:.2e} (no LR scaling for AdamW)")

        policy.train()
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=scaled_lr, weight_decay=cfg.optimizer.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda")

        # ---- resume --------------------------------------------------------
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_ckpt = checkpoint_dir / "checkpoint_latest.pt"
        start_step = 0

        if latest_ckpt.exists():
            log.info(f"Resuming from {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            policy.load_state_dict(ckpt["policy"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt["step"] + 1
            log.info(f"Resumed at step {start_step}")

        # ---- dataloader ----------------------------------------------------
        pin = "cuda" in cfg.device
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=cfg.train.num_workers, pin_memory=pin,
            persistent_workers=cfg.train.num_workers > 0,
        )
        dl_iter = iter(dl)
        camera_keys = dataset.meta.camera_keys
        grad_clip = cfg.train.grad_clip_norm
        ckpt_interval = cfg.train.get("checkpoint_interval", 5000)
        render_videos = cfg.train.get("render_eval_videos", False)

        log.info(f"Starting training: steps={cfg.train.steps}, "
                 f"batch={batch_size}, lr={scaled_lr:.2e}, AMP=on")

        # ---- training loop -------------------------------------------------
        for step in range(start_step, cfg.train.steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

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

            loss_val = loss.item()
            run.log_step(step, loss=loss_val)
            if use_wandb:
                wandb.log({"train/loss": loss_val, "train/batch_size": batch_size}, step=step)

            if torch.isnan(loss) or torch.isinf(loss):
                run.log_anomaly("NaN/Inf loss", step=step, loss=loss_val)
                raise RuntimeError(f"NaN/Inf loss at step {step}")

            if step % 100 == 0:
                peak_gb = _gb(torch.cuda.max_memory_allocated(device))
                log.info(f"Step {step}/{cfg.train.steps}  loss={loss_val:.4f}  VRAM={peak_gb:.2f}GB")

            # ---- checkpoint ------------------------------------------------
            if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                ckpt_data = {
                    "step": step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "batch_size": batch_size,
                    "lr": scaled_lr,
                }
                numbered = checkpoint_dir / f"checkpoint_{step + 1:07d}.pt"
                torch.save(ckpt_data, numbered)
                shutil.copy2(numbered, latest_ckpt)
                log.info(f"Checkpoint saved: {numbered.name}")
                if use_wandb:
                    wandb.log({"checkpoint/step": step + 1}, step=step)

            # ---- eval ------------------------------------------------------
            if (
                eval_env_cfg is not None
                and cfg.eval.eval_freq > 0
                and ((step + 1) % cfg.eval.eval_freq == 0 or step == cfg.train.steps - 1)
            ):
                log.info(f"Evaluating at step {step + 1}...")
                videos_dir = output_dir / f"eval/step_{step + 1}" if render_videos else None
                max_rendered = cfg.eval.n_episodes if render_videos else 0

                # Recreate env fresh — LIBERO's close() deletes internal state,
                # so reusing env objects across evals causes AttributeError on seed().
                eval_env = make_env(eval_env_cfg, n_envs=cfg.eval.batch_size)
                try:
                    policy.eval()
                    with torch.no_grad():
                        eval_info = eval_policy_all(
                            envs=eval_env,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=cfg.eval.n_episodes,
                            max_episodes_rendered=max_rendered,
                            videos_dir=videos_dir,
                            start_seed=cfg.seed,
                        )
                    avg_reward = eval_info["overall"]["avg_sum_reward"]
                    pc_success = eval_info["overall"]["pc_success"]
                    log.info(f"Eval step {step + 1}: reward={avg_reward:.2f}  success={pc_success:.1f}%")
                    run.log_step(step, eval_reward=avg_reward, eval_success=pc_success)
                    if use_wandb:
                        wandb.log({"eval/reward": avg_reward, "eval/success": pc_success}, step=step)
                except Exception as e:
                    log.warning(f"Eval at step {step + 1} failed (training unaffected): {e}")
                finally:
                    policy.train()
                    try:
                        eval_env.close()
                    except Exception:
                        pass

        # ---- final checkpoint ----------------------------------------------
        final_ckpt = checkpoint_dir / f"checkpoint_{cfg.train.steps:07d}_final.pt"
        torch.save({
            "step": cfg.train.steps - 1,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "batch_size": batch_size,
            "lr": scaled_lr,
        }, final_ckpt)
        shutil.copy2(final_ckpt, latest_ckpt)
        log.info(f"Final checkpoint: {final_ckpt}")

        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
