#!/usr/bin/env python
"""
Phase 3 native training loop using RunContext and Hydra.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.observability import RunContext
import torch
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy_config, make_policy, make_pre_post_processors

log = logging.getLogger(__name__)

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
        
        log.info("Loading dataset...")
        dataset_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
        
        log.info("Creating policy config...")
        policy_kwargs = OmegaConf.to_container(cfg.policy, resolve=True)
        policy_type = policy_kwargs.pop("type")
        policy_cfg = make_policy_config(policy_type, **policy_kwargs)
        
        # Determine delta_timestamps for dataset from policy config
        fps = dataset_kwargs.get("fps", 10)  # assume 10 fps if not provided
        if hasattr(policy_cfg, "observation_delta_indices"):
            dataset_kwargs["delta_timestamps"] = {
                "action": [i / fps for i in policy_cfg.action_delta_indices],
                "observation.image": [i / fps for i in policy_cfg.observation_delta_indices],
                "observation.state": [i / fps for i in policy_cfg.observation_delta_indices],
            }
            
        dataset = LeRobotDataset(**dataset_kwargs)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.train.num_workers,
            pin_memory=True if "cuda" in cfg.device else False
        )
        
        log.info("Creating policy...")
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg,
            dataset_stats=dataset.meta.stats
        )
        
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        policy.to(device)
        policy.train()
        
        log.info("Setting up optimizer...")
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
        
        log.info(f"Starting training loop for {cfg.train.steps} steps...")
        dataloader_iter = iter(dataloader)
        
        for step in range(cfg.train.steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            # LeRobot usually requires images to be float32 in [0, 1]
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            
            # preprocess batch
            batch = preprocessor(batch)
            
            # forward
            output = policy.forward(batch)
            if isinstance(output, tuple):
                loss = output[0]
            elif isinstance(output, dict) and "loss" in output:
                loss = output["loss"]
            else:
                loss = output
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            if cfg.train.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip_norm)
                
            optimizer.step()
            
            # log step
            run.log_step(step, loss=loss.item())
            
            if torch.isnan(loss) or torch.isinf(loss):
                run.log_anomaly("NaN/Inf loss detected", step=step, loss=loss.item())
                raise RuntimeError(f"NaN/Inf loss at step {step}")
                
            if step > 0 and step % 100 == 0:
                log.info(f"Step {step}/{cfg.train.steps} - loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
