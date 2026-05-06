"""Phase 2 smoke test: ManiSkill 3 GPU-parallel env.

Pass: 4 parallel PickCube-v1 envs step without error; reward shape is [4].
"""

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers envs
import torch

env = gym.make("PickCube-v1", num_envs=4, obs_mode="state", render_mode=None)
obs, _ = env.reset()
assert obs.shape == (4, 42), f"unexpected obs shape: {obs.shape}"

action = torch.zeros(env.action_space.shape)
obs, rew, _, _, _ = env.step(action)
assert rew.shape == (4,), f"unexpected reward shape: {rew.shape}"
env.close()

print("ManiSkill 3 smoke test PASSED — 4×PickCube-v1 GPU-parallel envs OK")
