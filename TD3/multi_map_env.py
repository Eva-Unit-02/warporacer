from __future__ import annotations

from pathlib import Path

import torch

try:
    from .racing_env import RacingEnv
except ImportError:
    from racing_env import RacingEnv


class MultiMapRacingEnv:
    def __init__(
        self,
        map_yamls: list[Path],
        num_envs: int,
        seed: int = 0,
        device: str | None = None,
    ):
        if not map_yamls:
            raise ValueError("At least one map must be provided")
        if num_envs < len(map_yamls):
            raise ValueError("num_envs must be at least the number of maps")

        self.map_yamls = [Path(path) for path in map_yamls]
        self.num_envs = int(num_envs)

        base = num_envs // len(self.map_yamls)
        extra = num_envs % len(self.map_yamls)
        self.env_counts = [base + (1 if idx < extra else 0) for idx in range(len(self.map_yamls))]

        self.envs: list[RacingEnv] = []
        for idx, (map_yaml, env_count) in enumerate(zip(self.map_yamls, self.env_counts)):
            if env_count <= 0:
                continue
            self.envs.append(
                RacingEnv(
                    map_yaml,
                    num_envs=env_count,
                    seed=seed + idx * 10_000,
                    device=device,
                )
            )

        self.device = self.envs[0].device
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self._splits = [env.num_envs for env in self.envs]

    def reset(self):
        obs_parts = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_parts.append(obs)
        return torch.cat(obs_parts, dim=0), {}

    def step(self, action: torch.Tensor):
        action_parts = torch.split(action, self._splits, dim=0)
        obs_parts = []
        reward_parts = []
        terminated_parts = []
        truncated_parts = []

        for env, env_action in zip(self.envs, action_parts):
            obs, reward, terminated, truncated, _ = env.step(env_action)
            obs_parts.append(obs)
            reward_parts.append(reward)
            terminated_parts.append(terminated)
            truncated_parts.append(truncated)

        return (
            torch.cat(obs_parts, dim=0),
            torch.cat(reward_parts, dim=0),
            torch.cat(terminated_parts, dim=0),
            torch.cat(truncated_parts, dim=0),
            {},
        )

    def save_state(self):
        return [env.save_state() for env in self.envs]

    def restore_state(self, snapshots):
        for env, snapshot in zip(self.envs, snapshots):
            env.restore_state(snapshot)
