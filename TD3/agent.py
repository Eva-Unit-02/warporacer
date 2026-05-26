from __future__ import annotations

import copy

import torch
import torch.nn as nn

try:
    from .config import ACT_DIM, OBS_DIM
except ImportError:
    from config import ACT_DIM, OBS_DIM


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.network = build_mlp(obs_dim, hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.network = build_mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        merged = torch.cat((obs, action), dim=-1)
        return self.network(merged).squeeze(-1)


class TD3Agent(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.actor = DeterministicActor(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q1 = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q2 = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def q_values(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def target_copy(self) -> "TD3Agent":
        target = copy.deepcopy(self)
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        return target
