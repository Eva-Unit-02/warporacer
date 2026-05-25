import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from .config import ACT_DIM, OBS_DIM
except ImportError:
    from config import ACT_DIM, OBS_DIM


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def _stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = torch.tanh(self.log_std(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._stats(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._stats(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SACAgent(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256):
        super().__init__()
        self.actor = SquashedGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q1 = QNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q2 = QNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor.sample(obs)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(obs)

    def make_target(self) -> "SACAgent":
        target = copy.deepcopy(self)
        for param in target.parameters():
            param.requires_grad_(False)
        return target
