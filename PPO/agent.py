import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import *

def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    LOGSTD_MIN, LOGSTD_MAX = -1.6, -0.3

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.full((1, act_dim), -0.5))

    def _dist(self, obs):
        mean = self.actor(obs)
        ls = self.log_std.expand_as(mean).clamp(self.LOGSTD_MIN, self.LOGSTD_MAX)
        return Normal(mean, ls.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act_value(self, obs, action=None):
        d = self._dist(obs)
        if action is None:
            action = d.sample()
        return action, d.log_prob(action).sum(-1), d.entropy().sum(-1), self.value(obs)

    def deterministic(self, obs):
        return self.actor(obs)
