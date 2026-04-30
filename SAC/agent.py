import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256, action_space=None):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc_mean = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(hidden, act_dim), std=0.01)

        if action_space is None:
            action_scale = torch.ones(act_dim, dtype=torch.float32)
            action_bias = torch.zeros(act_dim, dtype=torch.float32)
        else:
            action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32
            )
            action_bias = torch.tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32
            )
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def deterministic(self, obs):
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim + act_dim, hidden))
        self.fc2 = layer_init(nn.Linear(hidden, hidden))
        self.fc3 = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACAgent(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256, action_space=None):
        super().__init__()
        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=hidden,
            action_space=action_space,
        )
        self.q1 = SoftQNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q2 = SoftQNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q1_target = SoftQNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.q2_target = SoftQNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        self.copy_targets()

    def copy_targets(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def soft_update(self, tau: float):
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def set_critics_grad(self, requires_grad: bool):
        for critic in (self.q1, self.q2):
            for param in critic.parameters():
                param.requires_grad_(requires_grad)

    def sample_action(self, obs):
        return self.actor.sample(obs)

    def deterministic(self, obs):
        return self.actor.deterministic(obs)
