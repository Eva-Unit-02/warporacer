from __future__ import annotations

from collections import deque
from pathlib import Path
import math
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines

try:
    from .agent import TD3Agent
    from .config import ACT_DIM, CAR_LENGTH, CAR_WIDTH, DT, OBS_DIM
except ImportError:
    from agent import TD3Agent
    from config import ACT_DIM, CAR_LENGTH, CAR_WIDTH, DT, OBS_DIM


class RunningStats:
    def __init__(self, shape, device: torch.device | str):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, value: torch.Tensor):
        flat = value.reshape(-1, *self.mean.shape).float()
        batch_var, batch_mean = torch.var_mean(flat, dim=0, unbiased=False)
        batch_count = flat.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean.add_(delta, alpha=batch_count / total)
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * (self.count * batch_count / total)
        ) / total
        self.count = total
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, value: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return ((value - self.mean) * self.inv_std).clamp(-clip, clip)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device | str):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.cursor = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch = obs.shape[0]
        indices = (torch.arange(batch, device=self.obs.device) + self.cursor) % self.capacity
        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_obs[indices] = next_obs
        self.dones[indices] = dones
        self.cursor = (self.cursor + batch) % self.capacity
        self.size = min(self.size + batch, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.obs.device)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }


def soft_update(target: TD3Agent, source: TD3Agent, tau: float):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.lerp_(source_param, tau)


def record_rollout(env, agent, num_steps: int, out_path: Path, obs_rms: RunningStats | None = None):
    snapshot = env.save_state()
    was_training = agent.training
    agent.eval()
    try:
        track = env.map
        car_corners = np.array(
            [
                [-CAR_LENGTH * 0.5, -CAR_WIDTH * 0.5],
                [CAR_LENGTH * 0.5, -CAR_WIDTH * 0.5],
                [CAR_LENGTH * 0.5, CAR_WIDTH * 0.5],
                [-CAR_LENGTH * 0.5, CAR_WIDTH * 0.5],
            ]
        )

        def world_to_pixel(x: float, y: float) -> tuple[int, int]:
            col = int((x - track.origin_x) / track.resolution)
            row = int(track.height - 1 - (y - track.origin_y) / track.resolution)
            return col, row

        trail = deque(maxlen=300)
        raw_obs, _ = env.reset()
        obs = obs_rms.normalize(raw_obs) if obs_rms else raw_obs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(out_path), fps=int(1 / DT), macro_block_size=1) as writer:
            with torch.no_grad():
                for _ in range(num_steps):
                    action = agent.act(obs)
                    raw_obs, _, terminated, truncated, _ = env.step(action)
                    obs = obs_rms.normalize(raw_obs) if obs_rms else raw_obs

                    x = env.car_state_torch[0, 0].item()
                    y = env.car_state_torch[0, 1].item()
                    heading = env.car_state_torch[0, 4].item()
                    if bool(terminated[0].item()) or bool(truncated[0].item()):
                        trail.clear()
                    trail.append((x, y))

                    frame = cvtColor(track.raw, COLOR_GRAY2RGB)
                    if len(trail) > 1:
                        polyline = np.array([world_to_pixel(px, py) for px, py in trail], dtype=np.int32)
                        polylines(frame, [polyline], False, (0, 200, 0), 2)

                    rotation = np.array(
                        [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
                    )
                    polygon = car_corners @ rotation.T + np.array([x, y])
                    fillPoly(
                        frame,
                        [np.array([world_to_pixel(px, py) for px, py in polygon], dtype=np.int32)],
                        (255, 50, 50),
                    )
                    writer.append_data(frame)
    finally:
        env.restore_state(snapshot)
        agent.train(was_training)


def train(
    env,
    agent,
    iterations: int = 2000,
    buffer_size: int = 262_144,
    batch_size: int = 1024,
    learning_starts: int = 16_384,
    utd: float = 1.0,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    policy_delay: int = 2,
    log_dir: Path = Path("./logs_td3"),
    record_every: int = 250,
    record_steps: int = 1800,
):
    device = next(agent.parameters()).device
    num_envs = env.num_envs

    obs_rms = RunningStats((OBS_DIM,), device)
    replay = ReplayBuffer(buffer_size, OBS_DIM, ACT_DIM, device)
    target = agent.target_copy().to(device)

    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(
        list(agent.q1.parameters()) + list(agent.q2.parameters()),
        lr=critic_lr,
    )

    raw_obs, _ = env.reset()
    obs_rms.update(raw_obs)
    obs = obs_rms.normalize(raw_obs)

    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[float] = deque(maxlen=100)

    env_steps = 0
    critic_updates = 0
    actor_updates = 0
    started_at = time.time()
    last_log_at = started_at

    for iteration in range(iterations):
        with torch.no_grad():
            transition_obs = raw_obs.clone()
            if env_steps < learning_starts:
                action = torch.empty((num_envs, ACT_DIM), device=device).uniform_(-1.0, 1.0)
            else:
                action = agent.act(obs)
                action = (action + torch.randn_like(action) * exploration_noise).clamp(-1.0, 1.0)

            next_raw_obs, reward, terminated, truncated, _ = env.step(action)
            done = (terminated | truncated).to(dtype=torch.float32)
            replay.add(transition_obs, action, reward, next_raw_obs, done)

            episode_returns.add_(reward)
            episode_lengths.add_(1.0)
            finished = done.bool()
            if finished.any():
                recent_returns.extend(episode_returns[finished].detach().cpu().tolist())
                recent_lengths.extend(episode_lengths[finished].detach().cpu().tolist())
                episode_returns[finished] = 0.0
                episode_lengths[finished] = 0.0

            obs_rms.update(next_raw_obs)
            raw_obs = next_raw_obs
            obs = obs_rms.normalize(raw_obs)
            env_steps += num_envs

        stats = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "target_q": 0.0,
            "target_action_std": 0.0,
            "policy_action_std": 0.0,
        }
        local_updates = 0
        local_actor_updates = 0

        if replay.size >= max(batch_size, learning_starts):
            updates_per_iter = max(1, int(math.ceil(utd * num_envs / batch_size)))
            for _ in range(updates_per_iter):
                batch = replay.sample(batch_size)
                batch_obs = obs_rms.normalize(batch["obs"])
                batch_next_obs = obs_rms.normalize(batch["next_obs"])
                batch_actions = batch["actions"]
                batch_rewards = batch["rewards"]
                batch_dones = batch["dones"]

                with torch.no_grad():
                    smoothing = (torch.randn_like(batch_actions) * target_policy_noise).clamp(
                        -target_noise_clip, target_noise_clip
                    )
                    next_action = (target.act(batch_next_obs) + smoothing).clamp(-1.0, 1.0)
                    target_q1, target_q2 = target.q_values(batch_next_obs, next_action)
                    clipped_target_q = torch.minimum(target_q1, target_q2)
                    target_q = batch_rewards + gamma * (1.0 - batch_dones) * clipped_target_q

                q1, q2 = agent.q_values(batch_obs, batch_actions)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()

                stats["critic_loss"] += critic_loss.item()
                stats["q1"] += q1.mean().item()
                stats["q2"] += q2.mean().item()
                stats["target_q"] += target_q.mean().item()
                stats["target_action_std"] += next_action.std(dim=0).mean().item()

                critic_updates += 1
                local_updates += 1

                if critic_updates % policy_delay == 0:
                    policy_action = agent.act(batch_obs)
                    actor_loss = -agent.q1(batch_obs, policy_action).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()

                    soft_update(target, agent, tau)

                    stats["actor_loss"] += actor_loss.item()
                    stats["policy_action_std"] += policy_action.std(dim=0).mean().item()
                    actor_updates += 1
                    local_actor_updates += 1

        if local_updates > 0:
            stats["critic_loss"] /= local_updates
            stats["q1"] /= local_updates
            stats["q2"] /= local_updates
            stats["target_q"] /= local_updates
            stats["target_action_std"] /= local_updates
        if local_actor_updates > 0:
            stats["actor_loss"] /= local_actor_updates
            stats["policy_action_std"] /= local_actor_updates

        now = time.time()
        sps = int(num_envs / max(now - last_log_at, 1e-9))
        last_log_at = now

        log_data = {
            "iteration": iteration,
            "env_steps": env_steps,
            "buffer_size": replay.size,
            "buffer_fraction": replay.size / replay.capacity,
            "critic_updates": critic_updates,
            "actor_updates": actor_updates,
            "critic_loss": stats["critic_loss"],
            "actor_loss": stats["actor_loss"],
            "q1": stats["q1"],
            "q2": stats["q2"],
            "target_q": stats["target_q"],
            "target_action_std": stats["target_action_std"],
            "policy_action_std": stats["policy_action_std"],
            "exploration_noise": exploration_noise,
            "target_policy_noise": target_policy_noise,
            "target_noise_clip": target_noise_clip,
            "policy_delay": policy_delay,
            "obs_rms_count": obs_rms.count,
            "sps": sps,
        }
        if recent_returns:
            log_data["ep_return"] = float(np.mean(recent_returns))
            log_data["ep_length"] = float(np.mean(recent_lengths))

        try:
            wandb.log(log_data, step=env_steps)
        except Exception:
            pass

        if iteration % 10 == 0:
            mean_return = log_data.get("ep_return", float("nan"))
            print(
                f"[it {iteration:4d}] step={env_steps:>9d} sps={sps:>6d} "
                f"ret={mean_return:8.2f} critic={stats['critic_loss']:.4f} "
                f"actor={stats['actor_loss']:.4f} q={stats['q1']:.2f}/{stats['q2']:.2f}"
            )

        if record_every > 0 and (iteration + 1) % record_every == 0:
            video_path = log_dir / f"rollout_iter{iteration + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, video_path, obs_rms=obs_rms)
                wandb.log({"rollout": wandb.Video(str(video_path), format="mp4")}, step=env_steps)
            except Exception as exc:
                print(f"[rollout {iteration + 1}] failed: {exc}")

    elapsed = time.time() - started_at
    return elapsed, obs_rms, env_steps, critic_updates, actor_updates, target
