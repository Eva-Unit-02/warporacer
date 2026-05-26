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
        self.discounts = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.cursor = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        discounts: torch.Tensor,
    ):
        batch = obs.shape[0]
        indices = (torch.arange(batch, device=self.obs.device) + self.cursor) % self.capacity
        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_obs[indices] = next_obs
        self.dones[indices] = dones
        self.discounts[indices] = discounts
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
            "discounts": self.discounts[indices],
        }


class NStepBuilder:
    def __init__(self, num_envs: int, obs_dim: int, act_dim: int, n_step: int, gamma: float, device):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_step = max(1, int(n_step))
        self.gamma = gamma
        self.device = device
        self.obs = torch.zeros((self.n_step, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.n_step, num_envs, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.n_step, num_envs), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((self.n_step, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.n_step, num_envs), dtype=torch.float32, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.gamma_powers = gamma ** torch.arange(self.n_step, dtype=torch.float32, device=device)
        self.length_discounts = gamma ** torch.arange(self.n_step + 1, dtype=torch.float32, device=device)

    def append(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        if self.n_step > 1:
            self.obs[:-1].copy_(self.obs[1:].clone())
            self.actions[:-1].copy_(self.actions[1:].clone())
            self.rewards[:-1].copy_(self.rewards[1:].clone())
            self.next_obs[:-1].copy_(self.next_obs[1:].clone())
            self.dones[:-1].copy_(self.dones[1:].clone())

        self.obs[-1] = obs
        self.actions[-1] = actions
        self.rewards[-1] = rewards
        self.next_obs[-1] = next_obs
        self.dones[-1] = dones
        self.lengths.clamp_(max=self.n_step - 1).add_(1)
        self.ready = self.ready | (self.lengths >= self.n_step) | dones.bool()

    def pop_ready(self):
        ready = self.ready
        if not ready.any():
            return None

        lengths = self.lengths.clamp(min=1)
        oldest_rows = self.n_step - lengths
        env_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        out_obs = self.obs[oldest_rows[env_ids], env_ids]
        out_actions = self.actions[oldest_rows[env_ids], env_ids]

        rows = torch.arange(self.n_step, device=self.device).view(-1, 1)
        start_rows = oldest_rows.view(1, -1)
        local_steps = (rows - start_rows).clamp(min=0, max=self.n_step - 1)
        active = rows >= start_rows
        weights = self.gamma_powers[local_steps]
        rewards = (self.rewards * weights * active).sum(dim=0)

        active_dones = self.dones * active
        done_cumsum = active_dones.cumsum(dim=0)
        first_done_mask = (active_dones > 0.0) & (done_cumsum == 1.0)
        first_done_any = first_done_mask.any(dim=0)
        bootstrap_row = torch.where(
            first_done_any,
            first_done_mask.float().argmax(dim=0),
            self.n_step - 1,
        )

        out_next_obs = self.next_obs[bootstrap_row[env_ids], env_ids]
        out_rewards = rewards[env_ids]
        out_dones = first_done_any[env_ids].float()
        out_discounts = torch.where(
            first_done_any,
            torch.zeros_like(self.length_discounts[lengths]),
            self.length_discounts[lengths],
        )[env_ids]

        done_envs = self.dones[bootstrap_row[env_ids], env_ids] > 0.0
        self.ready[env_ids] = False
        self.lengths[env_ids] = torch.where(
            done_envs,
            torch.zeros_like(self.lengths[env_ids]),
            (self.lengths[env_ids] - 1).clamp_min(0),
        )
        self.ready[env_ids[~done_envs]] = self.lengths[env_ids[~done_envs]] >= self.n_step

        return out_obs, out_actions, out_rewards, out_next_obs, out_dones, out_discounts


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
                    height, width = frame.shape[:2]
                    if height % 2 != 0 or width % 2 != 0:
                        padded = np.zeros((height + height % 2, width + width % 2, 3), dtype=frame.dtype)
                        padded[:height, :width] = frame
                        frame = padded
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
    n_step: int = 5,
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
    record_env=None,
):
    device = next(agent.parameters()).device
    num_envs = env.num_envs

    obs_rms = RunningStats((OBS_DIM,), device)
    replay = ReplayBuffer(buffer_size, OBS_DIM, ACT_DIM, device)
    n_step_builder = NStepBuilder(num_envs, OBS_DIM, ACT_DIM, n_step, gamma, device)
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
            n_step_builder.append(transition_obs, action, reward, next_raw_obs, done)
            n_step_batch = n_step_builder.pop_ready()
            if n_step_batch is not None:
                replay.add(*n_step_batch)

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
                batch_discounts = batch["discounts"]

                with torch.no_grad():
                    smoothing = (torch.randn_like(batch_actions) * target_policy_noise).clamp(
                        -target_noise_clip, target_noise_clip
                    )
                    next_action = (target.act(batch_next_obs) + smoothing).clamp(-1.0, 1.0)
                    target_q1, target_q2 = target.q_values(batch_next_obs, next_action)
                    clipped_target_q = torch.minimum(target_q1, target_q2)
                    target_q = batch_rewards + batch_discounts * (1.0 - batch_dones) * clipped_target_q

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
            "n_step": n_step,
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
                rollout_env = env if record_env is None else record_env
                record_rollout(rollout_env, agent, record_steps, video_path, obs_rms=obs_rms)
                wandb.log({"rollout": wandb.Video(str(video_path), format="mp4")}, step=env_steps)
            except Exception as exc:
                print(f"[rollout {iteration + 1}] failed: {exc}")

    elapsed = time.time() - started_at
    return elapsed, obs_rms, env_steps, critic_updates, actor_updates, target
