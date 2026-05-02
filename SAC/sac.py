from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from config import *


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape).float()
        bv, bm = torch.var_mean(x, dim=0, unbiased=False)
        bc = x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta, alpha=bc / tot)
        self.var = (
            self.var * self.count + bv * bc + delta * delta * (self.count * bc / tot)
        ) / tot
        self.count = tot
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x, clip: float = 10.0):
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32)
        self.next_obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.empty((self.capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.terminated = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.truncated = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, actions, rewards, terminated, truncated):
        obs = obs.detach().to("cpu", dtype=torch.float32)
        next_obs = next_obs.detach().to("cpu", dtype=torch.float32)
        actions = actions.detach().to("cpu", dtype=torch.float32)
        rewards = rewards.detach().to("cpu", dtype=torch.float32).view(-1, 1)
        terminated = terminated.detach().to("cpu", dtype=torch.float32).view(-1, 1)
        truncated = truncated.detach().to("cpu", dtype=torch.float32).view(-1, 1)

        if obs.shape[0] > self.capacity:
            obs = obs[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            terminated = terminated[-self.capacity :]
            truncated = truncated[-self.capacity :]

        n = obs.shape[0]
        idx = (torch.arange(n) + self.pos) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.terminated[idx] = terminated
        self.truncated[idx] = truncated
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size):
        idx = torch.randint(self.size, (batch_size,))
        return ReplayBatch(
            observations=self.obs[idx].to(self.device),
            next_observations=self.next_obs[idx].to(self.device),
            actions=self.actions[idx].to(self.device),
            rewards=self.rewards[idx].to(self.device),
            terminated=self.terminated[idx].to(self.device),
            truncated=self.truncated[idx].to(self.device),
        )


def record_rollout(env, agent, num_steps, out_path, obs_rms=None):
    snap = env.save_state()
    actor_was_training = agent.actor.training
    agent.actor.eval()
    try:
        m = env.map
        corners = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        def w2p(x, y):
            return int((x - m.ox) / m.res), int(m.h - 1 - (y - m.oy) / m.res)

        trail = deque(maxlen=300)
        raw, _ = env.reset()
        obs = obs_rms.normalize(raw) if obs_rms else raw
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(out_path), fps=int(1 / DT), macro_block_size=1
        ) as writer:
            with torch.no_grad():
                for _ in range(num_steps):
                    action = agent.deterministic(obs)
                    raw, _, term, trunc, _ = env.step(action)
                    obs = obs_rms.normalize(raw) if obs_rms else raw
                    row = env.cars_buf[0].tolist()
                    x, y, psi = row[0], row[1], row[4]
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        trail.clear()
                    trail.append((x, y))
                    frame = cvtColor(m.raw, COLOR_GRAY2RGB)
                    if len(trail) > 1:
                        polylines(
                            frame,
                            [np.array([w2p(*p) for p in trail], dtype=np.int32)],
                            False,
                            (0, 200, 0),
                            2,
                        )
                    rot = np.array(
                        [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
                    )
                    world = corners @ rot.T + (x, y)
                    fillPoly(
                        frame,
                        [np.array([w2p(*p) for p in world], dtype=np.int32)],
                        (255, 50, 50),
                    )
                    writer.append_data(frame)
    finally:
        env.restore_state(snap)
        agent.actor.train(actor_was_training)


def train(
    env,
    agent,
    iterations,
    buffer_size,
    batch_size,
    gamma,
    tau,
    actor_lr,
    critic_lr,
    alpha_lr,
    learning_starts,
    policy_frequency,
    target_network_frequency, #decides how often target network for q network updates (copies of the current Q-networks that provide stable target values during training)
    updates_per_iter,
    autotune,
    alpha,
    log_dir,
    record_every,
    record_steps,
):
    device = next(agent.parameters()).device
    num_envs = env.num_envs
    obs_rms = RunningMeanStd((OBS_DIM,), device)
    replay = ReplayBuffer(buffer_size, OBS_DIM, ACT_DIM, device)

    critic_optimizer = torch.optim.Adam(
        list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=critic_lr
    )
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)

    if autotune:
        # target_entropy = -float(ACT_DIM)
        target_entropy = -1.0 # Set value
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha_value = log_alpha.exp().detach()
    else:
        target_entropy = None
        log_alpha = None
        alpha_optimizer = None
        alpha_value = torch.tensor(alpha, dtype=torch.float32, device=device)

    raw_obs, _ = env.reset()
    obs_rms.update(raw_obs)
    ep_ret = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    finished_rets, finished_lens = deque(maxlen=100), deque(maxlen=100)

    global_step = 0
    update_step = 0
    t0 = time.time()
    last_t = t0

    # TODO LEARN
    for it in range(iterations):
        with torch.no_grad():
            if global_step < learning_starts:
                actions = torch.empty((num_envs, ACT_DIM), device=device).uniform_(
                    -1.0, 1.0
                )
            else:
                norm_obs = obs_rms.normalize(raw_obs)
                actions, _, _ = agent.sample_action(norm_obs)

        next_raw_obs, rewards, term, trunc, _ = env.step(actions)
        terminated = term.float()
        truncated = trunc.float()

        replay.add(raw_obs, next_raw_obs, actions, rewards, terminated, truncated)

        ep_ret.add_(rewards)
        ep_len.add_(1.0)
        finished = (terminated > 0.0) | (truncated > 0.0)
        if finished.any():
            finished_rets.extend(ep_ret[finished].cpu().tolist())
            finished_lens.extend(ep_len[finished].cpu().tolist())
            ep_ret[finished] = 0.0
            ep_len[finished] = 0.0

        obs_rms.update(next_raw_obs)
        raw_obs = next_raw_obs
        global_step += num_envs

        gradient_steps = updates_per_iter or max(8, num_envs // max(batch_size, 1))
        stats = {
            "q1_loss": float("nan"),
            "q2_loss": float("nan"),
            "critic_loss": float("nan"),
            "actor_loss": float("nan"),
            "alpha_loss": float("nan"),
            "q1": float("nan"),
            "q2": float("nan"),
            "log_pi": float("nan"),
        }

        if global_step >= learning_starts and replay.size >= batch_size:
            for _ in range(gradient_steps):
                batch = replay.sample(batch_size)
                obs = obs_rms.normalize(batch.observations)
                next_obs = obs_rms.normalize(batch.next_observations)

                with torch.no_grad():
                    next_actions, next_log_pi, _ = agent.sample_action(next_obs)
                    q1_next = agent.q1_target(next_obs, next_actions)
                    q2_next = agent.q2_target(next_obs, next_actions)
                    min_q_next = torch.min(q1_next, q2_next) - alpha_value * next_log_pi
                    discount_mask = 1.0 - torch.maximum(
                        batch.terminated, batch.truncated
                    )
                    target_q = batch.rewards + discount_mask * gamma * min_q_next

                q1_pred = agent.q1(obs, batch.actions)
                q2_pred = agent.q2(obs, batch.actions)
                q1_loss = F.mse_loss(q1_pred, target_q)
                q2_loss = F.mse_loss(q2_pred, target_q)
                critic_loss = q1_loss + q2_loss

                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()
            

                update_step += 1

                if update_step % policy_frequency == 0:
                    agent.set_critics_grad(False)
                    pi, log_pi, _ = agent.sample_action(obs)
                    q1_pi = agent.q1(obs, pi)
                    q2_pi = agent.q2(obs, pi)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha_value * log_pi - min_q_pi).mean()

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, log_pi_alpha, _ = agent.sample_action(obs)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi_alpha + target_entropy)
                        ).mean()
                        alpha_optimizer.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha_value = log_alpha.exp().detach()
                        stats["alpha_loss"] = alpha_loss.item()

                    agent.set_critics_grad(True)

                    stats["actor_loss"] = actor_loss.item()
                    stats["log_pi"] = log_pi.mean().item()

                if update_step % target_network_frequency == 0:
                    agent.soft_update(tau)

                stats["q1_loss"] = q1_loss.item()
                stats["q2_loss"] = q2_loss.item()
                stats["critic_loss"] = critic_loss.item() * 0.5
                stats["q1"] = q1_pred.mean().item()
                stats["q2"] = q2_pred.mean().item()

        now = time.time()
        sps = int(num_envs / max(now - last_t, 1e-9))
        last_t = now
        log = {
            "critic_loss": stats["critic_loss"],
            "q1_loss": stats["q1_loss"],
            "q2_loss": stats["q2_loss"],
            "actor_loss": stats["actor_loss"],
            "alpha_loss": stats["alpha_loss"],
            "alpha": float(alpha_value.item()),
            "q1": stats["q1"],
            "q2": stats["q2"],
            "log_pi": stats["log_pi"],
            "replay_size": replay.size,
            "sps": sps,
            "global_step": global_step,
            "iteration": it,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))
        try:
            wandb.log(log, step=global_step)
        except Exception:
            pass

        if it % 10 == 0:
            er = log.get("ep_return", float("nan"))
            print(
                f"[it {it:4d}] step={global_step:>9d} sps={sps:>6d} "
                f"ret={er:8.2f} alpha={float(alpha_value.item()):.3f} "
                f"q={stats['q1']:.2f}/{stats['q2']:.2f}"
            )

        if record_every > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
                wandb.log(
                    {"rollout": wandb.Video(str(out), format="mp4")}, step=global_step
                )
            except Exception as e:
                print(f"[rollout {it + 1}] failed: {e}")

    alpha_scalar = float(alpha_value.item())
    log_alpha_cpu = log_alpha.detach().cpu() if log_alpha is not None else None
    return time.time() - t0, obs_rms, global_step, alpha_scalar, log_alpha_cpu
