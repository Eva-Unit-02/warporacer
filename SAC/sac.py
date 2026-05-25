from collections import deque
from pathlib import Path
import time

from cv2 import COLOR_GRAY2RGB, cvtColor, fillPoly, polylines
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb

try:
    from .agent import SACAgent
    from .config import ACT_DIM, DT, LENGTH, OBS_DIM, WIDTH
except ImportError:
    from agent import SACAgent
    from config import ACT_DIM, DT, LENGTH, OBS_DIM, WIDTH


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.inv_std = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        x = x.reshape(-1, *self.mean.shape).float()
        batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean.add_(delta, alpha=batch_count / total_count)
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * (self.count * batch_count / total_count)
        ) / total_count
        self.count = total_count
        self.inv_std = torch.rsqrt(self.var + 1e-8)

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return ((x - self.mean) * self.inv_std).clamp(-clip, clip)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch_size = obs.shape[0]
        idx = (torch.arange(batch_size, device=self.obs.device) + self.pos) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


def soft_update(target: SACAgent, source: SACAgent, tau: float):
    with torch.no_grad():
        for target_param, param in zip(target.q1.parameters(), source.q1.parameters()):
            target_param.lerp_(param, tau)
        for target_param, param in zip(target.q2.parameters(), source.q2.parameters()):
            target_param.lerp_(param, tau)


def record_rollout(env, agent, num_steps, out_path, obs_rms=None):
    snap = env.save_state()
    was_training = agent.training
    agent.eval()
    try:
        track = env.map
        corners = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        def world_to_px(x, y):
            return int((x - track.ox) / track.res), int(track.h - 1 - (y - track.oy) / track.res)

        trail = deque(maxlen=300)
        raw_obs, _ = env.reset()
        obs = obs_rms.normalize(raw_obs) if obs_rms else raw_obs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(out_path), fps=int(1 / DT), macro_block_size=1) as writer:
            with torch.no_grad():
                for _ in range(num_steps):
                    action = agent.deterministic(obs)
                    raw_obs, _, term, trunc, _ = env.step(action)
                    obs = obs_rms.normalize(raw_obs) if obs_rms else raw_obs
                    x, y, psi = env.cars_buf[0, 0].item(), env.cars_buf[0, 1].item(), env.cars_buf[0, 4].item()
                    if bool(term[0].item()) or bool(trunc[0].item()):
                        trail.clear()
                    trail.append((x, y))
                    frame = cvtColor(track.raw, COLOR_GRAY2RGB)
                    if len(trail) > 1:
                        polylines(
                            frame,
                            [np.array([world_to_px(*point) for point in trail], dtype=np.int32)],
                            False,
                            (0, 200, 0),
                            2,
                        )
                    rot = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
                    car_poly = corners @ rot.T + (x, y)
                    fillPoly(
                        frame,
                        [np.array([world_to_px(*point) for point in car_poly], dtype=np.int32)],
                        (255, 50, 50),
                    )
                    writer.append_data(frame)
    finally:
        env.restore_state(snap)
        agent.train(was_training)


def train(
    env,
    agent,
    iterations=2000,
    buffer_size=262_144,
    batch_size=1024,
    gamma=0.99,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    learning_starts=16_384,
    utd=1.0,
    target_entropy=None,
    log_dir=Path("./logs_sac"),
    record_every=250,
    record_steps=1800,
):
    device = next(agent.parameters()).device
    num_envs = env.num_envs
    obs_rms = RunningMeanStd((OBS_DIM,), device)
    replay = ReplayBuffer(buffer_size, OBS_DIM, ACT_DIM, device)
    target = agent.make_target().to(device)

    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(
        list(agent.q1.parameters()) + list(agent.q2.parameters()),
        lr=critic_lr,
    )
    log_alpha = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=alpha_lr)
    target_entropy = float(-ACT_DIM if target_entropy is None else target_entropy)

    raw_obs, _ = env.reset()
    obs_rms.update(raw_obs)
    obs = obs_rms.normalize(raw_obs)
    ep_ret = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    finished_rets, finished_lens = deque(maxlen=100), deque(maxlen=100)

    global_step = 0
    total_updates = 0
    last_t = time.time()
    start_t = last_t

    for it in range(iterations):
        with torch.no_grad():
            if global_step < learning_starts:
                action = torch.empty((num_envs, ACT_DIM), device=device).uniform_(-1.0, 1.0)
            else:
                action, _, _ = agent.sample(obs)
            raw_next_obs, reward, term, trunc, _ = env.step(action)
            done = (term | trunc).float()
            replay.add(raw_obs, action, reward, raw_next_obs, done)

            ep_ret.add_(reward)
            ep_len.add_(1.0)
            finished = done.bool()
            if finished.any():
                finished_rets.extend(ep_ret[finished].detach().cpu().tolist())
                finished_lens.extend(ep_len[finished].detach().cpu().tolist())
                ep_ret[finished] = 0.0
                ep_len[finished] = 0.0

            obs_rms.update(raw_next_obs)
            raw_obs = raw_next_obs
            obs = obs_rms.normalize(raw_obs)
            global_step += num_envs

        stats = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": log_alpha.detach().exp().item(),
            "q1": 0.0,
            "q2": 0.0,
            "target_q": 0.0,
        }
        update_count = 0

        if replay.size >= max(batch_size, learning_starts):
            updates_per_iter = max(1, int(round(utd * num_envs / batch_size)))
            for _ in range(updates_per_iter):
                batch = replay.sample(batch_size)
                batch_obs = obs_rms.normalize(batch["obs"])
                batch_next_obs = obs_rms.normalize(batch["next_obs"])
                batch_actions = batch["actions"]
                batch_rewards = batch["rewards"]
                batch_dones = batch["dones"]

                with torch.no_grad():
                    next_actions, next_logp, _ = agent.sample(batch_next_obs)
                    next_q1 = target.q1(batch_next_obs, next_actions)
                    next_q2 = target.q2(batch_next_obs, next_actions)
                    alpha = log_alpha.exp()
                    next_v = torch.min(next_q1, next_q2) - alpha * next_logp.squeeze(-1)
                    target_q = batch_rewards + gamma * (1.0 - batch_dones) * next_v

                q1 = agent.q1(batch_obs, batch_actions)
                q2 = agent.q2(batch_obs, batch_actions)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_opt.step()

                new_actions, logp, _ = agent.sample(batch_obs)
                q1_pi = agent.q1(batch_obs, new_actions)
                q2_pi = agent.q2(batch_obs, new_actions)
                q_pi = torch.min(q1_pi, q2_pi)
                alpha = log_alpha.exp()
                actor_loss = (alpha.detach() * logp.squeeze(-1) - q_pi).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(log_alpha * (logp.detach().squeeze(-1) + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

                soft_update(target, agent, tau)
                total_updates += 1
                update_count += 1

                stats["critic_loss"] += critic_loss.item()
                stats["actor_loss"] += actor_loss.item()
                stats["alpha_loss"] += alpha_loss.item()
                stats["alpha"] = log_alpha.detach().exp().item()
                stats["q1"] += q1.mean().item()
                stats["q2"] += q2.mean().item()
                stats["target_q"] += target_q.mean().item()

        if update_count > 0:
            for key in ("critic_loss", "actor_loss", "alpha_loss", "q1", "q2", "target_q"):
                stats[key] /= update_count

        now = time.time()
        sps = int(num_envs / max(now - last_t, 1e-9))
        last_t = now

        log = {
            "iteration": it,
            "env_steps": global_step,
            "buffer_size": replay.size,
            "gradient_steps": total_updates,
            "critic_loss": stats["critic_loss"],
            "actor_loss": stats["actor_loss"],
            "alpha_loss": stats["alpha_loss"],
            "alpha": stats["alpha"],
            "q1": stats["q1"],
            "q2": stats["q2"],
            "target_q": stats["target_q"],
            "sps": sps,
        }
        if finished_rets:
            log["ep_return"] = float(np.mean(finished_rets))
            log["ep_length"] = float(np.mean(finished_lens))
        try:
            wandb.log(log, step=global_step)
        except Exception:
            pass

        if it % 10 == 0:
            ret = log.get("ep_return", float("nan"))
            print(
                f"[it {it:4d}] step={global_step:>9d} sps={sps:>6d} "
                f"ret={ret:8.2f} alpha={stats['alpha']:.3f} "
                f"critic={stats['critic_loss']:.4f} actor={stats['actor_loss']:.4f}"
            )

        if record_every > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
                wandb.log({"rollout": wandb.Video(str(out), format="mp4")}, step=global_step)
            except Exception as exc:
                print(f"[rollout {it + 1}] failed: {exc}")

    return time.time() - start_t, obs_rms, global_step, total_updates, log_alpha.detach().exp().item()
