from pathlib import Path

import numpy as np
import torch
import wandb
from typer import run

from agent import SACAgent
from config import OBS_DIM
from racing_env import RacingEnv
from sac import record_rollout, train


def main(
    map_yaml: Path,
    num_envs: int = 512,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
    buffer_size: int = 250_000,
    batch_size: int = 256,
    learning_starts: int = 25_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    alpha_lr: float = 1e-3,
    policy_frequency: int = 2,
    target_network_frequency: int = 1,
    updates_per_iter: int = 2,
    autotune: bool = True,
    alpha: float = 0.2,
    obs_clip: float = 10.0,
    replay_device: str = "",
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = RacingEnv(
        map_yaml,
        num_envs=num_envs,
        seed=seed,
        device=device or None,
    )
    agent = SACAgent(obs_dim=OBS_DIM, action_space=env.action_space).to(env.device)
    resolved_replay_device = replay_device or env.device

    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"sac_seed{seed}_n{num_envs}",
                config={
                    "algorithm": "SAC",
                    "num_envs": num_envs,
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "map": str(map_yaml),
                    "device": env.device,
                    "replay_device": resolved_replay_device,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "learning_starts": learning_starts,
                    "gamma": gamma,
                    "tau": tau,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "alpha_lr": alpha_lr,
                    "policy_frequency": policy_frequency,
                    "target_network_frequency": target_network_frequency,
                    "updates_per_iter": updates_per_iter,
                    "autotune": autotune,
                    "alpha": alpha,
                    "obs_clip": obs_clip,
                },
            )
        except Exception as exc:
            print(f"[wandb] init failed: {exc}")

    elapsed, obs_rms, step, alpha_value, log_alpha = train(
        env,
        agent,
        total_timesteps=total_timesteps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        policy_frequency=policy_frequency,
        target_network_frequency=target_network_frequency,
        updates_per_iter=updates_per_iter,
        autotune=autotune,
        alpha=alpha,
        obs_clip=obs_clip,
        replay_device=resolved_replay_device,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(f"[done] {elapsed:.1f}s")

    torch.save(
        {
            "agent": agent.state_dict(),
            "obs_mean": obs_rms.mean.cpu(),
            "obs_var": obs_rms.var.cpu(),
            "obs_count": obs_rms.count,
            "alpha": alpha_value,
            "log_alpha": log_alpha,
        },
        log_dir / "agent_final.pt",
    )

    out = log_dir / "rollout_final.mp4"
    record_rollout(env, agent, record_steps, out, obs_rms=obs_rms, obs_clip=obs_clip)
    try:
        wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
    except Exception:
        pass


if __name__ == "__main__":
    run(main)
