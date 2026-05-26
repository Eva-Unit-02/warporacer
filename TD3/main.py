from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from typer import run

try:
    from .agent import TD3Agent
    from .config import ACT_DIM, OBS_DIM
    from .racing_env import RacingEnv
    from .td3 import record_rollout, train
except ImportError:
    from agent import TD3Agent
    from config import ACT_DIM, OBS_DIM
    from racing_env import RacingEnv
    from td3 import record_rollout, train


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 1000,
    seed: int = 0,
    hidden: int = 256,
    buffer_size: int = 262_144,
    batch_size: int = 1024,
    learning_starts: int = 100_000,
    utd: float = 1.0,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 2e-4,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    policy_delay: int = 2,
    log_dir: Path = Path("./logs_td3"),
    device: str = "",
    record_every: int = 200,
    record_steps: int = 1800,
    use_wandb: bool = True,
    wandb_project: str = "warporacer",
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
    agent = TD3Agent(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden).to(env.device)

    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                name=f"td3_seed{seed}_n{num_envs}",
                config={
                    "algorithm": "TD3",
                    "map": str(map_yaml),
                    "seed": seed,
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "hidden": hidden,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "learning_starts": learning_starts,
                    "utd": utd,
                    "gamma": gamma,
                    "tau": tau,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "exploration_noise": exploration_noise,
                    "target_policy_noise": target_policy_noise,
                    "target_noise_clip": target_noise_clip,
                    "policy_delay": policy_delay,
                    "device": env.device,
                },
            )
        except Exception as exc:
            print(f"[wandb] init failed: {exc}")

    elapsed, obs_rms, env_steps, critic_updates, actor_updates, target = train(
        env,
        agent,
        iterations=iterations,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        utd=utd,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        exploration_noise=exploration_noise,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        policy_delay=policy_delay,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(
        f"[done] {elapsed:.1f}s env_steps={env_steps} "
        f"critic_updates={critic_updates} actor_updates={actor_updates}"
    )

    checkpoint = {
        "agent": agent.state_dict(),
        "target": target.state_dict(),
        "obs_mean": obs_rms.mean.cpu(),
        "obs_var": obs_rms.var.cpu(),
        "obs_count": obs_rms.count,
        "env_steps": env_steps,
        "critic_updates": critic_updates,
        "actor_updates": actor_updates,
    }
    torch.save(checkpoint, log_dir / "agent_final.pt")

    final_video = log_dir / "rollout_final.mp4"
    record_rollout(env, agent, record_steps, final_video, obs_rms=obs_rms)
    try:
        wandb.log({"rollout_final": wandb.Video(str(final_video), format="mp4")}, step=env_steps)
    except Exception:
        pass


if __name__ == "__main__":
    run(main)
