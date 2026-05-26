from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import wandb
from typer import run

try:
    from .agent import TD3Agent
    from .config import ACT_DIM, OBS_DIM
    from .multi_map_env import MultiMapRacingEnv
    from .racing_env import RacingEnv
    from .td3 import record_rollout, train
except ImportError:
    from agent import TD3Agent
    from config import ACT_DIM, OBS_DIM
    from multi_map_env import MultiMapRacingEnv
    from racing_env import RacingEnv
    from td3 import record_rollout, train


def _build_training_env(map_yamls: list[Path], num_envs: int, seed: int, device: str):
    if len(map_yamls) == 1:
        return RacingEnv(
            map_yamls[0],
            num_envs=num_envs,
            seed=seed,
            device=device or None,
        )

    return MultiMapRacingEnv(
        map_yamls,
        num_envs=num_envs,
        seed=seed,
        device=device or None,
    )


def _record_all_maps(
    maps_dir: Path,
    log_dir: Path,
    agent: TD3Agent,
    obs_rms,
    record_steps: int,
    seed: int,
    device: str,
    env_steps: int,
):
    eval_maps = sorted(maps_dir.glob("*.yaml"))
    eval_dir = log_dir / "rollouts_all_maps"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for idx, eval_map in enumerate(eval_maps):
        eval_env = RacingEnv(
            eval_map,
            num_envs=1,
            seed=seed + 100_000 + idx,
            device=device,
        )
        video_path = eval_dir / f"{eval_map.stem}.mp4"
        record_rollout(eval_env, agent, record_steps, video_path, obs_rms=obs_rms)
        try:
            wandb.log(
                {f"rollout_{eval_map.stem}": wandb.Video(str(video_path), format="mp4")},
                step=env_steps,
            )
        except Exception:
            pass


def main(
    map_yamls: list[Path],
    num_envs: int = 4096,
    iterations: int = 16000,
    seed: int = 0,
    hidden: int = 256,
    buffer_size: int = 262_144,
    batch_size: int = 1024,
    learning_starts: int = 100_000,
    n_step: int = 20,
    utd: float = 1.0,
    gamma: float = 0.9965,
    tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 2e-4,
    exploration_noise: float = 0.1,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    policy_delay: int = 2,
    log_dir: Path = Path("./logs_td3"),
    device: str = "",
    record_every: int = 200000,
    record_steps: int = 1800,
    use_wandb: bool = True,
    wandb_project: str = "warporacer",
):
    if not map_yamls:
        raise ValueError("Provide at least one map yaml")

    map_yamls = [Path(path) for path in map_yamls]
    train_map_names = [str(path) for path in map_yamls]
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = _build_training_env(map_yamls, num_envs=num_envs, seed=seed, device=device)
    rollout_env = env if isinstance(env, RacingEnv) else env.envs[0]
    agent = TD3Agent(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden).to(env.device)

    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                name=f"td3_seed{seed}_n{num_envs}",
                config={
                    "algorithm": "TD3",
                    "maps": train_map_names,
                    "seed": seed,
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "hidden": hidden,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "learning_starts": learning_starts,
                    "n_step": n_step,
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
        n_step=n_step,
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
        record_env=rollout_env,
    )
    print(
        f"[done] {elapsed:.1f}s env_steps={env_steps} "
        f"critic_updates={critic_updates} actor_updates={actor_updates}"
    )

    checkpoint = {
        "agent": agent.state_dict(),
        "target": target.state_dict(),
        "train_maps": train_map_names,
        "obs_mean": obs_rms.mean.cpu(),
        "obs_var": obs_rms.var.cpu(),
        "obs_count": obs_rms.count,
        "env_steps": env_steps,
        "critic_updates": critic_updates,
        "actor_updates": actor_updates,
    }
    torch.save(checkpoint, log_dir / "agent_final.pt")

    maps_dir = Path(__file__).resolve().parent.parent / "maps"
    _record_all_maps(
        maps_dir=maps_dir,
        log_dir=log_dir,
        agent=agent,
        obs_rms=obs_rms,
        record_steps=record_steps,
        seed=seed,
        device=env.device,
        env_steps=env_steps,
    )


if __name__ == "__main__":
    run(main)
