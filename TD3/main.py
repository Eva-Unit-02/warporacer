from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import wandb
from typer import run

try:
    from .agent import TD3Agent
    from .config import ACT_DIM, OBS_DIM
    from .multi_map_env import MultiMapRacingEnv
    from .racing_env import RacingEnv
    from .td3 import RunningStats, record_rollout, train
except ImportError:
    from agent import TD3Agent
    from config import ACT_DIM, OBS_DIM
    from multi_map_env import MultiMapRacingEnv
    from racing_env import RacingEnv
    from td3 import RunningStats, record_rollout, train


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


def _record_maps_to_wandb(
    map_yamls: list[Path],
    agent: TD3Agent,
    obs_rms,
    record_steps: int,
    seed: int,
    device: str,
    env_steps: int,
    enabled: bool,
):
    if not enabled:
        print("[rollout skip] W&B disabled, so final rollout videos were not recorded")
        return

    skipped: list[str] = []
    with TemporaryDirectory(prefix="warporacer_td3_rollouts_") as temp_dir:
        temp_path = Path(temp_dir)
        for idx, eval_map in enumerate(map_yamls):
            try:
                eval_env = RacingEnv(
                    eval_map,
                    num_envs=1,
                    seed=seed + 100_000 + idx,
                    device=device,
                )
                video_path = temp_path / f"{eval_map.stem}.mp4"
                record_rollout(eval_env, agent, record_steps, video_path, obs_rms=obs_rms)
            except Exception as exc:
                message = f"{eval_map}: {type(exc).__name__}: {exc}"
                skipped.append(message)
                print(f"[rollout skip] {message}")
                continue

            try:
                wandb.log(
                    {f"rollout_{eval_map.stem}": wandb.Video(str(video_path), format="mp4")},
                    step=env_steps,
                )
            except Exception as exc:
                print(f"[wandb rollout] failed for {eval_map}: {exc}")

    if skipped:
        print("[rollout skip] skipped maps: " + "; ".join(skipped))


def _load_checkpoint(checkpoint_path: Path, agent: TD3Agent, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])

    obs_rms = RunningStats((OBS_DIM,), device)
    obs_rms.mean.copy_(checkpoint["obs_mean"].to(device))
    obs_rms.var.copy_(checkpoint["obs_var"].to(device))
    obs_rms.count = checkpoint["obs_count"]
    obs_rms.inv_std = torch.rsqrt(obs_rms.var + 1e-8)
    return checkpoint, obs_rms


def main(
    map_yamls: list[Path],
    num_envs: int = 4096,
    iterations: int = 8000,
    seed: int = 0,
    hidden: int = 256,
    buffer_size: int = 262_144,
    batch_size: int = 1024,
    learning_starts: int = 100_000,
    n_step: int = 20,
    utd: float = 1.0,
    gamma: float = 0.9963,
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
    checkpoint: Path | None = None,
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

    if checkpoint is not None:
        run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        agent = TD3Agent(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden).to(run_device)
        loaded_checkpoint, obs_rms = _load_checkpoint(checkpoint, agent, run_device)
        env_steps = int(loaded_checkpoint.get("env_steps", 0))
        eval_maps = [Path(path) for path in loaded_checkpoint.get("train_maps", train_map_names)]
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=f"td3_rollouts_seed{seed}",
                    config={
                        "algorithm": "TD3",
                        "checkpoint": str(checkpoint),
                        "maps": [str(path) for path in eval_maps],
                        "device": run_device,
                    },
                )
            except Exception as exc:
                print(f"[wandb] init failed: {exc}")
        _record_maps_to_wandb(
            map_yamls=eval_maps,
            agent=agent,
            obs_rms=obs_rms,
            record_steps=record_steps,
            seed=seed,
            device=run_device,
            env_steps=env_steps,
            enabled=use_wandb,
        )
        return

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

    _record_maps_to_wandb(
        map_yamls=map_yamls,
        agent=agent,
        obs_rms=obs_rms,
        record_steps=record_steps,
        seed=seed,
        device=env.device,
        env_steps=env_steps,
        enabled=use_wandb,
    )


if __name__ == "__main__":
    run(main)
