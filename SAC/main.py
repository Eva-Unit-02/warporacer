from pathlib import Path

import numpy as np
import torch
from typer import run
import wandb

try:
    from .agent import SACAgent
    from .config import ACT_DIM, OBS_DIM
    from .racing_env import RacingEnv
    from .sac import record_rollout, train
except ImportError:
    from agent import SACAgent
    from config import ACT_DIM, OBS_DIM
    from racing_env import RacingEnv
    from sac import record_rollout, train


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    hidden: int = 256,
    buffer_size: int = 262_144,
    batch_size: int = 1024,
    learning_starts: int = 16_384,
    utd: float = 1.0,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    log_dir: Path = Path("./logs_sac"),
    device: str = "",
    record_every: int = 250,
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
    agent = SACAgent(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=hidden).to(env.device)

    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                name=f"sac_seed{seed}_n{num_envs}",
                config={
                    "algorithm": "SAC",
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "seed": seed,
                    "map": str(map_yaml),
                    "hidden": hidden,
                    "buffer_size": buffer_size,
                    "batch_size": batch_size,
                    "learning_starts": learning_starts,
                    "utd": utd,
                    "gamma": gamma,
                    "tau": tau,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "alpha_lr": alpha_lr,
                    "device": env.device,
                },
            )
        except Exception as exc:
            print(f"[wandb] init failed: {exc}")

    elapsed, obs_rms, env_steps, gradient_steps, alpha = train(
        env,
        agent,
        iterations=iterations,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        learning_starts=learning_starts,
        utd=utd,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(f"[done] {elapsed:.1f}s env_steps={env_steps} gradient_steps={gradient_steps}")

    torch.save(
        {
            "agent": agent.state_dict(),
            "obs_mean": obs_rms.mean.cpu(),
            "obs_var": obs_rms.var.cpu(),
            "obs_count": obs_rms.count,
            "alpha": alpha,
        },
        log_dir / "agent_final.pt",
    )

    out = log_dir / "rollout_final.mp4"
    record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
    try:
        wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=env_steps)
    except Exception:
        pass


if __name__ == "__main__":
    run(main)
