from pathlib import Path
from typer import run

import numpy as np
import torch
import wandb

from agent import Agent
from config import OBS_DIM
from PPO import record_rollout, train
from racing_env import RacingEnv

def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
):
    # Setup Miscellaneous
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Setup Environment
    env = RacingEnv(
        map_yaml, 
        num_envs=num_envs,
        seed=seed,
        device=device or None,
    )   
    
    # Setup Agent
    agent = Agent(obs_dim=OBS_DIM).to(env.device)

    # Setup Wandb
    if use_wandb:
        try:
            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}",
                config={
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "seed": seed,
                    "map": str(map_yaml),
                },
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    # Train
    elapsed, obs_rms, ret_rms, step = train(
        env,
        agent,
        iterations=iterations,
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
        },
        log_dir / "agent_final.pt",
    )

    out = log_dir / "rollout_final.mp4"
    record_rollout(env, agent, record_steps, out, obs_rms=obs_rms)
    try:
        wandb.log({"rollout_final": wandb.Video(str(out), format="mp4")}, step=step)
    except Exception:
        pass


if __name__ == "__main__":
    run(main)
