"""End-to-end smoke test for DART-272 RL Control.

Runs a minimal configuration (2 envs, 2 steps, 1 iteration) on CPU to verify
the entire pipeline (env init -> reset -> step -> reward -> PPO update -> save)
works without crashing. Does not produce a useful policy; only checks that
all the moving parts hook together correctly.

Usage:
    python -m control._smoke_test
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from control.config import EnvArgs, GoalSchedulerArgs, PolicyArgs, RewardWeights, TrainArgs
from control.train import train


def main() -> None:
    # Tiny config that exercises the full pipeline quickly on CPU.
    # Resolve paths relative to repo root so the smoke test works from any cwd.
    repo_root = PROJECT_ROOT.parent
    save_dir = PROJECT_ROOT / "outputs" / "rl_smoke"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env_args = EnvArgs(
        checkpoint_path=str(PROJECT_ROOT / "outputs/mld_run2/checkpoint_best.pt"),
        seed_data_path=str(PROJECT_ROOT / "data/rl_seed"),
        data_root=str(repo_root / "humanml3d_272"),
        num_envs=2,
        max_steps=4,
        texts=["walk", "run", "hop on left leg"],
        guidance_scale=5.0,
        ddim_steps=2,            # only 2 DDIM steps to save time
        success_threshold=0.3,
        terminate_threshold=100.0,
        obs_goal_angle_clip=180.0,
        obs_goal_dist_clip=5.0,
        enable_export=False,
        export_interval=1,
        max_export=1,
        export_dir=str(save_dir / "rollouts"),
    )

    args = TrainArgs(
        env_args=env_args,
        policy_args=PolicyArgs(
            latent_dim=64,        # tiny policy
            n_blocks=1,
            activation="lrelu",
        ),
        goal_args=GoalSchedulerArgs(curriculum_interval=10000),
        reward_weights=RewardWeights(),
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        update_epochs=1,          # 1 epoch for speed
        minibatch_size=4,
        num_steps=2,              # 2-step rollout
        num_iterations=1,         # 1 iteration only
        anneal_lr=False,
        save_interval=1,
        export_interval=1,
        max_export=1,
        save_dir=str(save_dir),
        use_wandb=False,
        seed=1,
    )

    device = torch.device("cpu")
    print(f"[smoke] starting smoke test on {device}")
    t0 = time.time()
    train(args, device, resume=None)
    t1 = time.time()
    print(f"[smoke] training loop completed in {t1 - t0:.1f}s")

    # Verify checkpoint and config files were created
    expected = [
        save_dir / "config.json",
        save_dir / "checkpoint_last.pt",
        save_dir / "iter_1.pt",
    ]
    for path in expected:
        assert path.exists(), f"Expected output not found: {path}"
        print(f"[smoke] OK -> {path}")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
