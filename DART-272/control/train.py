"""PPO training loop for DART-272 RL control."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from control.config import EnvArgs, GoalSchedulerArgs, PolicyArgs, RewardWeights, TrainArgs
from control.env import Dart272Env
from control.policy import PolicyNetwork
from dart272.utils import ensure_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DART-272 RL control policy with PPO.")
    parser.add_argument("--checkpoint", type=str, default="DART-272/outputs/mld_run2/checkpoint_best.pt")
    parser.add_argument("--data-root", type=str, default="humanml3d_272")
    parser.add_argument("--seed-data-path", type=str, default="DART-272/data/rl_seed")
    parser.add_argument("--save-dir", type=str, default="DART-272/outputs/rl_control")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Total training iterations. If --total-timesteps is set, this is computed automatically.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment timesteps (like DART-main). Computes num_iterations = total_timesteps // (num_envs * num_steps).")
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--texts", type=str, nargs="+", default=["walk", "run", "hop on left leg"])
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-vloss", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--export-interval", type=int, default=100)
    parser.add_argument("--max-export", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use-wandb", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="dart272-control")
    parser.add_argument("--auto-create-seed-data", type=int, default=1)
    parser.add_argument("--seed-data-max-files", type=int, default=256)
    parser.add_argument("--obs-goal-angle-clip", type=float, default=180.0)
    parser.add_argument("--obs-goal-dist-clip", type=float, default=5.0)
    parser.add_argument("--success-threshold", type=float, default=0.3)
    parser.add_argument("--terminate-threshold", type=float, default=100.0)
    parser.add_argument("--goal-dist-min", type=float, default=0.5)
    parser.add_argument("--goal-dist-max-init", type=float, default=2.0)
    parser.add_argument("--goal-dist-max-delta", type=float, default=1.0)
    parser.add_argument("--goal-dist-max-clamp", type=float, default=5.0)
    parser.add_argument("--goal-angle-init", type=float, default=0.0)
    parser.add_argument("--goal-angle-delta", type=float, default=120.0)
    parser.add_argument("--goal-schedule-interval", type=int, default=10000)
    parser.add_argument("--weight-success", type=float, default=10.0)
    parser.add_argument("--weight-dist", type=float, default=1.0)
    parser.add_argument("--weight-foot-floor", type=float, default=1.0)
    parser.add_argument("--weight-skate", type=float, default=1.0)
    parser.add_argument("--weight-skate-delta", type=float, default=0.0)
    parser.add_argument("--weight-skate-max", type=float, default=1.0)
    parser.add_argument("--weight-skate-rigid", type=float, default=0.0)
    parser.add_argument("--weight-orient", type=float, default=1.0)
    parser.add_argument("--weight-rotation", type=float, default=0.0)
    parser.add_argument("--weight-jerk", type=float, default=0.0)
    parser.add_argument("--weight-delta", type=float, default=0.0)
    parser.add_argument("--policy-activation", type=str, default="lrelu")
    parser.add_argument("--use-tanh-scale", type=int, default=0)
    parser.add_argument("--use-zero-init", type=int, default=0)
    parser.add_argument(
        "--inference-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Autocast dtype for DDIM + VAE inference. Use fp16 on CUDA/MPS or bf16 on CPU for ~2x speedup.",
    )
    return parser.parse_args()


def prepare_seed_data(data_root: str, seed_data_path: str, max_files: int = 256) -> None:
    """Create a small RL seed directory from HumanML3D motion files if needed."""
    seed_dir = Path(seed_data_path)
    if seed_dir.exists() and any(seed_dir.glob("*.npy")):
        return

    data_root_path = Path(data_root)
    motion_dir = data_root_path / "motion_data"
    split_path = data_root_path / "split" / "train.txt"
    if not motion_dir.exists() or not split_path.exists():
        return

    seed_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    written = 0
    for sample_id in sample_ids:
        src = motion_dir / f"{sample_id}.npy"
        if not src.exists():
            continue
        motion = np.load(src)
        if motion.ndim == 2 and motion.shape[1] == 272 and motion.shape[0] >= 2:
            np.save(seed_dir / f"{sample_id}.npy", motion.astype(np.float32, copy=False))
            written += 1
        if written >= max_files:
            break


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with terminal zeroing and time-limit bootstrapping."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        bootstrap_nonterminal = 1.0 - terminations[step].float()
        recursive_nonterminal = 1.0 - (terminations[step] | truncations[step]).float()
        delta = rewards[step] + gamma * next_values[step] * bootstrap_nonterminal - values[step]
        lastgaelam = delta + gamma * gae_lambda * recursive_nonterminal * lastgaelam
        advantages[step] = lastgaelam
    returns = advantages + values
    return advantages, returns


def train(args: TrainArgs, device: torch.device, resume: str | None = None) -> None:
    save_dir = ensure_dir(args.save_dir)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    env = Dart272Env(args.env_args, device, args.goal_args, args.reward_weights)
    agent = PolicyNetwork(args.policy_args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    start_iteration = 1
    global_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        agent.load_state_dict(ckpt["policy_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "goal_scheduler" in ckpt:
            env.goal_scheduler.load_state_dict(ckpt["goal_scheduler"])
        start_iteration = int(ckpt.get("iteration", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))

    (save_dir / "config.json").write_text(
        json.dumps(
            {
                "env_args": asdict(args.env_args),
                "policy_args": asdict(args.policy_args),
                "goal_args": asdict(args.goal_args),
                "reward_weights": asdict(args.reward_weights),
                "train_args": {
                    k: v
                    for k, v in asdict(args).items()
                    if k not in {"env_args", "policy_args", "goal_args", "reward_weights"}
                },
            },
            indent=2,
        )
    )

    writer = SummaryWriter(log_dir=save_dir / "tb_logs")
    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=json.loads((save_dir / "config.json").read_text()))

    num_steps = args.num_steps
    num_envs = args.env_args.num_envs
    batch_size = num_steps * num_envs
    obs = torch.zeros((num_steps, num_envs, env.obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, env.action_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    next_values = torch.zeros((num_steps, num_envs), device=device)
    terminations = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
    truncations = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

    next_obs = env.reset()
    start_time = time.time()
    for iteration in tqdm(range(start_iteration, args.num_iterations + 1), desc="ppo"):
        env.global_iteration = iteration - 1
        if args.goal_args.curriculum_interval > 0 and iteration % args.goal_args.curriculum_interval == 0:
            env.curriculum_step()
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / max(args.num_iterations, 1)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_success = 0
        rollout_terminated = 0
        rollout_truncated = 0
        reward_logs: dict[str, list[torch.Tensor]] = {}

        for step in range(num_steps):
            obs[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards[step] = reward
            terminations[step] = terminated
            truncations[step] = truncated

            with torch.no_grad():
                transition_value = agent.get_value(next_obs)
                final_obs = info.get("final_observation")
                if final_obs is not None and truncated.any():
                    transition_value = transition_value.clone()
                    transition_value[truncated] = agent.get_value(final_obs[truncated])
                next_values[step] = transition_value

            global_step += num_envs

            rollout_success += info["num_success"]
            rollout_terminated += info["num_terminated"]
            rollout_truncated += info["num_truncated"]
            for key, value_tensor in info["reward_info"].items():
                reward_logs.setdefault(key, []).append(value_tensor.detach())

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            args.gamma,
            args.gae_lambda,
        )

        b_obs = obs.reshape(batch_size, env.obs_dim)
        b_actions = actions.reshape(batch_size, env.action_dim)
        b_logprobs = logprobs.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values.reshape(batch_size)

        clipfracs = []
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        for _ in range(args.update_epochs):
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        explained_var = np.nan
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        if var_y > 0:
            explained_var = 1.0 - np.var(y_true - y_pred) / var_y

        metrics = {
            "reward/mean": rewards.mean().item(),
            "done/num_success": rollout_success,
            "done/num_terminated": rollout_terminated,
            "done/num_truncated": rollout_truncated,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "losses/explained_variance": float(explained_var),
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS": int(global_step / max(time.time() - start_time, 1e-6)),
            "curriculum/dist_max": env.goal_scheduler.current_dist_max,
            "curriculum/angle_range": env.goal_scheduler.current_angle_range,
            "curriculum/weight_skate": env.reward_weights.weight_skate,
        }
        for key, tensors in reward_logs.items():
            metrics[f"reward/{key}"] = torch.stack(tensors).mean().item()
        for key, value in metrics.items():
            writer.add_scalar(key, value, global_step)
        if wandb_run is not None:
            wandb_run.log(metrics, step=global_step)

        if iteration % args.save_interval == 0 or iteration == args.num_iterations:
            payload = {
                "policy_state": agent.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "iteration": iteration,
                "global_step": global_step,
                "env_args": asdict(args.env_args),
                "policy_args": asdict(args.policy_args),
                "goal_args": asdict(args.goal_args),
                "reward_weights": asdict(args.reward_weights),
                "train_args": {
                    k: v
                    for k, v in asdict(args).items()
                    if k not in {"env_args", "policy_args", "goal_args", "reward_weights"}
                },
                "goal_scheduler": env.goal_scheduler.state_dict(),
            }
            torch.save(payload, save_dir / f"iter_{iteration}.pt")
            torch.save(payload, save_dir / "checkpoint_last.pt")

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    cli = parse_args()

    # Compute num_iterations from total_timesteps (DART-main style)
    if cli.total_timesteps is not None:
        cli.num_iterations = cli.total_timesteps // (cli.num_envs * cli.num_steps)
        print(f"[info] total_timesteps={cli.total_timesteps}, "
              f"num_envs={cli.num_envs}, num_steps={cli.num_steps} "
              f"=> num_iterations={cli.num_iterations}")
    elif cli.num_iterations is None:
        cli.num_iterations = 500  # fallback default

    if cli.auto_create_seed_data:
        prepare_seed_data(cli.data_root, cli.seed_data_path, cli.seed_data_max_files)
    device = resolve_device(cli.device)
    save_dir = Path(cli.save_dir)
    env_args = EnvArgs(
        checkpoint_path=cli.checkpoint,
        seed_data_path=cli.seed_data_path,
        data_root=cli.data_root,
        num_envs=cli.num_envs,
        max_steps=cli.max_episode_steps,
        texts=cli.texts,
        guidance_scale=cli.guidance_scale,
        ddim_steps=cli.ddim_steps,
        success_threshold=cli.success_threshold,
        terminate_threshold=cli.terminate_threshold,
        obs_goal_angle_clip=cli.obs_goal_angle_clip,
        obs_goal_dist_clip=cli.obs_goal_dist_clip,
        enable_export=True,
        export_interval=cli.export_interval,
        max_export=cli.max_export,
        export_dir=str(save_dir / "rollouts"),
        inference_dtype=cli.inference_dtype,
    )
    args = TrainArgs(
        env_args=env_args,
        policy_args=PolicyArgs(
            activation=cli.policy_activation,
            use_tanh_scale=bool(cli.use_tanh_scale),
            use_zero_init=bool(cli.use_zero_init),
        ),
        goal_args=GoalSchedulerArgs(
            dist_min=cli.goal_dist_min,
            dist_max_init=cli.goal_dist_max_init,
            dist_max_delta=cli.goal_dist_max_delta,
            dist_max_clamp=cli.goal_dist_max_clamp,
            angle_init=cli.goal_angle_init,
            angle_delta=cli.goal_angle_delta,
            curriculum_interval=cli.goal_schedule_interval,
        ),
        reward_weights=RewardWeights(
            weight_success=cli.weight_success,
            weight_dist=cli.weight_dist,
            weight_foot_floor=cli.weight_foot_floor,
            weight_skate=cli.weight_skate,
            weight_skate_delta=cli.weight_skate_delta,
            weight_skate_max=cli.weight_skate_max,
            weight_skate_rigid=cli.weight_skate_rigid,
            weight_orient=cli.weight_orient,
            weight_rotation=cli.weight_rotation,
            weight_jerk=cli.weight_jerk,
            weight_delta=cli.weight_delta,
        ),
        learning_rate=cli.learning_rate,
        gamma=cli.gamma,
        gae_lambda=cli.gae_lambda,
        clip_coef=cli.clip_coef,
        clip_vloss=bool(cli.clip_vloss),
        norm_adv=bool(cli.norm_adv),
        vf_coef=cli.vf_coef,
        ent_coef=cli.ent_coef,
        max_grad_norm=cli.max_grad_norm,
        target_kl=cli.target_kl,
        update_epochs=cli.update_epochs,
        minibatch_size=cli.minibatch_size,
        num_steps=cli.num_steps,
        num_iterations=cli.num_iterations,
        save_interval=cli.save_interval,
        export_interval=cli.export_interval,
        max_export=cli.max_export,
        save_dir=cli.save_dir,
        use_wandb=bool(cli.use_wandb),
        wandb_project=cli.wandb_project,
        seed=cli.seed,
    )
    train(args, device, resume=cli.resume)


if __name__ == "__main__":
    main()
