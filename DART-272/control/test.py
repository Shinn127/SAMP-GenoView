"""Inference and evaluation for trained DART-272 RL control policies."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from control.config import EnvArgs, GoalSchedulerArgs, PolicyArgs, RewardWeights
from control.env import Dart272Env
from control.policy import PolicyNetwork
from dart272.utils import ensure_dir, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DART-272 control policy on goal sequences.")
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--goal-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps-per-goal", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default=None, help="Override MLD checkpoint if absent from policy ckpt.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed-data-path", type=str, default=None)
    parser.add_argument(
        "--inference-dtype",
        type=str,
        default=None,
        choices=[None, "fp32", "fp16", "bf16"],
        help="Override inference dtype (DDIM + VAE) at test time. Defaults to checkpoint setting.",
    )
    return parser.parse_args()


def _dataclass_from_dict(cls, data: dict):
    valid = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
    return cls(**valid)


def load_policy_env(
    checkpoint_path: str,
    device: torch.device,
    num_envs: int,
    max_steps: int,
    overrides: argparse.Namespace,
) -> tuple[Dart272Env, PolicyNetwork, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    env_data = dict(ckpt.get("env_args", {}))
    policy_data = dict(ckpt.get("policy_args", {}))
    goal_data = dict(ckpt.get("goal_args", {}))
    reward_data = dict(ckpt.get("reward_weights", {}))

    if overrides.checkpoint is not None:
        env_data["checkpoint_path"] = overrides.checkpoint
    if overrides.data_root is not None:
        env_data["data_root"] = overrides.data_root
    if overrides.seed_data_path is not None:
        env_data["seed_data_path"] = overrides.seed_data_path
    if getattr(overrides, "inference_dtype", None) is not None:
        env_data["inference_dtype"] = overrides.inference_dtype
    missing = [key for key in ("checkpoint_path", "data_root", "seed_data_path") if key not in env_data]
    if missing:
        raise ValueError(
            "Policy checkpoint does not contain complete env_args. "
            f"Provide overrides for: {', '.join(missing)}"
        )

    env_data["num_envs"] = num_envs
    env_data["max_steps"] = max_steps * 100  # Prevent auto-truncation; test loop handles advancement
    env_data["enable_export"] = True
    env_data["export_dir"] = ""
    env_args = _dataclass_from_dict(EnvArgs, env_data)
    policy_args = _dataclass_from_dict(PolicyArgs, policy_data)
    goal_args = _dataclass_from_dict(GoalSchedulerArgs, goal_data) if goal_data else GoalSchedulerArgs()
    reward_weights = _dataclass_from_dict(RewardWeights, reward_data) if reward_data else RewardWeights()

    env = Dart272Env(env_args, device, goal_args, reward_weights)
    if "goal_scheduler" in ckpt:
        env.goal_scheduler.load_state_dict(ckpt["goal_scheduler"])
    policy = PolicyNetwork(policy_args).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    return env, policy, ckpt


def _goal_from_xy(location: list[float]) -> list[float]:
    if len(location) == 2:
        return [float(location[0]), 0.0, float(location[1])]
    if len(location) == 3:
        return [float(location[0]), 0.0, float(location[1])]
    raise ValueError(f"Goal location must contain 2 or 3 values, got {location}")


def _goal_from_world_y_up(position: list[float]) -> list[float]:
    if len(position) == 2:
        return [float(position[0]), 0.0, float(position[1])]
    if len(position) == 3:
        return [float(position[0]), 0.0, float(position[2])]
    raise ValueError(f"Waypoint position must contain 2 or 3 values, got {position}")


def load_goal_sequences(path: str, default_text: str = "walk") -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        sequences = data
    elif "goal_location" in data:
        sequences = [data]
    elif "goals" in data:
        sequences = [
            {
                "goal_location": [goal.get("xy", goal.get("location", goal.get("position"))) for goal in data["goals"]],
                "goal_text": [goal.get("text", default_text) for goal in data["goals"]],
            }
        ]
    elif "waypoints" in data:
        sequences = [
            {
                "goal_location": [_goal_from_world_y_up(wp["position"]) for wp in data["waypoints"]],
                "goal_text": [wp.get("text", default_text) for wp in data["waypoints"]],
            }
        ]
    else:
        raise ValueError(f"Unsupported goal JSON format: {path}")

    normalized = []
    for sequence in sequences:
        raw_locations = sequence["goal_location"]
        texts = sequence.get("goal_text", [default_text] * len(raw_locations))
        goals = []
        for loc in raw_locations:
            goals.append(loc if len(loc) == 3 and loc[1] == 0.0 else _goal_from_xy(loc))
        normalized.append({"goal_location": goals, "goal_text": texts})
    return normalized


def run_sequence(
    env: Dart272Env,
    policy: PolicyNetwork,
    goals: list[list[float]],
    texts: list[str],
    max_steps_per_goal: int,
) -> dict:
    device = env.device
    success_flags = [False] * len(goals)
    steps_per_goal = [0] * len(goals)
    current_goal = 0

    env_indices = torch.arange(env.num_envs, device=device)
    obs = env.reset(
        goal_location=torch.tensor(goals[current_goal], device=device).repeat(env.num_envs, 1),
        goal_texts=[texts[current_goal]] * env.num_envs,
    )

    total_steps = max_steps_per_goal * max(len(goals), 1)
    for _ in tqdm(range(total_steps), desc="test", leave=False):
        with torch.no_grad():
            action, _, _, _ = policy.get_action_and_value(obs)

        next_goal_tensor = None
        next_goal_texts = None
        if current_goal + 1 < len(goals):
            next_goal_tensor = torch.tensor(goals[current_goal + 1], device=device).repeat(env.num_envs, 1)
            next_goal_texts = [texts[current_goal + 1]] * env.num_envs

        obs, _, _, _, info = env.step(
            action,
            next_goal_location=next_goal_tensor,
            next_goal_texts=next_goal_texts,
            reset_text=True,
        )
        steps_per_goal[current_goal] += 1

        if bool(info["success_mask"][0].item()):
            success_flags[current_goal] = True
            current_goal += 1
            if current_goal >= len(goals):
                break
        elif steps_per_goal[current_goal] >= max_steps_per_goal:
            current_goal += 1
            if current_goal >= len(goals):
                break
            env.goal_location[:] = torch.tensor(goals[current_goal], device=device)
            env._set_goal_texts(env_indices, [texts[current_goal]] * env.num_envs)
            obs = env.get_observation()

    seq = env.sequences[0]
    world_joints = (
        torch.cat(seq["world_joints"], dim=0).numpy() if seq["world_joints"] else np.zeros((0, 22, 3), dtype=np.float32)
    )
    motion = torch.cat(seq["motion"], dim=0).numpy() if seq["motion"] else np.zeros((0, 272), dtype=np.float32)
    return {
        "motion": motion,
        "world_joints": world_joints,
        "root_translations": world_joints[:, 0, :] if world_joints.size else np.zeros((0, 3), dtype=np.float32),
        "root_orientations": torch.cat(seq["transf_rotmat"], dim=0).numpy()
        if seq["transf_rotmat"]
        else np.zeros((0, 3, 3), dtype=np.float32),
        "success_flags": success_flags,
        "steps_per_goal": steps_per_goal,
        "goals": goals,
        "texts": texts,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    env, policy, ckpt = load_policy_env(
        args.policy_checkpoint,
        device,
        args.num_envs,
        args.max_steps_per_goal,
        args,
    )
    output_dir = ensure_dir(args.output_dir or Path(args.policy_checkpoint).parent / "test_rollouts")
    sequences = load_goal_sequences(args.goal_json)

    for seq_idx, sequence in enumerate(sequences):
        result = run_sequence(
            env,
            policy,
            sequence["goal_location"],
            sequence["goal_text"],
            args.max_steps_per_goal,
        )
        result["policy_checkpoint"] = args.policy_checkpoint
        result["policy_args"] = ckpt.get("policy_args", asdict(PolicyArgs()))
        stem = f"{Path(args.goal_json).stem}_seq_{seq_idx}"
        with (output_dir / f"{stem}.pkl").open("wb") as f:
            pickle.dump(result, f)
        # Also save motion as .npy for direct GenoView visualization
        np.save(output_dir / f"{stem}.npy", result["motion"])
    print(f"saved test rollouts to {output_dir}")


if __name__ == "__main__":
    main()
