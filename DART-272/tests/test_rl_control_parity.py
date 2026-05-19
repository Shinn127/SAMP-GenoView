from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import MethodType
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.config import PolicyArgs, RewardWeights
from control.env import Dart272Env
from control.policy import PolicyNetwork
from control.reward import compute_reward
from control.test import load_goal_sequences
from control.train import compute_gae


class RLControlParityTests(unittest.TestCase):
    def test_policy_output_shapes(self) -> None:
        policy = PolicyNetwork(PolicyArgs())
        obs = torch.zeros(4, 1061)

        action, log_prob, entropy, value = policy.get_action_and_value(obs)

        self.assertEqual(action.shape, (4, 256))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(value.shape, (4,))

    def test_reward_uses_configurable_success_threshold(self) -> None:
        world_joints = torch.zeros(1, 2, 22, 3)
        world_joints[:, :, 0, 0] = torch.tensor([0.7, 0.85])
        prev_pelvis = torch.zeros(1, 3)
        goal = torch.tensor([[1.0, 0.0, 0.0]])

        _, success_loose, _ = compute_reward(
            world_joints,
            prev_pelvis,
            goal,
            ["walk"],
            RewardWeights(),
            success_threshold=0.2,
        )
        _, success_strict, _ = compute_reward(
            world_joints,
            prev_pelvis,
            goal,
            ["walk"],
            RewardWeights(),
            success_threshold=0.1,
        )

        self.assertEqual(success_loose.tolist(), [True])
        self.assertEqual(success_strict.tolist(), [False])

    def test_reward_includes_dart_main_skate_and_orientation_semantics(self) -> None:
        history_joints = torch.zeros(1, 1, 22, 3)
        world_joints = torch.zeros(1, 1, 22, 3)
        world_joints[:, :, 0, 0] = 1.0
        for joint_idx in (7, 8, 10, 11):
            world_joints[:, :, joint_idx, 0] = 1.0
        prev_pelvis = torch.zeros(1, 3)
        goal = torch.tensor([[1.0, 0.0, 0.0]])

        _, _, info = compute_reward(
            world_joints,
            prev_pelvis,
            goal,
            ["walk"],
            RewardWeights(),
            history_world_joints=history_joints,
        )

        self.assertTrue(torch.allclose(info["reward_orient"], torch.ones(1)))
        self.assertTrue(torch.allclose(info["reward_skate"], -torch.ones(1)))
        self.assertTrue(torch.allclose(info["reward_skate_rigid"], -torch.ones(1)))

    def test_compute_gae_zeros_terminated_and_bootstraps_truncated(self) -> None:
        rewards = torch.tensor([[1.0], [1.0]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[5.0], [7.0]])
        terminations = torch.tensor([[True], [False]])
        truncations = torch.tensor([[False], [True]])

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminations,
            truncations,
            gamma=1.0,
            gae_lambda=1.0,
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([[1.0], [8.0]])))
        self.assertTrue(torch.allclose(returns, torch.tensor([[1.0], [8.0]])))

    def test_success_without_next_goal_keeps_text_and_resamples_goal_only(self) -> None:
        env = object.__new__(Dart272Env)
        env.device = torch.device("cpu")
        env.goal_location = torch.zeros(2, 3)
        calls: list[tuple[list[int], bool]] = []

        def assign_random_goals(self: Dart272Env, batch_idx: torch.Tensor, reset_text: bool = True) -> None:
            calls.append((batch_idx.tolist(), reset_text))

        def set_goal_texts(self: Dart272Env, batch_idx: torch.Tensor, goal_texts: list[str]) -> None:
            raise AssertionError("text should be preserved when next_goal_texts is absent")

        env._assign_random_goals = MethodType(assign_random_goals, env)
        env._set_goal_texts = MethodType(set_goal_texts, env)

        env._handle_success_goals(torch.tensor([True, False]))

        self.assertEqual(calls, [([0], False)])

    def test_success_with_next_goal_updates_goal_and_text(self) -> None:
        env = object.__new__(Dart272Env)
        env.device = torch.device("cpu")
        env.goal_location = torch.zeros(2, 3)
        text_calls: list[tuple[list[int], list[str]]] = []

        def normalize_goal_coordinates(self: Dart272Env, goals: torch.Tensor) -> torch.Tensor:
            return goals

        def set_goal_texts(self: Dart272Env, batch_idx: torch.Tensor, goal_texts: list[str]) -> None:
            text_calls.append((batch_idx.tolist(), list(goal_texts)))

        env._normalize_goal_coordinates = MethodType(normalize_goal_coordinates, env)
        env._set_goal_texts = MethodType(set_goal_texts, env)

        env._handle_success_goals(
            torch.tensor([False, True]),
            next_goal_location=torch.tensor([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]),
            next_goal_texts=["walk", "run"],
            reset_text=True,
        )

        self.assertTrue(torch.allclose(env.goal_location[1], torch.tensor([3.0, 0.0, 4.0])))
        self.assertEqual(text_calls, [([1], ["walk", "run"])])

    def test_load_goal_sequences_accepts_dart_main_and_y_up_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "goals.json"
            path.write_text(
                json.dumps(
                    {
                        "goal_location": [[0.0, 4.0, 0.0], [1.0, 0.0, 2.0]],
                        "goal_text": ["walk", "run"],
                    }
                )
            )

            sequence = load_goal_sequences(str(path))[0]

        self.assertEqual(sequence["goal_location"], [[0.0, 0.0, 4.0], [1.0, 0.0, 2.0]])
        self.assertEqual(sequence["goal_text"], ["walk", "run"])


if __name__ == "__main__":
    unittest.main()
