"""Curriculum learning for goal placement in the RL control policy.

The GoalScheduler progressively increases goal difficulty by expanding
the maximum distance and angle range over training. Goals are sampled
in polar coordinates relative to the character's current position and
forward direction on the XZ plane (Y-up coordinate system).
"""

from __future__ import annotations

import math

import torch

from control.config import GoalSchedulerArgs


class GoalScheduler:
    """Curriculum learning for goal placement.

    Manages progressive difficulty by increasing the maximum goal distance
    and angle range over training. Goals are sampled in polar coordinates
    relative to the character's pelvis position and forward direction.

    Attributes:
        dist_min: Minimum goal distance (meters).
        current_dist_max: Current maximum goal distance (meters).
        current_angle_range: Current goal angle range (degrees).
        curriculum_steps_taken: Number of curriculum steps completed.
    """

    def __init__(self, args: GoalSchedulerArgs) -> None:
        """Initialize the goal scheduler with curriculum parameters.

        Args:
            args: Goal scheduler configuration containing initial values,
                  deltas, and clamp limits for distance and angle.
        """
        self.dist_min = args.dist_min
        self.dist_max_init = args.dist_max_init
        self.dist_max_delta = args.dist_max_delta
        self.dist_max_clamp = args.dist_max_clamp
        self.angle_init = args.angle_init
        self.angle_delta = args.angle_delta
        self.angle_max = args.angle_max
        self.curriculum_interval = args.curriculum_interval

        # Current curriculum state
        self.current_dist_max = self.dist_max_init
        self.current_angle_range = self.angle_init
        self.curriculum_steps_taken = 0

    def curriculum_step(self) -> None:
        """Increase difficulty by expanding distance and angle range.

        Increases the maximum goal distance by dist_max_delta (clamped at
        dist_max_clamp) and the angle range by angle_delta (clamped at
        angle_max). After reaching both clamps, this method is idempotent.
        """
        self.current_dist_max = min(
            self.current_dist_max + self.dist_max_delta,
            self.dist_max_clamp,
        )
        self.current_angle_range = min(
            self.current_angle_range + self.angle_delta,
            self.angle_max,
        )
        self.curriculum_steps_taken += 1

    def sample_goal(
        self,
        pelvis_pos: torch.Tensor,
        forward_dir: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample goal locations in polar coords relative to character.

        Goals are placed on the XZ plane (Y=0) at a random distance and
        angle relative to the character's current position and forward
        direction.

        Args:
            pelvis_pos: Current pelvis world position, shape [B, 3].
            forward_dir: Character forward direction (unit vector on XZ
                plane), shape [B, 3]. Y component should be ~0.
            batch_size: Number of goals to sample (must match B).

        Returns:
            Goal locations in world coordinates, shape [B, 3] with Y=0.
        """
        device = pelvis_pos.device

        # Sample distance uniformly from [dist_min, current_dist_max]
        dist = (
            torch.rand(batch_size, device=device)
            * (self.current_dist_max - self.dist_min)
            + self.dist_min
        )

        # Sample angle uniformly from [-current_angle_range/2, +current_angle_range/2]
        # Convert angle range from degrees to radians
        half_angle_rad = math.radians(self.current_angle_range / 2.0)
        angle = (
            torch.rand(batch_size, device=device) * 2.0 * half_angle_rad
            - half_angle_rad
        )

        # Convert polar to Cartesian offset in local frame
        # Forward direction is on XZ plane; we rotate it by the sampled angle
        # around the Y axis to get the goal direction.
        # Rotation around Y-axis: [cos(a), 0, -sin(a); 0, 1, 0; sin(a), 0, cos(a)]
        cos_a = torch.cos(angle)  # [B]
        sin_a = torch.sin(angle)  # [B]

        # Extract forward X and Z components (Y-up, so forward is on XZ plane)
        fwd_x = forward_dir[:, 0]  # [B]
        fwd_z = forward_dir[:, 2]  # [B]

        # Rotate forward direction by angle around Y axis
        goal_dir_x = fwd_x * cos_a + fwd_z * sin_a  # [B]
        goal_dir_z = -fwd_x * sin_a + fwd_z * cos_a  # [B]

        # Compute goal position: pelvis XZ + distance * rotated_direction
        goal_x = pelvis_pos[:, 0] + dist * goal_dir_x  # [B]
        goal_z = pelvis_pos[:, 2] + dist * goal_dir_z  # [B]

        # Goals are on the XZ plane (Y=0)
        goal = torch.zeros(batch_size, 3, device=device)
        goal[:, 0] = goal_x
        goal[:, 2] = goal_z

        return goal

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing.

        Returns:
            Dictionary containing current curriculum state.
        """
        return {
            "current_dist_max": self.current_dist_max,
            "current_angle_range": self.current_angle_range,
            "curriculum_steps_taken": self.curriculum_steps_taken,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint.

        Args:
            state: Dictionary from a previous state_dict() call.
        """
        self.current_dist_max = state["current_dist_max"]
        self.current_angle_range = state["current_angle_range"]
        self.curriculum_steps_taken = state["curriculum_steps_taken"]
