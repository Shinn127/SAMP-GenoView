"""Multi-component reward function for the RL control policy.

Computes rewards based on goal-reaching behavior, motion quality (foot contact,
skating), and orientation alignment. All computations use Y-up coordinates
consistent with the 272-dim HumanML3D representation.

Joint indices:
    - Pelvis: 0
    - L_Ankle: 7, R_Ankle: 8, L_Foot: 10, R_Foot: 11
"""

from __future__ import annotations

import torch

from control.config import RewardWeights

# Joint indices for foot contact/skating computation
FOOT_JOINT_INDICES = [7, 8, 10, 11]

# Contact threshold in meters — feet within this height are considered grounded
CONTACT_THRESHOLD = 0.03

# Default success threshold in meters
SUCCESS_THRESHOLD = 0.3


def compute_reward(
    world_joints: torch.Tensor,
    prev_pelvis_pos: torch.Tensor,
    goal_location: torch.Tensor,
    goal_texts: list[str],
    weights: RewardWeights,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute multi-component reward for goal-reaching locomotion.

    Args:
        world_joints: World-space joint positions [B, 8, 22, 3] (Y-up).
        prev_pelvis_pos: Previous step's pelvis world position [B, 3].
        goal_location: Target XY position with Y=0 [B, 3].
        goal_texts: Locomotion text per environment (length B).
        weights: Reward component weights.

    Returns:
        Tuple of (total_reward [B], success_mask [B], reward_info dict).
    """
    B = world_joints.shape[0]
    device = world_joints.device

    # Extract pelvis positions across all 8 frames: [B, 8, 3]
    pelvis_positions = world_joints[:, :, 0, :]

    # --- Distance reward ---
    reward_dist = _compute_distance_reward(
        prev_pelvis_pos, pelvis_positions, goal_location
    )

    # --- Success reward ---
    reward_success, success_mask = _compute_success_reward(
        pelvis_positions, goal_location
    )

    # --- Foot-floor contact penalty ---
    reward_foot_floor = _compute_foot_floor_penalty(world_joints)

    # --- Foot skating penalty ---
    reward_skate = _compute_skating_penalty(world_joints)

    # --- Orientation reward ---
    reward_orient = _compute_orientation_reward(
        prev_pelvis_pos, pelvis_positions, goal_location
    )

    # --- Combine rewards with configurable weights ---
    # Apply hop modifier: reduce foot-floor penalty weight by 0.1x when text contains "hop"
    hop_mask = torch.tensor(
        ["hop" in text.lower() for text in goal_texts],
        dtype=torch.float32,
        device=device,
    )
    # foot_floor_scale: 1.0 for non-hop, 0.1 for hop
    foot_floor_scale = 1.0 - 0.9 * hop_mask

    total_reward = (
        weights.weight_success * reward_success
        + weights.weight_dist * reward_dist
        + weights.weight_foot_floor * foot_floor_scale * reward_foot_floor
        + weights.weight_skate * reward_skate
        + weights.weight_orient * reward_orient
    )

    reward_info = {
        "reward_dist": reward_dist,
        "reward_success": reward_success,
        "reward_foot_floor": reward_foot_floor,
        "reward_skate": reward_skate,
        "reward_orient": reward_orient,
        "total_reward": total_reward,
    }

    return total_reward, success_mask, reward_info


def _compute_distance_reward(
    prev_pelvis_pos: torch.Tensor,
    pelvis_positions: torch.Tensor,
    goal_location: torch.Tensor,
) -> torch.Tensor:
    """Compute distance reward as XY-plane distance reduction.

    reward_dist = old_xy_dist - new_xy_dist (positive = moved closer).

    Args:
        prev_pelvis_pos: Previous pelvis position [B, 3].
        pelvis_positions: Current pelvis positions across frames [B, 8, 3].
        goal_location: Goal position [B, 3].

    Returns:
        Distance reward [B].
    """
    # Old distance: XY-plane distance from previous pelvis to goal
    old_xy_dist = torch.norm(
        prev_pelvis_pos[:, [0, 2]] - goal_location[:, [0, 2]], dim=-1
    )

    # New distance: XY-plane distance from last frame pelvis to goal
    new_pelvis_pos = pelvis_positions[:, -1, :]  # [B, 3]
    new_xy_dist = torch.norm(
        new_pelvis_pos[:, [0, 2]] - goal_location[:, [0, 2]], dim=-1
    )

    return old_xy_dist - new_xy_dist


def _compute_success_reward(
    pelvis_positions: torch.Tensor,
    goal_location: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute success reward based on minimum pelvis-goal distance across frames.

    Success = 1.0 if min XY distance across 8 frames < SUCCESS_THRESHOLD.

    Args:
        pelvis_positions: Pelvis positions [B, 8, 3].
        goal_location: Goal position [B, 3].

    Returns:
        Tuple of (success_reward [B], success_mask [B] as bool).
    """
    # XY distance from pelvis to goal for each frame: [B, 8]
    goal_xy = goal_location[:, [0, 2]].unsqueeze(1)  # [B, 1, 2]
    pelvis_xy = pelvis_positions[:, :, [0, 2]]  # [B, 8, 2]
    xy_distances = torch.norm(pelvis_xy - goal_xy, dim=-1)  # [B, 8]

    # Minimum distance across frames
    min_dist = xy_distances.min(dim=-1).values  # [B]

    success_mask = min_dist < SUCCESS_THRESHOLD
    reward_success = success_mask.float()

    return reward_success, success_mask


def _compute_foot_floor_penalty(
    world_joints: torch.Tensor,
) -> torch.Tensor:
    """Compute foot-floor contact penalty.

    Penalty = -sum over foot joints of max(0, min_height_across_frames - contact_threshold).

    Args:
        world_joints: World-space joints [B, 8, 22, 3].

    Returns:
        Foot-floor penalty [B] (negative values).
    """
    # Extract foot joint positions: [B, 8, 4, 3]
    foot_joints = world_joints[:, :, FOOT_JOINT_INDICES, :]

    # Foot heights (Y coordinate): [B, 8, 4]
    foot_heights = foot_joints[:, :, :, 1]

    # Minimum height across frames for each foot joint: [B, 4]
    min_heights = foot_heights.min(dim=1).values

    # Penalty: sum of max(0, min_height - threshold) across joints
    penalty_per_joint = torch.clamp(min_heights - CONTACT_THRESHOLD, min=0.0)
    penalty = -penalty_per_joint.sum(dim=-1)  # [B]

    return penalty


def _compute_skating_penalty(
    world_joints: torch.Tensor,
) -> torch.Tensor:
    """Compute foot skating penalty weighted by proximity to floor.

    Penalty = -mean of frame-to-frame XZ displacements weighted by
    max(0, 1.0 - foot_height / contact_threshold).

    Args:
        world_joints: World-space joints [B, 8, 22, 3].

    Returns:
        Skating penalty [B] (negative values).
    """
    # Extract foot joint positions: [B, 8, 4, 3]
    foot_joints = world_joints[:, :, FOOT_JOINT_INDICES, :]

    # Frame-to-frame XZ displacements: [B, 7, 4]
    # XZ indices are 0 (X) and 2 (Z) in Y-up coordinate system
    foot_xz = foot_joints[:, :, :, [0, 2]]  # [B, 8, 4, 2]
    displacements = torch.norm(
        foot_xz[:, 1:, :, :] - foot_xz[:, :-1, :, :], dim=-1
    )  # [B, 7, 4]

    # Foot heights at each frame (use average of consecutive frames for weighting)
    foot_heights = foot_joints[:, :, :, 1]  # [B, 8, 4]
    # Use the height at the start of each displacement interval
    heights_for_weight = foot_heights[:, :-1, :]  # [B, 7, 4]

    # Weight: max(0, 1.0 - height / threshold) — only penalize feet near floor
    proximity_weight = torch.clamp(
        1.0 - heights_for_weight / CONTACT_THRESHOLD, min=0.0
    )  # [B, 7, 4]

    # Weighted displacements
    weighted_displacements = displacements * proximity_weight  # [B, 7, 4]

    # Mean across frames and joints
    penalty = -weighted_displacements.mean(dim=(1, 2))  # [B]

    return penalty


def _compute_orientation_reward(
    prev_pelvis_pos: torch.Tensor,
    pelvis_positions: torch.Tensor,
    goal_location: torch.Tensor,
) -> torch.Tensor:
    """Compute orientation reward as cosine similarity between movement and goal directions.

    Uses XZ-plane (horizontal) directions in Y-up coordinate system.

    Args:
        prev_pelvis_pos: Previous pelvis position [B, 3].
        pelvis_positions: Current pelvis positions [B, 8, 3].
        goal_location: Goal position [B, 3].

    Returns:
        Orientation reward [B] in range [-1.0, 1.0].
    """
    # Movement direction in XZ plane: from previous pelvis to last frame pelvis
    new_pelvis_pos = pelvis_positions[:, -1, :]  # [B, 3]
    movement_xz = new_pelvis_pos[:, [0, 2]] - prev_pelvis_pos[:, [0, 2]]  # [B, 2]

    # Goal direction in XZ plane: from previous pelvis to goal
    goal_dir_xz = goal_location[:, [0, 2]] - prev_pelvis_pos[:, [0, 2]]  # [B, 2]

    # Normalize both vectors
    movement_norm = torch.norm(movement_xz, dim=-1, keepdim=True).clamp(min=1e-8)
    goal_norm = torch.norm(goal_dir_xz, dim=-1, keepdim=True).clamp(min=1e-8)

    movement_unit = movement_xz / movement_norm  # [B, 2]
    goal_unit = goal_dir_xz / goal_norm  # [B, 2]

    # Cosine similarity (dot product of unit vectors)
    cos_sim = (movement_unit * goal_unit).sum(dim=-1)  # [B]

    # Clamp to [-1, 1] for numerical safety
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    return cos_sim
