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

def compute_reward(
    world_joints: torch.Tensor,
    prev_pelvis_pos: torch.Tensor,
    goal_location: torch.Tensor,
    goal_texts: list[str],
    weights: RewardWeights,
    success_threshold: float = 0.3,
    history_world_joints: torch.Tensor | None = None,
    prev_motion_frame: torch.Tensor | None = None,
    future_motion: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute multi-component reward for goal-reaching locomotion.

    Args:
        world_joints: World-space joint positions [B, 8, 22, 3] (Y-up).
        prev_pelvis_pos: Previous step's pelvis world position [B, 3].
        goal_location: Target XY position with Y=0 [B, 3].
        goal_texts: Locomotion text per environment (length B).
        weights: Reward component weights.
        success_threshold: XY distance (meters) below which a goal is reached.
        history_world_joints: Previous history joints [B, H, 22, 3]. When
            provided, skate/rotation/jerk rewards include the transition from
            the last history frame to the first generated frame, matching
            DART-main.
        prev_motion_frame: Previous denormalized 272D frame [B, 272], used for
            delta-consistency reward.
        future_motion: Generated denormalized 272D frames [B, 8, 272], used for
            delta-consistency reward.

    Returns:
        Tuple of (total_reward [B], success_mask [B], reward_info dict).
    """
    device = world_joints.device

    # Extract pelvis positions across all 8 frames: [B, 8, 3]
    pelvis_positions = world_joints[:, :, 0, :]

    # --- Distance reward ---
    reward_dist = _compute_distance_reward(
        prev_pelvis_pos, pelvis_positions, goal_location
    )

    # --- Success reward ---
    reward_success, success_mask = _compute_success_reward(
        pelvis_positions, goal_location, success_threshold
    )

    # --- Foot-floor contact penalty ---
    reward_foot_floor = _compute_foot_floor_penalty(world_joints)

    # --- Foot skating penalty ---
    reward_skate, reward_skate_rigid = _compute_skating_penalty(
        world_joints, history_world_joints
    )

    # --- Orientation reward ---
    reward_orient = _compute_orientation_reward(
        prev_pelvis_pos, pelvis_positions, goal_location
    )

    all_world_joints = (
        torch.cat([history_world_joints, world_joints], dim=1)
        if history_world_joints is not None
        else world_joints
    )
    history_length = history_world_joints.shape[1] if history_world_joints is not None else 0
    reward_rotation = _compute_rotation_reward(all_world_joints, history_length)
    reward_jerk = _compute_jerk_reward(all_world_joints)
    reward_delta = _compute_delta_reward(prev_motion_frame, future_motion, world_joints.shape[0], device)

    # --- Combine rewards with configurable weights ---
    # DART-main reduces the foot-floor penalty for hopping and running.
    light_floor_mask = torch.tensor(
        [("hop" in text.lower()) or ("run" in text.lower()) for text in goal_texts],
        dtype=torch.float32,
        device=device,
    )
    foot_floor_scale = 1.0 - 0.9 * light_floor_mask

    total_reward = (
        weights.weight_success * reward_success
        + weights.weight_dist * reward_dist
        + weights.weight_foot_floor * foot_floor_scale * reward_foot_floor
        + weights.weight_skate * reward_skate
        + weights.weight_skate_rigid * reward_skate_rigid
        + weights.weight_orient * reward_orient
        + weights.weight_rotation * reward_rotation
        + weights.weight_jerk * reward_jerk
        + weights.weight_delta * reward_delta
    )

    reward_info = {
        "reward_dist": reward_dist,
        "reward_success": reward_success,
        "reward_foot_floor": reward_foot_floor,
        "reward_skate": reward_skate,
        "reward_skate_rigid": reward_skate_rigid,
        "reward_orient": reward_orient,
        "reward_rotation": reward_rotation,
        "reward_jerk": reward_jerk,
        "reward_delta": reward_delta,
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
    success_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute success reward based on minimum pelvis-goal distance across frames.

    Success = 1.0 if min XY distance across 8 frames < success_threshold.

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

    success_mask = min_dist < success_threshold
    reward_success = success_mask.float()

    return reward_success, success_mask


def _compute_foot_floor_penalty(
    world_joints: torch.Tensor,
) -> torch.Tensor:
    """Compute foot-floor contact penalty.

    Matches DART-main's floor term in Y-up coordinates:
    for every future frame, take the lowest foot, penalize deviation from the
    floor beyond a small contact threshold, then average over frames.

    Args:
        world_joints: World-space joints [B, 8, 22, 3].

    Returns:
        Foot-floor penalty [B] (negative values).
    """
    # Extract foot joint positions: [B, 8, 4, 3]
    foot_joints = world_joints[:, :, FOOT_JOINT_INDICES, :]

    # Foot heights (Y coordinate): [B, 8, 4]
    foot_heights = foot_joints[:, :, :, 1]

    lowest_foot_height = foot_heights.amin(dim=-1)  # [B, 8]
    clamped_dist_floor = (
        torch.abs(lowest_foot_height) - CONTACT_THRESHOLD
    ).clamp(min=0.0)
    penalty = -clamped_dist_floor.mean(dim=-1)  # [B]

    return penalty


def _compute_skating_penalty(
    world_joints: torch.Tensor,
    history_world_joints: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute foot skating penalty weighted by proximity to floor.

    Matches DART-main's skate term in Y-up coordinates. If history joints are
    provided, the first interval is last-history -> first-future.

    Args:
        world_joints: World-space joints [B, 8, 22, 3].

    Returns:
        Tuple of mean skating penalty and rigid/max skating penalty, both [B].
    """
    return _compute_skating_penalty_with_history(world_joints, history_world_joints)


def _compute_skating_penalty_with_history(
    world_joints: torch.Tensor,
    history_world_joints: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Extract foot joint positions: [B, 8, 4, 3]
    future_feet = world_joints[:, :, FOOT_JOINT_INDICES, :]
    if history_world_joints is not None:
        history_feet = history_world_joints[:, :, FOOT_JOINT_INDICES, :]
        all_feet = torch.cat([history_feet, future_feet], dim=1)
        start = history_feet.shape[1]
    else:
        all_feet = future_feet
        start = 1

    if all_feet.shape[1] < 2 or start >= all_feet.shape[1]:
        zeros = torch.zeros(world_joints.shape[0], device=world_joints.device)
        return zeros, zeros

    foot_diff = torch.norm(all_feet[:, start:] - all_feet[:, start - 1 : -1], dim=-1)
    foot_height = all_feet[:, :, :, 1]
    height_consecutive_max = torch.maximum(
        foot_height[:, start - 1 : -1],
        foot_height[:, start:],
    )
    skate = foot_diff * (
        2.0
        - 2.0
        ** (height_consecutive_max / CONTACT_THRESHOLD).clamp(min=0.0, max=1.0)
    )
    reward_skate = -skate.mean(dim=(1, 2))
    reward_skate_rigid = -skate.amax(dim=(1, 2))
    return reward_skate, reward_skate_rigid


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
        Orientation reward [B] in range [0.0, 1.0].
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

    cos_sim = (movement_unit * goal_unit).sum(dim=-1).clamp(-1.0, 1.0)

    # DART-main maps cosine similarity to [0, 1].
    return (cos_sim + 1.0) / 2.0


def _compute_rotation_reward(
    all_world_joints: torch.Tensor,
    history_length: int,
) -> torch.Tensor:
    """Penalize abrupt horizontal hip-axis rotation, matching DART-main."""
    B = all_world_joints.shape[0]
    device = all_world_joints.device
    if all_world_joints.shape[1] < 2:
        return torch.zeros(B, device=device)

    l_hips = all_world_joints[:, :, 1]
    r_hips = all_world_joints[:, :, 2]
    x_axis = r_hips - l_hips
    x_axis[:, :, 1] = 0.0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-8)

    start = history_length if history_length > 0 else 1
    if start >= x_axis.shape[1]:
        return torch.zeros(B, device=device)
    dot_product = torch.einsum(
        "bti,bti->bt", x_axis[:, start:], x_axis[:, start - 1 : -1]
    )
    return dot_product.mean(dim=-1) - 1.0


def _compute_jerk_reward(all_world_joints: torch.Tensor) -> torch.Tensor:
    """Compute DART-main's max joint jerk penalty."""
    B = all_world_joints.shape[0]
    device = all_world_joints.device
    if all_world_joints.shape[1] < 4:
        return torch.zeros(B, device=device)
    vel = all_world_joints[:, 1:] - all_world_joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    jerk = torch.abs(jerk).sum(dim=-1)
    return -jerk.amax(dim=(1, 2))


def _compute_delta_reward(
    prev_motion_frame: torch.Tensor | None,
    future_motion: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Approximate DART-main's delta consistency term for the 272D layout.

    The 272D representation stores joint velocities at [74:140]. We compare
    those predicted velocities against finite differences of local joint
    positions across the last history frame and generated future frames.
    """
    if prev_motion_frame is None or future_motion is None:
        return torch.zeros(batch_size, device=device)

    prev_joints = prev_motion_frame[:, 8:74].unsqueeze(1)
    future_joints = future_motion[:, :, 8:74]
    pred_joints_delta = future_motion[:, :, 74:140]
    calc_joints_delta = torch.cat([prev_joints, future_joints], dim=1)[:, 1:] - torch.cat(
        [prev_joints, future_joints], dim=1
    )[:, :-1]
    joints_delta_diff = (pred_joints_delta - calc_joints_delta).abs().mean(dim=-1).amax(dim=-1)
    return -joints_delta_diff
