"""Differentiable local-to-world coordinate transform for 272-dim motion features.

272-dim layout (from representation_272.py):
  [0:2]    root XZ velocity (no heading, local frame)
  [2:8]    heading angular velocity (6D rotation of frame-to-frame heading diff)
  [8:74]   joint positions (22 joints * 3, no heading, at XZ origin)
  [74:140] joint velocities (22 joints * 3)
  [140:272] joint rotations 6D + foot contact

This module provides a differentiable function to convert root-relative
joint positions to world coordinates by accumulating root velocity and heading.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        d6: [..., 6] tensor

    Returns:
        [..., 3, 3] rotation matrix
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3] columns are b1, b2, b3


def cumulative_rotation(heading_6d: torch.Tensor) -> torch.Tensor:
    """Accumulate frame-to-frame heading rotations into world heading.

    The 272-dim representation stores heading as a 6D rotation representing
    the yaw change from frame t-1 to frame t. Frame 0's heading_6d encodes
    the identity rotation [1,0,0, 0,1,0] (or close to it).

    Args:
        heading_6d: [T, 6] per-frame heading delta (6D rotation)

    Returns:
        world_heading: [T, 3, 3] cumulative world heading rotation matrices
    """
    T = heading_6d.shape[0]
    device = heading_6d.device

    # Convert each frame's 6D to rotation matrix
    delta_rot = rotation_6d_to_matrix(heading_6d)  # [T, 3, 3]

    # Accumulate: world_heading[t] = delta_rot[t] @ world_heading[t-1]
    # Frame 0: world_heading[0] = delta_rot[0] (which should be ~identity)
    world_heading = torch.zeros(T, 3, 3, device=device, dtype=heading_6d.dtype)
    world_heading[0] = delta_rot[0]
    for t in range(1, T):
        world_heading[t] = delta_rot[t] @ world_heading[t - 1]

    return world_heading


def cumulative_rotation_batched(heading_6d: torch.Tensor) -> torch.Tensor:
    """Batched version of cumulative_rotation (differentiable, no in-place ops).

    Args:
        heading_6d: [B, T, 6]

    Returns:
        world_heading: [B, T, 3, 3]
    """
    B, T, _ = heading_6d.shape

    delta_rot = rotation_6d_to_matrix(heading_6d)  # [B, T, 3, 3]

    # Accumulate without in-place operations for autograd compatibility
    headings = [delta_rot[:, 0]]  # First frame
    for t in range(1, T):
        headings.append(torch.bmm(delta_rot[:, t], headings[-1]))

    return torch.stack(headings, dim=1)  # [B, T, 3, 3]


def local_to_world_joints(motion_denorm: torch.Tensor) -> torch.Tensor:
    """Convert 272-dim root-relative motion to world-coordinate joint positions.

    This function is fully differentiable and can be used inside optimization loops.

    Args:
        motion_denorm: [B, T, 272] denormalized motion features

    Returns:
        world_joints: [B, T, 22, 3] joint positions in world coordinates
    """
    B, T, D = motion_denorm.shape
    device = motion_denorm.device

    # Extract components
    root_vel_xz = motion_denorm[:, :, 0:2]  # [B, T, 2] local XZ velocity
    heading_6d = motion_denorm[:, :, 2:8]   # [B, T, 6] heading delta (6D)
    local_joints = motion_denorm[:, :, 8:74].reshape(B, T, 22, 3)  # [B, T, 22, 3]

    # 1. Accumulate heading rotations
    world_heading = cumulative_rotation_batched(heading_6d)  # [B, T, 3, 3]

    # 2. Accumulate root XZ position in world frame
    # root_vel_xz is in the local (heading-removed) frame
    # We need to rotate it by the world heading to get world displacement
    # The heading rotation is around Y axis, so we only need the XZ components
    root_vel_3d = torch.zeros(B, T, 3, device=device, dtype=motion_denorm.dtype)
    root_vel_3d[:, :, 0] = root_vel_xz[:, :, 0]  # X
    root_vel_3d[:, :, 2] = root_vel_xz[:, :, 1]  # Z (second component maps to Z)

    # Rotate local velocity to world frame: world_vel = heading @ local_vel
    world_vel = torch.einsum("btij,btj->bti", world_heading, root_vel_3d)  # [B, T, 3]

    # Cumulative sum to get world position (frame 0 at origin)
    world_root_pos = torch.cumsum(world_vel, dim=1)  # [B, T, 3]
    # Y component of root position comes from the joint data (root joint Y)
    world_root_pos = torch.cat([
        world_root_pos[:, :, 0:1],
        local_joints[:, :, 0, 1:2],  # Root joint Y (height)
        world_root_pos[:, :, 2:3],
    ], dim=-1)  # [B, T, 3]

    # 3. Rotate local joints to world frame and add root translation
    # local_joints are already relative to root and heading-removed
    world_joints = torch.einsum("btij,btkj->btki", world_heading, local_joints)  # [B, T, 22, 3]
    world_joints = world_joints + world_root_pos.unsqueeze(2)  # [B, T, 22, 3]

    return world_joints


def local_to_world_root_trajectory(motion_denorm: torch.Tensor) -> torch.Tensor:
    """Extract just the world-space root (pelvis) trajectory.

    Lighter weight than full joint transform — useful for pelvis-only constraints.

    Args:
        motion_denorm: [B, T, 272]

    Returns:
        root_trajectory: [B, T, 3] world XYZ position of root
    """
    world_joints = local_to_world_joints(motion_denorm)
    return world_joints[:, :, 0, :]  # Joint 0 = Pelvis
