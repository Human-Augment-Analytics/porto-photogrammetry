# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def rotation_matrix_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate. Shape: (3,)
        b: The vector to rotate to. Shape: (3,)
    Returns:
        The rotation matrix. Shape: (3, 3)
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


def to4x4(pose: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate. Shape: (..., 3, 4)

    Returns:
        Camera poses with additional homogenous coordinate added. Shape: (..., 4, 4)
    """
    assert pose.shape[-2:] == (3, 4), f"Expected 3x4 pose, got {pose.shape}"
    # Build homogeneous row [0, 0, 0, 1] without in-place operations
    batch_shape = pose.shape[:-2]
    constants = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
    constants = constants.view(*([1] * len(batch_shape)), 1, 4).expand(*batch_shape, 1, 4)
    return torch.cat([pose, constants], dim=-2)


def inverse_pose(pose: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Calculate inverse of rigid body transformation(s).

    Args:
        pose: ``[..., 3/4, 4]`` pose(s) or a single ``[3/4, 4]`` pose. Either ``torch.Tensor`` or ``np.ndarray``.

    Returns:
        Inverse of input pose(s), same array type and shape as input.
    """
    is_numpy = isinstance(pose, np.ndarray)
    padded = pose.shape[-2] == 4
    squeeze = False
    if pose.ndim == 2:
        pose = pose[None] if is_numpy else pose.unsqueeze(0)
        squeeze = True

    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    rot = np.swapaxes(rotation, -2, -1) if is_numpy else rotation.transpose(-2, -1)
    t = -rot @ translation[..., None]
    if is_numpy:
        inv = np.concatenate([rot, t], axis=-1)
        if padded:
            inv = np.concatenate([inv, pose[..., 3:4, :]], axis=-2)
        if squeeze:
            inv = inv[0]
    else:
        inv = torch.cat([rot, t], -1)
        if padded:
            inv = torch.cat([inv, pose[..., 3:4, :]], -2)
        if squeeze:
            inv = inv.squeeze(0)
    return inv


def multiply_poses(pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right. Shape: (*batch, 3, 4)
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a. Shape: (*batch, 3, 4)

    Returns:
        Camera pose matrix where pose_a was applied to pose_b. Shape: (*batch, 3, 4)
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)
