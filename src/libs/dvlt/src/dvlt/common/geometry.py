# SPDX-FileCopyrightText: Copyright 2023 Google LLC
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file (the OpenCV-style radial/tangential undistortion solver
# `_compute_residual_and_jacobian` and `radial_and_tangential_undistort`) are
# adapted from MultiNeRF:
#   https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
# Original work licensed under the Apache License, Version 2.0:
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications by NVIDIA:
#   - Ported from NumPy/JAX to PyTorch with broadcasting semantics.
#   - Merged the Newton step into a vectorized batched implementation.

"""Geometry related functions."""

from typing import Tuple

import numpy as np
import open3d as o3d
import torch
from torch import Tensor

from dvlt.common.constants import EPS
from dvlt.common.pose import to4x4
from dvlt.common.projection import depth_to_points


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    """Applies transform to 3D points.

    Args:
        points (Tensor): 3D points of shape [..., 3] where [...] represents any number of batch dimensions.
        transform (Tensor): transforms of shape [..., 4, 4] or [..., 3, 4] where [...] should be (partially)
            broadcastable with points' batch dimensions. If shape is [..., 3, 4], it will be automatically
            converted to [..., 4, 4].

    Returns:
        Tensor: [..., 3] transformed points.

    Raises:
        ValueError: If points or transform have incorrect shape
    """
    # Validate inputs
    assert points.shape[-1] == 3, f"Points must be 3D (last dimension = 3), got shape: {points.shape}"

    # Handle different transform shapes
    if transform.shape[-2:] == (3, 4):
        transform = to4x4(transform)
    elif transform.shape[-2:] != (4, 4):
        raise ValueError(f"Transform last two dimensions must be (4, 4) or (3, 4), got shape: {transform.shape}")

    # Add homogeneous coordinates
    hom_coords = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)

    # Handle broadcasting by adding singleton dimensions
    points_batch_dims = hom_coords.ndim - 1  # exclude last dim
    transform_batch_dims = transform.ndim - 2  # exclude last two dims

    if points_batch_dims > transform_batch_dims:
        # Add singleton dims to transform: [..., 4, 4] -> [..., 1, ..., 1, 4, 4]
        for _ in range(points_batch_dims - transform_batch_dims):
            transform = transform.unsqueeze(-3)
    elif transform_batch_dims > points_batch_dims:
        # Add singleton dims to points: [..., 4] -> [..., 1, ..., 1, 4]
        for _ in range(transform_batch_dims - points_batch_dims):
            hom_coords = hom_coords.unsqueeze(-2)

    # Apply transformation using matrix multiplication
    points_transformed = torch.einsum("...i,...ji->...j", hom_coords, transform)
    return points_transformed[..., :3]


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.

    Adapted from MultiNeRF (Apache-2.0); see the file header for the full
    upstream notice. Original location:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-3,
    max_iterations: int = 10,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.

    Adapted from MultiNeRF (Apache-2.0); see the file header for the full
    upstream notice. Original location:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(torch.abs(denominator) > eps, x_numerator / denominator, torch.zeros_like(denominator))
        step_y = torch.where(torch.abs(denominator) > eps, y_numerator / denominator, torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


def normalize_with_norm(x: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """
    norm = torch.maximum(torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([EPS]).to(x))
    return x / norm, norm


def umeyama_alignment(x: Tensor, y: Tensor, with_scale: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes least squares solution parameters of a Sim(m) matrix to align point sets.

    Implements the Umeyama algorithm from:
    Umeyama, Shinji: "Least-squares estimation of transformation parameters
    between two point patterns." IEEE PAMI, 1991

    Args:
        x: Point set tensor of shape (m,n) where m is dimension and n is number of points
        y: Point set tensor of shape (m,n) where m is dimension and n is number of points
        with_scale: If True, also aligns scale. If False, keeps scale fixed at 1.0. Defaults to True.

    Returns:
        tuple:
            - r: Rotation matrix of shape (m,m)
            - t: Translation vector of shape (m,)
            - c: Scale factor (float)
    """
    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = torch.mean(x, dim=1)
    mean_y = torch.mean(y, dim=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (torch.norm(x - mean_x.unsqueeze(1)) ** 2)

    # covariance matrix, eq. 38 – computed in a vectorised manner for efficiency
    x_centered = x - mean_x.unsqueeze(1)
    y_centered = y - mean_y.unsqueeze(1)
    cov_xy = (y_centered @ x_centered.T) / float(n)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = torch.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = torch.eye(m, device=x.device, dtype=x.dtype)
    if torch.det(u) * torch.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1.0

    # rotation, eq. 40
    r = u @ s @ v

    # scale & translation, eq. 42 and 41
    c = (
        1.0 / sigma_x * torch.trace(torch.diag(d) @ s)
        if with_scale
        else torch.tensor(1.0, device=x.device, dtype=x.dtype)
    )
    t = mean_y - c * (r @ mean_x)

    return r, t, c


def depth_to_world_coords_points(
    depth_map: Tensor,
    extrinsics_c2w: Tensor,
    intrinsics: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (Tensor): Depth map of shape (H, W) or (..., H, W).
        extrinsics_c2w (Tensor): Camera extrinsic matrix of shape (3, 4) or (4, 4) or (..., 3, 4) or (..., 4, 4)
                                in camera-to-world (c2w) format.
        intrinsics (Tensor): Camera intrinsic matrix of shape (3, 3) or (..., 3, 3).
        eps (float, optional): Epsilon value for valid depth threshold. Defaults to 1e-8.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - World coordinates (..., H, W, 3)
            - Camera coordinates (..., H, W, 3)
            - Valid depth mask (..., H, W)
    """
    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_points(depth_map, intrinsics)  # Returns (..., H, W, 3)

    # Convert extrinsics to 4x4 if needed
    if extrinsics_c2w.shape[-2:] == (3, 4):
        transform = to4x4(extrinsics_c2w)
    else:  # Already (4, 4)
        transform = extrinsics_c2w

    # Apply the transformation to the camera coordinates
    # Both depth_to_points and transform_points support arbitrary shapes
    world_coords_points = transform_points(cam_coords_points, transform)

    return world_coords_points, cam_coords_points, point_mask


def voxel_downsample(points: Tensor, voxel_size: float) -> Tensor:
    """Voxel downsample a point cloud via Open3D, returns torch tensor on same device."""
    device = points.device
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.cpu().numpy()))
    pcd = pcd.voxel_down_sample(voxel_size)
    return torch.from_numpy(np.asarray(pcd.points)).float().to(device)


def icp_refine(pred_points: Tensor, gt_points: Tensor, threshold: float = 0.1) -> Tensor:
    """Refine predicted points with point-to-point ICP against ground truth.

    Follows the alignment protocol used in Pi3 / CUT3R evaluation.

    Args:
        pred_points: (N, 3) predicted point cloud (already coarsely aligned).
        gt_points: (M, 3) ground truth point cloud.
        threshold: Max correspondence distance for ICP.

    Returns:
        (N, 3) refined predicted points on the same device as input.
    """
    device = pred_points.device
    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_points.cpu().double().numpy()))
    pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points.cpu().double().numpy()))
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred,
        pcd_gt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_pred.transform(reg.transformation)
    return torch.from_numpy(np.asarray(pcd_pred.points)).float().to(device)
