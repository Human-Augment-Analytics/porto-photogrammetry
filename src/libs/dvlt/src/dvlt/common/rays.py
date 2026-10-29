# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are adapted from third-party sources, all under the
# Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
#
# (1) Ray / RANSAC / homography helpers — adapted from Depth-Anything-3
#     (https://github.com/ByteDance-Seed/Depth-Anything-3), specifically
#     ``utils/ray_utils.py``. Original notice:
#
#         Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#         Licensed under the Apache License, Version 2.0 (the "License");
#         you may not use this file except in compliance with the License.
#         You may obtain a copy of the License at
#             http://www.apache.org/licenses/LICENSE-2.0
#         Unless required by applicable law or agreed to in writing, software
#         distributed under the License is distributed on an "AS IS" BASIS,
#         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#         implied.
#
# (2) S² (unit-sphere) helpers — ``s2_geodesic_distance`` and
#     ``s2_logmap_at_z1`` — adapted from AnyCalib's ``Unit3`` manifold class
#     (https://github.com/javrtg/AnyCalib, ``anycalib/manifolds.py``).
#     AnyCalib is distributed under the Apache License, Version 2.0;
#     copyright 2025 Javier Tirado-Garín (AnyCalib authors).
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ray utilities and ray-to-pose conversion.

Includes RANSAC-based homography fitting for camera parameter estimation
from world-space rays (adapted from Depth-Anything-3's ``ray_utils.py``;
see the file header for license / attribution).

The S² tangent-space helpers (``s2_geodesic_distance``, ``s2_logmap_at_z1``)
are specializations of the S² manifold log-map for the north-pole basepoint,
used by ``CameraLoss._ray_loss`` to express the per-pixel ray loss as a
2-DoF Euclidean L1 in tangent coordinates rather than an angle-aware loss
in 3-DoF unit-vector space. Adapted from AnyCalib's ``Unit3`` class
(https://github.com/javrtg/AnyCalib); see the file header for license /
attribution.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# Two unit rays whose relative angle θ satisfies cos(θ) >= 1 - S2_EPS_PARALLEL are
# treated as parallel (avoids 0/0 in θ/sin(θ) at the north pole). 1e-4 ≈ 0.81°.
S2_EPS_PARALLEL = 1e-4


# =============================================================================
# S² (unit-sphere) manifold helpers
# =============================================================================


def s2_geodesic_distance(x: Tensor, y: Tensor) -> Tensor:
    """Geodesic distance between pairs of points on the unit sphere (S²).

    The geodesic distance between two points ``x`` and ``y`` on S² is::

        d(x, y) = acos(x^T y)   (the angle between the vectors)

    But ``acos`` is numerically unstable near 0 and π, and its gradient blows up
    at angle 0. We use the equivalent and more stable

        d(x, y) = 2 * asin(0.5 * ||x - y||)

    which follows from ``sin(θ/2) = sqrt(0.5(1 - cosθ))`` and ``||x - y||² = 2 - 2 cosθ``.
    Slightly less accurate for big angles (close to antipodal), but those are
    expected to be rare for our usage.

    Args:
        x: ``(..., 3)`` unit vectors.
        y: ``(..., 3)`` unit vectors.

    Returns:
        ``(...,)`` geodesic distances in radians, in ``[0, π]``.
    """
    chordal_dist = torch.linalg.norm(x - y, dim=-1)
    eps = torch.finfo(chordal_dist.dtype).eps
    geodesic_dist = 2 * torch.asin((0.5 * chordal_dist).clamp(0, 1 - eps))
    return geodesic_dist


def s2_logmap_at_z1(vecs: Tensor) -> Tensor:
    """Logarithmic map on S² with the tangent plane anchored at ``(0, 0, 1)``.

    Since the tangent basis at ``(0, 0, 1)`` is the canonical XY axes, the general
    logmap ``Log_x(y) = Basis_x^T θ/sin(θ) (y - cos(θ) x)`` collapses to::

        Log_z1(y) = (θ / sin(θ)) * (y_x, y_y)

    where ``θ`` is the angle between ``y`` and ``(0, 0, 1)``. The ``θ/sin(θ)``
    factor is the small-angle correction that makes a 2D L1 distance in tangent
    space match the S² geodesic distance to first order.

    Args:
        vecs: ``(..., 3)`` unit vectors to map to the tangent plane at ``z=+1``.

    Returns:
        ``(..., 2)`` tangent-plane coordinates ``(y_x, y_y) * θ/sin(θ)``.
    """
    assert vecs.shape[-1] == 3
    z1 = torch.tensor([0.0, 0.0, 1.0], device=vecs.device, dtype=vecs.dtype)
    # cosθ = z; treat near-parallel rays as having θ ≈ 0 to dodge 0/0.
    not_parallel = vecs[..., 2:] < 1 - S2_EPS_PARALLEL
    theta = torch.where(not_parallel, s2_geodesic_distance(z1, vecs)[..., None], 1)
    return torch.where(not_parallel, theta / torch.sin(theta) * vecs[..., :2], vecs[..., :2])


# =============================================================================
# Ray computation utilities
# =============================================================================


def compute_world_rays(
    extrinsics_c2w: Tensor,
    intrinsics: Tensor,
    H: int,
    W: int,
) -> Tensor:
    """
    Compute world-space rays from camera parameters.

    NOTE: ray directions are NOT normalized, so that
    world_point = ray_origin + depth * ray_direction (where depth is z-depth).

    Args:
        extrinsics_c2w: Camera-to-world extrinsics (B, S, 4, 4)
        intrinsics: Camera intrinsics (B, S, 3, 3)
        H, W: Image height and width

    Returns:
        rays: World-space rays (B, S, H, W, 6) where:
            - channels 0-2: ray direction (unnormalized, z-component encodes depth scale)
            - channels 3-5: ray origin (camera position in world space)
    """
    B, S = extrinsics_c2w.shape[:2]
    device = extrinsics_c2w.device
    dtype = extrinsics_c2w.dtype

    # Create pixel grid (H, W, 2)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    # Use pixel centers
    pixel_coords = torch.stack([x_grid + 0.5, y_grid + 0.5, torch.ones_like(x_grid)], dim=-1)  # (H, W, 3)

    # Compute ray directions in camera space
    # K_inv @ [x, y, 1] gives direction where z-component relates to depth scaling
    intrinsics_inv = torch.inverse(intrinsics.float()).to(dtype)  # (B, S, 3, 3)

    # Broadcast and compute: (B, S, 3, 3) @ (H, W, 3) -> (B, S, H, W, 3)
    ray_dirs_cam = torch.einsum("bsij,hwj->bshwi", intrinsics_inv, pixel_coords)

    # Transform ray directions to world space
    R = extrinsics_c2w[:, :, :3, :3]  # (B, S, 3, 3)
    ray_dirs_world = torch.einsum("bsij,bshwj->bshwi", R, ray_dirs_cam)

    # Ray origins are the camera positions in world space
    ray_origins = extrinsics_c2w[:, :, :3, 3]  # (B, S, 3)
    ray_origins = ray_origins[:, :, None, None, :].expand(B, S, H, W, 3)

    # Combine direction and origin into 6-channel rays
    rays = torch.cat([ray_dirs_world, ray_origins], dim=-1)  # (B, S, H, W, 6)

    return rays


# =============================================================================
# Homography fitting utilities (from DA3)
# =============================================================================


def _ql_decomposition(A: Tensor) -> Tuple[Tensor, Tensor]:
    """QL decomposition via QR decomposition with permutation."""
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device, dtype=A.dtype)
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def _find_homography_weighted_batch(src_pts_batch: Tensor, dst_pts_batch: Tensor, weights_batch: Tensor) -> Tensor:
    """
    Batch weighted least squares homography estimation.

    Args:
        src_pts_batch: (B, K, 2) source points
        dst_pts_batch: (B, K, 2) destination points
        weights_batch: (B, K) confidence weights

    Returns:
        H: (B, 3, 3) homography matrices
    """
    B, K, _ = src_pts_batch.shape
    w = weights_batch.sqrt().unsqueeze(2)  # (B, K, 1)
    x = src_pts_batch[:, :, 0:1]
    y = src_pts_batch[:, :, 1:2]
    u = dst_pts_batch[:, :, 0:1]
    v = dst_pts_batch[:, :, 1:2]
    zeros = torch.zeros_like(x)

    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=2)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=2)
    A = torch.cat([A1, A2], dim=1)  # (B, 2K, 9)

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].reshape(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H


def _find_homography_weighted(src_pts: Tensor, dst_pts: Tensor, weights: Tensor) -> Tensor:
    """Single-sample weighted least squares homography estimation."""
    N = src_pts.shape[0]
    if N < 4:
        return torch.eye(3, dtype=src_pts.dtype, device=src_pts.device)

    w = weights.sqrt().unsqueeze(1)  # (N, 1)
    x = src_pts[:, 0:1]
    y = src_pts[:, 1:2]
    u = dst_pts[:, 0:1]
    v = dst_pts[:, 1:2]
    zeros = torch.zeros_like(x)

    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=1)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=1)
    A = torch.cat([A1, A2], dim=0)  # (2N, 9)

    _, _, Vh = torch.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    H = H / H[-1, -1]
    return H


def _ransac_homography_batch(
    src_pts: Tensor,
    dst_pts: Tensor,
    weights: Tensor,
    n_sample: int,
    n_iter: int = 100,
    reproj_threshold: float = 0.2,
    num_sample_for_ransac: int = 8,
    max_inlier_num: int = 8000,
) -> Tensor:
    """
    Batch RANSAC homography estimation.

    Args:
        src_pts: (B, N, 2) source points
        dst_pts: (B, N, 2) destination points
        weights: (B, N) confidence weights
        n_sample: number of top-weighted points to sample from
        n_iter: RANSAC iterations
        reproj_threshold: inlier threshold
        num_sample_for_ransac: points per RANSAC sample
        max_inlier_num: max inliers for final fit

    Returns:
        H: (B, 3, 3) homography matrices
    """
    B, N, _ = src_pts.shape
    device = src_pts.device

    # Select top weighted points
    sorted_idx = torch.argsort(weights, descending=True, dim=1)  # (B, N)
    candidate_idx = sorted_idx[:, :n_sample]  # (B, n_sample)

    # Generate random sampling indices with a fixed local generator so pose
    # metrics are deterministic regardless of prior global RNG state (which
    # varies with model architecture / init order).
    _ransac_gen = torch.Generator(device=device).manual_seed(42)
    rand_sample_iters_idx = torch.stack(
        [torch.randperm(n_sample, device=device, generator=_ransac_gen)[:num_sample_for_ransac] for _ in range(n_iter)],
        dim=0,
    )  # (n_iter, num_sample_for_ransac)

    rand_idx = candidate_idx[:, rand_sample_iters_idx]  # (B, n_iter, num_sample_for_ransac)

    # Construct batch input
    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_iter, num_sample_for_ransac)
    src_pts_batch = src_pts[b_idx, rand_idx]  # (B, n_iter, num_sample_for_ransac, 2)
    dst_pts_batch = dst_pts[b_idx, rand_idx]
    weights_batch = weights[b_idx, rand_idx]

    # Batch fit homography
    cB, cN = src_pts_batch.shape[:2]
    H_batch = _find_homography_weighted_batch(
        src_pts_batch.flatten(0, 1), dst_pts_batch.flatten(0, 1), weights_batch.flatten(0, 1)
    )
    H_batch = H_batch.unflatten(0, (cB, cN))  # (B, n_iter, 3, 3)

    # Evaluate inliers
    src_homo = torch.cat([src_pts, torch.ones(B, N, 1, dtype=src_pts.dtype, device=device)], dim=2)
    src_homo_expand = src_homo.unsqueeze(1).expand(B, n_iter, N, 3)
    dst_pts_expand = dst_pts.unsqueeze(1).expand(B, n_iter, N, 2)
    weights_expand = weights.unsqueeze(1).expand(B, n_iter, N)

    H_batch_flat = H_batch.reshape(-1, 3, 3)
    src_homo_flat = src_homo_expand.reshape(-1, N, 3)
    proj = torch.bmm(src_homo_flat, H_batch_flat.transpose(1, 2))
    proj_xy = proj[:, :, :2] / proj[:, :, 2:3].clamp(min=1e-8)
    proj_xy = proj_xy.reshape(B, n_iter, N, 2)

    error = ((proj_xy - dst_pts_expand) ** 2).sum(dim=3).sqrt()
    inlier_mask = error < reproj_threshold
    total_score = (inlier_mask * weights_expand).sum(dim=2)

    # Select best and refit
    best_idx = torch.argmax(total_score, dim=1)
    best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_idx]

    H_inlier_list = []
    for b in range(B):
        mask = best_inlier_mask[b]
        inlier_src = src_pts[b][mask]
        inlier_dst = dst_pts[b][mask]
        inlier_weights = weights[b][mask]

        if inlier_src.shape[0] < 4:
            H_inlier_list.append(torch.eye(3, device=device, dtype=src_pts.dtype))
            continue

        # Limit inliers for efficiency
        if inlier_src.shape[0] > max_inlier_num:
            sorted_idx = torch.argsort(inlier_weights, descending=True)
            keep_len = max(int(len(sorted_idx) * 0.95), max_inlier_num)
            sorted_idx = sorted_idx[:keep_len]
            perm = torch.randperm(len(sorted_idx), device=device)[:max_inlier_num]
            sorted_idx = sorted_idx[perm]
            inlier_src = inlier_src[sorted_idx]
            inlier_dst = inlier_dst[sorted_idx]
            inlier_weights = inlier_weights[sorted_idx]

        H_inlier = _find_homography_weighted(inlier_src, inlier_dst, inlier_weights)
        H_inlier_list.append(H_inlier)

    return torch.stack(H_inlier_list, dim=0)


def _compute_rotation_intrinsics_batch(
    rays_origin: Tensor,
    rays_target: Tensor,
    weights: Tensor,
    z_threshold: float = 1e-4,
    reproj_threshold: float = 0.2,
    n_iter: int = 100,
    num_sample_for_ransac: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute optimal rotation and intrinsics from ray correspondences.

    Args:
        rays_origin: (B, N, 3) identity camera rays
        rays_target: (B, N, 3) predicted ray directions
        weights: (B, N) confidence weights
        z_threshold: threshold for valid z values
        reproj_threshold: RANSAC inlier threshold

    Returns:
        R: (B, 3, 3) rotation matrices
        focal_lengths: (B, 2) focal lengths (fx, fy)
        principal_points: (B, 2) principal points (cx, cy)
    """
    device = rays_origin.device
    B, N, _ = rays_origin.shape

    # Mask invalid z values
    z_mask = (torch.abs(rays_target[:, :, 2]) > z_threshold) & (torch.abs(rays_origin[:, :, 2]) > z_threshold)

    rays_origin = rays_origin.clone()
    rays_target = rays_target.clone()

    # Project to 2D (divide by z)
    for i in range(B):
        mask = z_mask[i]
        rays_origin[i, mask, 0] /= rays_origin[i, mask, 2]
        rays_origin[i, mask, 1] /= rays_origin[i, mask, 2]
        rays_target[i, mask, 0] /= rays_target[i, mask, 2]
        rays_target[i, mask, 1] /= rays_target[i, mask, 2]

    rays_origin_2d = rays_origin[:, :, :2]
    rays_target_2d = rays_target[:, :, :2]

    # Zero out invalid weights
    weights = weights.clone()
    weights[~z_mask] = 0

    # RANSAC parameters
    sample_ratio = 0.3
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))

    # Process in chunks for memory efficiency
    max_chunk_size = 2
    A_list = []
    for i in range(0, B, max_chunk_size):
        chunk_end = min(i + max_chunk_size, B)
        A = _ransac_homography_batch(
            rays_origin_2d[i:chunk_end],
            rays_target_2d[i:chunk_end],
            weights[i:chunk_end],
            n_sample=n_sample,
            n_iter=n_iter,
            reproj_threshold=reproj_threshold,
            num_sample_for_ransac=num_sample_for_ransac,
        )
        # Ensure positive determinant
        det_mask = torch.linalg.det(A) < 0
        A[det_mask] = -A[det_mask]
        A_list.append(A.to(device))

    A = torch.cat(A_list, dim=0)

    # Decompose homography into rotation and intrinsics
    R_list, f_list, pp_list = [], [], []
    for i in range(A.shape[0]):
        R, L = _ql_decomposition(A[i])
        L = L / L[2, 2]
        f = torch.stack([L[0, 0], L[1, 1]])
        pp = torch.stack([L[2, 0], L[2, 1]])
        R_list.append(R)
        f_list.append(f)
        pp_list.append(pp)

    R = torch.stack(R_list)
    f = torch.stack(f_list)
    pp = torch.stack(pp_list)

    return R, f, pp


def _create_identity_ray_grid_normalized(
    num_patches_y: int,
    num_patches_x: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Create identity camera rays in normalized coordinate system.

    The grid positions must match the positions sampled by F.interpolate(align_corners=True)
    when downsampling from (H, W) to (num_patches_y, num_patches_x).

    Uses a 2x2 normalized image space with identity intrinsics fx=fy=1, cx=cy=1.

    Args:
        num_patches_y, num_patches_x: patch grid dimensions
        H, W: original image dimensions
        device, dtype: tensor device and dtype

    Returns:
        rays: (num_patches_y, num_patches_x, 3) identity ray directions in normalized space
    """
    # F.interpolate with align_corners=True samples at pixel positions:
    # x_pixel = i * (W - 1) / (num_patches_x - 1) for i in [0, num_patches_x-1]
    # y_pixel = j * (H - 1) / (num_patches_y - 1) for j in [0, num_patches_y-1]
    #
    # But our rays are generated at pixel CENTERS: ray at index (i, j) is for pixel center (i+0.5, j+0.5)
    # So the interpolated ray at patch position (m, n) corresponds to pixel center position:
    # x = m * (W - 1) / (num_patches_x - 1) + 0.5
    # y = n * (H - 1) / (num_patches_y - 1) + 0.5
    #
    # Convert to normalized [0, 2] coordinates: x_norm = x * 2 / W, y_norm = y * 2 / H

    if num_patches_x > 1:
        x_pixel = torch.linspace(0, W - 1, num_patches_x, dtype=dtype, device=device)
    else:
        x_pixel = torch.tensor([(W - 1) / 2], dtype=dtype, device=device)

    if num_patches_y > 1:
        y_pixel = torch.linspace(0, H - 1, num_patches_y, dtype=dtype, device=device)
    else:
        y_pixel = torch.tensor([(H - 1) / 2], dtype=dtype, device=device)

    # Add 0.5 to convert from pixel index to pixel center (matching compute_world_rays)
    x_pixel = x_pixel + 0.5
    y_pixel = y_pixel + 0.5

    # Convert to normalized [0, 2] coordinates
    x_norm = x_pixel * 2.0 / W
    y_norm = y_pixel * 2.0 / H

    y_grid, x_grid = torch.meshgrid(y_norm, x_norm, indexing="ij")

    # For identity intrinsics K = [[1, 0, 1], [0, 1, 1], [0, 0, 1]] in normalized 2x2 space:
    # Ray direction at normalized coord (x, y) is K^-1 @ [x, y, 1] = [x-1, y-1, 1]
    ray_x = x_grid - 1.0  # cx = 1 in normalized space
    ray_y = y_grid - 1.0  # cy = 1 in normalized space
    ray_z = torch.ones_like(ray_x)

    rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)
    return rays


def _camray_to_caminfo(
    camray: Tensor,
    confidence: Tensor,
    H: int,
    W: int,
    reproj_threshold: float = 0.2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convert camera rays to camera parameters.

    Uses DA3's normalized coordinate system (2x2 image, fx=fy=1, cx=cy=1) for
    homography fitting, then converts results to pixel coordinates.

    Args:
        camray: (B, S, num_patches_y, num_patches_x, 6) world-space rays
        confidence: (B, S, num_patches_y, num_patches_x) confidence weights
        H, W: original image dimensions
        reproj_threshold: RANSAC reprojection threshold

    Returns:
        R: (B, S, 3, 3) rotation matrices
        T: (B, S, 3) translation vectors
        focal_lengths: (B, S, 2) focal lengths (fx, fy) in pixels
        principal_points: (B, S, 2) principal points (cx, cy) in pixels
    """
    B, S, num_patches_y, num_patches_x, _ = camray.shape
    device = camray.device
    dtype = camray.dtype

    # Create identity camera rays in normalized coordinate system
    # Grid positions must match F.interpolate(align_corners=True) sampling positions
    I_cam_rays = _create_identity_ray_grid_normalized(num_patches_y, num_patches_x, H, W, device, dtype)

    # Expand for batch dimensions
    I_cam_rays = I_cam_rays.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1, -1)

    # Flatten batch and spatial dimensions
    camray_flat = camray.flatten(0, 1).flatten(1, 2)  # (B*S, N, 6)
    I_cam_flat = I_cam_rays.flatten(0, 1).flatten(1, 2)  # (B*S, N, 3)
    conf_flat = confidence.flatten(0, 1).flatten(1, 2)  # (B*S, N)

    # Detach for RANSAC (no gradients through discrete operations)
    camray_flat = camray_flat.detach()
    I_cam_flat = I_cam_flat.detach()
    conf_flat = conf_flat.detach()

    # Normalize ray directions for homography fitting
    # The input rays may be unnormalized (DA3 convention), but homography fitting
    # works with normalized directions (dividing by z gives same 2D coords either way,
    # but normalized is more numerically stable)
    ray_dirs = camray_flat[:, :, :3]
    ray_dirs_normalized = F.normalize(ray_dirs, dim=-1)

    # Compute rotation and intrinsics via homography fitting
    # The homography maps normalized identity coords to predicted ray coords (both after /z)
    R, focal_lengths_norm, principal_points_norm = _compute_rotation_intrinsics_batch(
        I_cam_flat,
        ray_dirs_normalized,
        conf_flat,
        reproj_threshold=reproj_threshold,
    )

    # Compute translation as weighted average of ray origins
    T = (camray_flat[:, :, 3:] * conf_flat.unsqueeze(-1)).sum(dim=1) / (conf_flat.sum(dim=-1, keepdim=True) + 1e-8)

    # Reshape outputs
    R = R.reshape(B, S, 3, 3)
    T = T.reshape(B, S, 3)
    focal_lengths_norm = focal_lengths_norm.reshape(B, S, 2)
    principal_points_norm = principal_points_norm.reshape(B, S, 2)

    # Convert from normalized (2x2 image) to pixel coordinates
    # DA3 returns: 1/focal_lengths_norm (inverted), principal_points_norm + 1.0 (shifted)
    # These are in normalized [0, 2] coordinate space
    #
    # For normalized image [0, 2] with intrinsics K_norm = [[f_n, 0, c_n], [0, f_n, c_n], [0, 0, 1]]:
    #   f_pixel = f_norm * (pixel_dim / 2)
    #   c_pixel = c_norm * (pixel_dim / 2)

    # Invert to get normalized focal lengths (DA3 returns 1/f)
    # Clamp both min and max to avoid extreme values from ill-conditioned homographies
    # In normalized [0, 2] space: f_norm=1 corresponds to 45° FoV
    # - f_norm=0.2 -> very wide angle (~157° FoV)
    # - f_norm=10 -> telephoto (~11° FoV)
    focal_lengths_norm_inv = 1.0 / focal_lengths_norm.clamp(min=0.1, max=10.0)
    # Shift principal points (DA3 adds 1.0 to shift from [-1, 1] to [0, 2] range)
    principal_points_norm_shifted = principal_points_norm + 1.0

    # Scale to pixel coordinates
    focal_lengths = torch.stack(
        [
            focal_lengths_norm_inv[:, :, 0] * W / 2,
            focal_lengths_norm_inv[:, :, 1] * H / 2,
        ],
        dim=-1,
    )

    principal_points = torch.stack(
        [
            principal_points_norm_shifted[:, :, 0] * W / 2,
            principal_points_norm_shifted[:, :, 1] * H / 2,
        ],
        dim=-1,
    )

    return R, T, focal_lengths, principal_points


# =============================================================================
# Main ray-to-pose interface
# =============================================================================


def rays_to_pose(
    rays: Tensor,
    rays_conf: Tensor,
    H: int,
    W: int,
    patch_size: int = 16,
) -> Tuple[Tensor, Tensor]:
    """
    Convert world-space rays to camera extrinsics and intrinsics.

    Uses RANSAC-based homography fitting to estimate camera parameters
    from predicted world-space rays and their confidences.

    Args:
        rays: World-space rays (B, S, H, W, 6) - first 3 are direction, last 3 are origin
        rays_conf: Ray confidence (B, S, H, W)
        H, W: Image height and width
        patch_size: Patch size used for ray prediction

    Returns:
        extrinsics_c2w: (B, S, 4, 4) camera-to-world matrices
        intrinsics: (B, S, 3, 3) intrinsic matrices
    """
    B, S = rays.shape[:2]
    device = rays.device
    dtype = rays.dtype

    # Downsample rays to patch resolution
    ph, pw = H // patch_size, W // patch_size
    rays_patch = (
        F.interpolate(
            rays.view(B * S, H, W, 6).permute(0, 3, 1, 2),
            size=(ph, pw),
            mode="bilinear",
            align_corners=True,
        )
        .permute(0, 2, 3, 1)
        .view(B, S, ph, pw, 6)
    )

    conf_patch = F.interpolate(
        rays_conf.view(B * S, 1, H, W),
        size=(ph, pw),
        mode="bilinear",
        align_corners=True,
    ).view(B, S, ph, pw)

    # Convert rays to camera parameters
    with torch.no_grad():
        R, T, focal_lengths, principal_points = _camray_to_caminfo(
            rays_patch, confidence=conf_patch, H=H, W=W, reproj_threshold=0.2
        )

    # Build extrinsics (c2w)
    extrinsics_c2w = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).clone()
    extrinsics_c2w[:, :, :3, :3] = R
    extrinsics_c2w[:, :, :3, 3] = T

    # Build intrinsics (focal_lengths and principal_points are already in pixel units)
    intrinsics = torch.zeros(B, S, 3, 3, device=device, dtype=dtype)
    intrinsics[:, :, 0, 0] = focal_lengths[:, :, 0]  # fx
    intrinsics[:, :, 1, 1] = focal_lengths[:, :, 1]  # fy
    intrinsics[:, :, 0, 2] = principal_points[:, :, 0]  # cx
    intrinsics[:, :, 1, 2] = principal_points[:, :, 1]  # cy
    intrinsics[:, :, 2, 2] = 1.0

    return extrinsics_c2w, intrinsics
