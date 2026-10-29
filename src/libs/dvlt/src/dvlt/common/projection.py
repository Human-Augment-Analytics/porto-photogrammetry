# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection related functions."""

import torch
from torch import Tensor

from dvlt.common.types import TORCH_DEVICE


def inverse_pinhole(intrinsic_matrix: Tensor) -> Tensor:
    """Calculate inverse of pinhole projection matrix.

    Args:
        intrinsic_matrix (Tensor): [..., 3, 3] intrinsics or single [3, 3]
            intrinsics.

    Returns:
        Tensor:  Inverse of input intrinisics.
    """
    # Functional implementation without in-place writes so that autograd versions are preserved.
    # Accept both [..., 3, 3] and [3, 3].
    squeeze = False
    if intrinsic_matrix.dim() == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0)
        squeeze = True

    fx = intrinsic_matrix[..., 0, 0]
    fy = intrinsic_matrix[..., 1, 1]
    cx = intrinsic_matrix[..., 0, 2]
    cy = intrinsic_matrix[..., 1, 2]

    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)

    row0 = torch.stack((1.0 / fx, zeros, -cx / fx), dim=-1)
    row1 = torch.stack((zeros, 1.0 / fy, -cy / fy), dim=-1)
    row2 = torch.stack((zeros, zeros, ones), dim=-1)

    inv = torch.stack((row0, row1, row2), dim=-2)

    if squeeze:
        inv = inv.squeeze(0)
    return inv


def points_inside_image(
    points_coord: Tensor,
    depths: Tensor,
    images_hw: Tensor | tuple[int, int],
) -> Tensor:
    """Generate binary mask.

    Creates a mask that is true for all point coordiantes that lie inside the
    image,

    Args:
        points_coord (Tensor): 2D pixel coordinates of shape [..., 2].
        depths (Tensor): Associated depth of each 2D pixel coordinate.
        images_hw:  (Tensor| tuple[int, int]]) Associated tensor of image
                    dimensions, shape [..., 2] or single height, width pair.

    Returns:
        Tensor: Binary mask of points inside an image.
    """
    mask = torch.ones_like(depths)
    h: int | Tensor
    w: int | Tensor

    if isinstance(images_hw, tuple):
        h, w = images_hw
    else:
        h, w = images_hw[..., 0], images_hw[..., 1]
    mask = torch.logical_and(mask, depths > 0)
    mask = torch.logical_and(mask, points_coord[..., 0] > 0)
    mask = torch.logical_and(mask, points_coord[..., 0] < w - 1)
    mask = torch.logical_and(mask, points_coord[..., 1] > 0)
    mask = torch.logical_and(mask, points_coord[..., 1] < h - 1)
    return mask


def project_points(points: Tensor, intrinsics: Tensor) -> Tensor:
    """Project points to pixel coordinates with given intrinsics.

    Args:
        points: (..., 3) 3D coordinates. Supports arbitrary leading dimensions.
        intrinsics: (3, 3) or (..., 3, 3) intrinsic camera matrices where [...] should be
                   (partially) broadcastable with points' batch dimensions.

    Returns:
        Tensor: (..., 2) 2D pixel coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    assert points.shape[-1] == 3, "Input coordinates must be 3 dimensional!"

    # Validate intrinsics shape
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsics last two dimensions must be (3, 3), got shape: {intrinsics.shape}")

    # Convert to homogeneous coordinates (normalize by Z)
    hom_coords = points / points[..., 2:3]

    # Handle broadcasting by adding singleton dimensions (same pattern as transform_points)
    points_batch_dims = hom_coords.ndim - 1  # exclude last dim
    intrinsics_batch_dims = intrinsics.ndim - 2  # exclude last two dims

    if points_batch_dims > intrinsics_batch_dims:
        # Add singleton dims to intrinsics: [..., 3, 3] -> [..., 1, ..., 1, 3, 3]
        for _ in range(points_batch_dims - intrinsics_batch_dims):
            intrinsics = intrinsics.unsqueeze(-3)
    elif intrinsics_batch_dims > points_batch_dims:
        # Add singleton dims to points: [..., 3] -> [..., 1, ..., 1, 3]
        for _ in range(intrinsics_batch_dims - points_batch_dims):
            hom_coords = hom_coords.unsqueeze(-2)

    # Apply intrinsics using einsum (note: intrinsics is transposed in the einsum)
    pts_2d = torch.einsum("...i,...ji->...j", hom_coords, intrinsics)
    return pts_2d[..., :2]


def generate_depth_map(
    points: Tensor,
    intrinsics: Tensor,
    image_hw: tuple[int, int],
) -> Tensor:
    """Generate depth map for given pointcloud.

    Args:
        points: (N, 3) coordinates.
        intrinsics: (3, 3) intrinsic camera matrices.
        image_hw: (tuple[int,int]) height, width of the image

    Returns:
        Tensor: Projected depth map of the given pointcloud.
               Invalid depth has 0 values. When multiple points project to
               the same pixel, the one with the smallest depth value is kept.
    """
    pts_2d = project_points(points, intrinsics).round()
    depths = points[:, 2]
    depth_map = points.new_full(image_hw, float("inf"))
    mask = points_inside_image(pts_2d, depths, image_hw)
    pts_2d = pts_2d[mask].long()
    depths_masked = depths[mask]

    # Create indices for scatter operation
    indices = pts_2d[:, 1] * image_hw[1] + pts_2d[:, 0]
    depth_flat = depth_map.view(-1)

    # Using scatter_reduce to keep minimum depth values
    depth_flat.scatter_reduce_(0, indices, depths_masked, reduce="amin", include_self=True)

    # Replace inf with 0 for invalid depths
    depth_map = depth_map.view(image_hw)
    depth_map[depth_map == float("inf")] = 0.0

    return depth_map


def unproject_points(points: Tensor, depths: Tensor, intrinsics: Tensor) -> Tensor:
    """Un-projects pixel coordinates to 3D coordinates with given intrinsics.

    Args:
        points: (..., 2) 2D pixel coordinates. Supports arbitrary leading dimensions.
        depths: (...,) or (..., 1) depth values corresponding to points.
        intrinsics: (3, 3) or (..., 3, 3) intrinsic camera matrices where [...] should be
                   (partially) broadcastable with points' batch dimensions.

    Returns:
        Tensor: (..., 3) 3D coordinates.

    Raises:
        ValueError: Shape of input points is not valid for computation.
    """
    assert points.shape[-1] == 2, "Input coordinates must be 2 dimensional!"

    # Validate intrinsics shape
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsics last two dimensions must be (3, 3), got shape: {intrinsics.shape}")

    # Ensure depths has the same number of dimensions as points (add last dim if needed)
    if len(depths.shape) == len(points.shape) - 1:
        depths = depths.unsqueeze(-1)

    # Compute inverse intrinsics
    inv_intrinsics = inverse_pinhole(intrinsics)

    # Create homogeneous coordinates
    hom_coords = torch.cat([points, torch.ones_like(points)[..., 0:1]], -1)

    # Handle broadcasting by adding singleton dimensions (same pattern as transform_points)
    points_batch_dims = hom_coords.ndim - 1  # exclude last dim
    intrinsics_batch_dims = inv_intrinsics.ndim - 2  # exclude last two dims

    if points_batch_dims > intrinsics_batch_dims:
        # Add singleton dims to intrinsics: [..., 3, 3] -> [..., 1, ..., 1, 3, 3]
        for _ in range(points_batch_dims - intrinsics_batch_dims):
            inv_intrinsics = inv_intrinsics.unsqueeze(-3)
    elif intrinsics_batch_dims > points_batch_dims:
        # Add singleton dims to points: [..., 3] -> [..., 1, ..., 1, 3]
        for _ in range(intrinsics_batch_dims - points_batch_dims):
            hom_coords = hom_coords.unsqueeze(-2)

    # Apply inverse intrinsics using einsum (note: inv_intrinsics is transposed in the einsum)
    pts_3d = torch.einsum("...i,...ji->...j", hom_coords, inv_intrinsics)
    pts_3d *= depths
    return pts_3d


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates=True,
    device: TORCH_DEVICE = "cpu",
) -> Tensor:
    """Generates a coordinate grid for an image.
    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.
    Return:
        Tensor: returns a grid tensor with shape :math:`(H, W, 2)`.
    """
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float32)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float32)
    else:
        # Use arange for integer coordinates (more efficient than linspace)
        xs = torch.arange(width, device=device, dtype=torch.float32)
        ys = torch.arange(height, device=device, dtype=torch.float32)

    # This returns (width, height) grids which when stacked give (H, W, 2)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    base_grid = torch.stack([xx, yy], dim=-1)
    return base_grid


def depth_to_points(depth_maps: Tensor, intrinsics: Tensor) -> Tensor:
    """Convert depth map(s) to pointcloud(s).

    Args:
        depth_maps (Tensor): (..., H, W) depth values. Supports arbitrary leading dimensions.
        intrinsics (Tensor): (3, 3) or (..., 3, 3) intrinsic matrix.

    Returns:
        Tensor: (..., H, W, 3) 3D points with spatial structure preserved.
    """
    # Get spatial dimensions
    *batch_dims, height, width = depth_maps.shape

    # Create 2D grid points
    points2d = create_meshgrid(height, width, normalized_coordinates=False, device=depth_maps.device)
    # points2d is (H, W, 2)

    # For higher dimensional inputs, we need to flatten and then reshape
    if batch_dims:
        # Flatten batch dimensions and spatial dimensions for unproject_points
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Expand points2d to (batch_size, H*W, 2)
        points2d_flat = points2d.view(-1, 2).unsqueeze(0).repeat(batch_size, 1, 1)
        depths_flat = depth_maps.view(batch_size, -1)  # (batch_size, H*W)

        # Handle intrinsics
        if len(intrinsics.shape) == 2:
            # Single intrinsics for all batches
            intrinsics_expanded = intrinsics
        else:
            # Reshape intrinsics to match flattened batch
            intrinsics_expanded = intrinsics.view(batch_size, 3, 3)

        # Unproject to 3D points
        points_flat = unproject_points(points2d_flat, depths_flat, intrinsics_expanded)  # (batch_size, H*W, 3)

        # Reshape back to spatial structure
        points_ref = points_flat.view(*batch_dims, height, width, 3)
    else:
        # Simple case: just (H, W) depth map
        points_ref = unproject_points(points2d, depth_maps, intrinsics)

    return points_ref


@torch.jit.script
def fisheye624_unproject_helper(uv, params, max_iters: int = 5):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model. There is no analytical solution for the inverse of the project()
    function so this solves an optimization problem using Newton's method to get
    the inverse.
    Inputs:
        uv: BxNx2 tensor of 2D pixels to be unprojected
        params: Bx16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        xyz: BxNx3 tensor of 3D rays of uv points with z = 1.
    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model assumes fu=fv. This unproject function holds that:
    X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]
    and
    x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]
    Author: Daniel DeTone (ddetone@meta.com)
    """

    assert uv.ndim == 3, "Expected batched input shaped BxNx3"
    assert params.ndim == 2
    assert params.shape[-1] == 16 or params.shape[-1] == 15, "This model allows fx != fy"
    eps = 1e-6
    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    # Compute xr_yr using Newton's method.
    xr_yr = uv_dist.clone()  # Initial guess.
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        # Tangential terms.
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
        # Thin Prism terms.
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        # Compute the derivative of uv_dist w.r.t. xr_yr.
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = 1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = 1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (xr_yr[:, :, 0] * temp1)
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (xr_yr[:, :, 1] * temp1)
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (xr_yr[:, :, 0] * temp2)
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (xr_yr[:, :, 1] * temp2)
        # Compute 2x2 inverse manually here since torch.inverse() is very slow.
        # Because this is slow: inv = duv_dist_dxr_yr.inverse()
        # About a 10x reduction in speed with above line.
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        # Manually compute 2x2 @ 2x1 matrix multiply.
        # Because this is slow: step = (inv @ (uv_dist - uv_dist_est)[..., None])[..., 0]
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        # Newton step.
        xr_yr = xr_yr + step

    # Compute theta using Newton's method.
    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        # handle dthd_th close to 0.
        step = torch.where(dthd_th.abs() > eps, step, torch.sign(step) * eps * 10.0)
        th = th + step
    # Compute the ray direction using theta and xr_yr.
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    return ray


# unproject 2D point to 3D with fisheye624 model
def fisheye624_unproject(coords: Tensor, distortion_params: Tensor) -> Tensor:
    return fisheye624_unproject_helper(coords.unsqueeze(0), distortion_params[0].unsqueeze(0))


def intrinsics_from_rays(rays: Tensor) -> Tensor:
    """Estimate intrinsics from rays under the assumption that the camera model is pinhole
    Do a minimal least squares optimization that minimizes the reprojection error of the rays
    """
    B, S, H, W, _ = rays.shape

    # Create pixel coordinate grid (H, W, 2) with coordinates [u, v]
    # Use integer coordinates to match data loading convention (no +0.5)
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=rays.device, dtype=rays.dtype),
        torch.arange(W, device=rays.device, dtype=rays.dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([u_coords, v_coords], dim=-1)  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5  # Add 0.5 to get pixel centers

    intrinsics = torch.zeros((B, S, 3, 3), device=rays.device, dtype=rays.dtype)

    for i in range(B):
        for j in range(S):
            # Get rays for this image: (H, W, 3)
            rays_img = rays[i, j]  # (H, W, 3)

            # Flatten to (H*W, 3) and (H*W, 2)
            rays_flat = rays_img.reshape(-1, 3)  # (N, 3)
            pixels_flat = pixel_coords.reshape(-1, 2)  # (N, 2)

            # Filter out invalid rays (where z <= 0)
            valid_mask = rays_flat[:, 2] > 1e-6
            rays_valid = rays_flat[valid_mask]
            pixels_valid = pixels_flat[valid_mask]

            if rays_valid.shape[0] < 4:
                raise ValueError("Not enough valid rays for batch %d, image %d" % (i, j))

            # Compute normalized ray directions: (bx/bz, by/bz)
            ray_proj = rays_valid[:, :2] / rays_valid[:, 2:3]  # (N, 2)

            # Build linear system: A * [fx, fy, cx, cy]^T = b
            # For u: fx*(bx/bz) + cx = u  =>  [bx/bz, 0, 1, 0] * params = u
            # For v: fy*(by/bz) + cy = v  =>  [0, by/bz, 0, 1] * params = v
            N = ray_proj.shape[0]
            A = torch.zeros((2 * N, 4), device=rays.device, dtype=rays.dtype)
            b = torch.zeros((2 * N,), device=rays.device, dtype=rays.dtype)

            # Fill in A and b
            A[0::2, 0] = ray_proj[:, 0]  # fx coefficient for u equations
            A[0::2, 2] = 1.0  # cx coefficient for u equations
            A[1::2, 1] = ray_proj[:, 1]  # fy coefficient for v equations
            A[1::2, 3] = 1.0  # cy coefficient for v equations
            b[0::2] = pixels_valid[:, 0]  # u values
            b[1::2] = pixels_valid[:, 1]  # v values

            # Solve least squares: params = (A^T A)^{-1} A^T b
            params = torch.linalg.lstsq(A, b, rcond=None).solution

            # Extract parameters
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            # Clamp to reasonable values
            fx = torch.clamp(fx, min=1.0, max=10 * W)
            fy = torch.clamp(fy, min=1.0, max=10 * H)
            cx = torch.clamp(cx, min=0.0, max=W)
            cy = torch.clamp(cy, min=0.0, max=H)

            # Build intrinsic matrix
            intrinsics[i, j, 0, 0] = fx
            intrinsics[i, j, 1, 1] = fy
            intrinsics[i, j, 0, 2] = cx
            intrinsics[i, j, 1, 2] = cy
            intrinsics[i, j, 2, 2] = 1.0

    return intrinsics
