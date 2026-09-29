# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# :func:`create_uv_grid` is a port of MoGe's ``normalized_view_plane_uv``:
#   https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py
# Original work licensed under the MIT License:
#   https://opensource.org/licenses/MIT

"""Utility helpers for decoder heads.

Currently provides:

* :func:`create_uv_grid` — normalized UV coordinate grid (MoGe convention).
* :func:`extri_intri_to_pose_enc` / :func:`pose_enc_to_extri_intri` — pack and
  unpack camera extrinsics + intrinsics into a compact 9-channel encoding
  (translation 3 + quaternion 4 + vertical/horizontal field of view 2).
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from dvlt.common.amp import force_fp32
from dvlt.common.rotation import mat_to_quat, quat_to_mat


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a normalized UV grid of shape ``(width, height, 2)``.

    Top-left corner sits at ``(-aspect / diag, -1 / diag)`` and bottom-right
    at ``(aspect / diag, 1 / diag)`` where ``diag = sqrt(1 + aspect ** 2)``,
    matching MoGe's ``normalized_view_plane_uv`` convention.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv


# -----------------------------------------------------------------------------
# Camera pose encoding.
#
# Channel layout: ``[Tx, Ty, Tz, qx, qy, qz, qw, fov_h, fov_w]`` with the
# principal point fixed at the image centre. Single-encoding, parameter-free.
# -----------------------------------------------------------------------------

_FOV_EPS = 1e-3


def _fov_clamp_range() -> Tuple[float, float]:
    """Numerical bounds enforced on FoV channels before focal-length recovery."""
    return _FOV_EPS, float(torch.pi) - _FOV_EPS


def extri_intri_to_pose_enc(
    extrinsics: Tensor,
    intrinsics: Optional[Tensor] = None,
    image_size_hw: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """Pack extrinsics (and optionally intrinsics) into a compact pose encoding.

    Output layout:

    * 7-channel when ``intrinsics is None``: ``[T (3) | quat_xyzw (4)]``.
    * 9-channel otherwise: append ``[fov_h, fov_w]`` computed under the
      assumption of a centred principal point,
      ``fov_axis = 2 * atan((side_pixels / 2) / focal_axis)`` with ``fov_h``
      from image height and ``fy``, and ``fov_w`` from image width and ``fx``.

    Args:
        extrinsics: ``(..., 3, 4)`` ``[R | t]``; agnostic to whether ``R`` is
            camera-to-world or world-to-camera.
        intrinsics: optional ``(..., 3, 3)`` with ``fx`` at ``[..., 0, 0]`` and
            ``fy`` at ``[..., 1, 1]``.
        image_size_hw: ``(H, W)`` in pixels; required when ``intrinsics`` is
            provided.

    Returns:
        ``(..., 7)`` if ``intrinsics is None`` else ``(..., 9)``.
    """
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    quat_xyzw = mat_to_quat(rotation)

    components = [translation, quat_xyzw]

    if intrinsics is not None:
        if image_size_hw is None:
            raise ValueError("image_size_hw is required when intrinsics is provided")
        height, width = image_size_hw
        half_h = 0.5 * height
        half_w = 0.5 * width
        fov_h = 2.0 * torch.atan(half_h / intrinsics[..., 1, 1])
        fov_w = 2.0 * torch.atan(half_w / intrinsics[..., 0, 0])
        components.extend([fov_h.unsqueeze(-1), fov_w.unsqueeze(-1)])

    return torch.cat(components, dim=-1).float()


@force_fp32
def pose_enc_to_extri_intri(
    pose_encoding: Tensor,
    images_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Inverse of :func:`extri_intri_to_pose_enc`.

    Quaternions are L2-normalized internally so callers may pass raw network
    outputs. FoV channels are clamped to ``(eps, pi - eps)`` before being
    inverted into focal lengths to keep ``tan`` finite.

    Args:
        pose_encoding: ``(..., 7)`` (translation + quaternion only) or
            ``(..., 9)`` (additionally encoding ``fov_h``/``fov_w``).
        images_hw: ``(H, W)`` in pixels; consulted only when the encoding
            carries the trailing FoV pair.

    Returns:
        ``(extrinsics, intrinsics)``. ``extrinsics`` has shape ``(..., 3, 4)``.
        ``intrinsics`` is ``(..., 3, 3)`` when ``images_hw`` is given and the
        encoding contains FoV channels; otherwise ``None``.
    """
    translation = pose_encoding[..., :3]
    quat_xyzw = F.normalize(pose_encoding[..., 3:7], dim=-1)
    rotation = quat_to_mat(quat_xyzw)
    extrinsics = torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)

    if images_hw is None or pose_encoding.shape[-1] < 9:
        return extrinsics, None

    fov_lo, fov_hi = _fov_clamp_range()
    fov_h = pose_encoding[..., 7].clamp(fov_lo, fov_hi)
    fov_w = pose_encoding[..., 8].clamp(fov_lo, fov_hi)

    height, width = images_hw
    fy = (0.5 * height) / torch.tan(0.5 * fov_h)
    fx = (0.5 * width) / torch.tan(0.5 * fov_w)

    cx_value = 0.5 * width
    cy_value = 0.5 * height
    cx = torch.full_like(fx, cx_value)
    cy = torch.full_like(fy, cy_value)
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)

    row0 = torch.stack([fx, zero, cx], dim=-1)
    row1 = torch.stack([zero, fy, cy], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    intrinsics = torch.stack([row0, row1, row2], dim=-2)

    return extrinsics, intrinsics
