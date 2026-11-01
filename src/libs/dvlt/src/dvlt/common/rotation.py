# SPDX-License-Identifier: BSD-3-Clause
#
# `quat_to_mat`, `mat_to_quat`, `_sqrt_positive_part`, and
# `standardize_quaternion` are adapted from PyTorch3D
# (https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py),
# distributed under the BSD 3-Clause License:
#
#   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name Meta nor the names of its contributors may be used
#       to endorse or promote products derived from this software without
#       specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications from the PyTorch3D originals:
#   * switched from PyTorch3D's scalar-first (real-part-first, WXYZ /``rijk``)
#     quaternion convention to scalar-last (XYZW / ``ijkr``) to match the
#     convention used elsewhere in dvlt;
#   * ``so3_relative_angle`` and ``quaternion_slerp`` are new code authored
#     for dvlt and not part of the PyTorch3D originals.
#
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor


def quat_to_mat(quaternions: Tensor) -> Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape((*batch_dim, 9)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape((*batch_dim, 4))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: Tensor) -> Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: Tensor) -> Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def so3_relative_angle(R1: Tensor, R2: Tensor) -> Tensor:
    """Compute the angle between 2 rotation matrices.
    Args:
        R1: the 1st rotation matrix with shape (B, 3, 3)
        R2: the 2nd rotation matrix with shape (B, 3, 3)
    Returns:
        The rotation matrix angle in degrees with shape (B,)
    """
    # (Trace(R1.T @ R2) - 1)/2
    cos = (torch.einsum("...ii", R1.mT @ R2) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)
    diff = torch.abs(torch.arccos(cos))
    return diff


def quaternion_slerp(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation of two quaternions.

    Args:
        q1: The first quaternion with shape (..., 4)
        q2: The second quaternion with shape (..., 4)
        t: The interpolation factor with shape (..., 1)

    Returns:
        The interpolated quaternion with shape (..., 4)
    """
    q1 = standardize_quaternion(q1)
    q2 = standardize_quaternion(q2)

    # Ensure shortest path by flipping q2 when dot < 0
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # If the angle is very small, use linear interpolation and renormalize
    small_angle = sin_theta_0.abs() < 1e-8

    # Default to slerp
    s1 = torch.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = torch.sin(t * theta_0) / sin_theta_0
    out_slerp = s1 * q1 + s2 * q2

    # Lerp fallback for small angles
    out_lerp = (1.0 - t) * q1 + t * q2

    out = torch.where(small_angle, out_lerp, out_slerp)
    return standardize_quaternion(out)
