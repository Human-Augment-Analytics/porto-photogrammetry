# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

from dvlt.struct.cameras import Cameras


def extri_intri_to_cameras(extrinsics_c2w: Tensor, intrinsics: Tensor, image_size_hw: tuple[int, int]) -> Cameras:
    """
    Convert extrinsics (c2w, opencv camera coordinate system) and intrinsics to a Cameras object.

    Args:
        extrinsics_c2w: extrinsics [B, 3/4, 4] (c2w, opencv camera coordinate system)
        intrinsics: intrinsics [B, 3, 3]
        image_size_hw: image size (width, height)

    Returns:
        Cameras object
    """
    cameras = Cameras(
        camera_to_worlds=extrinsics_c2w[:, :3, :],
        fx=intrinsics[:, 0, 0, None],
        fy=intrinsics[:, 1, 1, None],
        cx=intrinsics[:, 0, 2, None],
        cy=intrinsics[:, 1, 2, None],
        width=image_size_hw[1],
        height=image_size_hw[0],
    )
    return cameras
