# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import numpy as np
import torch


EPS: float = np.finfo(float).eps * 4.0

# Colors
WHITE = torch.tensor([1.0, 1.0, 1.0])
BLACK = torch.tensor([0.0, 0.0, 0.0])


class DataField(str, Enum):
    """Field names for data batch dictionaries used throughout the codebase.

    Using this enum ensures consistency and helps with refactoring.

    Tensor shapes:
        - SEQ_NAME (list[str]): [B,] Sequence name identifier
        - IDS (Tensor): [B, S] Frame indices
        - ORIGINAL_SIZES (Tensor): [B, S, 2] Original image sizes (H, W)
        - SCALE_FACTOR (Tensor): [B] Scale factors for scene normalization
        - METRIC_SCALE (Tensor): [B] Whether the sequence GT is metric scale
        - IS_SYNTHETIC (Tensor): [B] Whether the sequence is synthetic or real

        - IMAGES (Tensor): [B, S, C, H, W] RGB images
        - DEPTHS (Tensor): [B, S, H, W] Depth maps

        - EXTRINSICS_C2W (Tensor): [B, S, 4, 4] Camera-to-world extrinsic matrices
        - INTRINSICS (Tensor): [B, S, 3, 3] Camera intrinsic matrices

        - WORLD_POINTS (Tensor): [B, S, H, W, 3] 3D points in world coordinates
        - WORLD_RAYS (Tensor): [B, S, H, W, 6] World-space rays (direction[3], origin[3])
        - POINT_MASKS (Tensor): [B, S, H, W] Boolean masks for valid 3D points

    where:
        - B: Batch size
        - S: Sequence length
        - C: Channel dimension (typically 3 for RGB)
        - H, W: Height and width of images
    """

    # Sequence metadata
    SEQ_NAME = "seq_name"
    IDS = "ids"
    ORIGINAL_SIZES = "original_sizes"
    SCALE_FACTOR = "scale_factor"
    METRIC_SCALE = "metric_scale"
    IS_SYNTHETIC = "is_synthetic"

    # Image data
    IMAGES = "images"
    DEPTHS = "depths"

    # Camera parameters
    EXTRINSICS_C2W = "extrinsics_c2w"
    INTRINSICS = "intrinsics"

    # 3D data
    WORLD_POINTS = "world_points"
    WORLD_RAYS = "world_rays"
    POINT_MASKS = "point_masks"

    # Augmentation control
    CONSISTENT_AUG = "consistent_aug"  # bool: whether to apply consistent augmentation across frames


class PredictionField(str, Enum):
    """Field names for model prediction dictionaries used throughout the codebase.

    Using this enum ensures consistency and helps with refactoring.

    Tensor shapes:
        - CAMERAS (List[Cameras]): List of B Cameras objects, each with shape [S]

        - DEPTHS (Tensor): [B, S, H, W] Predicted depth maps
        - DEPTHS_CONF (Tensor): [B, S, H, W] Confidence scores for depth predictions

        - WORLD_POINTS (Tensor): [B, S, H, W, 3] 3D world points derived from depth/pose/intrinsic
        - WORLD_POINTS_DIRECT (Tensor): [B, S, H, W, 3] Directly predicted 3D points in world coordinates
        - WORLD_POINTS_CONF (Tensor): [B, S, H, W] Confidence scores for 3D point predictions
        - WORLD_POINTS_DIRECT_CONF (Tensor): [B, S, H, W] Confidence scores for directly predicted 3D points

    where:
        - B: Batch size
        - S: Sequence length
        - H, W: Height and width of images
    """

    # Cameras
    CAMERAS = "cameras"

    # Depth related fields
    DEPTHS = "depths"
    DEPTHS_CONF = "depths_conf"

    # 3D geometry fields
    WORLD_POINTS = "world_points"  # 3D points derived from depth maps
    WORLD_POINTS_DIRECT = "world_points_direct"  # 3D points predicted directly
    WORLD_POINTS_DIRECT_CONF = "world_points_direct_conf"
