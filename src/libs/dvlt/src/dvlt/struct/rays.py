# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray bundle dataclass adapted from Nerfstudio."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor

from dvlt.struct.tensor import TensorDataclass


@dataclass
class RayBundle(TensorDataclass):
    """A bundle of ray parameters."""

    origins: Tensor
    """Ray origins (XYZ) with shape (*batch, 3)"""
    directions: Tensor
    """Unit ray direction vector with shape (*batch, 3)"""
    pixel_area: Tensor
    """Projected area of pixel a distance 1 away from origin with shape (*batch, 1)"""
    camera_indices: Optional[Tensor] = None
    """Camera indices with shape (*batch, 1) and dtype long"""
    nears: Optional[Tensor] = None
    """Distance along ray to start sampling with shape (*batch, 1)"""
    fars: Optional[Tensor] = None
    """Rays Distance along ray to stop sampling with shape (*batch, 1)"""
    metadata: Dict[str, Tensor] = field(default_factory=dict)
    """Additional metadata or data needed for interpolation with shape (num_rays, latent_dims), will mimic shape of rays"""
    times: Optional[Tensor] = None
    """Times at which rays are sampled with shape (*batch, 1)"""

    def set_camera_indices(self, camera_index: int) -> None:
        """Sets all the camera indices to a specific camera index.

        Args:
            camera_index: Camera index.
        """
        self.camera_indices = torch.ones_like(self.origins[..., 0:1]).long() * camera_index

    def __len__(self) -> int:
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int) -> "RayBundle":
        """Flattens RayBundle and extracts chunk given start and end indices.

        Args:
            start_idx: Start index of RayBundle chunk.
            end_idx: End index of RayBundle chunk.

        Returns:
            Flattened RayBundle with end_idx-start_idx rays.

        """
        return self.flatten()[start_idx:end_idx]
