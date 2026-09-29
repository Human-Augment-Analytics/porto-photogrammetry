# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common type definitions."""

from typing import Literal, Union

import torch


TORCH_DEVICE = Union[torch.device, str]

Colormaps = Literal["default", "turbo", "viridis", "magma", "inferno", "inferno_r", "cividis", "gray", "pca"]
