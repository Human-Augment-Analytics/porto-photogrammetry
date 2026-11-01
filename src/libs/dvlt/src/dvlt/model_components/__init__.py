# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable model building blocks shared across dvlt models.

The package is organized as flat modules at the top level (DINOv2 backbone,
pose-encoding helpers, head activations) plus the :mod:`layers` sub-package
(transformer primitives). All publicly used symbols are re-exported here so that callers can
write ``from dvlt.model_components import Block`` instead of digging into
submodule paths.
"""

from dvlt.model_components.dinov2 import vit_base, vit_giant2, vit_large, vit_small
from dvlt.model_components.head_activations import activate_head
from dvlt.model_components.layers.attention import Attention, set_attn_backend
from dvlt.model_components.layers.block import Block, DropPath, LayerScale, Mlp, NestedTensorBlock
from dvlt.model_components.layers.patch_embed import PatchEmbed
from dvlt.model_components.layers.rope import PositionGetter, RotaryPositionEmbedding2D
from dvlt.model_components.layers.swiglu_ffn import SwiGLUFFNFused

# LOCAL PATCH (augenblick): the `loss` sub-package (training losses) is not
# vendored; its imports and re-exports are removed.
from dvlt.model_components.pose_encoding import (
    create_uv_grid,
    extri_intri_to_pose_enc,
    pose_enc_to_extri_intri,
)


__all__ = [
    # Backbones
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_giant2",
    # Heads / pose helpers
    "activate_head",
    "create_uv_grid",
    "extri_intri_to_pose_enc",
    "pose_enc_to_extri_intri",
    # Transformer primitives
    "Attention",
    "set_attn_backend",
    "Block",
    "NestedTensorBlock",
    "DropPath",
    "LayerScale",
    "Mlp",
    "PatchEmbed",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
    "SwiGLUFFNFused",
]
