# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ResidualConvBlock and DecoderHead (conv mode) are adapted from MoGe
# (https://github.com/microsoft/MoGe), distributed by Microsoft under the MIT
# License. See THIRD_PARTY_LICENSES.md for the full license text.

"""Decoder heads for the DVLT model: spatial dense heads and camera head."""

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from dvlt.model_components import Block, LayerScale, create_uv_grid


def _concat_uv(x: Tensor, aspect_ratio: float) -> Tensor:
    """Concatenate normalized UV coordinates (MoGe convention) to a spatial feature map."""
    h, w = x.shape[-2:]
    uv = create_uv_grid(w, h, aspect_ratio=aspect_ratio, dtype=x.dtype, device=x.device)
    uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    return torch.cat([x, uv], dim=1)


class ResidualConvBlock(nn.Module):
    """Pre-norm residual conv block with GroupNorm and replicate padding.

    Architecture adapted from MoGe (https://github.com/microsoft/MoGe),
    distributed by Microsoft under the MIT License.
    """

    def __init__(self, channels: int, hidden_channels: int = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.layers = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(max(1, hidden_channels // 32), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x):
        return x + self.layers(x)


class DecoderHead(nn.Module):
    """Transformer decoder with configurable spatial output.

    Shared front-end: proj_in → transformer blocks → LayerNorm.
    Three output modes controlled by ``head_type``:

      "linear" — per-patch linear projection + pixel_shuffle.
      "conv"   — Progressive ConvTranspose2d upsampling with UV positional
                 encoding, adapted from MoGe's ``ConvStack`` decoder
                 (https://github.com/microsoft/MoGe, MIT License).
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        embed_dim=384,
        depth=2,
        num_heads=6,
        patch_size=14,
        mlp_ratio=4.0,
        rope=None,
        head_type="linear",
        dim_upsample=(256, 128, 64),
        num_res_blocks=2,
        hidden_dim_multiplier=2,
        last_conv_channels=32,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.head_type = head_type
        self.init_values = init_values

        # ---- shared transformer front-end ----
        self.proj_in = nn.Linear(in_dim, embed_dim)
        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    qk_norm=True,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        # ---- output stage ----
        if head_type == "conv":
            dim_upsample = list(dim_upsample)

            in_channels = [embed_dim] + dim_upsample[:-1]
            self.upsample_blocks = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, dim_upsample, strict=False):
                block = nn.ModuleList([self._make_upsampler(in_ch + 2, out_ch)])
                for _ in range(num_res_blocks):
                    block.append(ResidualConvBlock(out_ch, hidden_dim_multiplier * out_ch))
                self.upsample_blocks.append(block)

            self.output_block = nn.Sequential(
                nn.Conv2d(dim_upsample[-1] + 2, last_conv_channels, 3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, 1),
            )
        else:
            self.head = nn.Linear(embed_dim, out_dim * patch_size * patch_size)

        self._checkpoint_conv = False
        self._use_reentrant = False

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

    def reinit_transformer_front_end(self) -> None:
        """Reinitialize ``proj_in``, transformer ``blocks``, and final ``norm`` (not the output stage)."""
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        self.norm.reset_parameters()
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    m.reset_parameters()
            # Only reset LayerScale gamma if LayerScale is actually in use;
            # when init_values is falsy the Block replaces ls1/ls2 with Identity.
            if self.init_values:
                if isinstance(blk.ls1, LayerScale):
                    nn.init.constant_(blk.ls1.gamma, self.init_values)
                if isinstance(blk.ls2, LayerScale):
                    nn.init.constant_(blk.ls2.gamma, self.init_values)

    def output_stage_parameters(self):
        """Yield parameters belonging only to the output stage (not the transformer front-end)."""
        if self.head_type == "conv":
            yield from self.upsample_blocks.parameters()
            yield from self.output_block.parameters()
        else:
            yield from self.head.parameters()

    def replace_output_stage(
        self,
        new_head_type: str,
        dim_upsample=(256, 128, 64),
        num_res_blocks=2,
        hidden_dim_multiplier=2,
        last_conv_channels=32,
    ):
        """Replace the output stage while keeping the shared transformer front-end.

        Useful for swapping a pretrained linear head for a conv head while
        retaining the learned proj_in / transformer / norm weights.

        NOTE: If used with an optimizer, call this *before* the optimizer is
        created (the old parameter tensors become stale after replacement).
        """
        if self.head_type == "conv":
            out_dim = self.output_block[-1].out_channels
        else:
            out_dim = self.head.out_features // (self.patch_size**2)

        for attr in ("head", "upsample_blocks", "output_block"):
            if hasattr(self, attr):
                delattr(self, attr)

        self.head_type = new_head_type
        embed_dim = self.embed_dim
        patch_size = self.patch_size

        if new_head_type == "conv":
            dim_upsample = list(dim_upsample)
            in_channels = [embed_dim] + dim_upsample[:-1]
            self.upsample_blocks = nn.ModuleList()
            for in_ch, out_ch in zip(in_channels, dim_upsample, strict=False):
                block = nn.ModuleList([self._make_upsampler(in_ch + 2, out_ch)])
                for _ in range(num_res_blocks):
                    block.append(ResidualConvBlock(out_ch, hidden_dim_multiplier * out_ch))
                self.upsample_blocks.append(block)
            self.output_block = nn.Sequential(
                nn.Conv2d(dim_upsample[-1] + 2, last_conv_channels, 3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, 1),
            )
        else:
            self.head = nn.Linear(embed_dim, out_dim * patch_size * patch_size)

    def enable_gradient_checkpointing(self, use_reentrant: bool = False) -> None:
        """Enable gradient checkpointing for conv upsample stages."""
        self._checkpoint_conv = True
        self._use_reentrant = use_reentrant

    @staticmethod
    def _make_upsampler(in_channels: int, out_channels: int) -> nn.Sequential:
        up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
        )
        up[0].weight.data[:] = up[0].weight.data[:, :, :1, :1]
        return up

    def forward(self, x, H, W, patch_start_idx=0, pos=None):
        B = x.shape[0]
        ph, pw = H // self.patch_size, W // self.patch_size

        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x, pos=pos)
        features = x
        x = self.norm(x)[:, patch_start_idx:]

        if self.head_type == "conv":
            aspect_ratio = W / H
            x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)
            for block in self.upsample_blocks:
                x = _concat_uv(x, aspect_ratio)
                for layer in block:
                    if self._checkpoint_conv and self.training:
                        x = grad_checkpoint(layer, x, use_reentrant=self._use_reentrant)
                    else:
                        x = layer(x)
            x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
            x = _concat_uv(x, aspect_ratio)
            with torch.amp.autocast("cuda", enabled=False):
                x = self.output_block(x.float())
        else:
            with torch.amp.autocast("cuda", enabled=False):
                x = self.head(x.float())
                x = x.transpose(-1, -2).view(B, -1, ph, pw)
                x = F.pixel_shuffle(x, self.patch_size)

        return x, features


class SimpleCameraHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, pose_dim=9):
        super().__init__()
        self.pose_dim = pose_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_pose = nn.Linear(hidden_dim, pose_dim)

    def forward(self, cls_token, B, S):
        x = self.mlp(cls_token)
        with torch.amp.autocast("cuda", enabled=False):
            pose = self.fc_pose(x.float())
            pose[:, 7:] = F.relu(pose[:, 7:])
        return pose.view(B, S, self.pose_dim)
