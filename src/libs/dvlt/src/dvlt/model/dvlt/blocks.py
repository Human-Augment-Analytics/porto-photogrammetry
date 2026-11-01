# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth-scaled attention blocks and recurrent AA block for the DVLT model."""

from torch import Tensor, nn

from dvlt.model_components import Attention, DropPath, LayerScale, Mlp

from .depth_scaling import ContinuousDepthScaling, IntervalDepthScaling


class DepthScaledAttentionBlock(nn.Module):
    """Pre-norm attention + MLP with optional depth scaling and drop_path.

    gate_mode controls depth scaling behavior:
      "gated":    x += s_attn * attn(...); x += s_mlp * mlp(...); x = s_out * x
      "no_sout":  x += s_attn * attn(...); x += s_mlp * mlp(...)
      "none":     x += attn(...); x += mlp(...)

    time_mode selects the depth-scaling module (ignored when gate_mode="none"):
      "continuous": sinusoidal embedding of scalar t ∈ [0, 1]
      "interval":   sinusoidal embeddings of (t_now, t_next) concatenated
    """

    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=0.01,
        qk_norm=True,
        rope=None,
        drop_path=0.0,
        gate_mode="gated",
        time_mode="interval",
    ):
        super().__init__()
        self.gate_mode = gate_mode
        self.time_mode = time_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, qk_norm=qk_norm, rope=rope
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * ffn_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if gate_mode != "none":
            num_gates = 3 if gate_mode == "gated" else 2
            if time_mode == "interval":
                self.depth_scale = IntervalDepthScaling(dim, num_gates=num_gates)
            else:
                self.depth_scale = ContinuousDepthScaling(dim, num_gates=num_gates)
        else:
            self.depth_scale = None

    def forward(self, x: Tensor, k: Tensor = None, pos=None) -> Tensor:
        """x: (B, N, C), k: step index / t / t_pair tensor (or None to skip scaling)."""
        if self.depth_scale is not None and k is not None:
            s = self.depth_scale(k).unsqueeze(1)
            if self.gate_mode == "gated":
                s_attn, s_mlp, s_out = s.chunk(3, dim=-1)
            else:
                s_attn, s_mlp = s.chunk(2, dim=-1)
                s_out = None
        else:
            s_attn = s_mlp = s_out = None

        branch = self.ls1(self.attn(self.norm1(x), pos=pos))
        if s_attn is not None:
            branch = s_attn * branch
        x = x + self.drop1(branch)

        branch = self.ls2(self.mlp(self.norm2(x)))
        if s_mlp is not None:
            branch = s_mlp * branch
        x = x + self.drop2(branch)

        if s_out is not None:
            x = s_out * x
        return x


class LoopedAABlock(nn.Module):
    """Frame attention + global attention pair with optional depth scaling.

    Alternates intra-frame attention over the ``P`` per-frame tokens with
    inter-frame ("global") attention over the ``S * P`` flattened sequence.
    Built on dvlt's :class:`DepthScaledAttentionBlock`, with a
    step-conditioned depth-scaling gate.
    """

    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=0.01,
        qk_norm=True,
        rope=None,
        drop_path=0.0,
        gate_mode="gated",
        time_mode="interval",
    ):
        super().__init__()
        kw = dict(
            dim=dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            drop_path=drop_path,
            gate_mode=gate_mode,
            time_mode=time_mode,
        )
        self.frame_attn = DepthScaledAttentionBlock(**kw, rope=rope)
        self.global_attn = DepthScaledAttentionBlock(**kw, rope=None)

    def forward(self, x, k_frame, k_batch, rope_pos, B, S):
        """k_frame, k_batch: int/float tensors or None depending on mode."""
        _, P, C = x.shape
        x = x.reshape(B * S, P, C)
        x = self.frame_attn(x, k_frame, pos=rope_pos.reshape(B * S, P, 2))
        x = x.reshape(B, S * P, C)
        x = self.global_attn(x, k_batch, pos=None)
        return x.reshape(B * S, P, C)
