# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth-scaling modules for step-conditioned gating in the recurrent AA blocks."""

import math

import torch
from torch import Tensor, nn


class ContinuousDepthScaling(nn.Module):
    """Learned scaling parameterized by continuous time t ∈ [0, 1].

    Sinusoidal frequency schedule matches TimestepEmbedder (cos-then-sin,
    max_period=10000), followed by a projection to per-channel scale
    vectors centered at 1.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, num_gates: int = 2):
        super().__init__()
        half = hidden_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_gates * dim),
        )
        nn.init.normal_(self.proj[-1].weight, std=1e-4)
        nn.init.normal_(self.proj[-1].bias, std=1e-4)

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) float in [0, 1]."""
        args = t.unsqueeze(-1) * self.freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.proj(emb) + 1


class IntervalDepthScaling(nn.Module):
    """Learned scaling conditioned on an interval (t_now, t_next) in [0, 1]^2.

    Embeds both endpoints with sinusoidal frequencies, concatenates, and
    projects to per-channel scale vectors centered at 1. The model sees
    both where it is and where the next step lands, enabling dt-aware
    gating for adaptive step counts. Zero-initialized output so the gate
    starts at identity regardless of (t_now, t_next).
    """

    def __init__(self, dim: int, hidden_dim: int = 64, num_gates: int = 3):
        super().__init__()
        half = hidden_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_gates * dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def _sinusoidal(self, val: Tensor) -> Tensor:
        args = val.unsqueeze(-1) * self.freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, t_pair: Tensor) -> Tensor:
        """t_pair: (B, 2) float — [t_now, t_next]."""
        t_now, t_next = t_pair[:, 0], t_pair[:, 1]
        emb = torch.cat([self._sinusoidal(t_now), self._sinusoidal(t_next)], dim=-1)
        return self.proj(emb) + 1
