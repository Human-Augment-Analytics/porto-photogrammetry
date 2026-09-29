# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output activations for dvlt's decoder heads.

Heads emit raw tensors with channel-first layout ``(B, C, H, W)``.
:func:`activate_head` reorders them to channel-last ``(B, H, W, C)``,
optionally splits off a trailing single-channel confidence map, and applies
data / confidence activations chosen via typed enums.
"""

from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


# Conservative ceiling on the pre-exp argument so the result still fits
# comfortably inside bf16's dynamic range (bf16 saturates around exp(88)).
_EXP_INPUT_CEILING_BF16: float = 15.0


class DataActivation(Enum):
    """Per-pixel activation applied to the data channels of a head output."""

    IDENTITY = "identity"
    EXP_CLAMPED = "exp_clamped"


class ConfActivation(Enum):
    """Per-pixel activation applied to the confidence channel (if present)."""

    EXP_PLUS_ONE = "exp_plus_one"


def _coerce_data_activation(value) -> DataActivation:
    if isinstance(value, DataActivation):
        return value
    return DataActivation(value)


def _coerce_conf_activation(value) -> Optional[ConfActivation]:
    if value is None:
        return None
    if isinstance(value, ConfActivation):
        return value
    return ConfActivation(value)


def _apply_data_activation(data: Tensor, activation: DataActivation) -> Tensor:
    if activation is DataActivation.IDENTITY:
        return data
    if activation is DataActivation.EXP_CLAMPED:
        return torch.exp(data.clamp(max=_EXP_INPUT_CEILING_BF16))
    raise ValueError(f"Unhandled DataActivation: {activation!r}")


def _apply_conf_activation(conf: Tensor, activation: ConfActivation) -> Tensor:
    if activation is ConfActivation.EXP_PLUS_ONE:
        return torch.exp(conf) + 1.0
    raise ValueError(f"Unhandled ConfActivation: {activation!r}")


def _split_data_and_conf(channel_last: Tensor, has_conf: bool) -> Tuple[Tensor, Optional[Tensor]]:
    if not has_conf:
        return channel_last, None
    n_channels = channel_last.shape[-1]
    if n_channels < 2:
        raise ValueError(f"Need at least 2 channels to peel off a confidence map; got C={n_channels}")
    data_part, conf_part = torch.split(channel_last, [n_channels - 1, 1], dim=-1)
    return data_part, conf_part.squeeze(-1)


def activate_head(
    out: Tensor,
    activation="identity",
    conf_activation="exp_plus_one",
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply the data + (optional) confidence activations to a head output.

    Args:
        out: Raw head output, shape ``(B, C, H, W)``.
        activation: Either a :class:`DataActivation` member or one of its
            string values (``"identity"``, ``"exp_clamped"``).
        conf_activation: Either a :class:`ConfActivation` member, one of its
            string values (``"exp_plus_one"``), or ``None``. When ``None`` the
            entire channel axis is treated as data and the second return
            value is ``None``.

    Returns:
        ``(data, conf)``. ``data`` has shape ``(B, H, W, C_data)`` where
        ``C_data == C`` if ``conf_activation is None`` and ``C - 1`` otherwise.
        ``conf`` has shape ``(B, H, W)`` when a confidence channel was
        peeled off, otherwise ``None``.
    """
    data_act = _coerce_data_activation(activation)
    conf_act = _coerce_conf_activation(conf_activation)

    channel_last = out.movedim(1, -1).contiguous()

    raw_data, raw_conf = _split_data_and_conf(channel_last, has_conf=conf_act is not None)
    activated_data = _apply_data_activation(raw_data, data_act)
    activated_conf = _apply_conf_activation(raw_conf, conf_act) if raw_conf is not None else None

    return activated_data, activated_conf
