# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for AMP (Automatic Mixed Precision)."""

import functools

import torch


def force_fp32(func, device_type: str = "cuda") -> callable:
    """Force function to run in full precision (fp32) mode.

    This decorator disables any active autocast context manager during the execution
    of the decorated function, ensuring operations run in full precision.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function that runs in fp32 mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.amp.autocast(device_type=device_type, enabled=False):
            return func(*args, **kwargs)

    return wrapper
