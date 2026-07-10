"""Rolling-shutter-style row displacement."""
from __future__ import annotations

import math

import torch


def rolling_field(yy: torch.Tensor, amp_px: torch.Tensor, shear_px: torch.Tensor) -> torch.Tensor:
    shift = amp_px * torch.sin(math.pi * yy) + shear_px * yy
    zeros = torch.zeros_like(shift)
    return torch.cat([shift, zeros], dim=1)
