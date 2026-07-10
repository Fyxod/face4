"""Legacy DCT-basis coordinate warp.

This module parameterizes a spatial displacement field with cosine
basis functions. It does not modify image DCT coefficients and is not
used by the active FACE pipeline.
"""
from __future__ import annotations

import math

import torch


def dct_basis(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    basis = []
    for ky in range(size):
        for kx in range(size):
            if ky == 0 and kx == 0:
                continue
            value = torch.cos(math.pi * ky * (yy + 0.5) / height) * torch.cos(
                math.pi * kx * (xx + 0.5) / width
            )
            basis.append(value / value.square().mean().sqrt().clamp_min(1e-6))
    return torch.stack(basis, dim=0)
