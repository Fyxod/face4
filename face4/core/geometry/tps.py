"""Thin Plate Spline interpolation basis for control-point displacements."""
from __future__ import annotations

import torch


def tps_basis(size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    cy, cx = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )
    controls = torch.stack([cx.flatten(), cy.flatten()], dim=-1)
    n = int(controls.shape[0])
    distances = (controls[:, None] - controls[None, :]).square().sum(-1)
    kernel = distances * torch.log(distances.clamp_min(1e-8))
    kernel = kernel + torch.eye(n, device=device) * 1e-5
    polynomial = torch.cat([torch.ones(n, 1, device=device), controls], dim=1)
    system = torch.cat(
        [
            torch.cat([kernel, polynomial], dim=1),
            torch.cat([polynomial.T, torch.zeros(3, 3, device=device)], dim=1),
        ],
        dim=0,
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    query = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    qdist = (query[:, None] - controls[None, :]).square().sum(-1)
    qkernel = qdist * torch.log(qdist.clamp_min(1e-8))
    qpoly = torch.cat([torch.ones(query.shape[0], 1, device=device), query], dim=1)
    rhs = torch.cat([qkernel, qpoly], dim=1)
    inv = torch.linalg.inv(system)
    return rhs @ inv[:, :n]
