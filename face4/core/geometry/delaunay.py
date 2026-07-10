"""Fixed Delaunay barycentric interpolation for piecewise-affine displacement."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay


def delaunay_barycentric(size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gy, gx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    controls = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    triangulation = Delaunay(controls)
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    query = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    simplex = triangulation.find_simplex(query)
    simplex = np.maximum(simplex, 0)
    transform = triangulation.transform[simplex]
    bary = np.einsum("pij,pj->pi", transform[:, :2], query - transform[:, 2])
    bary = np.c_[bary, 1.0 - bary.sum(axis=1)]
    indices = triangulation.simplices[simplex].astype(np.int64)
    return (
        torch.from_numpy(indices).to(device=device),
        torch.from_numpy(bary.astype(np.float32)).to(device=device),
    )
