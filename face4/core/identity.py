"""edited-output ArcFace identity objective utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class IdentityReference:
    embedding_original: torch.Tensor
    original_embedding_norm: float


def prepare_identity_reference(arcface, original: torch.Tensor) -> IdentityReference:
    with torch.no_grad():
        embedding = arcface.embedding(original).detach()
    return IdentityReference(embedding_original=embedding, original_embedding_norm=float(embedding.norm(dim=1).mean().cpu()))


def identity_objective(arcface, perturbed: torch.Tensor, reference: IdentityReference) -> tuple[torch.Tensor, dict[str, Any]]:
    embedding = arcface.embedding(perturbed)
    cosine = F.cosine_similarity(reference.embedding_original, embedding, dim=1)
    cosine_raw = cosine.mean()
    Z = cosine_raw
    cosine_distance = 1.0 - cosine_raw
    l2 = torch.sqrt((embedding - reference.embedding_original).square().sum(dim=1).clamp_min(1e-12)).mean()
    angle = torch.acos(cosine_raw.clamp(-1.0, 1.0))
    score_pct = cosine_raw.clamp(0.0, 1.0) * 100.0
    terms = {
        "identity_cosine_similarity_raw": cosine_raw,
        "identity_cosine_distance": cosine_distance,
        "identity_l2_embedding_distance": l2,
        "identity_angle_radians": angle,
        "identity_angle_degrees": angle * (180.0 / math.pi),
        "identity_similarity_score_pct": score_pct,
        "original_embedding_norm": reference.original_embedding_norm,
        "perturbed_embedding_norm": embedding.norm(dim=1).mean(),
    }
    return Z, terms


def face_loss(Z: torch.Tensor) -> torch.Tensor:
    return Z
