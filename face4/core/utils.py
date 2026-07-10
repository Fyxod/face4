"""Small FACE helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps


def json_default(value: Any) -> Any:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_input_difference(original_path: Path, perturbed_path: Path, out_path: Path, amplify: float = 8.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not original_path.exists() or not perturbed_path.exists():
        Image.new("RGB", (512, 512), "#f3f4f6").save(out_path)
        return
    original = Image.open(original_path).convert("RGB")
    perturbed = Image.open(perturbed_path).convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(perturbed, original)
    diff = ImageEnhance.Brightness(diff).enhance(amplify)
    diff = ImageOps.autocontrast(diff, cutoff=0.5)
    diff.save(out_path, optimize=True)
