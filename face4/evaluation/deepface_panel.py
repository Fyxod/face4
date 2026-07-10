"""Post-run DeepFace identity panels.

DeepFace is evaluation-only and failures are recorded instead of raising.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


EVALUATORS = ("ArcFace", "Facenet512", "SFace")
PAIRS = (
    ("original_vs_perturbed_best_input", "original.png", "perturbed_best.png"),
    ("original_vs_perturbed_final_input", "original.png", "perturbed_final.png"),
    ("clean_edit_vs_perturbed_best_edit", "original_edited.png", "perturbed_best_edited.png"),
    ("clean_edit_vs_perturbed_final_edit", "original_edited.png", "perturbed_final_edited.png"),
    ("original_input_vs_clean_edit", "original.png", "original_edited.png"),
    ("perturbed_best_input_vs_perturbed_best_edit", "perturbed_best.png", "perturbed_best_edited.png"),
)


def run_deepface_panel(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from deepface import DeepFace
    except Exception as error:
        return [
            {
                "pair": pair_name,
                "model_name": model,
                "verified": None,
                "distance": None,
                "threshold": None,
                "confidence": None,
                "distance_metric": "cosine",
                "detector_backend": "skip",
                "error": f"DeepFace import failed: {error!r}",
            }
            for pair_name, _, _ in PAIRS
            for model in EVALUATORS
        ]

    for pair_name, left_name, right_name in PAIRS:
        left = run_dir / left_name
        right = run_dir / right_name
        for model in EVALUATORS:
            row: dict[str, Any] = {
                "pair": pair_name,
                "model_name": model,
                "distance_metric": "cosine",
                "detector_backend": "skip",
            }
            if not left.exists() or not right.exists():
                row.update({"verified": None, "distance": None, "threshold": None, "confidence": None, "error": "missing image"})
                rows.append(row)
                continue
            try:
                result = DeepFace.verify(
                    img1_path=str(left),
                    img2_path=str(right),
                    model_name=model,
                    detector_backend="skip",
                    distance_metric="cosine",
                    enforce_detection=False,
                    silent=True,
                )
                row.update(
                    {
                        "verified": result.get("verified"),
                        "distance": result.get("distance"),
                        "threshold": result.get("threshold"),
                        "confidence": result.get("confidence"),
                        "error": None,
                    }
                )
            except Exception as error:
                row.update({"verified": None, "distance": None, "threshold": None, "confidence": None, "error": repr(error)})
            rows.append(row)
    return rows


def write_identity_panel(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    import json

    (run_dir / "identity_panel.json").write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    if not rows:
        (run_dir / "identity_panel.csv").write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (run_dir / "identity_panel.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
