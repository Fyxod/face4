#!/usr/bin/env python3
"""Simple ArcFace evaluation between original, original_edit, and perturbed images."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from face4.models.arcface import ArcFaceIResNet100
from face4.core.identity import prepare_identity_reference
from face4.core.image_metrics import pil_to_tensor


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    # Ensure embeddings are 1D
    if embedding1.dim() > 1:
        embedding1 = embedding1.flatten()
    if embedding2.dim() > 1:
        embedding2 = embedding2.flatten()
    sim = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=1)
    return float(sim[0].item())


def evaluate_run(run_dir: Path, arcface, device):
    """Evaluate a single run directory."""
    original_path = run_dir / "original.png"
    original_edit_path = run_dir / "original_edited.png"
    perturbed_best_path = run_dir / "perturbed_best.png"
    perturbed_best_edit_path = run_dir / "perturbed_best_edited.png"
    perturbed_final_path = run_dir / "perturbed_final.png"
    perturbed_final_edit_path = run_dir / "perturbed_final_edited.png"

    if not all(p.exists() for p in [original_path, original_edit_path, perturbed_best_edit_path, perturbed_final_edit_path, perturbed_best_path, perturbed_final_path]):
        return None

    original = Image.open(original_path).convert("RGB")
    original_edit = Image.open(original_edit_path).convert("RGB")
    perturbed_best = Image.open(perturbed_best_path).convert("RGB")
    perturbed_best_edit = Image.open(perturbed_best_edit_path).convert("RGB")
    perturbed_final = Image.open(perturbed_final_path).convert("RGB")
    perturbed_final_edit = Image.open(perturbed_final_edit_path).convert("RGB")

    original_tensor = pil_to_tensor(original, device)
    original_edit_tensor = pil_to_tensor(original_edit, device)
    perturbed_best_tensor = pil_to_tensor(perturbed_best, device)
    perturbed_best_edit_tensor = pil_to_tensor(perturbed_best_edit, device)
    perturbed_final_tensor = pil_to_tensor(perturbed_final, device)
    perturbed_final_edit_tensor = pil_to_tensor(perturbed_final_edit, device)

    with torch.no_grad():
        emb_original = arcface.embedding(original_tensor)
        emb_original_edit = arcface.embedding(original_edit_tensor)
        emb_perturbed_best = arcface.embedding(perturbed_best_tensor)
        emb_perturbed_best_edit = arcface.embedding(perturbed_best_edit_tensor)
        emb_perturbed_final = arcface.embedding(perturbed_final_tensor)
        emb_perturbed_final_edit = arcface.embedding(perturbed_final_edit_tensor)

        # Original vs Original Edit
        orig_vs_edit = cosine_similarity(emb_original, emb_original_edit)

        # Original vs Perturbed Best/Final (edited)
        orig_vs_perturbed_best = cosine_similarity(emb_original, emb_perturbed_best_edit)
        orig_vs_perturbed_final = cosine_similarity(emb_original, emb_perturbed_final_edit)

        # Perturbed -> Perturbed Edit (identity preservation through edit)
        perturbed_best_vs_edit = cosine_similarity(emb_perturbed_best, emb_perturbed_best_edit)
        perturbed_final_vs_edit = cosine_similarity(emb_perturbed_final, emb_perturbed_final_edit)

    return {
        "original_vs_original_edit": orig_vs_edit,
        "original_vs_perturbed_best": orig_vs_perturbed_best,
        "original_vs_perturbed_final": orig_vs_perturbed_final,
        "perturbed_best_vs_perturbed_best_edit": perturbed_best_vs_edit,
        "perturbed_final_vs_perturbed_final_edit": perturbed_final_vs_edit,
    }


def main():
    run_root = Path("outputs/edited_output_identity_5/20260708_102529_edited_output_identity_all_sequential/runs/edited_output_identity/instructpix2pix_arcface_iresnet100")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arcface = ArcFaceIResNet100("models/arcface/iresnet100.pth", device)

    results = []
    for case_dir in sorted(run_root.iterdir()):
        if not case_dir.is_dir():
            continue
        result = evaluate_run(case_dir, arcface, device)
        if result:
            result["case"] = case_dir.name
            results.append(result)
            print(f"\n{case_dir.name}:")
            print(f"  Original vs Original Edit:           {result['original_vs_original_edit']:.4f} ({result['original_vs_original_edit']*100:.2f}%)")
            print(f"  Original vs Perturbed Best Edit:     {result['original_vs_perturbed_best']:.4f} ({result['original_vs_perturbed_best']*100:.2f}%)")
            print(f"  Original vs Perturbed Final Edit:    {result['original_vs_perturbed_final']:.4f} ({result['original_vs_perturbed_final']*100:.2f}%)")
            print(f"  Perturbed Best -> Best Edit:         {result['perturbed_best_vs_perturbed_best_edit']:.4f} ({result['perturbed_best_vs_perturbed_best_edit']*100:.2f}%)")
            print(f"  Perturbed Final -> Final Edit:       {result['perturbed_final_vs_perturbed_final_edit']:.4f} ({result['perturbed_final_vs_perturbed_final_edit']*100:.2f}%)")

    # Save summary
    output_path = Path("outputs/edited_output_identity_5/arcface_simple_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
