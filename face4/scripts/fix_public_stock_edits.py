"""Repair FACE4 public edited-output images with stock InstructPix2Pix replay.

FACE4 optimizes through a differentiable reconstruction of InstructPix2Pix so
gradients can flow into geometry parameters. Some saved ``perturbed_best`` edit
images from that gradient path can be wrapper-only artifacts. This script fixes
already completed runs by regenerating the public edited images with the normal
diffusers pipeline call:

``original_edited.png``
``perturbed_best_edited.png``
``perturbed_final_edited.png``

The old public images are preserved as explicit ``*_gradient_path*.png`` debug
artifacts. Reports that read the public filenames will then use the corrected
stock/no-grad images.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from face4.core.identity import identity_objective, prepare_identity_reference
from face4.core.image_metrics import image_metrics, pil_to_tensor, save_sheet
from face4.models.arcface import ArcFaceIResNet100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix FACE4 saved public edit images using stock InstructPix2Pix replay.")
    parser.add_argument("--results-root", default="outputs/edited_output_identity_2")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--run-folder", default=None)
    parser.add_argument("--arcface-checkpoint", default=None)
    parser.add_argument("--editor-dtype", default=None, choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be repaired without overwriting files.")
    return parser.parse_args()


def _dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.float16
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latest_run_root(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "runs" / "edited_output_identity").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No FACE4 run roots found under {results_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root:
        run_root = Path(args.run_root)
        if not run_root.exists():
            raise FileNotFoundError(f"Requested --run-root does not exist: {run_root}")
        return run_root
    results_root = Path(args.results_root)
    if args.run_folder:
        run_root = results_root / args.run_folder
        if not run_root.exists():
            raise FileNotFoundError(f"Requested --run-folder does not exist: {run_root}")
        return run_root
    return _latest_run_root(results_root)


def _case_run_dirs(run_root: Path) -> list[Path]:
    root = run_root / "runs" / "edited_output_identity"
    if not root.exists():
        raise FileNotFoundError(f"Missing FACE4 case root: {root}")
    return sorted(path for path in root.glob("*/*") if (path / "config_resolved.json").exists())


def _setting(config: dict[str, Any], summary: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in summary:
        return summary[key]
    if key in config:
        return config[key]
    settings = config.get("differentiable_instructpix2pix", {}).get("settings", {})
    aliases = {
        "editor_num_inference_steps": "num_inference_steps",
        "editor_guidance_scale": "guidance_scale",
        "editor_image_guidance_scale": "image_guidance_scale",
        "editor_dtype": "torch_dtype",
        "editor_model_id": "model_id",
    }
    if aliases.get(key) in settings:
        return settings[aliases[key]]
    return default


def _stock_edit(pipe, image: Image.Image, prompt: str, seed: int, steps: int, guidance: float, image_guidance: float, device: torch.device) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            image_guidance_scale=float(image_guidance),
            generator=generator,
        )
    return result.images[0].convert("RGB")


def _archive_once(source: Path, archive: Path, dry_run: bool) -> None:
    if not source.exists() or archive.exists():
        return
    if not dry_run:
        shutil.copy2(source, archive)


def _load_arcface(args: argparse.Namespace, case_dirs: list[Path], device: torch.device) -> ArcFaceIResNet100 | None:
    checkpoint = args.arcface_checkpoint
    if checkpoint is None:
        for run_dir in case_dirs:
            config = _read_json(run_dir / "config_resolved.json")
            checkpoint = config.get("arcface_checkpoint") or config.get("arcface", {}).get("arcface_checkpoint_path")
            if checkpoint:
                break
    if not checkpoint:
        print("[face4-fix] ArcFace checkpoint unavailable; stock Z fields will be skipped.")
        return None
    path = Path(checkpoint)
    if not path.exists():
        print(f"[face4-fix] ArcFace checkpoint missing: {path}; stock Z fields will be skipped.")
        return None
    return ArcFaceIResNet100(path, device)


def _identity_terms(arcface: ArcFaceIResNet100 | None, clean: Image.Image, best: Image.Image, final: Image.Image, device: torch.device) -> dict[str, Any]:
    if arcface is None:
        return {}
    with torch.no_grad():
        clean_tensor = pil_to_tensor(clean, device)
        best_tensor = pil_to_tensor(best, device)
        final_tensor = pil_to_tensor(final, device)
        reference = prepare_identity_reference(arcface, clean_tensor)
        best_z, best_terms = identity_objective(arcface, best_tensor, reference)
        final_z, final_terms = identity_objective(arcface, final_tensor, reference)
    return {
        "best_Z_stock_public": float(best_z.detach().float().cpu()),
        "final_Z_stock_public": float(final_z.detach().float().cpu()),
        "best_stock_public_identity_cosine_similarity_raw": float(best_terms["identity_cosine_similarity_raw"].detach().float().cpu()),
        "best_stock_public_identity_similarity_score_pct": float(best_terms["identity_similarity_score_pct"].detach().float().cpu()),
        "final_stock_public_identity_cosine_similarity_raw": float(final_terms["identity_cosine_similarity_raw"].detach().float().cpu()),
        "final_stock_public_identity_similarity_score_pct": float(final_terms["identity_similarity_score_pct"].detach().float().cpu()),
    }


def _prefixed_pair_terms(arcface: ArcFaceIResNet100 | None, left: Image.Image, right: Image.Image, prefix: str, device: torch.device) -> dict[str, Any]:
    if arcface is None:
        return {}
    with torch.no_grad():
        left_tensor = pil_to_tensor(left, device)
        right_tensor = pil_to_tensor(right, device)
        reference = prepare_identity_reference(arcface, left_tensor)
        _, terms = identity_objective(arcface, right_tensor, reference)
    mapping = {
        "identity_cosine_similarity_raw": "cosine_similarity_raw",
        "identity_cosine_distance": "cosine_distance",
        "identity_similarity_score_pct": "similarity_score_pct",
        "identity_l2_embedding_distance": "l2_embedding_distance",
        "identity_angle_degrees": "angle_degrees",
    }
    return {f"{prefix}_{new_key}": float(terms[old_key].detach().float().cpu()) for old_key, new_key in mapping.items() if old_key in terms}


def _repair_one(run_dir: Path, pipe, arcface: ArcFaceIResNet100 | None, device: torch.device, dry_run: bool) -> dict[str, Any]:
    config = _read_json(run_dir / "config_resolved.json")
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    spec = config.get("spec", {})
    prompt = spec.get("prompt") or summary.get("prompt")
    seed = int(spec.get("seed") or summary.get("seed"))
    face_id = spec.get("face_id") or summary.get("face_id")
    case_id = spec.get("case_id") or summary.get("case_id") or run_dir.name
    steps = int(_setting(config, summary, "editor_num_inference_steps", 20))
    guidance = float(_setting(config, summary, "editor_guidance_scale", 7.5))
    image_guidance = float(_setting(config, summary, "editor_image_guidance_scale", 1.5))

    required = ["original.png", "perturbed_best.png", "perturbed_final.png"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return {"case_id": case_id, "face_id": face_id, "prompt": prompt, "status": "skipped", "reason": f"missing {missing}", "run_dir": str(run_dir)}

    original = Image.open(run_dir / "original.png").convert("RGB")
    perturbed_best = Image.open(run_dir / "perturbed_best.png").convert("RGB")
    perturbed_final = Image.open(run_dir / "perturbed_final.png").convert("RGB")

    _archive_once(run_dir / "original_edited.png", run_dir / "original_edited_gradient_reference.png", dry_run)
    _archive_once(run_dir / "perturbed_best_edited.png", run_dir / "perturbed_best_edited_gradient_path.png", dry_run)
    _archive_once(run_dir / "perturbed_final_edited.png", run_dir / "perturbed_final_edited_gradient_path.png", dry_run)

    stock_clean = _stock_edit(pipe, original, prompt, seed, steps, guidance, image_guidance, device)
    stock_best = _stock_edit(pipe, perturbed_best, prompt, seed, steps, guidance, image_guidance, device)
    stock_final = _stock_edit(pipe, perturbed_final, prompt, seed, steps, guidance, image_guidance, device)

    if not dry_run:
        stock_clean.save(run_dir / "original_edited.png")
        stock_best.save(run_dir / "perturbed_best_edited.png")
        stock_final.save(run_dir / "perturbed_final_edited.png")

    input_metrics_best = image_metrics(original, perturbed_best)
    input_metrics_final = image_metrics(original, perturbed_final)
    output_metrics_best = image_metrics(stock_clean, stock_best)
    output_metrics_final = image_metrics(stock_clean, stock_final)
    identity = _identity_terms(arcface, stock_clean, stock_best, stock_final, device)
    pair_identity = {
        **_prefixed_pair_terms(arcface, original, stock_clean, "original_vs_original_edit_identity", device),
        **_prefixed_pair_terms(arcface, perturbed_best, stock_best, "perturbed_best_vs_perturbed_best_edit_identity", device),
        **_prefixed_pair_terms(arcface, perturbed_final, stock_final, "perturbed_final_vs_perturbed_final_edit_identity", device),
        **_prefixed_pair_terms(arcface, original, perturbed_best, "best_input_identity", device),
        **_prefixed_pair_terms(arcface, original, perturbed_final, "final_input_identity", device),
    }
    if identity:
        best_stock = float(identity["best_Z_stock_public"])
        final_stock = float(identity["final_Z_stock_public"])
        if final_stock <= best_stock:
            identity.update(
                {
                    "stock_public_Z_at_gradient_best": best_stock,
                    "best_saved_stock_public_Z": final_stock,
                    "best_saved_stock_public_source": "final_input",
                    "best_saved_stock_public_iter": summary.get("iters"),
                }
            )
        else:
            identity.update(
                {
                    "stock_public_Z_at_gradient_best": best_stock,
                    "best_saved_stock_public_Z": best_stock,
                    "best_saved_stock_public_source": "gradient_best_input",
                    "best_saved_stock_public_iter": summary.get("best_iter_by_Z"),
                }
            )

    old_best_path = run_dir / "perturbed_best_edited_gradient_path.png"
    old_final_path = run_dir / "perturbed_final_edited_gradient_path.png"
    gradient_comparison: dict[str, Any] = {}
    if old_best_path.exists():
        gradient_comparison.update({f"best_gradient_path_vs_stock_public_{k}": v for k, v in image_metrics(Image.open(old_best_path), stock_best).items()})
    if old_final_path.exists():
        gradient_comparison.update({f"final_gradient_path_vs_stock_public_{k}": v for k, v in image_metrics(Image.open(old_final_path), stock_final).items()})

    metrics: dict[str, Any] = {
        "case_id": case_id,
        "face_id": face_id,
        "prompt": prompt,
        "seed": seed,
        "status": "dry_run" if dry_run else "repaired",
        "run_dir": str(run_dir),
        "editor_num_inference_steps": steps,
        "editor_guidance_scale": guidance,
        "editor_image_guidance_scale": image_guidance,
        **{f"input_best_{key}": value for key, value in input_metrics_best.items()},
        **{f"input_final_{key}": value for key, value in input_metrics_final.items()},
        **{f"best_stock_public_output_{key}": value for key, value in output_metrics_best.items()},
        **{f"final_stock_public_output_{key}": value for key, value in output_metrics_final.items()},
        **identity,
        **pair_identity,
        **gradient_comparison,
    }

    if not dry_run:
        save_sheet(
            run_dir / "comparison_sheet.png",
            [
                ("Original", original),
                ("Perturbed Best", perturbed_best),
                ("Abs Difference x8", Image.open(run_dir / "input_difference_best.png") if (run_dir / "input_difference_best.png").exists() else perturbed_best),
                ("Combined Flow", Image.open(run_dir / "combined_flow_best.png") if (run_dir / "combined_flow_best.png").exists() else perturbed_best),
                ("DCT Difference x10", Image.open(run_dir / "dct_only_difference_x10.png") if (run_dir / "dct_only_difference_x10.png").exists() else Image.new("RGB", original.size, "black")),
                ("Clean Edit", stock_clean),
                ("Perturbed Edit", stock_best),
            ],
        )
        if "best_Z_gradient_path" not in summary and summary.get("best_Z") is not None:
            summary["best_Z_gradient_path"] = summary.get("best_Z")
        if "final_Z_gradient_path" not in summary and summary.get("final_Z") is not None:
            summary["final_Z_gradient_path"] = summary.get("final_Z")
        if identity:
            summary["best_Z"] = identity["best_saved_stock_public_Z"]
            summary["final_Z"] = identity["final_Z_stock_public"]
            summary["final_loss"] = identity["final_Z_stock_public"]
            summary["best_identity_cosine_similarity_raw"] = identity["best_stock_public_identity_cosine_similarity_raw"]
            summary["best_identity_similarity_score_pct"] = identity["best_stock_public_identity_similarity_score_pct"]
            summary["best_edit_identity_cosine_similarity_raw"] = identity["best_stock_public_identity_cosine_similarity_raw"]
            summary["best_edit_identity_similarity_score_pct"] = identity["best_stock_public_identity_similarity_score_pct"]
            summary["final_identity_cosine_similarity_raw"] = identity["final_stock_public_identity_cosine_similarity_raw"]
            summary["final_identity_similarity_score_pct"] = identity["final_stock_public_identity_similarity_score_pct"]
            summary["final_edit_identity_cosine_similarity_raw"] = identity["final_stock_public_identity_cosine_similarity_raw"]
            summary["final_edit_identity_similarity_score_pct"] = identity["final_stock_public_identity_similarity_score_pct"]
        summary.update(
            {
                "public_edit_images_regenerated_with_stock_pipeline": True,
                "public_edit_images_fixed_by": "face4.scripts.fix_public_stock_edits",
                "stock_public_repair_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "input_best_ssim": input_metrics_best["ssim"],
                "input_best_psnr": input_metrics_best["psnr"],
                "input_best_l2": input_metrics_best["l2"],
                "input_final_ssim": input_metrics_final["ssim"],
                "input_final_psnr": input_metrics_final["psnr"],
                "input_final_l2": input_metrics_final["l2"],
                "best_output_ssim": output_metrics_best["ssim"],
                "best_output_psnr": output_metrics_best["psnr"],
                "best_output_l2": output_metrics_best["l2"],
                "final_output_ssim": output_metrics_final["ssim"],
                "final_output_psnr": output_metrics_final["psnr"],
                "final_output_l2": output_metrics_final["l2"],
                "best_stock_public_output_ssim": output_metrics_best["ssim"],
                "best_stock_public_output_psnr": output_metrics_best["psnr"],
                "best_stock_public_output_l2": output_metrics_best["l2"],
                "final_stock_public_output_ssim": output_metrics_final["ssim"],
                "final_stock_public_output_psnr": output_metrics_final["psnr"],
                "final_stock_public_output_l2": output_metrics_final["l2"],
            }
        )
        summary.update(identity)
        summary.update(pair_identity)
        edit_meta = dict(summary.get("differentiable_instructpix2pix", {}))
        edit_meta.update(
            {
                "public_edit_images_regenerated_with_stock_pipeline": True,
                "public_edit_images_source": "StableDiffusionInstructPix2PixPipeline.__call__ under torch.inference_mode",
                "clean_edit_path": "original_edited.png",
                "clean_edit_gradient_reference_path": "original_edited_gradient_reference.png",
                "perturbed_best_edit_path": "perturbed_best_edited.png",
                "perturbed_final_edit_path": "perturbed_final_edited.png",
                "perturbed_best_edit_gradient_path": "perturbed_best_edited_gradient_path.png",
                "perturbed_final_edit_gradient_path": "perturbed_final_edited_gradient_path.png",
            }
        )
        summary["differentiable_instructpix2pix"] = edit_meta
        _write_json(summary_path, summary)
        _write_json(run_dir / "stock_public_repair_metrics.json", metrics)
        if (run_dir / "DONE.json").exists() and identity:
            done = _read_json(run_dir / "DONE.json")
            done["final_Z"] = identity["final_Z_stock_public"]
            done["public_edit_images_regenerated_with_stock_pipeline"] = True
            _write_json(run_dir / "DONE.json", done)
    return metrics


def main() -> None:
    args = parse_args()
    run_root = _resolve_run_root(args)
    case_dirs = _case_run_dirs(run_root)
    if args.limit is not None:
        case_dirs = case_dirs[: int(args.limit)]
    if not case_dirs:
        raise RuntimeError(f"No FACE4 case directories found under {run_root}")

    first_config = _read_json(case_dirs[0] / "config_resolved.json")
    first_summary = _read_json(case_dirs[0] / "summary.json") if (case_dirs[0] / "summary.json").exists() else {}
    model_id = _setting(first_config, first_summary, "editor_model_id", "timbrooks/instruct-pix2pix")
    dtype_name = args.editor_dtype or _setting(first_config, first_summary, "editor_dtype", "float16")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from diffusers import StableDiffusionInstructPix2PixPipeline

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=_dtype(dtype_name),
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    arcface = _load_arcface(args, case_dirs, device)

    rows = [_repair_one(run_dir, pipe, arcface, device, bool(args.dry_run)) for run_dir in case_dirs]
    out_dir = run_root / "stock_public_edit_repair"
    if not args.dry_run:
        _write_csv(out_dir / "stock_public_edit_repair.csv", rows)
        _write_json(
            out_dir / "stock_public_edit_repair_summary.json",
            {
                "run_root": str(run_root),
                "num_case_dirs": len(case_dirs),
                "num_repaired": sum(row.get("status") == "repaired" for row in rows),
                "num_skipped": sum(row.get("status") == "skipped" for row in rows),
                "rows": rows,
            },
        )
    print(f"[face4-fix] run root: {run_root}")
    print(f"[face4-fix] repaired: {sum(row.get('status') == 'repaired' for row in rows)}")
    print(f"[face4-fix] skipped: {sum(row.get('status') == 'skipped' for row in rows)}")
    if args.dry_run:
        print("[face4-fix] dry run only; no files were overwritten.")
    else:
        print(f"[face4-fix] wrote: {out_dir}")


if __name__ == "__main__":
    main()
