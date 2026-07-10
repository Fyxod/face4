"""Replay saved FACE4 perturbed inputs through stock InstructPix2Pix.

This is a sanity check for the differentiable editor path. It does not
optimize anything. It takes saved run artifacts such as ``perturbed_best.png``
and ``perturbed_final.png``, sends them through the normal diffusers
``StableDiffusionInstructPix2PixPipeline.__call__`` path, and compares those
stock outputs against the differentiable-wrapper outputs saved by FACE4.

Interpretation:

- If stock replay matches the saved wrapper edit, the saved edit is not a
  wrapper-only artifact.
- If stock replay does not match the saved wrapper edit, the differentiable
  path still has a mismatch for that perturbed input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from face4.core.image_metrics import image_metrics, save_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved FACE4 perturbed images through stock InstructPix2Pix.")
    parser.add_argument(
        "--results-root",
        default="outputs/smoke_timing",
        help="Parent result folder. Used to auto-pick the latest run if --run-root is omitted.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Specific FACE4 top-level run folder, e.g. outputs/smoke_timing/20260706_181028_edited_output_identity_all_sequential.",
    )
    parser.add_argument("--output-dir-name", default="stock_replay_verification")
    parser.add_argument("--editor-dtype", default=None, choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of case run folders to verify.")
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
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


def _case_run_dirs(run_root: Path) -> list[Path]:
    root = run_root / "runs" / "edited_output_identity"
    if not root.exists():
        raise FileNotFoundError(f"Missing FACE4 run case root: {root}")
    return sorted(path for path in root.glob("*/*") if (path / "config_resolved.json").exists())


def _load_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(run_dir / "config_resolved.json")
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    return config, summary


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


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _verify_one(run_dir: Path, pipe, device: torch.device, output_dir_name: str) -> dict[str, Any]:
    config, summary = _load_config(run_dir)
    spec = config.get("spec", {})
    prompt = spec.get("prompt") or summary.get("prompt")
    seed = int(spec.get("seed") or summary.get("seed"))
    face_id = spec.get("face_id") or summary.get("face_id")
    case_id = spec.get("case_id") or summary.get("case_id") or run_dir.name
    steps = int(_setting(config, summary, "editor_num_inference_steps", 20))
    guidance = float(_setting(config, summary, "editor_guidance_scale", 7.5))
    image_guidance = float(_setting(config, summary, "editor_image_guidance_scale", 1.5))

    original = Image.open(run_dir / "original.png").convert("RGB")
    perturbed_best = Image.open(run_dir / "perturbed_best.png").convert("RGB")
    perturbed_final = Image.open(run_dir / "perturbed_final.png").convert("RGB")
    saved_clean = Image.open(run_dir / "original_edited.png").convert("RGB")
    saved_best = Image.open(run_dir / "perturbed_best_edited.png").convert("RGB")
    saved_final = Image.open(run_dir / "perturbed_final_edited.png").convert("RGB")

    out_dir = run_dir / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_clean = _stock_edit(pipe, original, prompt, seed, steps, guidance, image_guidance, device)
    stock_best = _stock_edit(pipe, perturbed_best, prompt, seed, steps, guidance, image_guidance, device)
    stock_final = _stock_edit(pipe, perturbed_final, prompt, seed, steps, guidance, image_guidance, device)

    stock_clean.save(out_dir / "stock_original_edited.png")
    stock_best.save(out_dir / "stock_perturbed_best_edited.png")
    stock_final.save(out_dir / "stock_perturbed_final_edited.png")

    metrics = {
        **_prefixed("stock_vs_saved_clean", image_metrics(stock_clean, saved_clean)),
        **_prefixed("stock_vs_saved_best_wrapper", image_metrics(stock_best, saved_best)),
        **_prefixed("stock_vs_saved_final_wrapper", image_metrics(stock_final, saved_final)),
        **_prefixed("stock_clean_vs_stock_best", image_metrics(stock_clean, stock_best)),
        **_prefixed("stock_clean_vs_stock_final", image_metrics(stock_clean, stock_final)),
        **_prefixed("saved_clean_vs_saved_best", image_metrics(saved_clean, saved_best)),
        **_prefixed("saved_clean_vs_saved_final", image_metrics(saved_clean, saved_final)),
        **_prefixed("input_original_vs_best", image_metrics(original, perturbed_best)),
        **_prefixed("input_original_vs_final", image_metrics(original, perturbed_final)),
    }
    row: dict[str, Any] = {
        "case_id": case_id,
        "face_id": face_id,
        "prompt": prompt,
        "seed": seed,
        "edit_steps": steps,
        "guidance_scale": guidance,
        "image_guidance_scale": image_guidance,
        "run_dir": str(run_dir),
        "verification_dir": str(out_dir),
        "wrapper_best_matches_stock": metrics["stock_vs_saved_best_wrapper_ssim"] >= 0.98,
        "wrapper_final_matches_stock": metrics["stock_vs_saved_final_wrapper_ssim"] >= 0.98,
        "best_edit_disrupts_under_stock": metrics["stock_clean_vs_stock_best_ssim"] < 0.70,
        "final_edit_disrupts_under_stock": metrics["stock_clean_vs_stock_final_ssim"] < 0.70,
        **metrics,
    }
    _write_json(out_dir / "verification_metrics.json", row)
    save_sheet(
        out_dir / "stock_replay_sheet.png",
        [
            ("Original", original),
            ("Perturbed Best", perturbed_best),
            ("Perturbed Final", perturbed_final),
            ("Saved Clean Edit", saved_clean),
            ("Stock Clean Edit", stock_clean),
            ("Saved Best Edit", saved_best),
            ("Stock Best Edit", stock_best),
            ("Saved Final Edit", saved_final),
            ("Stock Final Edit", stock_final),
        ],
    )
    return row


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root) if args.run_root else _latest_run_root(Path(args.results_root))
    case_dirs = _case_run_dirs(run_root)
    if args.limit is not None:
        case_dirs = case_dirs[: int(args.limit)]
    if not case_dirs:
        raise RuntimeError(f"No case run dirs found under {run_root}")

    first_config, first_summary = _load_config(case_dirs[0])
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

    rows = [_verify_one(run_dir, pipe, device, args.output_dir_name) for run_dir in case_dirs]
    verify_root = run_root / args.output_dir_name
    _write_csv(verify_root / "stock_replay_verification.csv", rows)
    _write_json(
        verify_root / "stock_replay_verification_summary.json",
        {
            "run_root": str(run_root),
            "num_cases": len(rows),
            "num_wrapper_best_matches_stock": sum(bool(row["wrapper_best_matches_stock"]) for row in rows),
            "num_wrapper_final_matches_stock": sum(bool(row["wrapper_final_matches_stock"]) for row in rows),
            "num_best_edit_disrupts_under_stock": sum(bool(row["best_edit_disrupts_under_stock"]) for row in rows),
            "num_final_edit_disrupts_under_stock": sum(bool(row["final_edit_disrupts_under_stock"]) for row in rows),
            "rows": rows,
        },
    )
    lines = [
        "# FACE4 stock replay verification",
        "",
        f"- run root: `{run_root}`",
        f"- cases verified: {len(rows)}",
        f"- wrapper best edits matching stock replay: {sum(bool(row['wrapper_best_matches_stock']) for row in rows)} / {len(rows)}",
        f"- wrapper final edits matching stock replay: {sum(bool(row['wrapper_final_matches_stock']) for row in rows)} / {len(rows)}",
        f"- best perturbed edits disruptive under stock replay: {sum(bool(row['best_edit_disrupts_under_stock']) for row in rows)} / {len(rows)}",
        f"- final perturbed edits disruptive under stock replay: {sum(bool(row['final_edit_disrupts_under_stock']) for row in rows)} / {len(rows)}",
        "",
        "Interpretation: if wrapper-vs-stock SSIM is high, the saved differentiable edit is reproduced by the normal stock pipeline for that saved perturbed image.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']}",
                "",
                f"- prompt: {row['prompt']}",
                f"- stock vs saved best-wrapper SSIM: {row['stock_vs_saved_best_wrapper_ssim']:.6f}",
                f"- stock clean vs stock best SSIM: {row['stock_clean_vs_stock_best_ssim']:.6f}",
                f"- stock vs saved final-wrapper SSIM: {row['stock_vs_saved_final_wrapper_ssim']:.6f}",
                f"- stock clean vs stock final SSIM: {row['stock_clean_vs_stock_final_ssim']:.6f}",
                f"- sheet: `{Path(row['verification_dir']) / 'stock_replay_sheet.png'}`",
                "",
            ]
        )
    (verify_root / "stock_replay_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[face4-verify] wrote {verify_root}")


if __name__ == "__main__":
    main()
