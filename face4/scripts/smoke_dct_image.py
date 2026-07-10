"""Standalone smoke test for the image-domain DCT perturbation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from face4.core.geometry.dct_image import DCTImagePerturbation
from face4.core.image_metrics import pil_to_tensor, save_sheet, tensor_pair_metrics, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test FACE image-domain DCT perturbation.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-root", default="outputs/dct_image_smoke")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--frequency-mask", default="all_ac", choices=["all_ac", "low", "mid", "high"])
    parser.add_argument("--gain-limit", type=float, default=0.5)
    return parser.parse_args()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_delta_x10(path: Path, left: torch.Tensor, right: torch.Tensor) -> Image.Image:
    diff = (left.detach().float() - right.detach().float()).abs().mean(1)[0].mul(10.0).clamp(0, 1)
    image = Image.fromarray((diff.cpu().numpy() * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    image.save(path)
    return image


def _save_heatmap(path: Path, matrix, title: str) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = matrix.detach().float().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4.2, 3.7), dpi=130)
    im = ax.imshow(arr, cmap="magma", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("DCT v frequency index")
    ax.set_ylabel("DCT u frequency index")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Image.open(path).convert("RGB")


def _module(channels: int, args: argparse.Namespace, mask: str, enabled: bool = True) -> DCTImagePerturbation:
    return DCTImagePerturbation(
        channels=channels,
        block_size=args.block_size,
        gain_limit=args.gain_limit,
        frequency_mask_mode=mask,
        exclude_dc=True,
        enabled=enabled,
        device=torch.device("cpu"),
    )


def _set_gain(module: DCTImagePerturbation, value: float, mixed: bool = False) -> None:
    with torch.no_grad():
        if mixed:
            yy, xx = torch.meshgrid(
                torch.arange(module.block_size),
                torch.arange(module.block_size),
                indexing="ij",
            )
            signs = torch.where((yy + xx) % 2 == 0, 1.0, -1.0).to(module.dct_gain_raw)
            module.dct_gain_raw.copy_(value * signs[None])
        else:
            module.dct_gain_raw.fill_(value)
        module.project_()


def _run_example(original: torch.Tensor, module: DCTImagePerturbation, label: str, out_dir: Path) -> dict[str, Any]:
    output = module(original)
    image = tensor_to_pil(output.image)
    image.save(out_dir / f"{label}.png")
    diff = _save_delta_x10(out_dir / f"{label}_difference_x10.png", output.image, original)
    metrics = tensor_pair_metrics(output.image, original, prefix="")
    return {
        "label": label,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "mse": metrics["mse"],
        "max_abs_image_change": float((output.image.detach().float() - original.detach().float()).abs().max().cpu()),
        "coefficient_energy_before": output.stats["dct_input_coefficient_energy"],
        "coefficient_energy_after": output.stats["dct_output_coefficient_energy"],
        "relative_coefficient_energy_change": output.stats["dct_relative_energy_change"],
        "fraction_pixels_clipped": output.stats["dct_clipped_low_fraction"] + output.stats["dct_clipped_high_fraction"],
        "image": image,
        "diff": diff,
        "stats": output.stats,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pil = Image.open(args.image).convert("RGB")
    original = pil_to_tensor(pil, torch.device("cpu")).float()
    pil.save(output_root / "original.png")

    all_metrics: dict[str, Any] = {
        "image": str(args.image),
        "block_size": args.block_size,
        "gain_limit": args.gain_limit,
        "frequency_mask_requested": args.frequency_mask,
        "band_examples": {},
    }

    primary = _module(original.shape[1], args, args.frequency_mask)
    neutral = primary(original)
    neutral_error = float((neutral.image - original).abs().max().cpu())
    if neutral_error >= 1e-5:
        raise AssertionError(f"Neutral DCT reconstruction error too high: {neutral_error}")

    disabled = _module(original.shape[1], args, args.frequency_mask, enabled=False)
    disabled_out = disabled(original)
    disabled_error = float((disabled_out.image - original).abs().max().cpu())
    if disabled_error >= 1e-7:
        raise AssertionError(f"Disabled DCT changed image: {disabled_error}")
    if any(p.requires_grad for p in disabled.parameters()):
        raise AssertionError("Disabled DCT still has trainable parameters.")

    projection = _module(original.shape[1], args, args.frequency_mask)
    with torch.no_grad():
        projection.dct_gain_raw.fill_(args.gain_limit * 3.0)
    projection_stats = projection.project_()
    projected_max = float(projection.dct_gain_raw.max().cpu())
    if projected_max > args.gain_limit + 1e-8:
        raise AssertionError("DCT projection did not clamp high values.")
    if abs(float(projection.dct_gain_raw[:, 0, 0].abs().max().cpu())) > 1e-8:
        raise AssertionError("DCT DC gain was not preserved at zero.")

    grad_module = _module(original.shape[1], args, args.frequency_mask)
    _set_gain(grad_module, args.gain_limit * 0.25)
    grad_out = grad_module(original)
    grad_loss = grad_out.image.square().mean()
    grad_loss.backward()
    grad = grad_module.dct_gain_raw.grad
    grad_ok = bool(grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0)
    if not grad_ok:
        raise AssertionError("DCT gain gradient is missing, non-finite, or zero.")

    example_dir = output_root / args.frequency_mask
    example_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, fraction, mixed in [
        ("neutral_reconstruction", 0.0, False),
        ("positive_gain_25pct", 0.25, False),
        ("positive_gain_50pct", 0.50, False),
        ("positive_gain_100pct", 1.00, False),
        ("mixed_positive_negative", 0.75, True),
    ]:
        module = _module(original.shape[1], args, args.frequency_mask)
        _set_gain(module, args.gain_limit * fraction, mixed=mixed)
        rows.append(_run_example(original, module, label, example_dir))

    nonzero = rows[1]
    if nonzero["max_abs_image_change"] <= 0:
        raise AssertionError("Nonzero DCT gain did not change the image.")
    if abs(nonzero["relative_coefficient_energy_change"]) <= 0:
        raise AssertionError("Nonzero DCT gain did not change coefficient energy.")

    mask_img = _save_heatmap(example_dir / "dct_frequency_mask.png", primary.frequency_mask_2d, "DCT frequency mask")
    before = primary.spectrum_summary(original)
    after = _module(original.shape[1], args, args.frequency_mask)
    _set_gain(after, args.gain_limit)
    after_out = after(original)
    spectrum_before = _save_heatmap(example_dir / "dct_spectrum_before.png", before, "DCT spectrum before")
    spectrum_after = _save_heatmap(example_dir / "dct_spectrum_after.png", after.spectrum_summary(after_out.image), "DCT spectrum after")

    save_sheet(
        example_dir / "comparison_sheet.png",
        [
            ("Original", pil),
            ("Neutral DCT", rows[0]["image"]),
            ("25% gain", rows[1]["image"]),
            ("50% gain", rows[2]["image"]),
            ("100% gain", rows[3]["image"]),
            ("Mixed +/- gain", rows[4]["image"]),
            ("Difference x10", rows[3]["diff"]),
            ("Mask", mask_img),
            ("Spectrum before", spectrum_before),
            ("Spectrum after", spectrum_after),
        ],
        cell_width=256,
    )

    for mask in ["low", "mid", "high", "all_ac"]:
        band_dir = output_root / f"band_{mask}"
        band_dir.mkdir(parents=True, exist_ok=True)
        module = _module(original.shape[1], args, mask)
        _set_gain(module, args.gain_limit)
        band = _run_example(original, module, f"{mask}_gain_100pct", band_dir)
        band_mask = _save_heatmap(band_dir / "dct_frequency_mask.png", module.frequency_mask_2d, f"DCT mask {mask}")
        save_sheet(
            band_dir / "comparison_sheet.png",
            [("Original", pil), (f"{mask} gain", band["image"]), ("Difference x10", band["diff"]), ("Mask", band_mask)],
            cell_width=256,
        )
        all_metrics["band_examples"][mask] = {k: v for k, v in band.items() if k not in {"image", "diff", "stats"}}

    all_metrics.update(
        {
            "neutral_reconstruction_max_abs": neutral_error,
            "disabled_reconstruction_max_abs": disabled_error,
            "projection_num_clamped": projection_stats["dct_num_clamped"],
            "projected_max_gain": projected_max,
            "dc_gain_abs_max_after_projection": float(projection.dct_gain_raw[:, 0, 0].abs().max().cpu()),
            "gradient_reaches_dct_gain_raw": grad_ok,
            "primary_examples": [{k: v for k, v in row.items() if k not in {"image", "diff", "stats"}} for row in rows],
        }
    )
    _save_json(output_root / "metrics.json", all_metrics)
    print(f"[dct-smoke] wrote: {output_root}")
    print(f"[dct-smoke] neutral reconstruction max abs: {neutral_error:.8g}")
    print(f"[dct-smoke] gradient reaches dct_gain_raw: {grad_ok}")


if __name__ == "__main__":
    main()
