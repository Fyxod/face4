"""Generate a perturbed image using the limits from a geometry config file.

This script loads a geometry JSON config (e.g., configs/geometry_default.json)
and applies the maximum perturbation limits to produce a perturbed image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from face4.core.geometry.combined_face import CombinedFacePerturbation, FaceGeometryConfig, load_face_geometry_config
from face4.core.image_metrics import pil_to_tensor, tensor_to_pil


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maximize_parameters(geo: CombinedFacePerturbation, config: FaceGeometryConfig) -> None:
    """Set all enabled perturbation parameters to their maximum limits."""
    with torch.no_grad():
        # TPS
        if config.tps_enabled:
            geo.tps_raw[:] = geo.tps_limit_px

        # Delaunay
        if config.delaunay_enabled:
            geo.delaunay_raw[:] = geo.delaunay_limit_px

        # Rolling
        if config.rolling_enabled:
            geo.roll_params[:] = geo.rolling_limit_px

        # Polar
        if config.polar_enabled:
            geo.polar_params[0] = geo.polar_radial_limit_px
            geo.polar_params[1] = float(config.polar_twist_limit_rad)

        # B-spline
        if config.bspline_enabled:
            geo.bspline_raw[:] = geo.bspline_limit_px

        # Lens barrel
        if config.lens_barrel_enabled:
            geo.lens_barrel_k[:] = float(geo.lens_barrel_k_limit)

        # Lens pincushion
        if config.lens_pincushion_enabled:
            geo.lens_pincushion_k[:] = float(geo.lens_pincushion_k_limit)

        # Mobius
        if config.mobius_enabled:
            geo.mobius_params[:] = float(config.mobius_limit)

        # Laplacian
        if config.laplacian_enabled:
            geo.laplacian_raw[:] = geo.laplacian_limit_px

        # Geodesic
        if config.geodesic_enabled:
            geo.geodesic_params[:] = geo.geodesic_limit_px

        # Differential surface
        if config.differential_surface_enabled:
            geo.differential_surface_height[:] = float(config.differential_surface_height_limit)

        # DCT
        if config.dct_enabled:
            geo.dct_image.dct_gain_raw[:] = float(config.dct_gain_limit)

        # FFT Phase
        if config.fft_phase_enabled:
            geo.fft_phase.raw_phase[:] = float(config.fft_phase_limit_rad)

        geo.project_()


def generate_perturbed_image(
    image_path: Path,
    output_path: Path,
    config_path: Path | None,
    device: torch.device,
) -> dict:
    """Generate a perturbed image using the geometry config limits."""

    # Load config
    if config_path:
        config = load_face_geometry_config(config_path)
    else:
        config = FaceGeometryConfig()

    # Load image
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image, device)
    _, _, height, width = tensor.shape

    # Create geometry module
    geo = CombinedFacePerturbation(height, width, 3, device, seed=20260708, config=config)

    # Set parameters to maximum limits
    _maximize_parameters(geo, config)

    # Apply perturbation
    with torch.no_grad():
        perturbed, aux = geo(tensor)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed_pil = tensor_to_pil(perturbed)
    perturbed_pil.save(output_path)

    # Save diagnostics
    diagnostics = aux["diagnostics"]
    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    # Save config used
    config_used_path = output_path.with_suffix(".config.json")
    config_used = {
        "source_config": str(config_path) if config_path else "default",
        "face_geometry_config": config.__dict__,
    }
    config_used_path.write_text(json.dumps(config_used, indent=2), encoding="utf-8")

    # Save difference image (perturbed - original)
    diff_tensor = (perturbed - tensor).abs().clamp(0, 1)
    diff_pil = tensor_to_pil(diff_tensor)
    diff_path = output_path.with_name(f"{output_path.stem}_diff.png")
    diff_pil.save(diff_path)

    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a perturbed image using geometry config limits."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path. If not provided, uses sibling MAT data/face_002/instruct_512.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. If not provided, outputs to outputs/perturbed_from_config/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Geometry config JSON path. If not provided, uses configs/geometry_default.json.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage instead of CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()

    # Determine device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default image
    if args.image:
        image_path = args.image
    else:
        candidates = [
            root.parent / "mat" / "data" / "face_002" / "instruct_512.png",
            root / "perturbation_samples" / "face_002" / "original.png",
        ]
        image_path = next((path for path in candidates if path.exists()), None)
        if image_path is None:
            raise FileNotFoundError(
                "No input image found. Provide --image or place MAT beside FACE4 with "
                "mat/data/face_002/instruct_512.png."
            )

    # Default config
    if args.config:
        config_path = args.config
    else:
        config_path = root / "configs" / "geometry_default.json"

    # Default output
    if args.output:
        output_path = args.output
    else:
        timestamp = torch.cuda.current_device() if torch.cuda.is_available() else 0
        output_dir = root / "outputs" / "perturbed_from_config"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"perturbed_{image_path.stem}.png"

    print(f"[perturbed-from-config] Input: {image_path}")
    print(f"[perturbed-from-config] Config: {config_path}")
    print(f"[perturbed-from-config] Output: {output_path}")
    print(f"[perturbed-from-config] Device: {device}")

    diagnostics = generate_perturbed_image(image_path, output_path, config_path, device)

    print(f"[perturbed-from-config] Generated: {output_path}")
    print(f"[perturbed-from-config] Diagnostics: {output_path.with_suffix('.json')}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
