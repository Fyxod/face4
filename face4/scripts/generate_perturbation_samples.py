"""Generate visible sample strips for individual FACE4 perturbation families.

This script is intentionally model-free. It only applies geometric/image
perturbations to source images and writes sample strips under
`perturbation_samples/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from face4.core.geometry.combined_face import CombinedFacePerturbation, FaceGeometryConfig
from face4.core.image_metrics import flow_to_pil, pil_to_tensor, save_sheet, tensor_to_pil


PERTURBATION_NOTES = {
    "polar": {
        "title": "Polar coordinate shifts / perturbations",
        "note": "Implemented as radial expansion/compression plus radius-dependent twist around the image center. This is a direct differentiable image-plane coordinate warp.",
    },
    "bspline_bezier_ffd": {
        "title": "B-spline / Bezier-style free-form deformation",
        "note": "Implemented as a differentiable control-grid free-form deformation. This is the practical raster-image version of perturbing spline/curve control parameters; it does not vectorize the whole photo into literal Bezier curves.",
    },
    "lens_barrel": {
        "title": "Lens barrel distortion",
        "note": "Implemented as inverse-map radial lens distortion using a polynomial radius factor. The positive radial map is normalized to stay inside the finite image so samples do not become reflection/border artifacts.",
    },
    "lens_pincushion": {
        "title": "Lens pincushion distortion",
        "note": "Implemented with the opposite inverse-map radial sign from barrel distortion. The positive radial map is normalized to stay inside the finite image so samples do not become reflection/border artifacts.",
    },
    "mobius": {
        "title": "Möbius transform",
        "note": "Implemented on normalized image coordinates as a small complex-plane linear-fractional warp z'=(z+b)/(cz+1). This is differentiable and conformal away from singularities, with limits to avoid poles inside the image.",
    },
    "laplacian": {
        "title": "Laplacian-smoothed deformation",
        "note": "Laplacian smoothing itself is a mesh/field regularizer, not a unique warp. Here it is implemented as sparse handle/control displacements on a low-resolution grid, repeatedly diffused by a discrete Laplacian-like neighbor average before interpolation.",
    },
    "geodesic": {
        "title": "Geodesic-inspired deformation",
        "note": "The referenced Lu/Jain 3D face method is for 2.5D/3D facial surfaces. FACE4 only has RGB images, so this is a 2D surrogate using an elliptical face metric with radial/tangential flow.",
    },
    "differential_surface": {
        "title": "Differential-geometry-inspired surface warp",
        "note": "Differential geometry of surfaces is a 3D/surface theory, not directly a raster warp. This implementation treats a scalar height map h(x,y) as a simple Monge surface and uses its gradient as a coordinate displacement.",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_images(root: Path, face_id: str) -> list[Path]:
    parent = root.parent
    candidates = [
        parent / "mat" / "data" / face_id / "instruct_512.png",
        parent / "mat" / "data" / face_id / "master_1024.png",
        root / "data" / face_id / "instruct_512.png",
    ]
    candidates.extend(sorted(root.glob(f"outputs/**/{face_id}__*/original.png")))
    return [path for path in candidates if path.exists()]


def _resolve_images(root: Path, faces: list[str], explicit_image: str | None) -> dict[str, Path]:
    if explicit_image:
        path = Path(explicit_image)
        if not path.exists():
            raise FileNotFoundError(f"Explicit --image not found: {path}")
        return {path.stem: path}

    resolved: dict[str, Path] = {}
    for face_id in faces:
        found = _candidate_images(root, face_id)
        if found:
            resolved[face_id] = found[0]
    if not resolved:
        raise FileNotFoundError(
            "No sample images found. Expected MAT data beside this repo or existing FACE4 output originals. "
            "Pass --image PATH to force a specific image."
        )
    return resolved


def _base_config(**kwargs) -> FaceGeometryConfig:
    values = dict(
        tps_enabled=False,
        delaunay_enabled=False,
        rolling_enabled=False,
        dct_enabled=False,
        fft_phase_enabled=False,
        polar_enabled=False,
        bspline_enabled=False,
        lens_barrel_enabled=False,
        lens_pincushion_enabled=False,
        mobius_enabled=False,
        laplacian_enabled=False,
        geodesic_enabled=False,
        differential_surface_enabled=False,
        edge_falloff_px=0.0,
        init="neutral",
        spatial_padding_mode="border",
    )
    values.update(kwargs)
    return FaceGeometryConfig(**values)


def _sinusoidal_controls(shape: tuple[int, ...], limit: float, device: torch.device) -> torch.Tensor:
    _, channels, height, width = shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    x = torch.sin(torch.pi * yy) * torch.cos(torch.pi * xx)
    y = torch.cos(torch.pi * yy) * torch.sin(torch.pi * xx)
    control = torch.stack([x, y], dim=0)[None]
    return control[:, :channels] * float(limit)


def _laplacian_handle_controls(shape: tuple[int, ...], limit: float, device: torch.device) -> torch.Tensor:
    """Sparse handle-style control displacements for Laplacian samples."""

    _, channels, height, width = shape
    controls = torch.zeros(1, channels, height, width, device=device)
    handles = [
        (height // 3, width // 3, 0.85, -0.55),
        (height // 3, 2 * width // 3, -0.65, 0.35),
        (2 * height // 3, width // 2, 0.45, 0.90),
        (height // 2, width // 4, -0.35, 0.45),
    ]
    for row, col, dx, dy in handles:
        row = max(1, min(height - 2, int(row)))
        col = max(1, min(width - 2, int(col)))
        controls[0, 0, row, col] = float(dx) * float(limit)
        if channels > 1:
            controls[0, 1, row, col] = float(dy) * float(limit)
    return controls


def _surface_controls(shape: tuple[int, ...], limit: float, device: torch.device) -> torch.Tensor:
    _, _, height, width = shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    value = torch.exp(-((xx - 0.25).square() + (yy + 0.12).square()) / 0.25)
    value -= 0.75 * torch.exp(-((xx + 0.32).square() + (yy - 0.18).square()) / 0.18)
    return value[None, None] * float(limit)


def _configured_geometry(name: str, level: float, height: int, width: int, device: torch.device) -> CombinedFacePerturbation:
    if name == "polar":
        cfg = _base_config(
            polar_enabled=True,
            polar_radial_px_limit=40.0,
            polar_twist_limit_rad=0.85,
        )
    elif name == "bspline_bezier_ffd":
        cfg = _base_config(bspline_enabled=True, bspline_size=7, bspline_px_limit=48.0)
    elif name == "lens_barrel":
        cfg = _base_config(lens_barrel_enabled=True, lens_k_limit=0.28)
    elif name == "lens_pincushion":
        cfg = _base_config(lens_pincushion_enabled=True, lens_k_limit=0.28)
    elif name == "mobius":
        cfg = _base_config(mobius_enabled=True, mobius_limit=0.45)
    elif name == "laplacian":
        cfg = _base_config(
            laplacian_enabled=True,
            laplacian_size=11,
            laplacian_px_limit=72.0,
            laplacian_smoothing_steps=10,
            laplacian_smoothing_alpha=0.55,
        )
    elif name == "geodesic":
        cfg = _base_config(geodesic_enabled=True, geodesic_px_limit=56.0)
    elif name == "differential_surface":
        cfg = _base_config(
            differential_surface_enabled=True,
            differential_surface_size=8,
            differential_surface_height_limit=0.16,
            differential_surface_px_scale=20.0,
        )
    else:
        raise ValueError(f"Unknown sample perturbation: {name}")

    geo = CombinedFacePerturbation(height, width, 3, device, seed=20260708, config=cfg)
    with torch.no_grad():
        if name == "polar":
            geo.polar_params[:] = torch.tensor([level * 32.0, level * 0.65], device=device)
        elif name == "bspline_bezier_ffd":
            geo.bspline_raw[:] = _sinusoidal_controls(tuple(geo.bspline_raw.shape), level * 40.0, device)
        elif name == "lens_barrel":
            geo.lens_barrel_k[:] = level * 0.22
        elif name == "lens_pincushion":
            geo.lens_pincushion_k[:] = level * 0.22
        elif name == "mobius":
            geo.mobius_params[:] = torch.tensor([0.08, -0.04, 0.22, -0.18], device=device) * level
        elif name == "laplacian":
            geo.laplacian_raw[:] = _laplacian_handle_controls(tuple(geo.laplacian_raw.shape), level * 72.0, device)
        elif name == "geodesic":
            geo.geodesic_params[:] = torch.tensor([34.0, 18.0, -20.0, 12.0], device=device) * level
        elif name == "differential_surface":
            geo.differential_surface_height[:] = _surface_controls(
                tuple(geo.differential_surface_height.shape),
                level * 0.14,
                device,
            )
        geo.project_()
    return geo


def _apply_sample(image: Image.Image, name: str, level: float, device: torch.device) -> tuple[Image.Image, Image.Image, dict]:
    tensor = pil_to_tensor(image, device)
    _, _, height, width = tensor.shape
    geo = _configured_geometry(name, level, height, width, device)
    with torch.no_grad():
        perturbed, aux = geo(tensor)
    flow = flow_to_pil(aux["displacement"], scale_px=geo.component_limit_for_flow)
    return tensor_to_pil(perturbed), flow, aux["diagnostics"]


def _write_notes(output_root: Path, metadata: dict) -> None:
    lines = [
        "# FACE4 perturbation samples",
        "",
        "These samples are model-free visual checks for individual perturbation families. They do not run InstructPix2Pix or ArcFace.",
        "",
        "The optimizer config keeps the new components disabled by default. Enable them in `configs/geometry_default.json` when you want them active in a main run.",
        "",
        "## Perturbation notes",
        "",
    ]
    for name, item in PERTURBATION_NOTES.items():
        lines.extend([f"### {item['title']}", "", item["note"], ""])
    lines.extend(
        [
            "## Reference basis",
            "",
            "- Generic image warps are implemented as inverse coordinate maps sampled with bilinear interpolation, matching the standard remap/grid-sampling model.",
            "- Lens barrel and pincushion use radial camera-distortion style coordinate factors.",
            "- The B-spline/Bezier-style implementation uses a free-form deformation control grid, the practical raster-image analogue of perturbing spline control parameters.",
            "- Möbius uses the complex linear-fractional form `(a z + b) / (c z + d)`, restricted near identity for stability.",
            "- Laplacian smoothing is treated as sparse-handle displacement diffusion because Laplacian methods are normally mesh/field editing tools, not a standalone image-plane warp.",
            "- Geodesic and differential-surface items require actual 3D/surface data for literal implementations, so FACE4 provides RGB image-plane surrogates.",
            "",
        ]
    )
    lines.extend(["## Generated inputs", "", "```json", json.dumps(metadata, indent=2), "```", ""])
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_lens_grid_check(output_root: Path, device: torch.device) -> None:
    grid_dir = output_root / "lens_grid_check"
    grid_dir.mkdir(parents=True, exist_ok=True)
    width, height = 512, 320
    image = Image.new("RGB", (width, height), (210, 230, 244))
    draw = ImageDraw.Draw(image)
    for x in range(0, width + 1, 40):
        draw.line((x, 0, x, height), fill=(30, 50, 65), width=3)
    for y in range(0, height + 1, 40):
        draw.line((0, y, width, y), fill=(30, 50, 65), width=3)
    image.save(grid_dir / "grid_original.png")
    barrel, _, _ = _apply_sample(image, "lens_barrel", 1.0, device)
    pincushion, _, _ = _apply_sample(image, "lens_pincushion", 1.0, device)
    barrel.save(grid_dir / "grid_barrel.png")
    pincushion.save(grid_dir / "grid_pincushion.png")
    save_sheet(
        grid_dir / "grid_lens_check.png",
        [("undistorted", image), ("barrel", barrel), ("pincushion", pincushion)],
        cell_width=360,
    )


def generate_samples(root: Path, output_root: Path, faces: list[str], explicit_image: str | None, save_level_images: bool) -> dict:
    device = torch.device("cpu")
    output_root.mkdir(parents=True, exist_ok=True)
    images = _resolve_images(root, faces, explicit_image)
    levels = [0.25, 0.50, 0.75, 1.00]
    metadata: dict = {"images": {}, "perturbations": PERTURBATION_NOTES, "levels": levels}

    for image_key, path in images.items():
        image = Image.open(path).convert("RGB")
        image_dir = output_root / image_key
        image_dir.mkdir(parents=True, exist_ok=True)
        image.save(image_dir / "original.png")
        metadata["images"][image_key] = str(path)
        overview_entries: list[tuple[str, Image.Image]] = [("Original", image)]
        for name in PERTURBATION_NOTES:
            strip_images: list[tuple[str, Image.Image]] = [("Original", image)]
            flow_images: list[tuple[str, Image.Image]] = []
            diagnostics: list[dict] = []
            for level in levels:
                perturbed, flow, diag = _apply_sample(image, name, level, device)
                label = f"{level:.2f}x"
                if save_level_images:
                    perturbed.save(image_dir / f"{name}_{label}.png")
                    flow.save(image_dir / f"{name}_{label}_flow.png")
                strip_images.append((label, perturbed))
                flow_images.append((label, flow))
                diagnostics.append({"level": level, **diag})
            save_sheet(image_dir / f"{name}_strip.png", strip_images, cell_width=256)
            save_sheet(image_dir / f"{name}_flow_strip.png", flow_images, cell_width=256)
            (image_dir / f"{name}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            overview_entries.append((name.replace("_", " "), strip_images[-1][1]))
        save_sheet(image_dir / "all_perturbations_overview.png", overview_entries, cell_width=220)

    _write_notes(output_root, metadata)
    _write_lens_grid_check(output_root, device)
    (output_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FACE4 individual perturbation sample strips.")
    parser.add_argument("--output-root", default="perturbation_samples")
    parser.add_argument("--faces", nargs="*", default=["face_002", "face_005"])
    parser.add_argument("--image", default=None, help="Optional explicit image path. If set, --faces is ignored.")
    parser.add_argument("--save-level-images", action="store_true", help="Also save every individual level image/flow, not just strips.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    metadata = generate_samples(root, root / args.output_root, args.faces, args.image, args.save_level_images)
    print(f"[perturbation-samples] wrote {root / args.output_root}")
    print(json.dumps({"num_images": len(metadata["images"]), "images": metadata["images"]}, indent=2))


if __name__ == "__main__":
    main()
