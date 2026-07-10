"""Joint FACE perturbation: spatial geometry + image DCT + optional FFT phase."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dct_image import DCTImagePerturbation
from .delaunay import delaunay_barycentric
from .fft_phase import FFTPhasePerturbation
from .rolling import rolling_field
from .tps import tps_basis


@dataclass
class FaceGeometryConfig:
    init: str = "neutral"
    init_fraction: float = 0.05
    tps_size: int = 5
    delaunay_size: int = 5
    dct_block_size: int = 8
    dct_frequency_mask: str = "all_ac"
    dct_exclude_dc: bool = True
    fft_phase_size: int = 8
    bspline_size: int = 6
    laplacian_size: int = 7
    differential_surface_size: int = 7
    edge_falloff_px: float = 16.0
    tps_enabled: bool = True
    delaunay_enabled: bool = True
    rolling_enabled: bool = True
    dct_enabled: bool = True
    fft_phase_enabled: bool = True
    polar_enabled: bool = False
    bspline_enabled: bool = False
    lens_barrel_enabled: bool = False
    lens_pincushion_enabled: bool = False
    mobius_enabled: bool = False
    laplacian_enabled: bool = False
    geodesic_enabled: bool = False
    differential_surface_enabled: bool = False
    tps_norm_limit: float = 0.007
    delaunay_norm_limit: float = 0.010
    rolling_norm_limit: float = 0.009
    polar_radial_norm_limit: float = 0.020
    polar_twist_limit_rad: float = 0.35
    bspline_norm_limit: float = 0.020
    lens_k_limit: float = 0.35
    lens_barrel_k_limit: float | None = None
    lens_pincushion_k_limit: float | None = None
    mobius_limit: float = 0.20
    laplacian_norm_limit: float = 0.020
    geodesic_norm_limit: float = 0.030
    differential_surface_height_limit: float = 0.040
    differential_surface_px_scale: float = 12.0
    tps_px_limit: float | None = None
    delaunay_px_limit: float | None = None
    rolling_px_limit: float | None = None
    polar_radial_px_limit: float | None = None
    bspline_px_limit: float | None = None
    laplacian_px_limit: float | None = None
    geodesic_px_limit: float | None = None
    dct_gain_limit: float = 0.5
    fft_phase_limit_rad: float = math.pi
    laplacian_smoothing_steps: int = 8
    laplacian_smoothing_alpha: float = 0.35
    spatial_padding_mode: str = "reflection"
    max_combined_disp_px: float | None = None


def _limit_px(norm_limit: float, height: int, width: int) -> float:
    return float(norm_limit) * float(max(height, width))


def _configured_limit_px(px_limit: float | None, norm_limit: float, height: int, width: int) -> float:
    if px_limit is not None:
        return float(px_limit)
    return _limit_px(norm_limit, height, width)


def load_face_geometry_config(path: str | Path | None) -> FaceGeometryConfig:
    """Load a JSON geometry config.

    The file may use top-level dataclass keys and/or the friendlier nested
    structure used by `configs/geometry_default.json`.
    """

    if path is None:
        return FaceGeometryConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    allowed = set(FaceGeometryConfig.__dataclass_fields__)
    for key, value in payload.items():
        if key in allowed:
            values[key] = value
    for key, value in payload.get("sizes", {}).items():
        if key in allowed:
            values[key] = value
    for key, value in payload.get("global", {}).items():
        if key in allowed:
            values[key] = value
    components = payload.get("components", {})
    mapping = {
        "tps": "tps",
        "delaunay": "delaunay",
        "rolling": "rolling",
        "dct": "dct",
        "fft_phase": "fft_phase",
        "polar": "polar",
        "bspline": "bspline",
        "lens_barrel": "lens_barrel",
        "lens_pincushion": "lens_pincushion",
        "mobius": "mobius",
        "laplacian": "laplacian",
        "geodesic": "geodesic",
        "differential_surface": "differential_surface",
    }
    for name, prefix in mapping.items():
        block = components.get(name, {})
        if "enabled" in block:
            values[f"{prefix}_enabled"] = bool(block["enabled"])
        if name in {"tps", "delaunay", "rolling"}:
            if "norm_limit" in block:
                values[f"{prefix}_norm_limit"] = float(block["norm_limit"])
            if "px_limit" in block:
                values[f"{prefix}_px_limit"] = None if block["px_limit"] is None else float(block["px_limit"])
        elif name == "dct":
            if "block_size" in block:
                values["dct_block_size"] = int(block["block_size"])
            if "frequency_mask" in block:
                values["dct_frequency_mask"] = str(block["frequency_mask"])
            if "exclude_dc" in block:
                values["dct_exclude_dc"] = bool(block["exclude_dc"])
            if "gain_limit" in block:
                values["dct_gain_limit"] = float(block["gain_limit"])
        elif name == "fft_phase":
            if "phase_limit_rad" in block:
                values["fft_phase_limit_rad"] = float(block["phase_limit_rad"])
        elif name == "polar":
            if "radial_norm_limit" in block:
                values["polar_radial_norm_limit"] = float(block["radial_norm_limit"])
            if "radial_px_limit" in block:
                values["polar_radial_px_limit"] = None if block["radial_px_limit"] is None else float(block["radial_px_limit"])
            if "twist_limit_rad" in block:
                values["polar_twist_limit_rad"] = float(block["twist_limit_rad"])
        elif name == "bspline":
            if "size" in block:
                values["bspline_size"] = int(block["size"])
            if "norm_limit" in block:
                values["bspline_norm_limit"] = float(block["norm_limit"])
            if "px_limit" in block:
                values["bspline_px_limit"] = None if block["px_limit"] is None else float(block["px_limit"])
        elif name in {"lens_barrel", "lens_pincushion"}:
            if "k_limit" in block:
                values[f"{prefix}_k_limit"] = float(block["k_limit"])
        elif name == "mobius":
            if "limit" in block:
                values["mobius_limit"] = float(block["limit"])
        elif name == "laplacian":
            if "size" in block:
                values["laplacian_size"] = int(block["size"])
            if "norm_limit" in block:
                values["laplacian_norm_limit"] = float(block["norm_limit"])
            if "px_limit" in block:
                values["laplacian_px_limit"] = None if block["px_limit"] is None else float(block["px_limit"])
            if "smoothing_steps" in block:
                values["laplacian_smoothing_steps"] = int(block["smoothing_steps"])
            if "smoothing_alpha" in block:
                values["laplacian_smoothing_alpha"] = float(block["smoothing_alpha"])
        elif name == "geodesic":
            if "norm_limit" in block:
                values["geodesic_norm_limit"] = float(block["norm_limit"])
            if "px_limit" in block:
                values["geodesic_px_limit"] = None if block["px_limit"] is None else float(block["px_limit"])
        elif name == "differential_surface":
            if "size" in block:
                values["differential_surface_size"] = int(block["size"])
            if "height_limit" in block:
                values["differential_surface_height_limit"] = float(block["height_limit"])
            if "px_scale" in block:
                values["differential_surface_px_scale"] = float(block["px_scale"])
    return FaceGeometryConfig(**values)


def _field_stats(field: torch.Tensor, prefix: str) -> dict[str, float]:
    mag = torch.sqrt(field.detach().float().square().sum(dim=1))
    return {
        f"{prefix}_mean_disp": float(mag.mean().cpu()),
        f"{prefix}_max_disp": float(mag.max().cpu()),
        f"{prefix}_p95_disp": float(torch.quantile(mag.flatten(), 0.95).cpu()),
    }


def displacement_stats(field: torch.Tensor) -> dict[str, float]:
    mag = torch.sqrt(field.detach().float().square().sum(dim=1))
    return {
        "combined_max_disp_px": float(mag.max().cpu()),
        "combined_mean_disp_px": float(mag.mean().cpu()),
        "combined_p95_disp_px": float(torch.quantile(mag.flatten(), 0.95).cpu()),
    }


def smoothness_tv(field: torch.Tensor) -> torch.Tensor:
    return (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean() + (
        field[:, :, 1:] - field[:, :, :-1]
    ).abs().mean()


def jacobian_diagnostics(field: torch.Tensor) -> dict[str, float]:
    dx, dy = field[:, 0], field[:, 1]
    ddx = F.pad((dx[:, :, 2:] - dx[:, :, :-2]) / 2.0, (1, 1))
    dxy = F.pad((dx[:, 2:] - dx[:, :-2]) / 2.0, (0, 0, 1, 1))
    dyx = F.pad((dy[:, :, 2:] - dy[:, :, :-2]) / 2.0, (1, 1))
    ddy = F.pad((dy[:, 2:] - dy[:, :-2]) / 2.0, (0, 0, 1, 1))
    det = (1.0 + ddx) * (1.0 + ddy) - dxy * dyx
    return {
        "jacobian_det_min": float(det.detach().float().min().cpu()),
        "foldover_fraction": float((det.detach().float() < 0).float().mean().cpu()),
        "smoothness_tv": float(smoothness_tv(field.detach().float()).cpu()),
    }


def _upsample_control_field(control: torch.Tensor, height: int, width: int, mode: str = "bicubic") -> torch.Tensor:
    return F.interpolate(control, size=(height, width), mode=mode, align_corners=True)


def _control_boundary_mask(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    mask = torch.ones(*shape, device=device)
    if len(shape) >= 4:
        mask[:, :, 0] = 0
        mask[:, :, -1] = 0
        mask[:, :, :, 0] = 0
        mask[:, :, :, -1] = 0
    return mask


def _pixel_field_from_normalized_delta(
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    px = delta_x * float(max(width - 1, 1)) / 2.0
    py = delta_y * float(max(height - 1, 1)) / 2.0
    return torch.stack([px, py], dim=0)[None]


def _cap_field(field: torch.Tensor, cap_px: float | None) -> torch.Tensor:
    if cap_px is None or cap_px <= 0:
        return field
    magnitude = torch.sqrt(field.square().sum(dim=1, keepdim=True) + 1e-12)
    return field * torch.clamp(float(cap_px) / magnitude.clamp_min(1e-6), max=1.0)


class CombinedFacePerturbation(torch.nn.Module):
    """Combined differentiable FACE perturbation module.

    TPS, Delaunay, and rolling shutter are summed as coordinate fields and
    applied with grid_sample. A true image-domain DCT coefficient perturbation
    is then applied, followed by optional FFT phase.
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        device: torch.device,
        seed: int = 1234,
        config: FaceGeometryConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FaceGeometryConfig()
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.tps_limit_px = _configured_limit_px(self.config.tps_px_limit, self.config.tps_norm_limit, height, width)
        self.delaunay_limit_px = _configured_limit_px(
            self.config.delaunay_px_limit, self.config.delaunay_norm_limit, height, width
        )
        self.rolling_limit_px = _configured_limit_px(self.config.rolling_px_limit, self.config.rolling_norm_limit, height, width)
        self.polar_radial_limit_px = _configured_limit_px(
            self.config.polar_radial_px_limit, self.config.polar_radial_norm_limit, height, width
        )
        self.bspline_limit_px = _configured_limit_px(self.config.bspline_px_limit, self.config.bspline_norm_limit, height, width)
        self.laplacian_limit_px = _configured_limit_px(
            self.config.laplacian_px_limit, self.config.laplacian_norm_limit, height, width
        )
        self.geodesic_limit_px = _configured_limit_px(self.config.geodesic_px_limit, self.config.geodesic_norm_limit, height, width)
        self.lens_barrel_k_limit = (
            float(self.config.lens_barrel_k_limit)
            if self.config.lens_barrel_k_limit is not None
            else float(self.config.lens_k_limit)
        )
        self.lens_pincushion_k_limit = (
            float(self.config.lens_pincushion_k_limit)
            if self.config.lens_pincushion_k_limit is not None
            else float(self.config.lens_k_limit)
        )
        self.component_limit_for_flow = max(
            self.tps_limit_px,
            self.delaunay_limit_px,
            self.rolling_limit_px,
            self.polar_radial_limit_px,
            self.bspline_limit_px,
            self.laplacian_limit_px,
            self.geodesic_limit_px,
            float(self.config.differential_surface_height_limit) * float(max(height, width)),
            1.0,
        )

        generator = torch.Generator(device=device).manual_seed(seed + 9101)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1)[None])
        self.register_buffer("xx_norm", xx)
        self.register_buffer("yy_norm", yy)
        self.register_buffer("yy", yy[None, None])
        rr = torch.sqrt(xx.square() + yy.square()).clamp_min(1e-6)
        self.register_buffer("rr_norm", rr)
        self.register_buffer("unit_x", xx / rr)
        self.register_buffer("unit_y", yy / rr)
        self.register_buffer("ellipse_mask", torch.exp(-((xx / 0.62).square() + ((yy + 0.02) / 0.78).square()))[None, None])

        distances = torch.minimum(
            torch.minimum(torch.arange(width, device=device)[None], torch.arange(width - 1, -1, -1, device=device)[None]),
            torch.minimum(torch.arange(height, device=device)[:, None], torch.arange(height - 1, -1, -1, device=device)[:, None]),
        ).float()
        t = (distances / max(float(self.config.edge_falloff_px), 1.0)).clamp(0, 1)
        edge = t * t * (3 - 2 * t)
        self.register_buffer("edge", edge[None, None])

        self.register_buffer("tps_matrix", tps_basis(self.config.tps_size, height, width, device))
        delaunay_idx, delaunay_weight = delaunay_barycentric(self.config.delaunay_size, height, width, device)
        self.register_buffer("delaunay_idx", delaunay_idx)
        self.register_buffer("delaunay_weight", delaunay_weight)

        def init_tensor(shape, limit: float):
            if self.config.init == "small_random":
                return torch.randn(*shape, device=device, generator=generator) * (limit * self.config.init_fraction)
            return torch.zeros(*shape, device=device)

        self.tps_raw = torch.nn.Parameter(init_tensor((1, 2, self.config.tps_size, self.config.tps_size), self.tps_limit_px))
        self.delaunay_raw = torch.nn.Parameter(
            init_tensor((1, 2, self.config.delaunay_size, self.config.delaunay_size), self.delaunay_limit_px)
        )
        self.roll_params = torch.nn.Parameter(init_tensor((2,), self.rolling_limit_px))
        if self.config.init == "small_random":
            polar_init = torch.randn(2, device=device, generator=generator) * self.config.init_fraction
            polar_init = polar_init * torch.tensor(
                [self.polar_radial_limit_px, float(self.config.polar_twist_limit_rad)],
                device=device,
                dtype=polar_init.dtype,
            )
        else:
            polar_init = torch.zeros(2, device=device)
        self.polar_params = torch.nn.Parameter(polar_init)
        self.bspline_raw = torch.nn.Parameter(
            init_tensor((1, 2, self.config.bspline_size, self.config.bspline_size), self.bspline_limit_px)
        )
        self.lens_barrel_k = torch.nn.Parameter(init_tensor((1,), self.lens_barrel_k_limit))
        self.lens_pincushion_k = torch.nn.Parameter(init_tensor((1,), self.lens_pincushion_k_limit))
        self.mobius_params = torch.nn.Parameter(init_tensor((4,), float(self.config.mobius_limit)))
        self.laplacian_raw = torch.nn.Parameter(
            init_tensor((1, 2, self.config.laplacian_size, self.config.laplacian_size), self.laplacian_limit_px)
        )
        self.geodesic_params = torch.nn.Parameter(init_tensor((4,), self.geodesic_limit_px))
        self.differential_surface_height = torch.nn.Parameter(
            init_tensor(
                (1, 1, self.config.differential_surface_size, self.config.differential_surface_size),
                float(self.config.differential_surface_height_limit),
            )
        )
        self.dct_image = DCTImagePerturbation(
            channels=channels,
            block_size=self.config.dct_block_size,
            gain_limit=self.config.dct_gain_limit,
            frequency_mask_mode=self.config.dct_frequency_mask,
            exclude_dc=self.config.dct_exclude_dc,
            enabled=self.config.dct_enabled,
            device=device,
        )

        tps_mask = _control_boundary_mask(tuple(self.tps_raw.shape), device)
        self.register_buffer("tps_mask", tps_mask)
        delaunay_mask = _control_boundary_mask(tuple(self.delaunay_raw.shape), device)
        self.register_buffer("delaunay_mask", delaunay_mask)
        self.register_buffer("bspline_mask", _control_boundary_mask(tuple(self.bspline_raw.shape), device))
        self.register_buffer("laplacian_mask", _control_boundary_mask(tuple(self.laplacian_raw.shape), device))
        self.register_buffer(
            "differential_surface_mask",
            _control_boundary_mask(tuple(self.differential_surface_height.shape), device),
        )

        fft_init = 0.0 if self.config.init == "neutral" else 0.05 * torch.pi
        self.fft_phase = FFTPhasePerturbation(
            channels,
            self.config.fft_phase_size,
            float(fft_init),
            device,
            seed,
            max_phase_rad=self.config.fft_phase_limit_rad,
        )
        self.tps_raw.requires_grad_(self.config.tps_enabled)
        self.delaunay_raw.requires_grad_(self.config.delaunay_enabled)
        self.roll_params.requires_grad_(self.config.rolling_enabled)
        self.polar_params.requires_grad_(self.config.polar_enabled)
        self.bspline_raw.requires_grad_(self.config.bspline_enabled)
        self.lens_barrel_k.requires_grad_(self.config.lens_barrel_enabled)
        self.lens_pincushion_k.requires_grad_(self.config.lens_pincushion_enabled)
        self.mobius_params.requires_grad_(self.config.mobius_enabled)
        self.laplacian_raw.requires_grad_(self.config.laplacian_enabled)
        self.geodesic_params.requires_grad_(self.config.geodesic_enabled)
        self.differential_surface_height.requires_grad_(self.config.differential_surface_enabled)
        self.fft_phase.raw_phase.requires_grad_(self.config.fft_phase_enabled)
        self.project_()

    def _zero_field(self) -> torch.Tensor:
        return self.base_grid.new_zeros((1, 2, self.height, self.width))

    def _tps_field(self) -> torch.Tensor:
        if not self.config.tps_enabled:
            return self._zero_field()
        controls = (self.tps_raw.clamp(-self.tps_limit_px, self.tps_limit_px) * self.tps_mask).reshape(1, 2, -1)
        field = torch.einsum("pn,bcn->bcp", self.tps_matrix, controls)
        return field.reshape(1, 2, self.height, self.width)

    def _delaunay_field(self) -> torch.Tensor:
        if not self.config.delaunay_enabled:
            return self._zero_field()
        controls = (self.delaunay_raw.clamp(-self.delaunay_limit_px, self.delaunay_limit_px) * self.delaunay_mask).reshape(1, 2, -1)
        gathered = controls[:, :, self.delaunay_idx.flatten()].reshape(1, 2, -1, 3)
        field = (gathered * self.delaunay_weight[None, None]).sum(-1)
        return field.reshape(1, 2, self.height, self.width)

    def _rolling_field(self) -> torch.Tensor:
        if not self.config.rolling_enabled:
            return self._zero_field()
        params = self.roll_params.clamp(-self.rolling_limit_px, self.rolling_limit_px)
        return rolling_field(self.yy, params[0], params[1])

    def _polar_field(self) -> torch.Tensor:
        """Polar radial/twist perturbation in image coordinates."""

        if not self.config.polar_enabled:
            return self._zero_field()
        radial_px = self.polar_params[0].clamp(-self.polar_radial_limit_px, self.polar_radial_limit_px)
        twist = self.polar_params[1].clamp(-float(self.config.polar_twist_limit_rad), float(self.config.polar_twist_limit_rad))
        r = self.rr_norm.clamp(0, math.sqrt(2.0))
        radial = radial_px * r.square()
        radial_field = torch.stack([radial * self.unit_x, radial * self.unit_y], dim=0)[None]

        angle = twist * r.square()
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_rot = self.xx_norm * cos_a - self.yy_norm * sin_a
        y_rot = self.xx_norm * sin_a + self.yy_norm * cos_a
        twist_field = _pixel_field_from_normalized_delta(
            x_rot - self.xx_norm,
            y_rot - self.yy_norm,
            self.height,
            self.width,
        )
        return radial_field + twist_field

    def _bspline_field(self) -> torch.Tensor:
        """B-spline/Bezier-style free-form control-grid displacement."""

        if not self.config.bspline_enabled:
            return self._zero_field()
        controls = self.bspline_raw.clamp(-self.bspline_limit_px, self.bspline_limit_px) * self.bspline_mask
        return _cap_field(_upsample_control_field(controls, self.height, self.width), self.bspline_limit_px)

    def _lens_field(self, parameter: torch.Tensor, limit: float, sign: float, enabled: bool) -> torch.Tensor:
        if not enabled:
            return self._zero_field()
        if float(limit) <= 0.0:
            return self._zero_field()
        k = parameter[0].clamp(-float(limit), float(limit)) * float(sign)
        x = self.xx_norm
        y = self.yy_norm
        r2 = x.square() + y.square()
        # Positive inverse radial coefficients can otherwise request samples
        # outside the finite source image. Normalize by the corner factor so
        # the visualization/optimization does not degenerate into padding
        # streaks while retaining the radial lens curvature.
        safe_scale = torch.where(k > 0, 1.0 / (1.0 + 2.0 * k), torch.ones_like(k))
        factor = safe_scale * (1.0 + k * r2)
        return _pixel_field_from_normalized_delta(
            x * factor - x,
            y * factor - y,
            self.height,
            self.width,
        )

    def _lens_barrel_field(self) -> torch.Tensor:
        # grid_sample uses inverse mapping. With the bounded radial map in
        # _lens_field, positive k produces the visible barrel pattern: grid
        # lines bulge outward from the center.
        return self._lens_field(self.lens_barrel_k, self.lens_barrel_k_limit, 1.0, self.config.lens_barrel_enabled)

    def _lens_pincushion_field(self) -> torch.Tensor:
        # Opposite sign of barrel: grid lines pinch inward toward the center.
        return self._lens_field(
            self.lens_pincushion_k,
            self.lens_pincushion_k_limit,
            -1.0,
            self.config.lens_pincushion_enabled,
        )

    def _mobius_field(self) -> torch.Tensor:
        """Small complex-plane Möbius/homography-like warp.

        Uses z' = (z + b) / (c z + 1), with trainable complex b and c. The
        identity is b=c=0. Limits are intentionally small to avoid denominator
        singularities in image coordinates.
        """

        if not self.config.mobius_enabled:
            return self._zero_field()
        params = self.mobius_params.clamp(-float(self.config.mobius_limit), float(self.config.mobius_limit))
        b = torch.complex(params[0], params[1])
        c = torch.complex(params[2], params[3])
        z = torch.complex(self.xx_norm.float(), self.yy_norm.float())
        denom = c * z + torch.ones((), device=z.device, dtype=z.dtype)
        denom_abs = denom.abs().clamp_min(0.25)
        denom = denom / denom.abs().clamp_min(1e-6) * denom_abs
        z2 = (z + b) / denom
        return _pixel_field_from_normalized_delta(
            z2.real - self.xx_norm,
            z2.imag - self.yy_norm,
            self.height,
            self.width,
        ).to(dtype=self.base_grid.dtype)

    def _laplacian_field(self) -> torch.Tensor:
        """Laplacian-smoothed displacement component.

        Laplacian smoothing is a mesh/field regularization operation rather
        than a unique image warp. Here it is implemented as a trainable coarse
        handle/control displacement field repeatedly diffused on the coarse
        grid with a discrete Laplacian-like neighbor average, then
        interpolated to the image. This keeps it distinct from the B-spline
        component, which directly interpolates its control grid.
        """

        if not self.config.laplacian_enabled:
            return self._zero_field()
        field = self.laplacian_raw.clamp(-self.laplacian_limit_px, self.laplacian_limit_px) * self.laplacian_mask
        alpha = float(self.config.laplacian_smoothing_alpha)
        alpha = max(0.0, min(alpha, 1.0))
        for _ in range(max(0, int(self.config.laplacian_smoothing_steps))):
            neighbor = F.avg_pool2d(field, kernel_size=3, stride=1, padding=1)
            field = (1.0 - alpha) * field + alpha * neighbor
            field = field * self.laplacian_mask
        field = _upsample_control_field(field, self.height, self.width)
        return _cap_field(field, self.laplacian_limit_px)

    def _geodesic_field(self) -> torch.Tensor:
        """2D geodesic-inspired face deformation surrogate.

        The referenced 3D face work uses actual facial surfaces/range data. For
        RGB-only FACE4 runs, this component uses an elliptical face metric and
        radial/tangential coordinate flow as a practical image-plane surrogate.
        """

        if not self.config.geodesic_enabled:
            return self._zero_field()
        params = self.geodesic_params.clamp(-self.geodesic_limit_px, self.geodesic_limit_px)
        x = self.xx_norm
        y = self.yy_norm + 0.04
        rx = 0.62
        ry = 0.78
        rho = torch.sqrt((x / rx).square() + (y / ry).square()).clamp_min(1e-6)
        ux = (x / (rx * rx)) / rho
        uy = (y / (ry * ry)) / rho
        tangent_x = -uy
        tangent_y = ux
        ring = torch.exp(-((rho - 0.72) / 0.34).square())
        center = torch.exp(-(rho / 0.85).square())
        radial = params[0] * ring + params[2] * center
        tangential = params[1] * ring + params[3] * center
        field = torch.stack(
            [radial * ux + tangential * tangent_x, radial * uy + tangential * tangent_y],
            dim=0,
        )[None]
        return _cap_field(field * self.ellipse_mask, self.geodesic_limit_px)

    def _differential_surface_field(self) -> torch.Tensor:
        """Differential-geometry-inspired surface-gradient coordinate warp.

        A scalar height patch h(x,y) is treated as a simple Monge surface; the
        image-plane displacement is proportional to its gradient. This is a 2D
        differentiable surrogate, not a full 3D surface reconstruction.
        """

        if not self.config.differential_surface_enabled:
            return self._zero_field()
        h = self.differential_surface_height.clamp(
            -float(self.config.differential_surface_height_limit),
            float(self.config.differential_surface_height_limit),
        )
        h = h * self.differential_surface_mask
        height = F.interpolate(h, size=(self.height, self.width), mode="bicubic", align_corners=True)
        dx = F.pad((height[:, :, :, 2:] - height[:, :, :, :-2]) / 2.0, (1, 1))
        dy = F.pad((height[:, :, 2:] - height[:, :, :-2]) / 2.0, (0, 0, 1, 1))
        field = torch.cat([dx, dy], dim=1) * float(self.config.differential_surface_px_scale) * float(max(self.height, self.width))
        cap = float(self.config.differential_surface_height_limit) * float(max(self.height, self.width))
        return _cap_field(field, cap)

    def spatial_fields(self) -> dict[str, torch.Tensor]:
        return {
            "tps": self._tps_field() * self.edge,
            "delaunay": self._delaunay_field() * self.edge,
            "rolling": self._rolling_field() * self.edge,
            "polar": self._polar_field() * self.edge,
            "bspline": self._bspline_field() * self.edge,
            "lens_barrel": self._lens_barrel_field() * self.edge,
            "lens_pincushion": self._lens_pincushion_field() * self.edge,
            "mobius": self._mobius_field() * self.edge,
            "laplacian": self._laplacian_field() * self.edge,
            "geodesic": self._geodesic_field() * self.edge,
            "differential_surface": self._differential_surface_field() * self.edge,
        }

    def spatial_warp(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        fields = self.spatial_fields()
        displacement = sum(fields.values())
        if self.config.max_combined_disp_px is not None and self.config.max_combined_disp_px > 0:
            magnitude = torch.sqrt(displacement.square().sum(dim=1, keepdim=True) + 1e-12)
            cap = float(self.config.max_combined_disp_px)
            displacement = displacement * torch.clamp(cap / magnitude.clamp_min(1e-6), max=1.0)
        grid = self.base_grid.clone()
        grid[..., 0] += 2.0 * displacement[:, 0] / max(self.width - 1, 1)
        grid[..., 1] += 2.0 * displacement[:, 1] / max(self.height - 1, 1)
        padding_mode = str(self.config.spatial_padding_mode)
        if padding_mode not in {"zeros", "border", "reflection"}:
            padding_mode = "reflection"
        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True).clamp(0, 1)
        return warped, displacement, fields

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        spatial, displacement, fields = self.spatial_warp(image)
        dct_output = self.dct_image(spatial)
        dct_image = dct_output.image
        dct_delta = dct_output.delta
        dct_stats = dct_output.stats
        if self.config.fft_phase_enabled:
            perturbed, fft_delta, fft_stats = self.fft_phase(dct_image)
        else:
            perturbed = dct_image
            fft_delta = torch.zeros_like(dct_image)
            fft_stats = {
                "fft_phase_norm": 0.0,
                "fft_phase_mean_abs": 0.0,
                "fft_phase_max_abs": 0.0,
                "legacy_fft_strength_equivalent": 0.0,
                "fft_spatial_delta_mse": 0.0,
            }
        diagnostics = self.diagnostics(displacement, fields)
        diagnostics.update(dct_stats)
        diagnostics.update(fft_stats if isinstance(fft_stats, dict) else fft_stats.__dict__)
        return perturbed, {
            "spatial": spatial,
            "dct_image": dct_image,
            "dct_delta": dct_delta,
            "displacement": displacement,
            "fields": fields,
            "fft_delta": fft_delta,
            "diagnostics": diagnostics,
        }

    def diagnostics(self, displacement: torch.Tensor, fields: dict[str, torch.Tensor]) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(displacement_stats(displacement))
        out.update(jacobian_diagnostics(displacement))
        for name, field in fields.items():
            out.update(_field_stats(field, name))
        return out

    def grad_norms(self) -> dict[str, float]:
        def norm(parameters) -> float:
            values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            if not values:
                return 0.0
            return float(torch.stack(values).sum().sqrt().cpu())

        return {
            "tps_grad_norm": norm([self.tps_raw]),
            "delaunay_grad_norm": norm([self.delaunay_raw]),
            "rolling_grad_norm": norm([self.roll_params]),
            "polar_grad_norm": norm([self.polar_params]),
            "bspline_grad_norm": norm([self.bspline_raw]),
            "lens_barrel_grad_norm": norm([self.lens_barrel_k]),
            "lens_pincushion_grad_norm": norm([self.lens_pincushion_k]),
            "mobius_grad_norm": norm([self.mobius_params]),
            "laplacian_grad_norm": norm([self.laplacian_raw]),
            "geodesic_grad_norm": norm([self.geodesic_params]),
            "differential_surface_grad_norm": norm([self.differential_surface_height]),
            "dct_gain_grad_norm": norm([self.dct_image.dct_gain_raw]),
            "fft_phase_grad_norm": norm([self.fft_phase.raw_phase]),
            "total_grad_norm": norm(list(self.parameters())),
        }

    def _param_stats(self, tensor: torch.Tensor, limit: float, prefix: str) -> dict[str, float | int]:
        data = tensor.detach().float()
        return {
            f"{prefix}_param_min": float(data.min().cpu()),
            f"{prefix}_param_max": float(data.max().cpu()),
            f"{prefix}_param_mean_abs": float(data.abs().mean().cpu()),
            f"{prefix}_num_at_min": int((data <= -limit + 1e-8).sum().cpu()),
            f"{prefix}_num_at_max": int((data >= limit - 1e-8).sum().cpu()),
        }

    def parameter_diagnostics(self) -> dict[str, float | int | str]:
        stats: dict[str, float | int | str] = {}
        stats.update(self._param_stats(self.tps_raw, self.tps_limit_px, "tps"))
        stats.update(self._param_stats(self.delaunay_raw, self.delaunay_limit_px, "delaunay"))
        stats.update(self._param_stats(self.roll_params, self.rolling_limit_px, "rolling"))
        stats.update(self._param_stats(self.polar_params[:1], self.polar_radial_limit_px, "polar_radial"))
        stats.update(self._param_stats(self.polar_params[1:], float(self.config.polar_twist_limit_rad), "polar_twist"))
        stats.update(self._param_stats(self.bspline_raw, self.bspline_limit_px, "bspline"))
        stats.update(self._param_stats(self.lens_barrel_k, self.lens_barrel_k_limit, "lens_barrel"))
        stats.update(self._param_stats(self.lens_pincushion_k, self.lens_pincushion_k_limit, "lens_pincushion"))
        stats.update(self._param_stats(self.mobius_params, float(self.config.mobius_limit), "mobius"))
        stats.update(self._param_stats(self.laplacian_raw, self.laplacian_limit_px, "laplacian"))
        stats.update(self._param_stats(self.geodesic_params, self.geodesic_limit_px, "geodesic"))
        stats.update(
            self._param_stats(
                self.differential_surface_height,
                float(self.config.differential_surface_height_limit),
                "differential_surface",
            )
        )
        stats.update(self.dct_image.parameter_diagnostics())
        phase = self.fft_phase.raw_phase.detach().float()
        phase_limit = float(self.config.fft_phase_limit_rad)
        stats.update(
            {
                "fft_phase_num_at_min": int((phase <= -phase_limit + 1e-8).sum().cpu()),
                "fft_phase_num_at_max": int((phase >= phase_limit - 1e-8).sum().cpu()),
                "tps_enabled": int(self.config.tps_enabled),
                "delaunay_enabled": int(self.config.delaunay_enabled),
                "rolling_enabled": int(self.config.rolling_enabled),
                "dct_enabled": int(self.config.dct_enabled),
                "fft_phase_enabled": int(self.config.fft_phase_enabled),
                "polar_enabled": int(self.config.polar_enabled),
                "bspline_enabled": int(self.config.bspline_enabled),
                "lens_barrel_enabled": int(self.config.lens_barrel_enabled),
                "lens_pincushion_enabled": int(self.config.lens_pincushion_enabled),
                "mobius_enabled": int(self.config.mobius_enabled),
                "laplacian_enabled": int(self.config.laplacian_enabled),
                "geodesic_enabled": int(self.config.geodesic_enabled),
                "differential_surface_enabled": int(self.config.differential_surface_enabled),
            }
        )
        return stats

    def theta_state(self) -> dict[str, Any]:
        """Return only trainable perturbation parameters plus metadata.

        This intentionally excludes large fixed buffers such as TPS matrices,
        DCT bases, grids, and Delaunay interpolation weights. Those buffers are
        deterministic from config + image size and made previous `.pt` files
        unnecessarily huge.
        """

        return {
            "format": "FACE_theta_only_v2_dct_image",
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "config": self.config.__dict__.copy(),
            "limits": self.limits_dict(),
            "theta": {
                "tps_raw": self.tps_raw.detach().cpu().clone(),
                "delaunay_raw": self.delaunay_raw.detach().cpu().clone(),
                "polar_params": self.polar_params.detach().cpu().clone(),
                "bspline_raw": self.bspline_raw.detach().cpu().clone(),
                "lens_barrel_k": self.lens_barrel_k.detach().cpu().clone(),
                "lens_pincushion_k": self.lens_pincushion_k.detach().cpu().clone(),
                "mobius_params": self.mobius_params.detach().cpu().clone(),
                "laplacian_raw": self.laplacian_raw.detach().cpu().clone(),
                "geodesic_params": self.geodesic_params.detach().cpu().clone(),
                "differential_surface_height": self.differential_surface_height.detach().cpu().clone(),
                "dct_gain_raw": self.dct_image.dct_gain_raw.detach().cpu().clone(),
                "roll_params": self.roll_params.detach().cpu().clone(),
                "fft_phase_raw_phase": self.fft_phase.raw_phase.detach().cpu().clone(),
            },
            "dct_metadata": self.dct_image.metadata(),
        }

    def project_(self) -> dict[str, Any]:
        with torch.no_grad():
            blocks = [
                ("tps", self.tps_raw, self.tps_limit_px, self.config.tps_enabled),
                ("delaunay", self.delaunay_raw, self.delaunay_limit_px, self.config.delaunay_enabled),
                ("rolling", self.roll_params, self.rolling_limit_px, self.config.rolling_enabled),
                ("bspline", self.bspline_raw, self.bspline_limit_px, self.config.bspline_enabled),
                ("lens_barrel", self.lens_barrel_k, self.lens_barrel_k_limit, self.config.lens_barrel_enabled),
                ("lens_pincushion", self.lens_pincushion_k, self.lens_pincushion_k_limit, self.config.lens_pincushion_enabled),
                ("mobius", self.mobius_params, float(self.config.mobius_limit), self.config.mobius_enabled),
                ("laplacian", self.laplacian_raw, self.laplacian_limit_px, self.config.laplacian_enabled),
                ("geodesic", self.geodesic_params, self.geodesic_limit_px, self.config.geodesic_enabled),
                (
                    "differential_surface",
                    self.differential_surface_height,
                    float(self.config.differential_surface_height_limit),
                    self.config.differential_surface_enabled,
                ),
            ]
            total_params = 0
            total_clamped = 0
            total_at_min = 0
            total_at_max = 0
            components = []
            self.polar_params.nan_to_num_(0.0)
            if self.config.polar_enabled:
                polar_limits = torch.tensor(
                    [self.polar_radial_limit_px, float(self.config.polar_twist_limit_rad)],
                    device=self.polar_params.device,
                    dtype=self.polar_params.dtype,
                )
                before = (self.polar_params < -polar_limits) | (self.polar_params > polar_limits)
                total_clamped += int(before.sum().item())
                self.polar_params.copy_(torch.maximum(torch.minimum(self.polar_params, polar_limits), -polar_limits))
                at_min = int((self.polar_params <= -polar_limits + 1e-8).sum().item())
                at_max = int((self.polar_params >= polar_limits - 1e-8).sum().item())
                total_at_min += at_min
                total_at_max += at_max
                total_params += self.polar_params.numel()
                if at_min or at_max:
                    components.append("polar")
            else:
                self.polar_params.zero_()
            for name, parameter, limit, enabled in blocks:
                parameter.nan_to_num_(0.0)
                if not enabled:
                    parameter.zero_()
                    continue
                before_low = parameter < -limit
                before_high = parameter > limit
                total_clamped += int((before_low | before_high).sum().item())
                parameter.clamp_(-limit, limit)
                at_min = int((parameter <= -limit + 1e-8).sum().item())
                at_max = int((parameter >= limit - 1e-8).sum().item())
                total_at_min += at_min
                total_at_max += at_max
                total_params += parameter.numel()
                if at_min or at_max:
                    components.append(name)
            if self.config.bspline_enabled:
                self.bspline_raw.mul_(self.bspline_mask)
            if self.config.laplacian_enabled:
                self.laplacian_raw.mul_(self.laplacian_mask)
            if self.config.differential_surface_enabled:
                self.differential_surface_height.mul_(self.differential_surface_mask)
            dct_projection = self.dct_image.project_()
            if self.config.dct_enabled:
                dct_params = int(self.dct_image.frequency_mask.sum().item())
                total_params += dct_params
                total_clamped += int(dct_projection.get("dct_num_clamped", 0))
                dct_diag = self.dct_image.parameter_diagnostics()
                total_at_min += int(dct_diag.get("dct_num_at_min", 0))
                total_at_max += int(dct_diag.get("dct_num_at_max", 0))
                if dct_diag.get("dct_num_at_min", 0) or dct_diag.get("dct_num_at_max", 0):
                    components.append("dct")
            if self.config.fft_phase_enabled:
                fft_stats = self.fft_phase.project_()
                phase = self.fft_phase.raw_phase
                total_params += phase.numel()
                total_clamped += int(fft_stats.get("fft_phase_num_clamped", 0))
                total_at_min += int(fft_stats.get("fft_phase_num_at_min", 0))
                total_at_max += int(fft_stats.get("fft_phase_num_at_max", 0))
                if fft_stats.get("fft_phase_num_at_min", 0) or fft_stats.get("fft_phase_num_at_max", 0):
                    components.append("fft_phase")
            else:
                self.fft_phase.raw_phase.zero_()
                fft_stats = {
                    "fft_phase_num_clamped": 0,
                    "fft_phase_num_at_min": 0,
                    "fft_phase_num_at_max": 0,
                }
            return {
                "num_total_params": int(total_params),
                "num_clamped_total": int(total_clamped),
                "fraction_clamped_total": float(total_clamped / max(total_params, 1)),
                "num_at_min_total": int(total_at_min),
                "num_at_max_total": int(total_at_max),
                "components_at_boundary": ",".join(sorted(set(components))),
                **dct_projection,
                **fft_stats,
            }

    def limits_dict(self) -> dict[str, Any]:
        return {
            "tps_enabled": bool(self.config.tps_enabled),
            "delaunay_enabled": bool(self.config.delaunay_enabled),
            "rolling_enabled": bool(self.config.rolling_enabled),
            "dct_enabled": bool(self.config.dct_enabled),
            "fft_phase_enabled": bool(self.config.fft_phase_enabled),
            "polar_enabled": bool(self.config.polar_enabled),
            "bspline_enabled": bool(self.config.bspline_enabled),
            "lens_barrel_enabled": bool(self.config.lens_barrel_enabled),
            "lens_pincushion_enabled": bool(self.config.lens_pincushion_enabled),
            "mobius_enabled": bool(self.config.mobius_enabled),
            "laplacian_enabled": bool(self.config.laplacian_enabled),
            "geodesic_enabled": bool(self.config.geodesic_enabled),
            "differential_surface_enabled": bool(self.config.differential_surface_enabled),
            "tps_limit_px": self.tps_limit_px,
            "delaunay_limit_px": self.delaunay_limit_px,
            "rolling_limit_px": self.rolling_limit_px,
            "polar_radial_limit_px": self.polar_radial_limit_px,
            "polar_twist_limit_rad": float(self.config.polar_twist_limit_rad),
            "bspline_limit_px": self.bspline_limit_px,
            "lens_k_limit": float(self.config.lens_k_limit),
            "lens_barrel_k_limit": self.lens_barrel_k_limit,
            "lens_pincushion_k_limit": self.lens_pincushion_k_limit,
            "mobius_limit": float(self.config.mobius_limit),
            "laplacian_limit_px": self.laplacian_limit_px,
            "geodesic_limit_px": self.geodesic_limit_px,
            "differential_surface_height_limit": float(self.config.differential_surface_height_limit),
            "differential_surface_px_scale": float(self.config.differential_surface_px_scale),
            "tps_norm_limit": self.config.tps_norm_limit,
            "delaunay_norm_limit": self.config.delaunay_norm_limit,
            "rolling_norm_limit": self.config.rolling_norm_limit,
            "polar_radial_norm_limit": self.config.polar_radial_norm_limit,
            "bspline_norm_limit": self.config.bspline_norm_limit,
            "laplacian_norm_limit": self.config.laplacian_norm_limit,
            "geodesic_norm_limit": self.config.geodesic_norm_limit,
            "dct_mode": "block_frequency_gain",
            "dct_block_size": self.config.dct_block_size,
            "dct_gain_limit": self.config.dct_gain_limit,
            "dct_frequency_mask": self.config.dct_frequency_mask,
            "dct_selected_frequency_count": self.dct_image.selected_frequency_count,
            "dct_exclude_dc": self.config.dct_exclude_dc,
            "fft_phase_limit_rad": float(self.config.fft_phase_limit_rad),
            "max_combined_disp_px": self.config.max_combined_disp_px,
            "spatial_padding_mode": str(self.config.spatial_padding_mode),
        }
