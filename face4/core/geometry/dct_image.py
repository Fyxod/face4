"""Differentiable blockwise image-domain DCT frequency perturbation.

The active FACE DCT module operates on image DCT coefficients. It is not
a coordinate displacement field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def orthonormal_dct_matrix(block_size: int, device: torch.device) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix C[u, x]."""

    n = int(block_size)
    u = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(n, device=device, dtype=torch.float32)[None, :]
    alpha = torch.full((n, 1), math.sqrt(2.0 / n), device=device, dtype=torch.float32)
    alpha[0, 0] = math.sqrt(1.0 / n)
    return alpha * torch.cos(math.pi * (2.0 * x + 1.0) * u / (2.0 * n))


def frequency_mask(block_size: int, mode: str = "all_ac", exclude_dc: bool = True, device: torch.device | None = None) -> torch.Tensor:
    """Build an N x N frequency mask.

    Frequency bands are defined by u + v:

    - low: 1 <= u + v <= 3
    - mid: 4 <= u + v <= 7
    - high: u + v >= 8
    - all_ac: every coefficient except DC when exclude_dc is true
    """

    n = int(block_size)
    dev = device or torch.device("cpu")
    u = torch.arange(n, device=dev)[:, None]
    v = torch.arange(n, device=dev)[None, :]
    band = u + v
    mode = str(mode).lower()
    if mode == "all_ac":
        mask = torch.ones((n, n), device=dev, dtype=torch.float32)
    elif mode == "low":
        mask = ((band >= 1) & (band <= 3)).float()
    elif mode == "mid":
        mask = ((band >= 4) & (band <= 7)).float()
    elif mode == "high":
        mask = (band >= 8).float()
    else:
        raise ValueError(f"Unsupported DCT frequency mask mode: {mode}")
    if exclude_dc:
        mask[0, 0] = 0.0
    return mask


@dataclass
class DCTForwardOutput:
    image: torch.Tensor
    delta: torch.Tensor
    stats: dict[str, Any]


class DCTImagePerturbation(torch.nn.Module):
    """Shared block-frequency gain perturbation for RGB images."""

    def __init__(
        self,
        channels: int,
        block_size: int,
        gain_limit: float,
        frequency_mask_mode: str,
        exclude_dc: bool,
        enabled: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.block_size = int(block_size)
        self.gain_limit = float(gain_limit)
        self.frequency_mask_mode = str(frequency_mask_mode)
        self.exclude_dc = bool(exclude_dc)
        self.enabled = bool(enabled)
        self.register_buffer("dct_matrix", orthonormal_dct_matrix(self.block_size, device))
        mask_2d = frequency_mask(self.block_size, self.frequency_mask_mode, self.exclude_dc, device)
        self.register_buffer("frequency_mask_2d", mask_2d)
        self.register_buffer("frequency_mask", mask_2d[None].repeat(self.channels, 1, 1))
        self.dct_gain_raw = torch.nn.Parameter(torch.zeros(self.channels, self.block_size, self.block_size, device=device))
        self.dct_gain_raw.requires_grad_(self.enabled)
        self.project_()

    @property
    def selected_frequency_count(self) -> int:
        return int(self.frequency_mask.detach().sum().cpu())

    def _pad(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        n = self.block_size
        h, w = image.shape[-2:]
        pad_h = (n - h % n) % n
        pad_w = (n - w % n) % n
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return image, (pad_h, pad_w)

    def _blocks(self, image: torch.Tensor) -> torch.Tensor:
        n = self.block_size
        return image.unfold(2, n, n).unfold(3, n, n)

    def dct2(self, blocks: torch.Tensor) -> torch.Tensor:
        c = self.dct_matrix.to(device=blocks.device, dtype=blocks.dtype)
        return torch.einsum("ux,bcijxy,vy->bcijuv", c, blocks, c)

    def idct2(self, coeffs: torch.Tensor) -> torch.Tensor:
        c = self.dct_matrix.to(device=coeffs.device, dtype=coeffs.dtype)
        return torch.einsum("ux,bcijuv,vy->bcijxy", c, coeffs, c)

    def image_to_coefficients(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        padded, pads = self._pad(image.float())
        blocks = self._blocks(padded)
        coeffs = self.dct2(blocks)
        return coeffs, pads, padded.shape[-2:]

    def coefficients_to_image(self, coeffs: torch.Tensor, pads: tuple[int, int], padded_hw: tuple[int, int]) -> torch.Tensor:
        blocks = self.idct2(coeffs)
        b, c, bh, bw, n, _ = blocks.shape
        image = blocks.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, c, bh * n, bw * n)
        pad_h, pad_w = pads
        h, w = padded_hw
        image = image[..., : h - pad_h if pad_h else h, : w - pad_w if pad_w else w]
        return image

    def effective_gain(self) -> torch.Tensor:
        gain = self.dct_gain_raw.clamp(-self.gain_limit, self.gain_limit)
        return gain * self.frequency_mask.to(device=gain.device, dtype=gain.dtype)

    def _band_mask(self, mode: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return frequency_mask(self.block_size, mode, exclude_dc=(mode != "all_ac"), device=device).to(dtype=dtype)

    def _energy(self, coeffs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        value = coeffs.float().square()
        if mask is not None:
            value = value * mask[None, None, None, None]
            denom = mask.sum().clamp_min(1.0) * coeffs.shape[0] * coeffs.shape[1] * coeffs.shape[2] * coeffs.shape[3]
            return value.sum() / denom
        return value.mean()

    def _disabled_stats(self, image: torch.Tensor) -> dict[str, Any]:
        zero = 0.0
        return {
            "dct_enabled": 0,
            "dct_gain_min": zero,
            "dct_gain_max": zero,
            "dct_gain_mean": zero,
            "dct_gain_mean_abs": zero,
            "dct_gain_l2": zero,
            "dct_num_at_min": 0,
            "dct_num_at_max": 0,
            "dct_num_clamped": 0,
            "dct_selected_frequency_count": 0,
            "dct_frequency_mask_mode": self.frequency_mask_mode,
            "dct_input_coefficient_energy": zero,
            "dct_output_coefficient_energy": zero,
            "dct_coefficient_delta_l1": zero,
            "dct_coefficient_delta_l2": zero,
            "dct_relative_energy_change": zero,
            "dct_spatial_delta_mse": zero,
            "dct_spatial_delta_l1": zero,
            "dct_spatial_delta_l2": zero,
            "dct_spatial_delta_max_abs": zero,
            "dct_clipped_low_fraction": zero,
            "dct_clipped_high_fraction": zero,
            "dct_dc_energy": zero,
            "dct_low_frequency_energy_before": zero,
            "dct_low_frequency_energy_after": zero,
            "dct_mid_frequency_energy_before": zero,
            "dct_mid_frequency_energy_after": zero,
            "dct_high_frequency_energy_before": zero,
            "dct_high_frequency_energy_after": zero,
        }

    def forward(self, image: torch.Tensor) -> DCTForwardOutput:
        if not self.enabled:
            return DCTForwardOutput(image=image, delta=torch.zeros_like(image), stats=self._disabled_stats(image))

        coeffs, pads, padded_hw = self.image_to_coefficients(image)
        gain = self.effective_gain().to(device=coeffs.device, dtype=coeffs.dtype)
        modified = coeffs * (1.0 + gain[None, :, None, None])
        reconstructed_unclipped = self.coefficients_to_image(modified, pads, padded_hw).to(dtype=image.dtype)
        reconstructed = reconstructed_unclipped.clamp(0, 1)
        delta = reconstructed - image
        stats = self.stats(coeffs, modified, image, reconstructed, reconstructed_unclipped)
        return DCTForwardOutput(image=reconstructed, delta=delta, stats=stats)

    def stats(
        self,
        coeffs_before: torch.Tensor,
        coeffs_after: torch.Tensor,
        image_before: torch.Tensor,
        image_after: torch.Tensor,
        image_after_unclipped: torch.Tensor,
    ) -> dict[str, Any]:
        gain = self.effective_gain().detach().float()
        selected = self.frequency_mask.detach().bool()
        selected_gain = gain[selected] if selected.any() else gain.flatten()[:0]
        coeff_delta = (coeffs_after - coeffs_before).detach().float()
        spatial_delta = (image_after - image_before).detach().float()
        input_energy = self._energy(coeffs_before.detach())
        output_energy = self._energy(coeffs_after.detach())
        rel = (output_energy - input_energy) / input_energy.clamp_min(1e-12)
        device = coeffs_before.device
        dtype = coeffs_before.dtype
        dc_mask = torch.zeros((self.block_size, self.block_size), device=device, dtype=dtype)
        dc_mask[0, 0] = 1.0
        low_mask = self._band_mask("low", device, dtype)
        mid_mask = self._band_mask("mid", device, dtype)
        high_mask = self._band_mask("high", device, dtype)
        return {
            "dct_enabled": 1,
            "dct_gain_min": float(selected_gain.min().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_max": float(selected_gain.max().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_mean": float(selected_gain.mean().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_mean_abs": float(selected_gain.abs().mean().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_l2": float(selected_gain.square().sum().sqrt().cpu()) if selected_gain.numel() else 0.0,
            "dct_num_at_min": int((selected_gain <= -self.gain_limit + 1e-8).sum().cpu()) if selected_gain.numel() else 0,
            "dct_num_at_max": int((selected_gain >= self.gain_limit - 1e-8).sum().cpu()) if selected_gain.numel() else 0,
            "dct_selected_frequency_count": self.selected_frequency_count,
            "dct_frequency_mask_mode": self.frequency_mask_mode,
            "dct_input_coefficient_energy": float(input_energy.cpu()),
            "dct_output_coefficient_energy": float(output_energy.cpu()),
            "dct_coefficient_delta_l1": float(coeff_delta.abs().mean().cpu()),
            "dct_coefficient_delta_l2": float(coeff_delta.square().mean().sqrt().cpu()),
            "dct_relative_energy_change": float(rel.cpu()),
            "dct_spatial_delta_mse": float(spatial_delta.square().mean().cpu()),
            "dct_spatial_delta_l1": float(spatial_delta.abs().mean().cpu()),
            "dct_spatial_delta_l2": float(spatial_delta.square().mean().sqrt().cpu()),
            "dct_spatial_delta_max_abs": float(spatial_delta.abs().max().cpu()),
            "dct_clipped_low_fraction": float((image_after_unclipped.detach().float() < 0).float().mean().cpu()),
            "dct_clipped_high_fraction": float((image_after_unclipped.detach().float() > 1).float().mean().cpu()),
            "dct_dc_energy": float(self._energy(coeffs_before.detach(), dc_mask).cpu()),
            "dct_low_frequency_energy_before": float(self._energy(coeffs_before.detach(), low_mask).cpu()),
            "dct_low_frequency_energy_after": float(self._energy(coeffs_after.detach(), low_mask).cpu()),
            "dct_mid_frequency_energy_before": float(self._energy(coeffs_before.detach(), mid_mask).cpu()),
            "dct_mid_frequency_energy_after": float(self._energy(coeffs_after.detach(), mid_mask).cpu()),
            "dct_high_frequency_energy_before": float(self._energy(coeffs_before.detach(), high_mask).cpu()),
            "dct_high_frequency_energy_after": float(self._energy(coeffs_after.detach(), high_mask).cpu()),
        }

    def parameter_diagnostics(self) -> dict[str, Any]:
        gain = self.effective_gain().detach().float()
        selected = self.frequency_mask.detach().bool()
        selected_gain = gain[selected] if selected.any() else gain.flatten()[:0]
        return {
            "dct_enabled": int(self.enabled),
            "dct_gain_min": float(selected_gain.min().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_max": float(selected_gain.max().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_mean": float(selected_gain.mean().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_mean_abs": float(selected_gain.abs().mean().cpu()) if selected_gain.numel() else 0.0,
            "dct_gain_l2": float(selected_gain.square().sum().sqrt().cpu()) if selected_gain.numel() else 0.0,
            "dct_num_at_min": int((selected_gain <= -self.gain_limit + 1e-8).sum().cpu()) if selected_gain.numel() else 0,
            "dct_num_at_max": int((selected_gain >= self.gain_limit - 1e-8).sum().cpu()) if selected_gain.numel() else 0,
            "dct_selected_frequency_count": self.selected_frequency_count if self.enabled else 0,
            "dct_frequency_mask_mode": self.frequency_mask_mode,
        }

    def project_(self) -> dict[str, int]:
        with torch.no_grad():
            self.dct_gain_raw.nan_to_num_(0.0)
            if not self.enabled:
                self.dct_gain_raw.zero_()
                return {"dct_num_clamped": 0}
            if self.exclude_dc:
                self.dct_gain_raw[:, 0, 0] = 0.0
            before_low = self.dct_gain_raw < -self.gain_limit
            before_high = self.dct_gain_raw > self.gain_limit
            num_clamped = int((before_low | before_high).sum().item())
            self.dct_gain_raw.clamp_(-self.gain_limit, self.gain_limit)
            if self.exclude_dc:
                self.dct_gain_raw[:, 0, 0] = 0.0
            return {"dct_num_clamped": num_clamped}

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "block_frequency_gain",
            "block_size": self.block_size,
            "gain_limit": self.gain_limit,
            "frequency_mask_mode": self.frequency_mask_mode,
            "selected_frequency_count": self.selected_frequency_count,
            "exclude_dc": self.exclude_dc,
        }

    def spectrum_summary(self, image: torch.Tensor) -> torch.Tensor:
        coeffs, _, _ = self.image_to_coefficients(image)
        return torch.log1p(coeffs.detach().float().abs()).mean(dim=(0, 1, 2, 3))

    def gain_heatmap(self) -> torch.Tensor:
        return self.effective_gain().detach().float().abs().mean(dim=0)
