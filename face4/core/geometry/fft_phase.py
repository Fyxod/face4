"""Differentiable FFT phase perturbation."""
from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F


@dataclass
class FFTPhaseStats:
    fft_phase_norm: float
    fft_phase_mean_abs: float
    fft_phase_max_abs: float
    legacy_fft_strength_equivalent: float
    fft_spatial_delta_mse: float


class FFTPhasePerturbation(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        phase_size: int,
        init_scale: float,
        device: torch.device,
        seed: int,
        max_phase_rad: float = math.pi,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(seed + 5051)
        self.max_phase_rad = float(max_phase_rad)
        self.raw_phase = torch.nn.Parameter(
            torch.randn(1, channels, phase_size, phase_size, device=device, generator=generator) * init_scale
        )

    def phase(self, height: int, width: int) -> torch.Tensor:
        phase = self.raw_phase.clamp(-self.max_phase_rad, self.max_phase_rad)
        phase = phase - phase.mean(dim=(-2, -1), keepdim=True)
        phase = F.interpolate(phase, size=(height, width), mode="bicubic", align_corners=True)
        return phase.clamp(-self.max_phase_rad, self.max_phase_rad)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, FFTPhaseStats]:
        height, width = image.shape[-2:]
        phase = self.phase(height, width).to(dtype=torch.float32)
        spectrum = torch.fft.fftshift(torch.fft.fft2(image.float(), norm="ortho"), dim=(-2, -1))
        perturbed = spectrum * torch.exp(1j * phase)
        output = torch.fft.ifft2(torch.fft.ifftshift(perturbed, dim=(-2, -1)), norm="ortho").real
        output = output.clamp(0, 1).to(dtype=image.dtype)
        delta = output.float() - image.float()
        phase_abs = phase.detach().float().abs()
        stats = FFTPhaseStats(
            fft_phase_norm=float(phase.detach().float().square().mean().sqrt().cpu()),
            fft_phase_mean_abs=float(phase_abs.mean().cpu()),
            fft_phase_max_abs=float(phase_abs.max().cpu()),
            legacy_fft_strength_equivalent=float((phase_abs.mean() / math.pi * 1_000_000.0).cpu()),
            fft_spatial_delta_mse=float(delta.detach().float().square().mean().cpu()),
        )
        return output, delta, stats

    def project_(self) -> dict[str, int]:
        with torch.no_grad():
            self.raw_phase.nan_to_num_(0.0)
            before_low = self.raw_phase < -self.max_phase_rad
            before_high = self.raw_phase > self.max_phase_rad
            self.raw_phase.clamp_(-self.max_phase_rad, self.max_phase_rad)
            return {
                "fft_phase_num_clamped": int((before_low | before_high).sum().item()),
                "fft_phase_num_at_min": int((self.raw_phase <= -self.max_phase_rad + 1e-8).sum().item()),
                "fft_phase_num_at_max": int((self.raw_phase >= self.max_phase_rad - 1e-8).sum().item()),
            }
