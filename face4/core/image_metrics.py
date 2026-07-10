"""Image and tensor metrics used by WOOD."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def pil_to_tensor(image: Image.Image, device, dtype=None):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def tensor_to_pil(tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))


def make_blank_like(image: Image.Image, blank_value: float) -> Image.Image:
    value = int(max(0.0, min(1.0, float(blank_value))) * 255 + 0.5)
    return Image.new("RGB", image.size, (value, value, value))


def tensor_pair_metrics(left, right, prefix: str = "") -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    a = left.detach().float().clamp(0, 1)
    b = right.detach().float().clamp(0, 1)
    mse_t = (a - b).square().mean()
    mse = float(mse_t.cpu())
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    l2 = float(torch.sqrt(mse_t.clamp_min(1e-12)).cpu())

    # Windowed SSIM (11x11 Gaussian, sigma=1.5), rather than FACE3's single
    # global mean/variance approximation that was misleadingly named SSIM.
    coords = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    gaussian = torch.exp(-(coords.square()) / (2.0 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = torch.outer(gaussian, gaussian)
    kernel = kernel_2d.expand(a.shape[1], 1, 11, 11)
    padded_a = F.pad(a, (5, 5, 5, 5), mode="reflect")
    padded_b = F.pad(b, (5, 5, 5, 5), mode="reflect")
    mu_a = F.conv2d(padded_a, kernel, groups=a.shape[1])
    mu_b = F.conv2d(padded_b, kernel, groups=b.shape[1])
    var_a = F.conv2d(padded_a * padded_a, kernel, groups=a.shape[1]) - mu_a.square()
    var_b = F.conv2d(padded_b * padded_b, kernel, groups=b.shape[1]) - mu_b.square()
    cov = F.conv2d(padded_a * padded_b, kernel, groups=a.shape[1]) - mu_a * mu_b
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2)
    )
    ssim = float(ssim_map.mean().clamp(-1, 1).cpu())
    return {
        f"{prefix}mse": mse,
        f"{prefix}psnr": float(psnr),
        f"{prefix}l2": l2,
        f"{prefix}ssim": ssim,
    }


def image_metrics(left: Image.Image, right: Image.Image) -> dict[str, float]:
    width = min(left.width, right.width)
    height = min(left.height, right.height)
    a = np.asarray(left.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    b = np.asarray(right.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        ssim = float(max(0.0, 1.0 - math.sqrt(mse) * 4.0))
    return {
        "ssim": ssim,
        "psnr": float(psnr),
        "l2": float(math.sqrt(mse)),
        "mse": mse,
        "mean_abs": float(np.abs(a - b).mean()),
        "max_abs": float(np.abs(a - b).max()),
    }


def flow_to_pil(flow, scale_px: float | None = None) -> Image.Image:
    import torch

    value = flow.detach().float().cpu()[0]
    mag = torch.sqrt(value.square().sum(0) + 1e-12)
    max_mag = float(mag.max().item()) if scale_px is None else max(scale_px, 1e-6)
    max_mag = max(max_mag, 1e-6)
    array = np.stack(
        [
            np.clip(value[0].numpy() / max_mag * 0.5 + 0.5, 0, 1),
            np.clip(value[1].numpy() / max_mag * 0.5 + 0.5, 0, 1),
            np.clip(mag.numpy() / max_mag, 0, 1),
        ],
        axis=-1,
    )
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def delta_to_pil(delta) -> Image.Image:
    value = delta.detach().float().abs().mean(1, keepdim=False)[0].cpu().numpy()
    max_value = max(float(value.max()), 1e-8)
    array = np.clip(value / max_value, 0, 1)
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")


def save_sheet(path: Path, images: list[tuple[str, Image.Image]], cell_width: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        return
    width = cell_width or max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    label_h = 30
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(images):
        resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        x = idx * width
        canvas.paste(resized, (x, 0))
        draw.text((x + 4, height + 8), label, fill="black")
    canvas.save(path, quality=92)


def flatten_for_json(value: Any) -> Any:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu())
    if isinstance(value, dict):
        return {k: flatten_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [flatten_for_json(v) for v in value]
    return value
