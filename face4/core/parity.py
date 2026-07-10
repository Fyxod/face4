"""Correctness gates for the FACE4 edited-output objective."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .identity import identity_objective
from .image_metrics import image_metrics, pil_to_tensor, tensor_to_pil


class CorrectnessGateError(RuntimeError):
    """A systemic forward/gradient parity failure that must stop the matrix."""


@dataclass
class ParityThresholds:
    exact_max_abs: float = 1.0 / 255.0 + 1e-6
    exact_min_ssim: float = 0.999
    native_pil_min_ssim: float = 0.990
    max_Z_gap: float = 0.001


def _pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    metrics = image_metrics(tensor_to_pil(left), tensor_to_pil(right))
    metrics["tensor_max_abs"] = float((left.detach().float() - right.detach().float()).abs().max().cpu())
    metrics["tensor_mean_abs"] = float((left.detach().float() - right.detach().float()).abs().mean().cpu())
    return metrics


def run_editor_parity_gate(
    editor,
    image: torch.Tensor,
    prompt: str,
    seed: int,
    *,
    arcface=None,
    identity_reference=None,
    thresholds: ParityThresholds | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare grad, no-grad, stock-tensor, and normal PIL pipeline forwards.

    The gate exercises the *same branch used by optimization*: its input has
    ``requires_grad=True`` and backward is executed.  FACE3's old test wrapped
    the differentiable call in ``torch.no_grad`` and therefore missed the
    broken gradient-only image-latent branch.
    """

    limits = thresholds or ParityThresholds()
    canonical = editor.canonical_input(image.detach())
    probe = canonical.detach().clone().requires_grad_(True)
    grad_output = editor.edit_tensor(probe, prompt, seed)
    probe_loss = grad_output.float().square().mean()
    input_grad = torch.autograd.grad(probe_loss, probe, retain_graph=False, create_graph=False)[0]
    finite_grad = bool(torch.isfinite(input_grad).all().item())
    grad_norm = float(input_grad.float().norm().detach().cpu())

    with torch.no_grad():
        no_grad_output = editor.edit_tensor(canonical, prompt, seed).detach()
    stock_tensor_output = editor.stock_edit_tensor(canonical, prompt, seed).detach()
    canonical_pil = tensor_to_pil(canonical)
    stock_pil_output = editor.stock_edit_pil(canonical_pil, prompt, seed)
    stock_pil_tensor = pil_to_tensor(stock_pil_output, canonical.device)

    grad_detached = grad_output.detach()
    pairs = {
        "grad_vs_no_grad": _pair(grad_detached, no_grad_output),
        "grad_vs_stock_tensor": _pair(grad_detached, stock_tensor_output),
        "stock_tensor_vs_native_pil": _pair(stock_tensor_output, stock_pil_tensor),
    }

    z_values: dict[str, float | None] = {
        "Z_grad": None,
        "Z_no_grad": None,
        "Z_stock_tensor": None,
        "Z_stock_native_pil": None,
    }
    if arcface is not None and identity_reference is not None:
        with torch.no_grad():
            for name, value in (
                ("Z_grad", grad_detached),
                ("Z_no_grad", no_grad_output),
                ("Z_stock_tensor", stock_tensor_output),
                ("Z_stock_native_pil", stock_pil_tensor),
            ):
                z, _ = identity_objective(arcface, value, identity_reference)
                z_values[name] = float(z.detach().float().cpu())

    exact_pairs_pass = all(
        pairs[name]["tensor_max_abs"] <= limits.exact_max_abs and pairs[name]["ssim"] >= limits.exact_min_ssim
        for name in ("grad_vs_no_grad", "grad_vs_stock_tensor")
    )
    native_pil_pass = pairs["stock_tensor_vs_native_pil"]["ssim"] >= limits.native_pil_min_ssim
    z_present = [value for value in z_values.values() if value is not None]
    z_gap = max(z_present) - min(z_present) if z_present else 0.0
    z_pass = z_gap <= limits.max_Z_gap
    passed = bool(finite_grad and grad_norm > 0.0 and exact_pairs_pass and native_pil_pass and z_pass)

    report: dict[str, Any] = {
        "passed": passed,
        "prompt": prompt,
        "seed": int(seed),
        "thresholds": asdict(limits),
        "input_gradient_finite": finite_grad,
        "input_gradient_norm": grad_norm,
        "pairs": pairs,
        **z_values,
        "max_Z_gap": z_gap,
        "checks": {
            "finite_nonzero_input_gradient": finite_grad and grad_norm > 0.0,
            "grad_no_grad_exact_forward_parity": (
                pairs["grad_vs_no_grad"]["tensor_max_abs"] <= limits.exact_max_abs
                and pairs["grad_vs_no_grad"]["ssim"] >= limits.exact_min_ssim
            ),
            "grad_stock_tensor_exact_forward_parity": (
                pairs["grad_vs_stock_tensor"]["tensor_max_abs"] <= limits.exact_max_abs
                and pairs["grad_vs_stock_tensor"]["ssim"] >= limits.exact_min_ssim
            ),
            "stock_tensor_native_pil_parity": native_pil_pass,
            "Z_parity": z_pass,
        },
    }
    images = {
        "canonical_input": canonical_pil,
        "grad_output": tensor_to_pil(grad_detached),
        "no_grad_output": tensor_to_pil(no_grad_output),
        "stock_tensor_output": tensor_to_pil(stock_tensor_output),
        "stock_native_pil_output": stock_pil_output,
    }
    return report, images


def run_checkpoint_gradient_gate(
    editor,
    image: torch.Tensor,
    prompt: str,
    seed: int,
    *,
    min_gradient_cosine: float = 0.995,
    max_relative_l2: float = 0.10,
    min_output_ssim: float = 0.999,
) -> dict[str, Any]:
    """Compare input gradients with Diffusers checkpointing on and off.

    This catches bugs that preserve the forward image but change backward
    recomputation, including FACE3's late-bound denoising timestep closure.
    It is intended for the short A6000 correctness smoke, not every long-run
    iteration.
    """

    if not hasattr(editor.unet, "enable_gradient_checkpointing") or not hasattr(
        editor.unet, "disable_gradient_checkpointing"
    ):
        return {"passed": False, "error": "UNet does not expose checkpoint enable/disable methods"}

    canonical = editor.canonical_input(image.detach())
    height, width = canonical.shape[-2:]
    yy = torch.linspace(-1.0, 1.0, height, device=canonical.device, dtype=torch.float32).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=canonical.device, dtype=torch.float32).view(1, 1, 1, width)
    pattern = (0.37 * xx + 0.63 * yy).expand(1, canonical.shape[1], height, width)

    def one(enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled:
            editor.unet.enable_gradient_checkpointing()
        else:
            editor.unet.disable_gradient_checkpointing()
        probe = canonical.detach().clone().requires_grad_(True)
        output = editor.edit_tensor(probe, prompt, seed)
        scalar = (output.float() * pattern).mean()
        gradient = torch.autograd.grad(scalar, probe, retain_graph=False, create_graph=False)[0]
        return output.detach(), gradient.detach()

    try:
        checkpoint_output, checkpoint_grad = one(True)
        plain_output, plain_grad = one(False)
    finally:
        if editor.settings.enable_gradient_checkpointing:
            editor.unet.enable_gradient_checkpointing()

    output_metrics = _pair(checkpoint_output, plain_output)
    checkpoint_flat = checkpoint_grad.float().flatten()
    plain_flat = plain_grad.float().flatten()
    gradient_cosine = float(
        torch.nn.functional.cosine_similarity(checkpoint_flat.unsqueeze(0), plain_flat.unsqueeze(0), dim=1).cpu()
    )
    diff_l2 = float((checkpoint_flat - plain_flat).norm().cpu())
    reference_l2 = float(plain_flat.norm().clamp_min(1e-12).cpu())
    relative_l2 = diff_l2 / reference_l2
    finite = bool(torch.isfinite(checkpoint_grad).all().item() and torch.isfinite(plain_grad).all().item())
    passed = bool(
        finite
        and output_metrics["ssim"] >= min_output_ssim
        and gradient_cosine >= min_gradient_cosine
        and relative_l2 <= max_relative_l2
    )
    return {
        "passed": passed,
        "finite_gradients": finite,
        "checkpoint_gradient_norm": float(checkpoint_flat.norm().cpu()),
        "plain_gradient_norm": reference_l2,
        "gradient_cosine_similarity": gradient_cosine,
        "gradient_relative_l2_error": relative_l2,
        "output_metrics": output_metrics,
        "thresholds": {
            "min_gradient_cosine": min_gradient_cosine,
            "max_relative_l2": max_relative_l2,
            "min_output_ssim": min_output_ssim,
        },
    }
