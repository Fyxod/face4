"""Exact autograd-enabled InstructPix2Pix execution for FACE4.

FACE3 maintained a second, hand-written denoising loop.  Its gradient-only
image-conditioning branch was not the same function as Diffusers' stock
pipeline.  FACE4 instead unwraps only the outer ``torch.no_grad`` decorator
from the installed Diffusers pipeline and executes that pipeline body itself.
All preprocessing, image-latent preparation, timesteps, guidance, scheduler
steps, and decoding therefore come from the installed Diffusers version.

The only backward approximation is the explicitly documented straight-
through estimator (STE) for an 8-bit image roundtrip.  Its forward value is
exactly the value written to PNG and subsequently supplied to stock replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import inspect
from typing import Any

import torch
from PIL import Image


@dataclass
class DifferentiableInstructSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    eta: float = 0.0
    enable_gradient_checkpointing: bool = True
    enable_vae_slicing: bool = True
    quantize_input_8bit: bool = True
    quantize_output_8bit: bool = True


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return aliases[name.lower()]


def quantize_8bit_ste(image: torch.Tensor) -> torch.Tensor:
    """Return an exact 8-bit forward value with an identity STE backward.

    The forward tensor is ``round(clamp(x, 0, 1) * 255) / 255``.  Gradients
    through the rounding operation use the identity straight-through
    estimator.  This prevents the optimizer from targeting fractional values
    that disappear when the perturbed input is saved as an 8-bit PNG.
    """

    clipped = image.clamp(0.0, 1.0)
    quantized = torch.round(clipped * 255.0) / 255.0
    return clipped + (quantized - clipped).detach()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class DifferentiableInstructPix2Pix:
    """Autograd path that executes the exact installed Diffusers call body."""

    def __init__(self, device: torch.device, settings: DifferentiableInstructSettings | None = None) -> None:
        self.device = device
        self.settings = settings or DifferentiableInstructSettings()
        self.dtype = _dtype(self.settings.torch_dtype)
        self.pipe = self._load_pipeline()
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))

        decorated_call = type(self.pipe).__call__
        self._exact_call = inspect.unwrap(decorated_call)
        if self._exact_call is decorated_call:
            raise RuntimeError(
                "Could not unwrap StableDiffusionInstructPix2PixPipeline.__call__. "
                "FACE4 will not fall back silently to a reconstructed editor."
            )

    def _load_pipeline(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if self.settings.enable_vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if self.settings.enable_gradient_checkpointing and hasattr(pipe.unet, "enable_gradient_checkpointing"):
            pipe.unet.enable_gradient_checkpointing()
        return pipe

    def metadata(self) -> dict[str, Any]:
        scheduler_config = dict(getattr(self.scheduler, "config", {}) or {})
        return {
            "editor": self.settings.model_id,
            "differentiable_editor": True,
            "implementation": "inspect.unwrap(installed Diffusers pipeline __call__)",
            "reconstructed_denoising_loop": False,
            "stock_pipeline_math_used_for_gradient": True,
            "settings": asdict(self.settings),
            "dtype": str(self.dtype).replace("torch.", ""),
            "vae_scale_factor": self.vae_scale_factor,
            "scheduler_class": type(self.scheduler).__name__,
            "scheduler_config": scheduler_config,
            "package_versions": {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "diffusers": _package_version("diffusers"),
                "transformers": _package_version("transformers"),
                "accelerate": _package_version("accelerate"),
            },
        }

    def canonical_input(self, image: torch.Tensor) -> torch.Tensor:
        if self.settings.quantize_input_8bit:
            return quantize_8bit_ste(image)
        return image.clamp(0.0, 1.0)

    def canonical_output(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float().clamp(0.0, 1.0)
        if self.settings.quantize_output_8bit:
            return quantize_8bit_ste(image)
        return image

    def _call_kwargs(self, image: torch.Tensor, prompt: str, seed: int) -> dict[str, Any]:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": int(self.settings.num_inference_steps),
            "guidance_scale": float(self.settings.guidance_scale),
            "image_guidance_scale": float(self.settings.image_guidance_scale),
            "eta": float(self.settings.eta),
            "generator": generator,
            "output_type": "pt",
            "return_dict": True,
        }

    def edit_tensor(self, image: torch.Tensor, prompt: str, seed: int) -> torch.Tensor:
        """Edit ``image`` with the exact pipeline body while preserving autograd."""

        canonical = self.canonical_input(image)
        result = self._exact_call(self.pipe, **self._call_kwargs(canonical, prompt, seed))
        edited = result.images
        if not torch.is_tensor(edited):
            raise TypeError(f"Diffusers output_type='pt' returned {type(edited)!r}, expected torch.Tensor")
        edited = self.canonical_output(edited)
        if torch.is_grad_enabled() and image.requires_grad and not edited.requires_grad:
            raise RuntimeError(
                "The installed Diffusers pipeline detached the edited tensor. "
                "FACE4 refuses to optimize a non-differentiable result."
            )
        return edited

    @torch.inference_mode()
    def stock_edit_tensor(self, image: torch.Tensor, prompt: str, seed: int) -> torch.Tensor:
        """Run the decorated stock pipeline on the same canonical tensor input."""

        canonical = self.canonical_input(image.detach())
        result = self.pipe(**self._call_kwargs(canonical, prompt, seed))
        if not torch.is_tensor(result.images):
            raise TypeError(f"Stock Diffusers output_type='pt' returned {type(result.images)!r}")
        return self.canonical_output(result.images)

    @torch.inference_mode()
    def stock_edit_pil(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Run the ordinary PIL-input/PIL-output Diffusers pipeline call."""

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=int(self.settings.num_inference_steps),
            guidance_scale=float(self.settings.guidance_scale),
            image_guidance_scale=float(self.settings.image_guidance_scale),
            eta=float(self.settings.eta),
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return result.images[0].convert("RGB")
