"""Downstream InstructPix2Pix edit evaluation for face4.

This module is evaluation-only. It is never used in the ArcFace optimization
loss or gradient loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class InstructEvalSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5


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


class InstructPix2PixEvaluator:
    name = "instructpix2pix_downstream_eval"

    def __init__(self, device: torch.device, settings: InstructEvalSettings | None = None) -> None:
        self.device = device
        self.settings = settings or InstructEvalSettings()
        self.pipe = self._load()

    def _load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=_dtype(self.settings.torch_dtype),
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
        return pipe

    @torch.inference_mode()
    def generate_edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=self.settings.num_inference_steps,
            guidance_scale=self.settings.guidance_scale,
            image_guidance_scale=self.settings.image_guidance_scale,
            generator=generator,
        )
        return result.images[0].convert("RGB")

    def metadata(self) -> dict[str, Any]:
        return {
            "downstream_editor": self.settings.model_id,
            "downstream_eval_only": True,
            "num_inference_steps": self.settings.num_inference_steps,
            "guidance_scale": self.settings.guidance_scale,
            "image_guidance_scale": self.settings.image_guidance_scale,
        }
