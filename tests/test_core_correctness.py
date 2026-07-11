from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from face4.core.identity import prepare_identity_reference
from face4.core.geometry.combined_face import CombinedFacePerturbation, load_face_geometry_config
from face4.core.image_metrics import pil_to_tensor, tensor_to_pil
from face4.core.parity import ParityThresholds, run_checkpoint_gradient_gate, run_editor_parity_gate
from face4.core.runner_correct import _next_backward_scale
from face4.models.differentiable_instruct import (
    DifferentiableInstructPix2Pix,
    DifferentiableInstructSettings,
    quantize_8bit_ste,
)
from face4.models.iresnet import iresnet100


class _FakeEditor:
    def __init__(self):
        class Toggle:
            def enable_gradient_checkpointing(inner_self):
                inner_self.enabled = True

            def disable_gradient_checkpointing(inner_self):
                inner_self.enabled = False

        self.unet = Toggle()
        self.settings = SimpleNamespace(enable_gradient_checkpointing=True)

    def canonical_input(self, image):
        return quantize_8bit_ste(image)

    def edit_tensor(self, image, prompt, seed):
        del prompt, seed
        return quantize_8bit_ste(image * 0.75 + 0.1)

    @torch.inference_mode()
    def stock_edit_tensor(self, image, prompt, seed):
        return self.edit_tensor(image, prompt, seed)

    @torch.inference_mode()
    def stock_edit_pil(self, image, prompt, seed):
        tensor = pil_to_tensor(image, torch.device("cpu"))
        return tensor_to_pil(self.edit_tensor(tensor, prompt, seed))


class _FakeArcFace(nn.Module):
    def embedding(self, image):
        pooled = F.adaptive_avg_pool2d(image.float(), (2, 2)).flatten(1)
        return F.normalize(pooled + 1e-3, p=2, dim=1)


class _FakeNativePILDivergentEditor(_FakeEditor):
    """Exact tensor paths agree while native PIL replay differs on purpose."""

    @torch.inference_mode()
    def stock_edit_pil(self, image, prompt, seed):
        tensor = pil_to_tensor(image, torch.device("cpu"))
        edited = self.edit_tensor(tensor, prompt, seed)
        return tensor_to_pil(torch.flip(edited, dims=(1,)))


class _FakePipeline:
    def __init__(self):
        self.vae = SimpleNamespace(config=SimpleNamespace(), enable_slicing=lambda: None)
        self.unet = _FakeEditor().unet
        self.text_encoder = SimpleNamespace()
        self.tokenizer = SimpleNamespace()
        self.scheduler = SimpleNamespace(config={"name": "fake"})
        self.vae_scale_factor = 8

    @torch.no_grad()
    def __call__(self, *, image, output_type, **kwargs):
        del kwargs
        value = image * 0.75 + 0.1
        if output_type == "pt":
            return SimpleNamespace(images=value)
        if output_type == "pil":
            return SimpleNamespace(images=[tensor_to_pil(value)])
        raise ValueError(output_type)


class _FakeExactEditor(DifferentiableInstructPix2Pix):
    def _load_pipeline(self):
        return _FakePipeline()


class CoreCorrectnessTests(unittest.TestCase):
    def test_backward_scale_backoff_reaches_and_stays_at_minimum(self):
        scale = 65536.0
        observed = []
        while scale > 1.0:
            scale = _next_backward_scale(scale, 1.0, 0.5)
            observed.append(scale)
        self.assertEqual(len(observed), 16)
        self.assertEqual(observed[-1], 1.0)
        self.assertEqual(_next_backward_scale(1.0, 1.0, 0.5), 1.0)

    def test_ste_forward_is_exact_8bit_and_backward_is_identity(self):
        value = torch.tensor([0.12345, 0.501, -0.2, 1.2], requires_grad=True)
        quantized = quantize_8bit_ste(value)
        expected = torch.round(value.detach().clamp(0, 1) * 255.0) / 255.0
        self.assertTrue(torch.equal(quantized.detach(), expected))
        quantized.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.tensor([1.0, 1.0, 0.0, 0.0])))

    def test_iresnet_downsample_matches_insightface_conv_then_bn(self):
        model = iresnet100()
        block = model.layer1[0].downsample
        self.assertIsInstance(block[0], nn.Conv2d)
        self.assertIsInstance(block[1], nn.BatchNorm2d)

    def test_unwrapped_stock_call_preserves_autograd(self):
        editor = _FakeExactEditor(
            torch.device("cpu"),
            DifferentiableInstructSettings(
                torch_dtype="float32",
                num_inference_steps=2,
                enable_gradient_checkpointing=False,
            ),
        )
        image = torch.rand(1, 3, 16, 16, requires_grad=True)
        edited = editor.edit_tensor(image, "prompt", 9)
        self.assertTrue(edited.requires_grad)
        edited.mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertGreater(float(image.grad.norm()), 0.0)
        stock = editor.stock_edit_tensor(image.detach(), "prompt", 9)
        self.assertTrue(torch.equal(edited.detach(), stock.detach()))

    def test_parity_gate_exercises_grad_and_stock_paths(self):
        rng = np.random.default_rng(7)
        image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        tensor = pil_to_tensor(image, torch.device("cpu"))
        editor = _FakeEditor()
        arcface = _FakeArcFace()
        with torch.no_grad():
            clean = editor.edit_tensor(tensor, "prompt", 123)
        reference = prepare_identity_reference(arcface, clean)
        report, _ = run_editor_parity_gate(
            editor,
            tensor,
            "prompt",
            123,
            arcface=arcface,
            identity_reference=reference,
            thresholds=ParityThresholds(exact_min_ssim=0.9999, native_pil_min_ssim=0.9999),
        )
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["input_gradient_norm"], 0.0)
        self.assertTrue(all(report["checks"].values()), report)
        checkpoint_report = run_checkpoint_gradient_gate(editor, tensor, "prompt", 123)
        self.assertTrue(checkpoint_report["passed"], checkpoint_report)

    def test_parity_gate_separates_exact_tensor_and_native_pil_Z_tolerances(self):
        rng = np.random.default_rng(17)
        image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        tensor = pil_to_tensor(image, torch.device("cpu"))
        editor = _FakeNativePILDivergentEditor()
        arcface = _FakeArcFace()
        with torch.no_grad():
            clean = editor.edit_tensor(tensor, "prompt", 123)
        reference = prepare_identity_reference(arcface, clean)

        diagnostic, _ = run_editor_parity_gate(
            editor,
            tensor,
            "prompt",
            123,
            arcface=arcface,
            identity_reference=reference,
            thresholds=ParityThresholds(
                exact_min_ssim=0.9999,
                native_pil_min_ssim=-1.0,
                exact_max_Z_gap=1e-8,
                native_pil_max_Z_gap=2.0,
            ),
        )
        self.assertTrue(diagnostic["passed"], diagnostic)
        self.assertEqual(diagnostic["exact_tensor_max_Z_gap"], 0.0)
        self.assertGreater(diagnostic["native_pil_Z_gap"], 0.0)
        self.assertTrue(diagnostic["checks"]["exact_tensor_Z_parity"])

        rejected, _ = run_editor_parity_gate(
            editor,
            tensor,
            "prompt",
            123,
            arcface=arcface,
            identity_reference=reference,
            thresholds=ParityThresholds(
                exact_min_ssim=0.9999,
                native_pil_min_ssim=-1.0,
                exact_max_Z_gap=1e-8,
                native_pil_max_Z_gap=0.0,
            ),
        )
        self.assertFalse(rejected["passed"], rejected)
        self.assertTrue(rejected["checks"]["exact_tensor_Z_parity"])
        self.assertFalse(rejected["checks"]["native_pil_Z_parity"])

        compatibility = ParityThresholds(native_pil_max_Z_gap=2.0, max_Z_gap=0.25)
        self.assertEqual(compatibility.resolved_native_pil_max_Z_gap(), 0.25)

    def test_extended_geometry_components_have_finite_forward_and_backward(self):
        torch.manual_seed(29)
        config = load_face_geometry_config("configs/geometry_extended_all.json")
        config.init = "small_random"
        geometry = CombinedFacePerturbation(32, 32, 3, torch.device("cpu"), seed=31, config=config)
        image = torch.rand(1, 3, 32, 32)
        output, aux = geometry(image)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(aux["displacement"]).all())
        output.square().mean().backward()
        enabled_gradients = [
            parameter.grad
            for parameter in geometry.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(enabled_gradients)
        self.assertTrue(all(gradient is not None for gradient in enabled_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in enabled_gradients))

    def test_tps_parameter_gradient_matches_finite_difference(self):
        # Keep the sampled image fixed: for some random images this particular
        # directional derivative is close enough to zero that float32 finite
        # differences become dominated by cancellation noise.
        torch.manual_seed(2)
        config = load_face_geometry_config("configs/geometry_default.json")
        for name in (
            "delaunay_enabled",
            "rolling_enabled",
            "dct_enabled",
            "fft_phase_enabled",
            "polar_enabled",
            "bspline_enabled",
            "lens_barrel_enabled",
            "lens_pincushion_enabled",
            "mobius_enabled",
            "laplacian_enabled",
            "geodesic_enabled",
            "differential_surface_enabled",
        ):
            setattr(config, name, False)
        config.tps_enabled = True
        config.init = "small_random"
        geometry = CombinedFacePerturbation(32, 32, 3, torch.device("cpu"), seed=11, config=config)
        image = torch.rand(1, 3, 32, 32)
        weight = torch.linspace(-1, 1, 32).view(1, 1, 1, 32)

        output, _ = geometry(image)
        scalar = (output * weight).mean()
        scalar.backward()
        index = (0, 0, 2, 2)
        autograd_value = float(geometry.tps_raw.grad[index])
        base = float(geometry.tps_raw.detach()[index])
        eps = 1e-3
        with torch.no_grad():
            geometry.tps_raw[index] = base + eps
            plus = float((geometry(image)[0] * weight).mean())
            geometry.tps_raw[index] = base - eps
            minus = float((geometry(image)[0] * weight).mean())
            geometry.tps_raw[index] = base
        finite_difference = (plus - minus) / (2 * eps)
        self.assertAlmostEqual(autograd_value, finite_difference, delta=max(1e-5, abs(finite_difference) * 0.08))


if __name__ == "__main__":
    unittest.main()
