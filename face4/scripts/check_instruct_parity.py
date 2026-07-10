"""Fail-fast FACE4 grad/no-grad/stock InstructPix2Pix parity check."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from face4.core.cases import Case, resolve_image_path
from face4.core.identity import prepare_identity_reference
from face4.core.image_metrics import pil_to_tensor, save_sheet
from face4.core.logging import write_json
from face4.core.parity import ParityThresholds, run_checkpoint_gradient_gate, run_editor_parity_gate
from face4.models.arcface import ArcFaceIResNet100
from face4.models.differentiable_instruct import DifferentiableInstructPix2Pix, DifferentiableInstructSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the exact FACE4 gradient path against stock InstructPix2Pix.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--face-id", default="face_002")
    parser.add_argument("--prompt", default="add black sunglasses")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--edit-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--editor-dtype", default="float16")
    parser.add_argument("--output-root", default="outputs/correctness_check")
    parser.add_argument("--exact-min-ssim", type=float, default=0.999)
    parser.add_argument("--native-pil-min-ssim", type=float, default=0.990)
    parser.add_argument("--max-z-gap", type=float, default=0.001)
    parser.add_argument("--backward-scale", type=float, default=65536.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The full FACE4 editor parity check requires CUDA.")
    device = torch.device("cuda:0")
    case = Case(args.face_id, args.prompt)
    seed = case.seed if args.seed is None else int(args.seed)
    image_path = resolve_image_path(Path(args.mat_root), args.face_id)
    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image, device)

    editor = DifferentiableInstructPix2Pix(
        device,
        DifferentiableInstructSettings(
            torch_dtype=args.editor_dtype,
            num_inference_steps=args.edit_steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            enable_gradient_checkpointing=True,
            quantize_input_8bit=True,
            quantize_output_8bit=True,
        ),
    )
    arcface = ArcFaceIResNet100(args.arcface_checkpoint, device)
    with torch.no_grad():
        clean = editor.edit_tensor(tensor, args.prompt, seed).detach()
    reference = prepare_identity_reference(arcface, clean)
    report, images = run_editor_parity_gate(
        editor,
        tensor,
        args.prompt,
        seed,
        arcface=arcface,
        identity_reference=reference,
        thresholds=ParityThresholds(
            exact_min_ssim=args.exact_min_ssim,
            native_pil_min_ssim=args.native_pil_min_ssim,
            max_Z_gap=args.max_z_gap,
        ),
        backward_scale=args.backward_scale,
    )
    print("[face4-parity] comparing checkpointed and non-checkpointed input gradients")
    checkpoint_gradient = run_checkpoint_gradient_gate(
        editor, tensor, args.prompt, seed, backward_scale=args.backward_scale
    )
    report["checkpoint_gradient_parity"] = checkpoint_gradient
    report["passed"] = bool(report["passed"] and checkpoint_gradient.get("passed", False))
    output = Path(args.output_root) / f"{args.face_id}__{case.slug}__seed_{seed}__steps_{args.edit_steps}"
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "original.png")
    for name, value in images.items():
        value.save(output / f"{name}.png")
    save_sheet(
        output / "parity_sheet.png",
        [
            ("Original", image),
            ("Grad exact", images["grad_output"]),
            ("No-grad exact", images["no_grad_output"]),
            ("Stock tensor", images["stock_tensor_output"]),
            ("Stock PIL", images["stock_native_pil_output"]),
        ],
    )
    report["editor"] = editor.metadata()
    report["arcface"] = arcface.metadata()
    write_json(output / "parity_report.json", report)
    print(f"[face4-parity] passed={report['passed']} output={output}")
    print(f"[face4-parity] checks={report['checks']}")
    print(f"[face4-parity] Z gap={report['max_Z_gap']}")
    print(f"[face4-parity] checkpoint gradients={checkpoint_gradient}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
