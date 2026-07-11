"""Timing smoke for face4."""
from __future__ import annotations

import argparse

from face4.core.runner import RunConfig, run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACE timing smoke.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry_default.json")
    parser.add_argument("--output-root", default="outputs/smoke_timing")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--edit-steps", type=int, default=2, help="Differentiable InstructPix2Pix denoising steps for smoke timing.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--editor-dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--editor-model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--enable-editor-gradient-checkpointing", dest="enable_editor_gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--disable-editor-gradient-checkpointing", dest="enable_editor_gradient_checkpointing", action="store_false")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--all-cases", action="store_true")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init", choices=["neutral", "small_random"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backward-scale", type=float, default=65536.0)
    parser.add_argument("--backward-scale-min", type=float, default=1.0)
    parser.add_argument("--backward-scale-backoff", type=float, default=0.5)
    parser.add_argument("--backward-scale-max-retries", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        mat_root=args.mat_root,
        arcface_checkpoint=args.arcface_checkpoint,
        output_root=args.output_root,
        geometry_config_path=args.geometry_config,
        iters=args.iters,
        lr=args.lr,
        init=args.init,
        seed=args.seed,
        quick=args.quick,
        all_cases=args.all_cases,
        mode="smoke_timing",
        skip_deepface=True,
        editor_model_id=args.editor_model_id,
        editor_dtype=args.editor_dtype,
        edit_steps=args.edit_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        enable_editor_gradient_checkpointing=True if args.enable_editor_gradient_checkpointing is None else args.enable_editor_gradient_checkpointing,
        stock_validation_every=1,
        backward_scale=args.backward_scale,
        backward_scale_min=args.backward_scale_min,
        backward_scale_backoff=args.backward_scale_backoff,
        backward_scale_max_retries=args.backward_scale_max_retries,
    )
    summary = run_matrix(cfg)
    estimates = summary.get("time_estimates", {})
    print(f"[face4-timing] wrote: {summary['output_root']}")
    for key in ("estimated_runtime_seconds_for_50_iterations", "estimated_runtime_seconds_for_100_iterations", "estimated_runtime_seconds_for_150_iterations", "estimated_runtime_seconds_for_400_iterations"):
        value = estimates.get(key)
        if value is not None:
            print(f"[face4-timing] {key}: {value / 60:.2f} min")


if __name__ == "__main__":
    main()
