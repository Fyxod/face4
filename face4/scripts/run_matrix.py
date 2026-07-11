"""Run the full FACE edited-output ArcFace identity matrix."""
from __future__ import annotations

import argparse

from face4.core.runner import RunConfig, run_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACE edited-output ArcFace identity optimization matrix.")
    parser.add_argument("--mat-root", required=True)
    parser.add_argument("--arcface-checkpoint", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry_default.json")
    parser.add_argument("--output-root", default="outputs/edited_output_identity_exact")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--edit-steps", type=int, default=20, help="Exact InstructPix2Pix denoising steps inside each optimization iteration.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--editor-dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--editor-model-id", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--enable-editor-gradient-checkpointing", dest="enable_editor_gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--disable-editor-gradient-checkpointing", dest="enable_editor_gradient_checkpointing", action="store_false")
    parser.add_argument("--init", choices=["neutral", "small_random"], default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional override; default preserves each case's deterministic seed.")
    parser.add_argument("--stock-validation-every", type=int, default=25)
    parser.add_argument("--parity-max-z-gap", type=float, default=0.001)
    parser.add_argument("--backward-scale", type=float, default=65536.0)
    parser.add_argument("--backward-scale-min", type=float, default=1.0)
    parser.add_argument("--backward-scale-backoff", type=float, default=0.5)
    parser.add_argument("--backward-scale-max-retries", type=int, default=20)
    parser.add_argument("--resume-run-root", default=None, help="Resume a specific timestamped run root; skips DONE cases and archives incomplete cases before rerun.")
    parser.add_argument("--resume-latest", action="store_true", help="Resume the latest timestamped run under --output-root.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-deepface", action="store_true")
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
        quick=False,
        all_cases=True,
        mode="run_matrix",
        skip_deepface=args.skip_deepface,
        force=args.force,
        editor_model_id=args.editor_model_id,
        editor_dtype=args.editor_dtype,
        edit_steps=args.edit_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        enable_editor_gradient_checkpointing=True if args.enable_editor_gradient_checkpointing is None else args.enable_editor_gradient_checkpointing,
        stock_validation_every=args.stock_validation_every,
        parity_max_Z_gap=args.parity_max_z_gap,
        backward_scale=args.backward_scale,
        backward_scale_min=args.backward_scale_min,
        backward_scale_backoff=args.backward_scale_backoff,
        backward_scale_max_retries=args.backward_scale_max_retries,
        resume_run_root=args.resume_run_root,
        resume_latest=args.resume_latest,
    )
    summary = run_matrix(cfg)
    print(f"[face4-run] wrote: {summary['output_root']}")


if __name__ == "__main__":
    main()
