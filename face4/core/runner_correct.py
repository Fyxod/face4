"""Correctness-gated FACE4 optimization runner.

This module deliberately lives beside the preserved FACE3 runner so the old
implementation remains inspectable while FACE4 dispatches only this path.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image

from .cases import resolve_image_path
from .identity import face_loss, identity_objective, prepare_identity_reference
from .image_metrics import (
    flow_to_pil,
    image_metrics,
    pil_to_tensor,
    save_sheet,
    tensor_pair_metrics,
    tensor_to_pil,
)
from .logging import append_jsonl, nvidia_smi_memory_gb, write_csv, write_json
from .parity import CorrectnessGateError, ParityThresholds, run_editor_parity_gate
from .runtime import torch_peak_gb
from .utils import save_input_difference
from .runner import (
    _component_flow_images,
    _edit_terms_aliases,
    _float_terms,
    _history_fields_ok,
    _identity_pair_metrics,
    _prefixed_identity_terms,
    _save_checkpoint,
)


def _clone_aux(aux: dict[str, Any]) -> dict[str, Any]:
    return {
        "spatial": aux["spatial"].detach().clone(),
        "dct_image": aux["dct_image"].detach().clone(),
        "dct_delta": aux["dct_delta"].detach().clone(),
        "displacement": aux["displacement"].detach().clone(),
        "fields": {key: value.detach().clone() for key, value in aux["fields"].items()},
        "fft_delta": aux["fft_delta"].detach().clone(),
        "diagnostics": dict(aux["diagnostics"]),
    }


def _capture_candidate(row, geometry, perturbed, perturbed_edit, aux) -> dict[str, Any]:
    return {
        "row": dict(row),
        "theta_state": geometry.theta_state(),
        "perturbed": perturbed.detach().clone(),
        "perturbed_edit": perturbed_edit.detach().clone(),
        "aux": _clone_aux(aux),
    }


def _write_preflight(output_dir: Path, report: dict[str, Any], images: dict[str, Image.Image]) -> None:
    root = output_dir / "correctness_preflight"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "parity_report.json", report)
    for name, image in images.items():
        image.save(root / f"{name}.png")
    save_sheet(
        root / "parity_sheet.png",
        [
            ("Canonical input", images["canonical_input"]),
            ("Grad exact", images["grad_output"]),
            ("No-grad exact", images["no_grad_output"]),
            ("Stock tensor", images["stock_tensor_output"]),
            ("Stock native PIL", images["stock_native_pil_output"]),
        ],
    )


def _stock_Z(editor, arcface, perturbed, prompt: str, seed: int, stock_reference):
    import torch

    with torch.no_grad():
        edited = editor.stock_edit_tensor(perturbed, prompt, seed)
        value, terms = identity_objective(arcface, edited, stock_reference)
    return float(value.detach().float().cpu()), _float_terms(terms), edited.detach()


def _validate_backward_scale_config(cfg) -> None:
    if not math.isfinite(cfg.backward_scale) or cfg.backward_scale <= 0.0:
        raise ValueError(f"backward_scale must be finite and positive, got {cfg.backward_scale!r}")
    if not math.isfinite(cfg.backward_scale_min) or cfg.backward_scale_min <= 0.0:
        raise ValueError(f"backward_scale_min must be finite and positive, got {cfg.backward_scale_min!r}")
    if cfg.backward_scale_min > cfg.backward_scale:
        raise ValueError(
            f"backward_scale_min ({cfg.backward_scale_min}) cannot exceed "
            f"backward_scale ({cfg.backward_scale})"
        )
    if not math.isfinite(cfg.backward_scale_backoff) or not 0.0 < cfg.backward_scale_backoff < 1.0:
        raise ValueError(
            "backward_scale_backoff must be finite and strictly between 0 and 1, "
            f"got {cfg.backward_scale_backoff!r}"
        )
    if cfg.backward_scale_max_retries < 0:
        raise ValueError(
            f"backward_scale_max_retries must be nonnegative, got {cfg.backward_scale_max_retries!r}"
        )


def _exact_Z_tolerance(cfg) -> float:
    """Tolerance for mathematically identical tensor-path forwards."""

    return float(cfg.parity_exact_max_Z_gap)


def _native_pil_Z_tolerance(cfg) -> float:
    """Tolerance for public PIL save/reload/pipeline replay."""

    legacy = getattr(cfg, "parity_max_Z_gap", None)
    return float(cfg.parity_native_pil_max_Z_gap if legacy is None else legacy)


def _next_backward_scale(current: float, minimum: float, backoff: float) -> float:
    """Return a strictly smaller scale, bounded below by ``minimum``."""

    current = float(current)
    minimum = float(minimum)
    candidate = max(minimum, current * float(backoff))
    if candidate >= current:
        return current
    return candidate


def _make_row(
    *,
    iteration: int,
    updates_completed: int,
    phase: str,
    Z,
    terms: dict[str, Any],
    spec,
    run_seed: int,
    cfg,
    started: float,
    seconds_iter: float,
    perturbed,
    original_tensor,
    input_identity_terms: dict[str, Any],
    aux: dict[str, Any],
    grad_norms: dict[str, float],
    geometry,
    projection: dict[str, Any],
    best_Z: float,
    best_iter: int,
    stock_Z: float | None,
) -> dict[str, Any]:
    metrics = tensor_pair_metrics(perturbed, original_tensor, prefix="")
    current_Z = float(Z.detach().float().cpu())
    row: dict[str, Any] = {
        "iter": int(iteration),
        "optimizer_updates_completed": int(updates_completed),
        "phase": phase,
        "Z": current_Z,
        "loss": current_Z,
        "Z_stock_validation": stock_Z,
        "Z_stock_gap": None if stock_Z is None else abs(current_Z - float(stock_Z)),
        "best_Z_so_far": float(best_Z),
        "best_iter_so_far": int(best_iter),
        "learning_rate": cfg.lr,
        "seed": run_seed,
        "face_id": spec.case.face_id,
        "prompt": spec.case.prompt,
        "case_id": spec.case.slug,
        "seconds_iter": float(seconds_iter),
        "seconds_elapsed": time.monotonic() - started,
        "peak_vram_gb": torch_peak_gb(),
        "psnr_to_original": metrics["psnr"],
        "ssim_to_original": metrics["ssim"],
        "mse_to_original": metrics["mse"],
        "l2_to_original": metrics["l2"],
        "editor_num_inference_steps": cfg.edit_steps,
        "editor_forward": "exact_installed_diffusers_pipeline_body",
        **_prefixed_identity_terms("input_identity", input_identity_terms),
        **_float_terms(terms),
        **_float_terms(_edit_terms_aliases(terms)),
        **aux["diagnostics"],
        **grad_norms,
        **geometry.parameter_diagnostics(),
        **projection,
    }
    row["total_geometry_grad_norm"] = row.get("total_grad_norm", 0.0)
    return row


def optimize_one_correct(spec, cfg, arcface, editor, device, output_dir: Path) -> dict[str, Any]:
    import torch

    from .geometry.combined_face import CombinedFacePerturbation, FaceGeometryConfig, load_face_geometry_config

    _validate_backward_scale_config(cfg)

    done_path = output_dir / "DONE.json"
    if done_path.exists() and not cfg.force:
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            print(f"[face4] skip completed run: {output_dir}")
            from .logging import read_json

            return read_json(summary_path)
        raise RuntimeError(f"DONE.json exists but summary.json is missing: {output_dir}")
    if cfg.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    run_seed = int(spec.run_seed if cfg.seed is None else cfg.seed)
    image_path = resolve_image_path(Path(cfg.mat_root), spec.case.face_id)
    print(f"[face4] running {spec.slug} image={image_path} seed={run_seed}")
    original = Image.open(image_path).convert("RGB")
    original.save(output_dir / "original.png")
    original_tensor = pil_to_tensor(original, device)
    canonical_original = editor.canonical_input(original_tensor)
    input_identity_reference = prepare_identity_reference(arcface, canonical_original)

    geometry_config = load_face_geometry_config(cfg.geometry_config_path) if cfg.geometry_config_path else FaceGeometryConfig()
    if cfg.init:
        geometry_config.init = cfg.init
    torch.manual_seed(run_seed)
    geometry = CombinedFacePerturbation(
        original_tensor.shape[-2],
        original_tensor.shape[-1],
        original_tensor.shape[1],
        device,
        seed=run_seed,
        config=geometry_config,
    )
    trainable = [parameter for parameter in geometry.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Geometry configuration has no enabled trainable parameters.")
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    projection = geometry.project_()

    # Persist the resolved geometry and execution settings before doing any
    # expensive editor work. In particular, a failed parity preflight must
    # remain reproducible even when the user later edits the source JSON.
    config_payload = {
        **asdict(cfg),
        "spec": {
            "experiment": "edited_output_identity_exact",
            "model": spec.model,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seed": run_seed,
            "image_path": str(image_path),
        },
        "objective": "Z = cosine_similarity(ArcFace(exact clean edit), ArcFace(exact perturbed edit))",
        "loss": "loss = Z",
        "Z_forward": "exact installed Diffusers pipeline body with 8-bit STE forward values",
        "parity_preflight": {"status": "pending"},
        "parity_threshold_semantics": {
            "exact_tensor_Z_gap": _exact_Z_tolerance(cfg),
            "native_pil_public_replay_Z_gap": _native_pil_Z_tolerance(cfg),
        },
        "arcface": arcface.metadata(),
        "differentiable_instructpix2pix": editor.metadata(),
        "geometry_config_path": cfg.geometry_config_path,
        "geometry_config_resolved": geometry_config.__dict__.copy(),
        "geometry_limits": geometry.limits_dict(),
        "model_weights_frozen": True,
        "no_visual_counter_loss": True,
    }
    write_json(output_dir / "config_resolved.json", config_payload)

    print(f"[face4] generating exact clean edit reference: prompt={spec.case.prompt!r} steps={cfg.edit_steps}")
    with torch.no_grad():
        clean_edit_tensor = editor.edit_tensor(canonical_original, spec.case.prompt, run_seed).detach()
    clean_edit = tensor_to_pil(clean_edit_tensor)
    clean_edit.save(output_dir / "original_edited_exact_reference.png")
    reference = prepare_identity_reference(arcface, clean_edit_tensor)
    np.save(output_dir / "embedding_clean_edit_exact.npy", reference.embedding_original.detach().cpu().numpy().astype("float32"))

    with torch.no_grad():
        preflight_input_raw, _ = geometry(original_tensor)
        preflight_input = editor.canonical_input(preflight_input_raw)
    thresholds = ParityThresholds(
        exact_max_abs=cfg.parity_exact_max_abs,
        exact_min_ssim=cfg.parity_exact_min_ssim,
        native_pil_min_ssim=cfg.parity_native_pil_min_ssim,
        exact_max_Z_gap=_exact_Z_tolerance(cfg),
        native_pil_max_Z_gap=_native_pil_Z_tolerance(cfg),
    )
    print("[face4] running mandatory grad/no-grad/stock parity preflight")
    parity_report, parity_images = run_editor_parity_gate(
        editor,
        preflight_input,
        spec.case.prompt,
        run_seed,
        arcface=arcface,
        identity_reference=reference,
        thresholds=thresholds,
        backward_scale=cfg.backward_scale,
    )
    _write_preflight(output_dir, parity_report, parity_images)
    config_payload["parity_preflight"] = parity_report
    write_json(output_dir / "config_resolved.json", config_payload)
    if cfg.require_parity_preflight and not parity_report["passed"]:
        raise CorrectnessGateError(f"Editor parity preflight failed: {parity_report['checks']}")

    stock_clean_tensor = editor.stock_edit_tensor(canonical_original, spec.case.prompt, run_seed)
    stock_reference = prepare_identity_reference(arcface, stock_clean_tensor)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    effective_backward_scale = float(cfg.backward_scale)
    backward_scale_reductions_total = 0
    backward_retries_total = 0
    max_backward_retries_at_state = 0

    for step in range(cfg.iters):
        iter_started = time.monotonic()
        backward_retry_count = 0
        attempted_backward_scales: list[float] = []
        while True:
            optimizer.zero_grad(set_to_none=True)
            raw_perturbed, aux = geometry(original_tensor)
            perturbed = editor.canonical_input(raw_perturbed)
            perturbed_edit = editor.edit_tensor(perturbed, spec.case.prompt, run_seed)
            Z, terms = identity_objective(arcface, perturbed_edit, reference)
            loss = face_loss(Z)
            if not bool(torch.isfinite(loss).item() and torch.isfinite(perturbed_edit).all().item()):
                raise FloatingPointError(f"Non-finite Z/loss at optimizer state {step}")

            # Manual fp16 loss scaling prevents the tiny edited-output
            # identity gradient from underflowing inside the frozen VAE/UNet.
            # If a particular state overflows, retry that exact state with a
            # lower scale. Geometry and Adam remain untouched until a finite
            # unscaled gradient is available, so no optimizer update is lost
            # or silently skipped and the mathematical loss remains Z.
            attempted_backward_scales.append(float(effective_backward_scale))
            (loss * float(effective_backward_scale)).backward()
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(float(effective_backward_scale))
            nonfinite_gradients = [
                index
                for index, parameter in enumerate(trainable)
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            ]
            if not nonfinite_gradients:
                break

            next_scale = _next_backward_scale(
                effective_backward_scale,
                cfg.backward_scale_min,
                cfg.backward_scale_backoff,
            )
            if backward_retry_count >= cfg.backward_scale_max_retries or next_scale >= effective_backward_scale:
                raise CorrectnessGateError(
                    f"Non-finite geometry gradients after adaptive loss-scale retries at state {step}; "
                    f"parameter indices={nonfinite_gradients} attempted_scales={attempted_backward_scales} "
                    f"minimum={cfg.backward_scale_min} max_retries={cfg.backward_scale_max_retries}"
                )
            print(
                f"[face4] non-finite fp16 backward at state {step} with scale "
                f"{effective_backward_scale:g}; retrying the same state with {next_scale:g}"
            )
            effective_backward_scale = next_scale
            backward_retry_count += 1
            backward_retries_total += 1
            backward_scale_reductions_total += 1

        max_backward_retries_at_state = max(max_backward_retries_at_state, backward_retry_count)
        grad_norms = geometry.grad_norms()
        if not np.isfinite(grad_norms.get("total_grad_norm", 0.0)) or grad_norms.get("total_grad_norm", 0.0) <= 0.0:
            raise CorrectnessGateError(
                f"No finite nonzero geometry gradient reached the optimizer at state {step}: {grad_norms}; "
                f"effective_backward_scale={effective_backward_scale} retries={backward_retry_count}"
            )

        with torch.no_grad():
            _, input_identity_terms = identity_objective(arcface, perturbed, input_identity_reference)
        current_Z = float(Z.detach().float().cpu())
        candidate_is_best = best is None or current_Z < float(best["row"]["Z"])
        proposed_best_Z = current_Z if candidate_is_best else float(best["row"]["Z"])
        proposed_best_iter = step if candidate_is_best else int(best["row"]["iter"])

        stock_Z = None
        if step == 0 or (cfg.stock_validation_every > 0 and step % cfg.stock_validation_every == 0):
            stock_Z, _, _ = _stock_Z(editor, arcface, perturbed, spec.case.prompt, run_seed, stock_reference)
            if abs(stock_Z - current_Z) > _exact_Z_tolerance(cfg):
                raise CorrectnessGateError(
                    f"Stock Z parity failed at state {step}: exact={current_Z:.8f} stock={stock_Z:.8f} "
                    f"gap={abs(stock_Z-current_Z):.8f} exact_tensor_tolerance={_exact_Z_tolerance(cfg):.8f}"
                )

        row = _make_row(
            iteration=step,
            updates_completed=step,
            phase="optimization",
            Z=Z,
            terms=terms,
            spec=spec,
            run_seed=run_seed,
            cfg=cfg,
            started=started,
            seconds_iter=time.monotonic() - iter_started,
            perturbed=perturbed,
            original_tensor=canonical_original,
            input_identity_terms=input_identity_terms,
            aux=aux,
            grad_norms=grad_norms,
            geometry=geometry,
            projection=projection,
            best_Z=proposed_best_Z,
            best_iter=proposed_best_iter,
            stock_Z=stock_Z,
        )
        row["backward_scale"] = float(effective_backward_scale)
        row["backward_scale_initial"] = float(cfg.backward_scale)
        row["backward_retry_count"] = int(backward_retry_count)
        row["backward_scale_reductions_total"] = int(backward_scale_reductions_total)
        row["backward_attempted_scales"] = ",".join(f"{value:g}" for value in attempted_backward_scales)
        rows.append(row)
        append_jsonl(output_dir / "history.jsonl", row)
        if candidate_is_best:
            best = _capture_candidate(row, geometry, perturbed, perturbed_edit, aux)
        if step % max(1, cfg.checkpoint_every) == 0:
            _save_checkpoint(output_dir, step, perturbed.detach(), aux, row, geometry)

        # Only now change theta.  Everything stored above describes one exact,
        # internally consistent pre-update state.
        optimizer.step()
        projection = geometry.project_()

    # Evaluate theta_N once after the Nth optimizer update and append it as the
    # final state.  It is not mixed with the pre-update row from step N-1.
    final_started = time.monotonic()
    with torch.no_grad():
        final_raw, final_aux = geometry(original_tensor)
        final_perturbed_tensor = editor.canonical_input(final_raw)
        final_edit_tensor_exact = editor.edit_tensor(final_perturbed_tensor, spec.case.prompt, run_seed).detach()
        final_Z_exact, final_terms_exact = identity_objective(arcface, final_edit_tensor_exact, reference)
        _, final_input_identity_terms = identity_objective(arcface, final_perturbed_tensor, input_identity_reference)
    final_Z_stock_tensor, _, _ = _stock_Z(
        editor, arcface, final_perturbed_tensor, spec.case.prompt, run_seed, stock_reference
    )
    final_Z_value = float(final_Z_exact.detach().float().cpu())
    if abs(final_Z_stock_tensor - final_Z_value) > _exact_Z_tolerance(cfg):
        raise CorrectnessGateError(
            f"Final stock Z parity failed: exact={final_Z_value:.8f} stock={final_Z_stock_tensor:.8f} "
            f"gap={abs(final_Z_stock_tensor-final_Z_value):.8f} "
            f"exact_tensor_tolerance={_exact_Z_tolerance(cfg):.8f}"
        )
    final_is_best = best is None or final_Z_value < float(best["row"]["Z"])
    final_best_Z = final_Z_value if final_is_best else float(best["row"]["Z"])
    final_best_iter = cfg.iters if final_is_best else int(best["row"]["iter"])
    final_row = _make_row(
        iteration=cfg.iters,
        updates_completed=cfg.iters,
        phase="final_evaluation",
        Z=final_Z_exact,
        terms=final_terms_exact,
        spec=spec,
        run_seed=run_seed,
        cfg=cfg,
        started=started,
        seconds_iter=time.monotonic() - final_started,
        perturbed=final_perturbed_tensor,
        original_tensor=canonical_original,
        input_identity_terms=final_input_identity_terms,
        aux=final_aux,
        grad_norms={key: 0.0 for key in geometry.grad_norms()},
        geometry=geometry,
        projection=projection,
        best_Z=final_best_Z,
        best_iter=final_best_iter,
        stock_Z=final_Z_stock_tensor,
    )
    final_row["backward_scale"] = float(effective_backward_scale)
    final_row["backward_scale_initial"] = float(cfg.backward_scale)
    final_row["backward_retry_count"] = 0
    final_row["backward_scale_reductions_total"] = int(backward_scale_reductions_total)
    final_row["backward_attempted_scales"] = ""
    rows.append(final_row)
    append_jsonl(output_dir / "history.jsonl", final_row)
    if final_is_best:
        best = _capture_candidate(final_row, geometry, final_perturbed_tensor, final_edit_tensor_exact, final_aux)
    _save_checkpoint(output_dir, cfg.iters, final_perturbed_tensor, final_aux, final_row, geometry)
    if best is None:
        raise RuntimeError("No finite candidate was recorded.")

    best_perturbed = tensor_to_pil(best["perturbed"])
    final_perturbed = tensor_to_pil(final_perturbed_tensor)
    best_perturbed.save(output_dir / "perturbed_best.png")
    final_perturbed.save(output_dir / "perturbed_final.png")
    tensor_to_pil(best["perturbed_edit"]).save(output_dir / "perturbed_best_edited_exact.png")
    tensor_to_pil(final_edit_tensor_exact).save(output_dir / "perturbed_final_edited_exact.png")

    # Public artifacts use the ordinary decorated PIL pipeline.  These are
    # validation replays of the same 8-bit inputs optimized above.
    stock_clean_edit = editor.stock_edit_pil(tensor_to_pil(canonical_original), spec.case.prompt, run_seed)
    stock_best_edit = editor.stock_edit_pil(best_perturbed, spec.case.prompt, run_seed)
    stock_final_edit = editor.stock_edit_pil(final_perturbed, spec.case.prompt, run_seed)
    stock_clean_edit.save(output_dir / "original_edited.png")
    stock_best_edit.save(output_dir / "perturbed_best_edited.png")
    stock_final_edit.save(output_dir / "perturbed_final_edited.png")

    with torch.no_grad():
        stock_clean_public_tensor = pil_to_tensor(stock_clean_edit, device)
        stock_best_public_tensor = pil_to_tensor(stock_best_edit, device)
        stock_final_public_tensor = pil_to_tensor(stock_final_edit, device)
        public_reference = prepare_identity_reference(arcface, stock_clean_public_tensor)
        best_Z_stock_public, best_terms_stock_public = identity_objective(arcface, stock_best_public_tensor, public_reference)
        final_Z_stock_public, final_terms_stock_public = identity_objective(arcface, stock_final_public_tensor, public_reference)
        original_vs_original_edit_identity = _identity_pair_metrics(
            arcface, canonical_original, stock_clean_public_tensor, "original_vs_original_edit_identity"
        )
        best_input_identity = _identity_pair_metrics(
            arcface, canonical_original, best["perturbed"], "best_input_identity"
        )
        final_input_identity = _identity_pair_metrics(
            arcface, canonical_original, final_perturbed_tensor, "final_input_identity"
        )
        perturbed_best_vs_edit_identity = _identity_pair_metrics(
            arcface, best["perturbed"], stock_best_public_tensor, "perturbed_best_vs_perturbed_best_edit_identity"
        )
        perturbed_final_vs_edit_identity = _identity_pair_metrics(
            arcface, final_perturbed_tensor, stock_final_public_tensor, "perturbed_final_vs_perturbed_final_edit_identity"
        )

    clean_exact_vs_stock = image_metrics(clean_edit, stock_clean_edit)
    best_exact_vs_stock = image_metrics(tensor_to_pil(best["perturbed_edit"]), stock_best_edit)
    final_exact_vs_stock = image_metrics(tensor_to_pil(final_edit_tensor_exact), stock_final_edit)
    best_stock_Z_value = float(best_Z_stock_public.detach().float().cpu())
    final_stock_Z_value = float(final_Z_stock_public.detach().float().cpu())
    best_Z_gap = abs(best_stock_Z_value - float(best["row"]["Z"]))
    final_Z_gap = abs(final_stock_Z_value - final_Z_value)
    if (
        best_exact_vs_stock["ssim"] < cfg.parity_native_pil_min_ssim
        or final_exact_vs_stock["ssim"] < cfg.parity_native_pil_min_ssim
        or best_Z_gap > _native_pil_Z_tolerance(cfg)
        or final_Z_gap > _native_pil_Z_tolerance(cfg)
    ):
        raise CorrectnessGateError(
            "Final public replay parity failed: "
            f"best_ssim={best_exact_vs_stock['ssim']:.6f}, final_ssim={final_exact_vs_stock['ssim']:.6f}, "
            f"best_Z_gap={best_Z_gap:.6f}, final_Z_gap={final_Z_gap:.6f}, "
            f"native_pil_Z_tolerance={_native_pil_Z_tolerance(cfg):.6f}"
        )

    _component_flow_images(final_aux, output_dir, geometry.component_limit_for_flow, geometry)
    flow_to_pil(best["aux"]["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow_best.png")
    flow_to_pil(final_aux["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow_final.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_best.png", output_dir / "input_difference_best.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_final.png", output_dir / "input_difference_final.png")
    write_json(
        output_dir / "geometry_params_final.json",
        {"limits": geometry.limits_dict(), "parameter_diagnostics": geometry.parameter_diagnostics(), "last_projection": projection},
    )
    write_json(
        output_dir / "geometry_params_best.json",
        {"best_iter_by_Z": best["row"]["iter"], "best_Z": best["row"]["Z"], "state_is_pre_update_and_consistent": True},
    )
    np.save(output_dir / "embedding_perturbed_edit_best_exact.npy", arcface.embedding(best["perturbed_edit"]).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_final_exact.npy", arcface.embedding(final_edit_tensor_exact).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_best_stock.npy", arcface.embedding(stock_best_public_tensor).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_final_stock.npy", arcface.embedding(stock_final_public_tensor).detach().cpu().numpy().astype("float32"))

    save_sheet(
        output_dir / "comparison_sheet.png",
        [
            ("Original", original),
            ("Perturbed Best", best_perturbed),
            ("Abs Difference x8", Image.open(output_dir / "input_difference_best.png")),
            ("Clean Stock Edit", stock_clean_edit),
            ("Perturbed Stock Edit", stock_best_edit),
        ],
    )

    if not cfg.skip_deepface:
        from ..evaluation.deepface_panel import run_deepface_panel, write_identity_panel

        panel_rows = run_deepface_panel(output_dir)
        write_identity_panel(output_dir, panel_rows)
    else:
        panel_rows = []
        write_json(output_dir / "identity_panel.json", [{"status": "skipped"}])
        (output_dir / "identity_panel.csv").write_text("status\nskipped\n", encoding="utf-8")

    write_csv(output_dir / "history.csv", rows)
    elapsed = time.monotonic() - started
    input_metrics_best = image_metrics(original, best_perturbed)
    input_metrics_final = image_metrics(original, final_perturbed)
    output_metrics_best = image_metrics(stock_clean_edit, stock_best_edit)
    output_metrics_final = image_metrics(stock_clean_edit, stock_final_edit)
    optimization_rows = [row for row in rows if row["phase"] == "optimization"]
    summary = {
        "status": "done",
        "experiment": "edited_output_identity_exact",
        "model": spec.model,
        "face_id": spec.case.face_id,
        "prompt": spec.case.prompt,
        "case_id": spec.case.slug,
        "seed": run_seed,
        "iters": cfg.iters,
        "optimizer_updates_completed": cfg.iters,
        "backward_scale_initial": float(cfg.backward_scale),
        "backward_scale_final": float(effective_backward_scale),
        "backward_retries_total": int(backward_retries_total),
        "backward_scale_reductions_total": int(backward_scale_reductions_total),
        "max_backward_retries_at_state": int(max_backward_retries_at_state),
        "Z_definition": "ArcFace cosine similarity between exact clean and perturbed edited outputs",
        "loss": "loss = Z",
        "Z_is_stock_equivalent_forward": True,
        "final_Z": final_Z_value,
        "final_loss": final_Z_value,
        "best_Z": float(best["row"]["Z"]),
        "best_iter_by_Z": int(best["row"]["iter"]),
        "final_Z_stock_public": final_stock_Z_value,
        "best_Z_stock_public": best_stock_Z_value,
        "final_Z_stock_gap": final_Z_gap,
        "best_Z_stock_gap": best_Z_gap,
        "final_exact_vs_stock_public_ssim": final_exact_vs_stock["ssim"],
        "best_exact_vs_stock_public_ssim": best_exact_vs_stock["ssim"],
        "clean_exact_vs_stock_public_ssim": clean_exact_vs_stock["ssim"],
        "parity_preflight_passed": parity_report["passed"],
        "parity_preflight": parity_report,
        "public_edit_images_regenerated_with_stock_pipeline": True,
        "best_selection_source": "minimum exact stock-equivalent Z across every optimized state",
        "state_bookkeeping": "Z, image, aux, theta, and parameter diagnostics are captured before the corresponding optimizer update",
        "editor_num_inference_steps": cfg.edit_steps,
        "final_identity_cosine_similarity_raw": final_stock_Z_value,
        "best_identity_cosine_similarity_raw": best_stock_Z_value,
        "final_identity_similarity_score_pct": float(
            _float_terms(final_terms_stock_public)["identity_similarity_score_pct"]
        ),
        "best_identity_similarity_score_pct": float(
            _float_terms(best_terms_stock_public)["identity_similarity_score_pct"]
        ),
        "final_stock_public_identity_cosine_similarity_raw": final_stock_Z_value,
        "best_stock_public_identity_cosine_similarity_raw": best_stock_Z_value,
        "final_stock_public_identity_similarity_score_pct": float(
            _float_terms(final_terms_stock_public)["identity_similarity_score_pct"]
        ),
        "best_stock_public_identity_similarity_score_pct": float(
            _float_terms(best_terms_stock_public)["identity_similarity_score_pct"]
        ),
        **best_input_identity,
        **final_input_identity,
        **original_vs_original_edit_identity,
        **perturbed_best_vs_edit_identity,
        **perturbed_final_vs_edit_identity,
        "mean_seconds_iter": float(sum(row["seconds_iter"] for row in optimization_rows) / max(len(optimization_rows), 1)),
        "elapsed_seconds": elapsed,
        "final_psnr_to_original": final_row["psnr_to_original"],
        "final_ssim_to_original": final_row["ssim_to_original"],
        "final_mse_to_original": final_row["mse_to_original"],
        "input_best_ssim": input_metrics_best["ssim"],
        "input_best_psnr": input_metrics_best["psnr"],
        "input_best_l2": input_metrics_best["l2"],
        "input_final_ssim": input_metrics_final["ssim"],
        "input_final_psnr": input_metrics_final["psnr"],
        "input_final_l2": input_metrics_final["l2"],
        "best_output_ssim": output_metrics_best["ssim"],
        "best_output_psnr": output_metrics_best["psnr"],
        "best_output_l2": output_metrics_best["l2"],
        "final_output_ssim": output_metrics_final["ssim"],
        "final_output_psnr": output_metrics_final["psnr"],
        "final_output_l2": output_metrics_final["l2"],
        "final_combined_max_disp_px": final_row["combined_max_disp_px"],
        "final_combined_mean_disp_px": final_row["combined_mean_disp_px"],
        "final_combined_p95_disp_px": final_row["combined_p95_disp_px"],
        "final_fraction_clamped_total": final_row["fraction_clamped_total"],
        "all_required_history_fields_populated": _history_fields_ok(final_row),
        "clamp_project_logic_active": final_row["num_total_params"] > 0,
        "arcface": arcface.metadata(),
        "differentiable_instructpix2pix": editor.metadata(),
        "deepface_rows": len(panel_rows),
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "run_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "DONE.json",
        {
            "status": "done",
            "elapsed_seconds": elapsed,
            "final_Z": final_Z_value,
            "best_Z": float(best["row"]["Z"]),
            "parity_passed": True,
        },
    )
    return summary
