"""Run orchestration for FACE smoke and edited-output ArcFace identity jobs."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import shutil

import numpy as np
from PIL import Image

from .cases import RunSpec, build_matrix, print_resolved_cases, resolve_image_path
from .identity import face_loss, identity_objective, prepare_identity_reference
from .image_metrics import delta_to_pil, flow_to_pil, image_metrics, pil_to_tensor, save_sheet, tensor_pair_metrics, tensor_to_pil
from .logging import append_jsonl, nvidia_smi_memory_gb, read_json, write_csv, write_json
from .parity import CorrectnessGateError
from .runtime import torch_device, torch_peak_gb
from .utils import save_input_difference


@dataclass
class RunConfig:
    mat_root: str
    arcface_checkpoint: str
    output_root: str
    geometry_config_path: str = "configs/geometry_default.json"
    iters: int = 150
    lr: float = 0.1
    seed: int | None = None
    init: str | None = None
    quick: bool = False
    all_cases: bool = False
    mode: str = "run_matrix"
    skip_deepface: bool = False
    force: bool = False
    source_url: str | None = None
    checkpoint_every: int = 25
    editor_model_id: str = "timbrooks/instruct-pix2pix"
    editor_dtype: str = "float16"
    edit_steps: int = 4
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    edit_eta: float = 0.0
    enable_editor_gradient_checkpointing: bool = True
    quantize_input_8bit: bool = True
    quantize_output_8bit: bool = True
    require_parity_preflight: bool = True
    parity_exact_max_abs: float = 1.0 / 255.0 + 1e-6
    parity_exact_min_ssim: float = 0.999
    parity_native_pil_min_ssim: float = 0.990
    parity_max_Z_gap: float = 0.001
    stock_validation_every: int = 25
    backward_scale: float = 65536.0
    backward_scale_min: float = 1.0
    backward_scale_backoff: float = 0.5
    backward_scale_max_retries: int = 20
    resume_run_root: str | None = None
    resume_latest: bool = False


REQUIRED_HISTORY_FIELDS = {
    "iter",
    "Z",
    "loss",
    "best_Z_so_far",
    "best_iter_so_far",
    "learning_rate",
    "seed",
    "face_id",
    "prompt",
    "case_id",
    "seconds_iter",
    "seconds_elapsed",
    "peak_vram_gb",
    "identity_cosine_similarity_raw",
    "identity_cosine_distance",
    "identity_similarity_score_pct",
    "edit_identity_cosine_similarity_raw",
    "edit_identity_cosine_distance",
    "edit_identity_similarity_score_pct",
    "identity_l2_embedding_distance",
    "identity_angle_radians",
    "identity_angle_degrees",
    "original_embedding_norm",
    "clean_edit_embedding_norm",
    "perturbed_embedding_norm",
    "perturbed_edit_embedding_norm",
    "editor_num_inference_steps",
    "psnr_to_original",
    "ssim_to_original",
    "mse_to_original",
    "l2_to_original",
    "combined_max_disp_px",
    "combined_mean_disp_px",
    "combined_p95_disp_px",
    "jacobian_det_min",
    "foldover_fraction",
    "smoothness_tv",
    "tps_mean_disp",
    "tps_max_disp",
    "tps_p95_disp",
    "tps_param_min",
    "tps_param_max",
    "tps_param_mean_abs",
    "tps_grad_norm",
    "tps_num_at_min",
    "tps_num_at_max",
    "delaunay_mean_disp",
    "delaunay_max_disp",
    "delaunay_p95_disp",
    "delaunay_param_min",
    "delaunay_param_max",
    "delaunay_param_mean_abs",
    "delaunay_grad_norm",
    "delaunay_num_at_min",
    "delaunay_num_at_max",
    "rolling_mean_disp",
    "rolling_max_disp",
    "rolling_p95_disp",
    "rolling_param_min",
    "rolling_param_max",
    "rolling_param_mean_abs",
    "rolling_grad_norm",
    "rolling_num_at_min",
    "rolling_num_at_max",
    "dct_enabled",
    "dct_gain_min",
    "dct_gain_max",
    "dct_gain_mean",
    "dct_gain_mean_abs",
    "dct_gain_l2",
    "dct_gain_grad_norm",
    "dct_num_at_min",
    "dct_num_at_max",
    "dct_num_clamped",
    "dct_selected_frequency_count",
    "dct_frequency_mask_mode",
    "dct_input_coefficient_energy",
    "dct_output_coefficient_energy",
    "dct_coefficient_delta_l1",
    "dct_coefficient_delta_l2",
    "dct_relative_energy_change",
    "dct_spatial_delta_mse",
    "dct_spatial_delta_l1",
    "dct_spatial_delta_l2",
    "dct_spatial_delta_max_abs",
    "dct_clipped_low_fraction",
    "dct_clipped_high_fraction",
    "dct_dc_energy",
    "dct_low_frequency_energy_before",
    "dct_low_frequency_energy_after",
    "dct_mid_frequency_energy_before",
    "dct_mid_frequency_energy_after",
    "dct_high_frequency_energy_before",
    "dct_high_frequency_energy_after",
    "fft_phase_norm",
    "fft_phase_mean_abs",
    "fft_phase_max_abs",
    "fft_phase_grad_norm",
    "fft_phase_num_at_min",
    "fft_phase_num_at_max",
    "legacy_fft_strength_equivalent",
    "fft_spatial_delta_mse",
    "num_total_params",
    "num_clamped_total",
    "fraction_clamped_total",
    "num_at_min_total",
    "num_at_max_total",
    "components_at_boundary",
    "total_geometry_grad_norm",
    "backward_scale",
    "backward_retry_count",
    "backward_scale_reductions_total",
    "input_identity_cosine_similarity_raw",
    "input_identity_cosine_distance",
    "input_identity_similarity_score_pct",
    "input_identity_l2_embedding_distance",
}


def _run_dir(root: Path, spec: RunSpec) -> Path:
    return root / "runs" / "edited_output_identity" / spec.model / spec.case.slug


def _latest_run_root(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "runs" / "edited_output_identity").exists()
    ] if output_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No resumable FACE4 run roots found under {output_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _archive_incomplete_case_dir(run_dir: Path) -> None:
    if not run_dir.exists() or (run_dir / "DONE.json").exists():
        return
    if not any(run_dir.iterdir()):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_root = run_dir.parent / "_incomplete_case_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{run_dir.name}_{stamp}"
    counter = 1
    while archive.exists():
        archive = archive_root / f"{run_dir.name}_{stamp}_{counter:02d}"
        counter += 1
    print(f"[face4] archiving incomplete case before rerun: {run_dir} -> {archive}")
    shutil.move(str(run_dir), str(archive))


def _float_terms(terms: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in terms.items():
        if hasattr(value, "detach"):
            out[key] = float(value.detach().float().cpu())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _prefixed_identity_terms(prefix: str, terms: dict[str, Any]) -> dict[str, float]:
    flat = _float_terms(terms)
    mapping = {
        "identity_cosine_similarity_raw": "cosine_similarity_raw",
        "identity_cosine_distance": "cosine_distance",
        "identity_similarity_score_pct": "similarity_score_pct",
        "identity_l2_embedding_distance": "l2_embedding_distance",
        "identity_angle_radians": "angle_radians",
        "identity_angle_degrees": "angle_degrees",
        "original_embedding_norm": "left_embedding_norm",
        "perturbed_embedding_norm": "right_embedding_norm",
    }
    return {f"{prefix}_{new_key}": flat[old_key] for old_key, new_key in mapping.items() if old_key in flat}


def _identity_pair_metrics(arcface, left, right, prefix: str) -> dict[str, float]:
    reference = prepare_identity_reference(arcface, left)
    _, terms = identity_objective(arcface, right, reference)
    return _prefixed_identity_terms(prefix, terms)


def _history_fields_ok(row: dict[str, Any]) -> bool:
    return REQUIRED_HISTORY_FIELDS.issubset(set(row))


def _save_scaled_delta(path: Path, delta, scale: float = 10.0) -> None:
    import torch

    value = delta.detach().float().abs().mean(1, keepdim=False)[0].mul(float(scale)).clamp(0, 1)
    array = value.cpu().numpy()
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB").save(path)


def _save_matrix_heatmap(path: Path, matrix, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = matrix.detach().float().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=140)
    im = ax.imshow(arr, cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("DCT v frequency index")
    ax.set_ylabel("DCT u frequency index")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _dct_visual_images(aux: dict[str, Any], out_dir: Path, geometry) -> None:
    if "dct_image" not in aux or "dct_delta" not in aux:
        return
    tensor_to_pil(aux["dct_image"]).save(out_dir / "dct_only_perturbed.png")
    delta_to_pil(aux["dct_delta"]).save(out_dir / "dct_only_difference.png")
    _save_scaled_delta(out_dir / "dct_only_difference_x10.png", aux["dct_delta"], scale=10.0)
    dct = geometry.dct_image
    _save_matrix_heatmap(out_dir / "dct_gain_heatmap.png", dct.gain_heatmap(), "DCT gain |mean over RGB|")
    _save_matrix_heatmap(out_dir / "dct_frequency_mask.png", dct.frequency_mask_2d, f"DCT frequency mask: {dct.frequency_mask_mode}", vmin=0, vmax=1)
    before = dct.spectrum_summary(aux["spatial"])
    after = dct.spectrum_summary(aux["dct_image"])
    vmax = float(max(before.max().cpu(), after.max().cpu()).item()) if before.numel() and after.numel() else None
    _save_matrix_heatmap(out_dir / "dct_spectrum_before.png", before, "DCT spectrum before", vmin=0, vmax=vmax)
    _save_matrix_heatmap(out_dir / "dct_spectrum_after.png", after, "DCT spectrum after", vmin=0, vmax=vmax)
    _save_matrix_heatmap(out_dir / "dct_spectrum_difference.png", (after - before).abs(), "DCT spectrum |after-before|")


def _component_flow_images(aux: dict[str, Any], out_dir: Path, scale_px: float, geometry=None) -> None:
    flow_to_pil(aux["displacement"], scale_px).save(out_dir / "combined_flow.png")
    for name, field in aux["fields"].items():
        flow_to_pil(field, scale_px).save(out_dir / f"{name}_flow.png")
    delta_to_pil(aux["fft_delta"]).save(out_dir / "fft_phase_visualization.png")
    if geometry is not None:
        _dct_visual_images(aux, out_dir, geometry)


def _save_checkpoint(run_dir: Path, iteration: int, perturbed, aux: dict[str, Any], row: dict[str, Any], geometry) -> None:
    ckpt = run_dir / "checkpoints" / f"iter_{iteration:03d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(perturbed).save(ckpt / "perturbed.png")
    flow_to_pil(aux["displacement"], geometry.component_limit_for_flow).save(ckpt / "combined_flow.png")
    write_json(ckpt / "metrics.json", row)
    write_json(
        ckpt / "geometry_params.json",
        {
            "limits": geometry.limits_dict(),
            "parameter_diagnostics": geometry.parameter_diagnostics(),
        },
    )


def _arcface(device, checkpoint_path: str, source_url: str | None):
    from face4.models.arcface import ArcFaceIResNet100

    return ArcFaceIResNet100(checkpoint_path, device, source_url=source_url)


def _editor(device, cfg: RunConfig):
    from face4.models.differentiable_instruct import DifferentiableInstructPix2Pix, DifferentiableInstructSettings

    settings = DifferentiableInstructSettings(
        model_id=cfg.editor_model_id,
        torch_dtype=cfg.editor_dtype,
        num_inference_steps=cfg.edit_steps,
        guidance_scale=cfg.guidance_scale,
        image_guidance_scale=cfg.image_guidance_scale,
        eta=cfg.edit_eta,
        enable_gradient_checkpointing=cfg.enable_editor_gradient_checkpointing,
        quantize_input_8bit=cfg.quantize_input_8bit,
        quantize_output_8bit=cfg.quantize_output_8bit,
    )
    return DifferentiableInstructPix2Pix(device, settings=settings)


def _edit_terms_aliases(terms: dict[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for key, value in terms.items():
        if key.startswith("identity_"):
            aliases[f"edit_{key}"] = value
    if "original_embedding_norm" in terms:
        aliases["clean_edit_embedding_norm"] = terms["original_embedding_norm"]
    if "perturbed_embedding_norm" in terms:
        aliases["perturbed_edit_embedding_norm"] = terms["perturbed_embedding_norm"]
    return aliases


def _optimize_one_face3_legacy(spec: RunSpec, cfg: RunConfig, arcface, editor, device, output_dir: Path) -> dict[str, Any]:
    """Preserved FACE3 implementation for audit/reference; never dispatched."""
    import torch

    from face4.core.geometry.combined_face import CombinedFacePerturbation, FaceGeometryConfig, load_face_geometry_config

    output_dir.mkdir(parents=True, exist_ok=True)
    done_path = output_dir / "DONE.json"
    if done_path.exists() and not cfg.force:
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            print(f"[face4] skip completed run: {output_dir}")
            return read_json(summary_path)
        raise RuntimeError(f"DONE.json exists but summary.json is missing: {output_dir}. Use --force after inspecting.")

    started = time.monotonic()
    mat_root = Path(cfg.mat_root)
    image_path = resolve_image_path(mat_root, spec.case.face_id)
    print(f"[face4] running {spec.slug} image={image_path}")

    original = Image.open(image_path).convert("RGB")
    original.save(output_dir / "original.png")
    original_tensor = pil_to_tensor(original, device)
    input_identity_reference = prepare_identity_reference(arcface, original_tensor)

    geometry_config = load_face_geometry_config(cfg.geometry_config_path) if cfg.geometry_config_path else FaceGeometryConfig()
    if cfg.init:
        geometry_config.init = cfg.init
    torch.manual_seed(spec.run_seed)
    geometry = CombinedFacePerturbation(
        original_tensor.shape[-2],
        original_tensor.shape[-1],
        original_tensor.shape[1],
        device,
        seed=spec.run_seed,
        config=geometry_config,
    )
    optimizer = torch.optim.Adam([p for p in geometry.parameters() if p.requires_grad], lr=cfg.lr)
    projection = geometry.project_()
    print(f"[face4] generating fixed clean edit reference: prompt={spec.case.prompt!r} steps={cfg.edit_steps}")
    with torch.no_grad():
        clean_edit_tensor = editor.edit_tensor(original_tensor, spec.case.prompt, spec.run_seed).detach()
    clean_edit = tensor_to_pil(clean_edit_tensor)
    clean_edit.save(output_dir / "original_edited_gradient_reference.png")
    clean_edit.save(output_dir / "original_edited.png")
    reference = prepare_identity_reference(arcface, clean_edit_tensor)
    reference.embedding_original.detach().cpu().numpy().astype("float32").tofile(output_dir / "embedding_clean_edit.raw")
    np.save(output_dir / "embedding_clean_edit.npy", reference.embedding_original.detach().cpu().numpy().astype("float32"))

    config_payload = {
        **asdict(cfg),
        "spec": {
            "experiment": "edited_output_identity",
            "model": spec.model,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seed": spec.run_seed,
            "image_path": str(image_path),
        },
        "experiment_description": "White-box edited-output identity optimization. Gradients flow through differentiable InstructPix2Pix editing and frozen ArcFace into perturbation parameters.",
        "objective": "Z = cosine_similarity(ArcFace(original_edit), ArcFace(perturbed_edit))",
        "loss": "loss = Z",
        "arcface_objective_prompt_conditioned": True,
        "prompt_usage": "Prompt conditions the differentiable InstructPix2Pix edit inside the optimization objective.",
        "differentiable_instructpix2pix": editor.metadata(),
        "clean_edit_reference": "original input edited once with fixed prompt/seed/settings, then detached as ArcFace reference",
        "no_landmarks_alignment_or_detection": True,
        "no_visual_counter_loss": True,
        "model_weights_frozen": True,
        "optimized_parameters": "spatial_geometry_plus_dct_image_frequency_parameters",
        "arcface": arcface.metadata(),
        "geometry_config_path": cfg.geometry_config_path,
        "geometry_config_resolved": geometry_config.__dict__.copy(),
        "geometry_limits": geometry.limits_dict(),
    }
    write_json(output_dir / "config_resolved.json", config_payload)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    with torch.no_grad():
        p0, aux0 = geometry(original_tensor)
        metrics0 = tensor_pair_metrics(p0, original_tensor, prefix="")
        _, input_identity_terms0 = identity_objective(arcface, p0, input_identity_reference)
        edit0 = editor.edit_tensor(p0, spec.case.prompt, spec.run_seed).detach()
        Z0, terms0 = identity_objective(arcface, edit0, reference)
        row0 = {
            "iter": 0,
            "Z": float(Z0.detach().float().cpu()),
            "loss": float(face_loss(Z0).detach().float().cpu()),
            "best_Z_so_far": float(Z0.detach().float().cpu()),
            "best_iter_so_far": 0,
            "learning_rate": cfg.lr,
            "seed": spec.run_seed,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seconds_iter": 0.0,
            "seconds_elapsed": 0.0,
            "peak_vram_gb": torch_peak_gb(),
            "psnr_to_original": metrics0["psnr"],
            "ssim_to_original": metrics0["ssim"],
            "mse_to_original": metrics0["mse"],
            "l2_to_original": metrics0["l2"],
            **_prefixed_identity_terms("input_identity", input_identity_terms0),
            **_float_terms(terms0),
            **_float_terms(_edit_terms_aliases(terms0)),
            **aux0["diagnostics"],
            **{key: 0.0 for key in geometry.grad_norms()},
            **geometry.parameter_diagnostics(),
            **projection,
        }
        row0["editor_num_inference_steps"] = cfg.edit_steps
        row0["total_geometry_grad_norm"] = 0.0
        _save_checkpoint(output_dir, 0, p0, aux0, row0, geometry)
        rows.append(row0)
        append_jsonl(output_dir / "history.jsonl", row0)
        best = {
            "row": row0,
            "theta_state": geometry.theta_state(),
            "perturbed": p0.detach().clone(),
            "perturbed_edit": edit0.detach().clone(),
            "aux": {
                "spatial": aux0["spatial"].detach().clone(),
                "dct_image": aux0["dct_image"].detach().clone(),
                "dct_delta": aux0["dct_delta"].detach().clone(),
                "displacement": aux0["displacement"].detach().clone(),
                "fields": {k: v.detach().clone() for k, v in aux0["fields"].items()},
                "fft_delta": aux0["fft_delta"].detach().clone(),
            },
        }

    for iteration in range(1, cfg.iters + 1):
        iter_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        perturbed, aux = geometry(original_tensor)
        perturbed_edit = editor.edit_tensor(perturbed, spec.case.prompt, spec.run_seed)
        Z, terms = identity_objective(arcface, perturbed_edit, reference)
        loss = face_loss(Z)
        finite = bool(torch.isfinite(loss).item() and torch.isfinite(perturbed).all().item() and torch.isfinite(perturbed_edit).all().item())
        if not finite:
            raise FloatingPointError(f"Non-finite Z/loss at iteration {iteration}")
        loss.backward()
        grad_norms = geometry.grad_norms()
        optimizer.step()
        projection = geometry.project_()

        with torch.no_grad():
            metrics_original = tensor_pair_metrics(perturbed, original_tensor, prefix="")
            _, input_identity_terms = identity_objective(arcface, perturbed, input_identity_reference)
        seconds_iter = time.monotonic() - iter_started
        current_Z = float(Z.detach().float().cpu())
        prev_best = best["row"]["Z"] if best is not None else 1e30
        row: dict[str, Any] = {
            "iter": iteration,
            "Z": current_Z,
            "loss": float(loss.detach().float().cpu()),
            "best_Z_so_far": float(min(prev_best, current_Z)),
            "best_iter_so_far": iteration if best is None or current_Z < prev_best else best["row"]["iter"],
            "learning_rate": cfg.lr,
            "seed": spec.run_seed,
            "face_id": spec.case.face_id,
            "prompt": spec.case.prompt,
            "case_id": spec.case.slug,
            "seconds_iter": seconds_iter,
            "seconds_elapsed": time.monotonic() - started,
            "peak_vram_gb": torch_peak_gb(),
            "psnr_to_original": metrics_original["psnr"],
            "ssim_to_original": metrics_original["ssim"],
            "mse_to_original": metrics_original["mse"],
            "l2_to_original": metrics_original["l2"],
            **_prefixed_identity_terms("input_identity", input_identity_terms),
            **_float_terms(terms),
            **_float_terms(_edit_terms_aliases(terms)),
            **aux["diagnostics"],
            **grad_norms,
            **geometry.parameter_diagnostics(),
            **projection,
        }
        row["editor_num_inference_steps"] = cfg.edit_steps
        row["total_geometry_grad_norm"] = row.get("total_grad_norm", 0.0)
        rows.append(row)
        append_jsonl(output_dir / "history.jsonl", row)
        if row["Z"] < best["row"]["Z"]:
            best = {
                "row": row,
                "theta_state": geometry.theta_state(),
                "perturbed": perturbed.detach().clone(),
                "perturbed_edit": perturbed_edit.detach().clone(),
                "aux": {
                    "spatial": aux["spatial"].detach().clone(),
                    "dct_image": aux["dct_image"].detach().clone(),
                    "dct_delta": aux["dct_delta"].detach().clone(),
                    "displacement": aux["displacement"].detach().clone(),
                    "fields": {k: v.detach().clone() for k, v in aux["fields"].items()},
                    "fft_delta": aux["fft_delta"].detach().clone(),
                },
            }
        if iteration % max(1, cfg.checkpoint_every) == 0 or iteration == cfg.iters:
            _save_checkpoint(output_dir, iteration, perturbed.detach(), aux, row, geometry)

    if not rows or best is None:
        raise RuntimeError("No finite optimization iteration completed.")

    with torch.no_grad():
        final_perturbed_tensor, final_aux = geometry(original_tensor)
        final_edit_tensor_gradient_path = editor.edit_tensor(final_perturbed_tensor, spec.case.prompt, spec.run_seed).detach()
        final_Z_gradient_path, final_terms_gradient_path = identity_objective(arcface, final_edit_tensor_gradient_path, reference)

    final_perturbed = tensor_to_pil(final_perturbed_tensor)
    best_perturbed = tensor_to_pil(best["perturbed"])
    final_perturbed.save(output_dir / "perturbed_final.png")
    best_perturbed.save(output_dir / "perturbed_best.png")
    tensor_to_pil(best["perturbed_edit"]).save(output_dir / "perturbed_best_edited_gradient_path.png")
    tensor_to_pil(final_edit_tensor_gradient_path).save(output_dir / "perturbed_final_edited_gradient_path.png")

    # Public edited images must come from the normal stock/no-grad diffusers
    # pipeline. The differentiable edit path above is only the optimization
    # path; for some perturbed inputs it can produce wrapper-only artifacts.
    stock_clean_edit = editor.stock_edit_pil(original, spec.case.prompt, spec.run_seed)
    stock_best_edit = editor.stock_edit_pil(best_perturbed, spec.case.prompt, spec.run_seed)
    stock_final_edit = editor.stock_edit_pil(final_perturbed, spec.case.prompt, spec.run_seed)
    stock_clean_edit.save(output_dir / "original_edited.png")
    stock_best_edit.save(output_dir / "perturbed_best_edited.png")
    stock_final_edit.save(output_dir / "perturbed_final_edited.png")

    with torch.no_grad():
        stock_clean_tensor = pil_to_tensor(stock_clean_edit, device)
        stock_best_tensor = pil_to_tensor(stock_best_edit, device)
        stock_final_tensor = pil_to_tensor(stock_final_edit, device)
        stock_best_input_tensor = pil_to_tensor(best_perturbed, device)
        stock_final_input_tensor = pil_to_tensor(final_perturbed, device)
        stock_reference = prepare_identity_reference(arcface, stock_clean_tensor)
        best_Z_stock_public, best_terms_stock_public = identity_objective(arcface, stock_best_tensor, stock_reference)
        final_Z_stock_public, final_terms_stock_public = identity_objective(arcface, stock_final_tensor, stock_reference)
        original_vs_original_edit_identity = _identity_pair_metrics(
            arcface, original_tensor, stock_clean_tensor, "original_vs_original_edit_identity"
        )
        best_input_identity = _identity_pair_metrics(arcface, original_tensor, stock_best_input_tensor, "best_input_identity")
        final_input_identity = _identity_pair_metrics(arcface, original_tensor, stock_final_input_tensor, "final_input_identity")
        perturbed_best_vs_perturbed_best_edit_identity = _identity_pair_metrics(
            arcface, stock_best_input_tensor, stock_best_tensor, "perturbed_best_vs_perturbed_best_edit_identity"
        )
        perturbed_final_vs_perturbed_final_edit_identity = _identity_pair_metrics(
            arcface, stock_final_input_tensor, stock_final_tensor, "perturbed_final_vs_perturbed_final_edit_identity"
        )
        clean_stock_vs_gradient_reference = image_metrics(stock_clean_edit, clean_edit)
        best_gradient_vs_stock = image_metrics(stock_best_edit, Image.open(output_dir / "perturbed_best_edited_gradient_path.png"))
        final_gradient_vs_stock = image_metrics(stock_final_edit, Image.open(output_dir / "perturbed_final_edited_gradient_path.png"))
        if float(final_Z_stock_public.detach().float().cpu()) <= float(best_Z_stock_public.detach().float().cpu()):
            best_saved_stock_public_Z = final_Z_stock_public
            best_saved_stock_public_source = "final_input"
            best_saved_stock_public_iter = cfg.iters
        else:
            best_saved_stock_public_Z = best_Z_stock_public
            best_saved_stock_public_source = "gradient_best_input"
            best_saved_stock_public_iter = best["row"]["iter"]

    _component_flow_images(final_aux, output_dir, geometry.component_limit_for_flow, geometry)
    flow_to_pil(best["aux"]["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow_best.png")
    (output_dir / "combined_flow.png").replace(output_dir / "combined_flow_final.png")
    # Keep a conventional alias for report scripts.
    flow_to_pil(final_aux["displacement"], geometry.component_limit_for_flow).save(output_dir / "combined_flow.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_best.png", output_dir / "input_difference_best.png")
    save_input_difference(output_dir / "original.png", output_dir / "perturbed_final.png", output_dir / "input_difference_final.png")

    # Do not write replay .pt tensors by default; previous FACE runs showed
    # these are easy to push accidentally. JSON diagnostics below are enough
    # for report/debug metadata.
    write_json(output_dir / "geometry_params_final.json", {"limits": geometry.limits_dict(), "parameter_diagnostics": geometry.parameter_diagnostics(), "last_projection": projection})
    write_json(output_dir / "geometry_params_best.json", {"best_iter_by_Z": best["row"]["iter"], "best_Z": best["row"]["Z"]})
    np.save(output_dir / "embedding_perturbed_edit_final_gradient_path.npy", arcface.embedding(final_edit_tensor_gradient_path).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_best_gradient_path.npy", arcface.embedding(best["perturbed_edit"]).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_final.npy", arcface.embedding(stock_final_tensor).detach().cpu().numpy().astype("float32"))
    np.save(output_dir / "embedding_perturbed_edit_best.npy", arcface.embedding(stock_best_tensor).detach().cpu().numpy().astype("float32"))

    edit_metadata: dict[str, Any] = {
        **editor.metadata(),
        "separate_downstream_eval_used": False,
        "edit_outputs_used_in_loss": True,
        "public_edit_images_regenerated_with_stock_pipeline": True,
        "public_edit_images_source": "StableDiffusionInstructPix2PixPipeline.__call__ under torch.inference_mode",
        "clean_edit_path": "original_edited.png",
        "clean_edit_gradient_reference_path": "original_edited_gradient_reference.png",
        "perturbed_best_edit_path": "perturbed_best_edited.png",
        "perturbed_final_edit_path": "perturbed_final_edited.png",
        "perturbed_best_edit_gradient_path": "perturbed_best_edited_gradient_path.png",
        "perturbed_final_edit_gradient_path": "perturbed_final_edited_gradient_path.png",
        "best_selection_source": "minimum differentiable gradient-path Z; public image is stock replay of that selected input",
    }

    input_metrics_best = image_metrics(original, best_perturbed)
    input_metrics_final = image_metrics(original, final_perturbed)
    output_metrics_best = image_metrics(Image.open(output_dir / "original_edited.png"), Image.open(output_dir / "perturbed_best_edited.png"))
    output_metrics_final = image_metrics(Image.open(output_dir / "original_edited.png"), Image.open(output_dir / "perturbed_final_edited.png"))
    save_sheet(
        output_dir / "comparison_sheet.png",
        [
            ("Original", original),
            ("Perturbed Best", best_perturbed),
            ("Abs Difference x8", Image.open(output_dir / "input_difference_best.png")),
            ("Combined Flow", Image.open(output_dir / "combined_flow_best.png")),
            ("DCT Difference x10", Image.open(output_dir / "dct_only_difference_x10.png")),
            ("Clean Edit", Image.open(output_dir / "original_edited.png")),
            ("Perturbed Edit", Image.open(output_dir / "perturbed_best_edited.png")),
        ],
    )

    if not cfg.skip_deepface:
        from face4.evaluation.deepface_panel import run_deepface_panel, write_identity_panel

        panel_rows = run_deepface_panel(output_dir)
        write_identity_panel(output_dir, panel_rows)
    else:
        panel_rows = []
        write_json(output_dir / "identity_panel.json", [{"status": "skipped"}])
        (output_dir / "identity_panel.csv").write_text("status\nskipped\n", encoding="utf-8")

    write_csv(output_dir / "history.csv", rows)
    elapsed = time.monotonic() - started
    final_row = rows[-1]
    optimization_rows = [row for row in rows if int(row.get("iter", 0)) > 0]
    summary = {
        "status": "done",
        "experiment": "edited_output_identity",
        "model": spec.model,
        "face_id": spec.case.face_id,
        "prompt": spec.case.prompt,
        "case_id": spec.case.slug,
        "seed": spec.run_seed,
        "iters": cfg.iters,
        "Z_definition": "ArcFace cosine similarity between original edited output and perturbed edited output",
        "loss": "loss = Z",
        "arcface_objective_prompt_conditioned": True,
        "differentiable_instructpix2pix_in_gradient_loop": True,
        "editor_num_inference_steps": cfg.edit_steps,
        "editor_guidance_scale": cfg.guidance_scale,
        "editor_image_guidance_scale": cfg.image_guidance_scale,
        "final_Z": float(final_Z_stock_public.detach().float().cpu()),
        "final_loss": float(final_Z_stock_public.detach().float().cpu()),
        "best_iter_by_Z": best["row"]["iter"],
        "best_Z": float(best_saved_stock_public_Z.detach().float().cpu()),
        "final_Z_stock_public": float(final_Z_stock_public.detach().float().cpu()),
        "best_Z_stock_public": float(best_Z_stock_public.detach().float().cpu()),
        "stock_public_Z_at_gradient_best": float(best_Z_stock_public.detach().float().cpu()),
        "best_saved_stock_public_Z": float(best_saved_stock_public_Z.detach().float().cpu()),
        "best_saved_stock_public_source": best_saved_stock_public_source,
        "best_saved_stock_public_iter": best_saved_stock_public_iter,
        "final_Z_gradient_path": float(final_Z_gradient_path.detach().float().cpu()),
        "best_Z_gradient_path": best["row"]["Z"],
        "public_edit_images_regenerated_with_stock_pipeline": True,
        "clean_stock_vs_gradient_reference_ssim": clean_stock_vs_gradient_reference["ssim"],
        "best_gradient_path_vs_stock_public_ssim": best_gradient_vs_stock["ssim"],
        "final_gradient_path_vs_stock_public_ssim": final_gradient_vs_stock["ssim"],
        "final_stock_public_identity_cosine_similarity_raw": float(_float_terms(final_terms_stock_public)["identity_cosine_similarity_raw"]),
        "final_stock_public_identity_similarity_score_pct": float(_float_terms(final_terms_stock_public)["identity_similarity_score_pct"]),
        "best_stock_public_identity_cosine_similarity_raw": float(_float_terms(best_terms_stock_public)["identity_cosine_similarity_raw"]),
        "best_stock_public_identity_similarity_score_pct": float(_float_terms(best_terms_stock_public)["identity_similarity_score_pct"]),
        "final_identity_cosine_similarity_raw": float(_float_terms(final_terms_stock_public)["identity_cosine_similarity_raw"]),
        "final_identity_similarity_score_pct": float(_float_terms(final_terms_stock_public)["identity_similarity_score_pct"]),
        "best_identity_cosine_similarity_raw": float(_float_terms(best_terms_stock_public)["identity_cosine_similarity_raw"]),
        "best_identity_similarity_score_pct": float(_float_terms(best_terms_stock_public)["identity_similarity_score_pct"]),
        "final_edit_identity_cosine_similarity_raw": float(_float_terms(final_terms_stock_public)["identity_cosine_similarity_raw"]),
        "final_edit_identity_similarity_score_pct": float(_float_terms(final_terms_stock_public)["identity_similarity_score_pct"]),
        "best_edit_identity_cosine_similarity_raw": float(_float_terms(best_terms_stock_public)["identity_cosine_similarity_raw"]),
        "best_edit_identity_similarity_score_pct": float(_float_terms(best_terms_stock_public)["identity_similarity_score_pct"]),
        **best_input_identity,
        **final_input_identity,
        **original_vs_original_edit_identity,
        **perturbed_best_vs_perturbed_best_edit_identity,
        **perturbed_final_vs_perturbed_final_edit_identity,
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
        "best_stock_public_output_ssim": output_metrics_best["ssim"],
        "best_stock_public_output_psnr": output_metrics_best["psnr"],
        "best_stock_public_output_l2": output_metrics_best["l2"],
        "final_stock_public_output_ssim": output_metrics_final["ssim"],
        "final_stock_public_output_psnr": output_metrics_final["psnr"],
        "final_stock_public_output_l2": output_metrics_final["l2"],
        "final_combined_max_disp_px": final_row["combined_max_disp_px"],
        "final_combined_mean_disp_px": final_row["combined_mean_disp_px"],
        "final_combined_p95_disp_px": final_row["combined_p95_disp_px"],
        "final_dct_gain_mean_abs": final_row.get("dct_gain_mean_abs"),
        "final_dct_relative_energy_change": final_row.get("dct_relative_energy_change"),
        "final_dct_spatial_delta_mse": final_row.get("dct_spatial_delta_mse"),
        "final_dct_spatial_delta_l1": final_row.get("dct_spatial_delta_l1"),
        "final_dct_spatial_delta_max_abs": final_row.get("dct_spatial_delta_max_abs"),
        "dct_metadata": geometry.dct_image.metadata(),
        "final_fraction_clamped_total": final_row["fraction_clamped_total"],
        "all_required_history_fields_populated": _history_fields_ok(final_row),
        "clamp_project_logic_active": final_row["num_total_params"] > 0,
        "arcface": arcface.metadata(),
        "differentiable_instructpix2pix": edit_metadata,
        "deepface_rows": len(panel_rows),
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "run_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "DONE.json", {"status": "done", "elapsed_seconds": elapsed, "final_Z": summary["final_Z"]})
    return summary


def optimize_one(spec: RunSpec, cfg: RunConfig, arcface, editor, device, output_dir: Path) -> dict[str, Any]:
    """Dispatch the correctness-gated FACE4 runner."""

    from .runner_correct import optimize_one_correct

    return optimize_one_correct(spec, cfg, arcface, editor, device, output_dir)


def _time_estimates(summaries: list[dict[str, Any]], wall_seconds: float, observed_iters: int) -> dict[str, Any]:
    if not summaries:
        return {
            "estimates_valid": False,
            "invalid_reason": "no completed runs; timing estimates are unavailable",
            "observed_completed_runs": 0,
            "observed_mean_seconds_per_iteration_per_run": None,
            "estimated_full_matrix_seconds_per_iteration": None,
            "estimated_fixed_overhead_seconds": None,
            "estimated_runtime_seconds_for_50_iterations": None,
            "estimated_runtime_seconds_for_100_iterations": None,
            "estimated_runtime_seconds_for_150_iterations": None,
            "estimated_runtime_seconds_for_400_iterations": None,
        }
    mean_seconds = [float(row.get("mean_seconds_iter", 0.0)) for row in summaries]
    observed_iter_seconds = sum(mean_seconds)
    fixed_overhead = max(0.0, float(wall_seconds) - float(observed_iters) * observed_iter_seconds)
    completed = max(len(summaries), 1)
    scale_to_full = 4.0 / completed
    full_iter_seconds = observed_iter_seconds * scale_to_full
    full_overhead = fixed_overhead * scale_to_full
    return {
        "estimates_valid": True,
        "observed_completed_runs": len(summaries),
        "observed_mean_seconds_per_iteration_per_run": float(sum(mean_seconds) / max(len(mean_seconds), 1)),
        "estimated_full_matrix_seconds_per_iteration": full_iter_seconds,
        "estimated_fixed_overhead_seconds": full_overhead,
        "estimated_runtime_seconds_for_50_iterations": full_overhead + 50 * full_iter_seconds,
        "estimated_runtime_seconds_for_100_iterations": full_overhead + 100 * full_iter_seconds,
        "estimated_runtime_seconds_for_150_iterations": full_overhead + 150 * full_iter_seconds,
        "estimated_runtime_seconds_for_400_iterations": full_overhead + 400 * full_iter_seconds,
    }


def _write_top_summary(run_root: Path, cfg: RunConfig, started: float, summaries: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    wall = time.monotonic() - started
    status = "done" if not failures else "failed"
    estimates = _time_estimates(summaries, wall, cfg.iters)
    payload = {
        "status": status,
        "mode": cfg.mode,
        "experiment": "edited_output_identity",
        "iters": cfg.iters,
        "editor_num_inference_steps": cfg.edit_steps,
        "quick": cfg.quick,
        "all_cases": cfg.all_cases,
        "execution": "sequential",
        "wall_seconds": wall,
        "num_runs_attempted": len(summaries) + len(failures),
        "num_runs_completed": len(summaries),
        "num_failures": len(failures),
        "failures": failures,
        "summaries": summaries,
        "time_estimates": estimates,
        "peak_vram_gb": torch_peak_gb(),
        "nvidia_smi_memory_gb": nvidia_smi_memory_gb(),
        "all_per_iteration_logging_fields_populated": all(s.get("all_required_history_fields_populated", False) for s in summaries),
        "clamp_project_logic_active": all(s.get("clamp_project_logic_active", False) for s in summaries),
        "output_root": str(run_root),
    }
    write_json(run_root / "summary.json", payload)
    lines = [
        f"# FACE4 {cfg.mode} summary",
        "",
        f"- status: {status}",
        "- experiment: edited_output_identity",
        "- execution: sequential",
        f"- iterations per run: {cfg.iters}",
        f"- InstructPix2Pix edit steps in gradient loop: {cfg.edit_steps}",
        f"- runs attempted: {payload['num_runs_attempted']}",
        f"- runs completed: {payload['num_runs_completed']}",
        f"- failures: {payload['num_failures']}",
        f"- wall seconds: {wall:.2f}",
        f"- peak VRAM GB: {payload.get('peak_vram_gb')}",
        f"- all required per-iteration fields populated: {payload['all_per_iteration_logging_fields_populated']}",
        f"- clamp/project logic active: {payload['clamp_project_logic_active']}",
        "",
    ]
    if estimates.get("estimates_valid"):
        lines.insert(-4, f"- observed mean seconds/iteration/run: {estimates['observed_mean_seconds_per_iteration_per_run']:.3f}")
        lines.insert(-4, f"- estimated 150-iteration full matrix: {estimates['estimated_runtime_seconds_for_150_iterations'] / 60:.1f} min")
    else:
        lines.insert(-4, f"- timing estimates: unavailable ({estimates.get('invalid_reason')})")
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure.get('spec')}: {failure.get('error')}")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_matrix(cfg: RunConfig) -> dict[str, Any]:
    started = time.monotonic()
    label = "quick" if cfg.quick else "all"
    if cfg.resume_run_root:
        root = Path(cfg.resume_run_root)
        print(f"[face4] resuming explicit run root: {root}")
    elif cfg.resume_latest:
        root = _latest_run_root(Path(cfg.output_root))
        print(f"[face4] resuming latest run root under {cfg.output_root}: {root}")
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        root = Path(cfg.output_root) / f"{run_id}_edited_output_identity_{label}_sequential"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "launcher_config.json", asdict(cfg))
    print_resolved_cases(Path(cfg.mat_root))
    specs = build_matrix(quick=cfg.quick)
    if cfg.all_cases:
        specs = build_matrix(quick=False)
    device = torch_device()
    arcface = _arcface(device, cfg.arcface_checkpoint, cfg.source_url)
    editor = _editor(device, cfg)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = _run_dir(root, spec)
        try:
            if (cfg.resume_run_root or cfg.resume_latest) and run_dir.exists() and not (run_dir / "DONE.json").exists():
                _archive_incomplete_case_dir(run_dir)
            summaries.append(optimize_one(spec, cfg, arcface, editor, device, run_dir))
        except Exception as error:
            failures.append({"spec": spec.slug, "error": repr(error), "run_dir": str(run_dir)})
            write_json(run_dir / "FAILED.json", {"status": "failed", "error": repr(error)})
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if isinstance(error, CorrectnessGateError):
                print(f"[face4] systemic correctness gate failed; stopping matrix: {error}")
                break
    payload = _write_top_summary(root, cfg, started, summaries, failures)
    if failures:
        raise RuntimeError(f"FACE4 matrix completed with {len(failures)} failure(s); see {root / 'summary.json'}")
    return payload
