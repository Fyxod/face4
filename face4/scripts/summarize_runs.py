"""Summarize FACE edited-output ArcFace identity runs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageOps


TITLE = "FACE4: Edited-output ArcFace White-box Optimization"
SUBTITLE = "Exact stock-equivalent InstructPix2Pix edit identity results with geometric perturbations"
AUTHOR = "Parth Katiyar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FACE4 result folders.")
    parser.add_argument("--results-root", default="outputs/edited_output_identity_exact")
    parser.add_argument("--output-root", default="outputs/reports/edited_output_identity_exact")
    parser.add_argument("--run-folder", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--compress-images", action="store_true")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation.")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}"


def slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_", "/", "."}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_run_root(results_root: Path) -> Path:
    candidates = []
    for child in sorted(results_root.iterdir()) if results_root.exists() else []:
        if not child.is_dir():
            continue
        if list(child.glob("runs/edited_output_identity/instructpix2pix_arcface_iresnet100/*/summary.json")):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No FACE4 run roots found under {results_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_run_root(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root)
    if args.run_root:
        path = Path(args.run_root)
        if path.exists():
            return path
        print(f"[face-report] requested run root missing, falling back to latest: {path}")
    if args.run_folder:
        path = results_root / args.run_folder
        if path.exists():
            return path
        print(f"[face-report] requested run folder missing, falling back to latest: {path}")
    return find_latest_run_root(results_root)


def collect_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for summary_path in sorted(run_root.glob("runs/edited_output_identity/instructpix2pix_arcface_iresnet100/*/summary.json")):
        run_dir = summary_path.parent
        summary = read_json(summary_path)
        config = read_json(run_dir / "config_resolved.json") if (run_dir / "config_resolved.json").exists() else {}
        history_rows = read_csv_rows(run_dir / "history.csv")
        spec = config.get("spec", {})
        face_id = str(summary.get("face_id") or spec.get("face_id") or run_dir.name.split("__")[0])
        prompt = str(summary.get("prompt") or spec.get("prompt") or "")
        images = {
            "original": run_dir / "original.png",
            "perturbed_best": run_dir / "perturbed_best.png",
            "input_difference": run_dir / "input_difference_best.png",
            "combined_flow": run_dir / "combined_flow_best.png",
            "clean_edit": run_dir / "original_edited.png",
            "perturbed_edit": run_dir / "perturbed_best_edited.png",
            "comparison_sheet": run_dir / "comparison_sheet.png",
        }
        for label, path in images.items():
            if not path.exists():
                missing.append({"case": f"{face_id} / {prompt}", "artifact": label, "path": str(path)})
        runs.append(
            {
                "face_id": face_id,
                "prompt": prompt,
                "case": f"{face_id} / {prompt}",
                "case_slug": slug(f"{face_id}_{prompt}"),
                "run_dir": run_dir.as_posix(),
                "summary": summary,
                "config": config,
                "history_rows": history_rows,
                "images": images,
            }
        )
    return runs, missing


def make_strip(run: dict[str, Any], output_root: Path, compress: bool) -> str:
    out_dir = output_root / "strips"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (360, 360) if compress else (512, 512)
    diff_dir = output_root / "edit_diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)

    def edit_difference(left_path: Path, right_path: Path, name: str) -> Path:
        out = diff_dir / f"{name}_{run['case_slug']}.jpg"
        if not left_path.exists() or not right_path.exists():
            Image.new("RGB", size, "#f3f4f6").save(out, quality=82, optimize=True)
            return out
        left = Image.open(left_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        right = Image.open(right_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        diff = ImageChops.difference(left, right)
        diff = ImageEnhance.Brightness(diff).enhance(8.0)
        diff = ImageOps.autocontrast(diff, cutoff=0.5)
        diff.save(out, quality=86, optimize=True)
        return out

    original_edit_diff = edit_difference(run["images"]["clean_edit"], run["images"]["original"], "original_edit_minus_original")
    perturbed_edit_diff = edit_difference(
        run["images"]["perturbed_edit"],
        run["images"]["perturbed_best"],
        "perturbed_edit_minus_perturbed",
    )
    perturbed_edit_original_edit_diff = edit_difference(
        run["images"]["perturbed_edit"],
        run["images"]["clean_edit"],
        "perturbed_edit_minus_original_edit",
    )
    labels = [
        ("Original", run["images"]["original"]),
        ("Perturbed Best", run["images"]["perturbed_best"]),
        ("Abs Difference x8", run["images"]["input_difference"]),
        ("Clean Edit", run["images"]["clean_edit"]),
        ("Original Edit - Original", original_edit_diff),
        ("Perturbed Edit", run["images"]["perturbed_edit"]),
        ("Perturbed Edit - Perturbed", perturbed_edit_diff),
        ("Perturbed Edit - Original Edit", perturbed_edit_original_edit_diff),
    ]
    cells = []
    for label, path in labels:
        if path.exists():
            img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", size, "#f3f4f6")
        cells.append((label, img))
    label_h = 34
    canvas = Image.new("RGB", (size[0] * len(cells), size[1] + label_h), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(cells):
        x = idx * size[0]
        canvas.paste(img, (x, 0))
        draw.text((x + 8, size[1] + 9), label, fill="black")
    ext = "jpg" if compress else "png"
    path = out_dir / f"face_{run['case_slug']}.{ext}"
    if compress:
        canvas.save(path, quality=82, optimize=True)
    else:
        canvas.save(path, optimize=True)
    return path.relative_to(output_root).as_posix()


def plot_lines(path: Path, title: str, ylabel: str, runs: list[dict[str, Any]], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=125)
    for run in runs:
        xs, ys = [], []
        for row in run["history_rows"]:
            x = to_float(row.get("iter"))
            y = to_float(row.get(key))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            plt.plot(xs, ys, linewidth=1.8, label=f"{run['face_id']} / {run['prompt'].replace('add ', '')}")
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(path: Path, title: str, runs: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 5.5), dpi=125)
    for run in runs:
        summary = run["summary"]
        x = to_float(summary.get(x_key))
        y = to_float(summary.get(y_key))
        if x is not None and y is not None:
            plt.scatter([x], [y], s=70, label=f"{run['face_id']} / {run['prompt'].replace('add ', '')}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _mean_series_by_iter(runs: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    values_by_iter: dict[float, list[float]] = defaultdict(list)
    for run in runs:
        for row in run["history_rows"]:
            x = to_float(row.get("iter"))
            y = to_float(row.get(key))
            if x is not None and y is not None:
                values_by_iter[x].append(y)
    xs, ys = [], []
    for x in sorted(values_by_iter):
        vals = values_by_iter[x]
        if vals:
            xs.append(x)
            ys.append(sum(vals) / len(vals))
    return xs, ys


def component_enabled(run: dict[str, Any], component: str) -> bool:
    """Return whether a geometry component was enabled for this saved run.

    Prefer the immutable resolved configuration saved with the run.  Older
    result folders may not contain it, so fall back to the per-iteration
    ``*_enabled`` diagnostic already written to history.csv.
    """
    resolved = run.get("config", {}).get("geometry_config_resolved", {})
    config_value = resolved.get(f"{component}_enabled")
    if config_value is not None:
        return bool(config_value)
    history_key = f"{component}_enabled"
    return any((to_float(row.get(history_key)) or 0.0) > 0.5 for row in run.get("history_rows", []))


def plot_mean_geometry_lines(
    path: Path,
    title: str,
    ylabel: str,
    runs: list[dict[str, Any]],
    keys: list[tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=125)
    plotted = False
    for key, label, component in keys:
        enabled_runs = [run for run in runs if component_enabled(run, component)]
        xs, ys = _mean_series_by_iter(enabled_runs, key)
        if xs:
            plotted = True
            plt.plot(xs, ys, linewidth=2.0, label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def any_history_key(runs: list[dict[str, Any]], key: str) -> bool:
    return any(to_float(row.get(key)) is not None for run in runs for row in run["history_rows"])


def summary_value(summary: dict[str, Any], preferred: str, fallback: str) -> Any:
    value = summary.get(preferred)
    return summary.get(fallback) if value is None or value == "" else value


def best_saved_stock_z(summary: dict[str, Any]) -> Any:
    explicit = summary.get("best_saved_stock_public_Z")
    if explicit is not None and explicit != "":
        return explicit
    final_z = to_float(summary.get("final_Z_stock_public") or summary.get("final_Z"))
    best_stock = to_float(summary.get("best_Z_stock_public") or summary.get("best_Z"))
    if final_z is not None and best_stock is not None:
        return min(final_z, best_stock)
    return summary.get("best_Z")


def plot_components(path: Path, runs: list[dict[str, Any]]) -> None:
    keys = ["tps_mean_disp", "delaunay_mean_disp", "rolling_mean_disp"]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.2), dpi=125)
    for key in keys:
        xs, ys = [], []
        for run in runs:
            values = [to_float(row.get(key)) for row in run["history_rows"]]
            values = [v for v in values if v is not None]
            if values:
                xs.append(run["case"].replace("add ", ""))
                ys.append(values[-1])
        if ys:
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=key)
    plt.title("Final component diagnostics by run")
    plt.ylabel("raw diagnostic value")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_graphs(runs: list[dict[str, Any]], output_root: Path) -> list[dict[str, str]]:
    graph_dir = output_root / "graphs"
    graphs = []
    specs = [
        ("Z vs iteration", "Z", "Z", "z_vs_iteration.png"),
        ("Loss vs iteration", "loss", "loss", "loss_vs_iteration.png"),
        ("PSNR to original vs iteration", "psnr_to_original", "PSNR", "psnr_vs_iteration.png"),
        ("SSIM to original vs iteration", "ssim_to_original", "SSIM", "ssim_vs_iteration.png"),
        (
            "Input ArcFace identity similarity vs iteration",
            "input_identity_similarity_score_pct",
            "original vs perturbed identity similarity (%)",
            "input_identity_similarity_vs_iteration.png",
        ),
    ]
    for title, key, ylabel, name in specs:
        if key == "input_identity_similarity_score_pct" and not any_history_key(runs, key):
            continue
        path = graph_dir / name
        plot_lines(path, title, ylabel, runs, key)
        graphs.append({"title": title, "path": path.relative_to(output_root).as_posix()})
    component_path = graph_dir / "geometry_component_diagnostics_vs_iteration.png"
    plot_mean_geometry_lines(
        component_path,
        "Geometry component diagnostics vs iteration",
        "raw diagnostic value",
        runs,
        [
            ("tps_mean_disp", "TPS", "tps"),
            ("delaunay_mean_disp", "Delaunay", "delaunay"),
            ("rolling_mean_disp", "Rolling shutter", "rolling"),
            ("dct_gain_mean_abs", "DCT gain", "dct"),
            ("fft_phase_mean_abs", "FFT phase", "fft_phase"),
            ("polar_mean_disp", "Polar", "polar"),
            ("bspline_mean_disp", "B-spline / Bezier", "bspline"),
            ("lens_barrel_mean_disp", "Lens barrel", "lens_barrel"),
            ("lens_pincushion_mean_disp", "Lens pincushion", "lens_pincushion"),
            ("mobius_mean_disp", "Mobius", "mobius"),
            ("laplacian_mean_disp", "Laplacian", "laplacian"),
            ("geodesic_mean_disp", "Geodesic", "geodesic"),
            (
                "differential_surface_mean_disp",
                "Differential surface",
                "differential_surface",
            ),
        ],
    )
    graphs.append({"title": "Geometry component diagnostics vs iteration", "path": component_path.relative_to(output_root).as_posix()})
    return graphs


def table_html(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    for _, label in cols:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _ in cols:
            value = row.get(key, "")
            parts.append(f"<td>{html.escape(fmt(value) if isinstance(value, (int, float)) else str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def build_tables(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run = []
    identity_rows = []
    for run in runs:
        summary = run["summary"]
        per_run.append(
            {
                "face_id": run["face_id"],
                "prompt": run["prompt"],
                "final_Z": summary.get("final_Z"),
                "best_Z": summary.get("best_Z"),
                "final_Z_stock_public": summary.get("final_Z_stock_public"),
                "best_Z_stock_public": summary.get("best_Z_stock_public"),
                "final_Z_stock_gap": summary.get("final_Z_stock_gap"),
                "best_Z_stock_gap": summary.get("best_Z_stock_gap"),
                "best_iter_by_Z": summary.get("best_iter_by_Z"),
                "final_identity_cosine_similarity_raw": summary_value(
                    summary, "final_stock_public_identity_cosine_similarity_raw", "final_identity_cosine_similarity_raw"
                ),
                "final_identity_similarity_score_pct": summary_value(
                    summary, "final_stock_public_identity_similarity_score_pct", "final_identity_similarity_score_pct"
                ),
                "ssim_to_original": summary.get("final_ssim_to_original"),
                "psnr_to_original": summary.get("final_psnr_to_original"),
                "best_input_identity_similarity_score_pct": summary.get("best_input_identity_similarity_score_pct"),
                "final_input_identity_similarity_score_pct": summary.get("final_input_identity_similarity_score_pct"),
                "original_vs_original_edit_identity_similarity_score_pct": summary.get("original_vs_original_edit_identity_similarity_score_pct"),
                "perturbed_best_vs_perturbed_best_edit_identity_similarity_score_pct": summary.get(
                    "perturbed_best_vs_perturbed_best_edit_identity_similarity_score_pct"
                ),
                "perturbed_final_vs_perturbed_final_edit_identity_similarity_score_pct": summary.get(
                    "perturbed_final_vs_perturbed_final_edit_identity_similarity_score_pct"
                ),
                "output_ssim": summary_value(summary, "best_stock_public_output_ssim", "best_output_ssim"),
                "output_l2": summary_value(summary, "best_stock_public_output_l2", "best_output_l2"),
                "max_disp_px": summary.get("final_combined_max_disp_px"),
                "dct_gain_mean_abs": summary.get("final_dct_gain_mean_abs"),
                "dct_energy_change": summary.get("final_dct_relative_energy_change"),
                "dct_spatial_delta_mse": summary.get("final_dct_spatial_delta_mse"),
                "fraction_clamped": summary.get("final_fraction_clamped_total"),
                "seconds_per_iter": summary.get("mean_seconds_iter"),
                "public_edits_stock_regenerated": summary.get("public_edit_images_regenerated_with_stock_pipeline", False),
                "run_dir": run["run_dir"],
            }
        )
        panel_path = Path(run["run_dir"]) / "identity_panel.csv"
        for row in read_csv_rows(panel_path):
            row = dict(row)
            row["face_id"] = run["face_id"]
            row["prompt"] = run["prompt"]
            identity_rows.append(row)
    return per_run, identity_rows


def build_html(data: dict[str, Any]) -> str:
    per_cols = [
        ("face_id", "face"),
        ("prompt", "prompt"),
        ("final_Z", "final Z"),
        ("best_Z", "best Z"),
        ("final_Z_stock_public", "final stock Z"),
        ("best_Z_stock_public", "best stock Z"),
        ("best_Z_stock_gap", "best Z gap"),
        ("final_identity_cosine_similarity_raw", "final edit cosine sim"),
        ("final_identity_similarity_score_pct", "final edit score %"),
        ("original_vs_original_edit_identity_similarity_score_pct", "orig vs orig-edit %"),
        ("perturbed_best_vs_perturbed_best_edit_identity_similarity_score_pct", "perturbed vs best-edit %"),
        ("best_input_identity_similarity_score_pct", "orig vs perturbed %"),
        ("ssim_to_original", "SSIM"),
        ("psnr_to_original", "PSNR"),
        ("output_ssim", "edit output SSIM"),
        ("max_disp_px", "max disp px"),
        ("dct_gain_mean_abs", "DCT gain mean abs"),
        ("dct_energy_change", "DCT energy change"),
        ("fraction_clamped", "fraction clamped"),
        ("public_edits_stock_regenerated", "stock public edits"),
    ]
    css = """
    body { margin:0; font-family: Inter, "Segoe UI", Arial, sans-serif; color:#17202a; background:white; }
    main { max-width:1180px; margin:0 auto; padding:34px 28px 70px; }
    h1 { font-size:34px; margin:0 0 6px; }
    h2 { margin-top:48px; padding-top:18px; border-top:2px solid #d7dde5; }
    h3 { margin-top:30px; }
    .subtitle,.small { color:#5d6d7e; }
    .card { border:1px solid #d7dde5; border-radius:12px; padding:18px; margin:20px 0; background:white; }
    table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0 22px; }
    th,td { border:1px solid #d7dde5; padding:7px 9px; vertical-align:top; }
    th { background:#f6f8fb; text-align:left; }
    .strip { width:100%; border:1px solid #d7dde5; border-radius:10px; display:block; }
    .graph-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }
    figure { border:1px solid #d7dde5; border-radius:10px; padding:12px; margin:0; }
    figure img { width:100%; display:block; }
    figcaption { font-weight:650; margin-bottom:10px; }
    .path { font-family:Consolas, monospace; font-size:12px; word-break:break-all; }
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>FACE4 report</title><style>",
        css,
        "</style></head><body><main>",
        f"<h1>{html.escape(TITLE)}</h1><p class='subtitle'>{html.escape(SUBTITLE)}</p><p class='small'>Author: {html.escape(AUTHOR)}</p>",
        "<div class='card'><p>FACE4 optimizes <code>Z = cosine_similarity</code> between frozen ArcFace iResNet-100 embeddings of the clean InstructPix2Pix edit and the perturbed InstructPix2Pix edit. The loss is exactly <code>loss = Z</code>. InstructPix2Pix and ArcFace weights are frozen; only perturbation parameters are optimized. TPS, Delaunay, and rolling are coordinate perturbations. DCT is a blockwise image-frequency coefficient perturbation, not a flow field.</p></div>",
        "<div class='card'><p>The iteration graph shows the exact stock-equivalent <code>Z</code> used for optimization. Normal decorated stock-pipeline Z is retained as a parity measurement; the configured correctness gate stops the run if the two values diverge beyond tolerance.</p></div>",
        "<h2>1. Run matrix</h2><p>Four prompt-labeled cases are retained. The prompt conditions the differentiable InstructPix2Pix edit inside the optimization objective.</p>",
        table_html(data["per_run_rows"], per_cols),
        "<h2>2. Case image strips</h2>",
    ]
    for run in data["runs"]:
        parts.append(
            f"<div class='card'><h3>{html.escape(run['face_id'])} — {html.escape(run['prompt'])}</h3>"
            f"<img class='strip' src='{html.escape(run['strip_path'])}' alt='strip'>"
            f"<p class='path'>{html.escape(run['run_dir'])}</p></div>"
        )
    parts.append("<h2>3. Graphs</h2><div class='graph-grid'>")
    for graph in data["graphs"]:
        parts.append(f"<figure><figcaption>{html.escape(graph['title'])}</figcaption><a href='{html.escape(graph['path'])}'><img src='{html.escape(graph['path'])}'></a></figure>")
    parts.append("</div></main></body></html>")
    return "\n".join(parts)


def build_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"# {TITLE}",
        "",
        SUBTITLE,
        "",
        "FACE4 optimizes `Z = cosine_similarity(ArcFace(original_edit), ArcFace(perturbed_edit))` with `loss = Z`. DCT is reported as an image-frequency coefficient perturbation, not a spatial flow.",
        "",
        "The Z iteration graph is the exact stock-equivalent objective used for optimization. Decorated stock-pipeline Z is reported separately as a parity check.",
        "",
        "## Image strips",
        "",
    ]
    for run in data["runs"]:
        lines.extend([f"### {run['face_id']} / {run['prompt']}", "", f"![strip]({run['strip_path']})", ""])
    lines.extend(["## Graphs", ""])
    for graph in data["graphs"]:
        lines.extend([f"### {graph['title']}", "", f"![{graph['title']}]({graph['path']})", ""])
    return "\n".join(lines)


def make_pdf(data: dict[str, Any], output_root: Path, pdf_path: Path, compress_images: bool) -> None:
    """Build a WOOD-style PDF report with tables, strips, and graphs."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    story: list[Any] = []

    def p(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 0.08 * inch))

    def add_table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], font_size: int = 6) -> None:
        if not rows:
            p("No rows available.")
            return
        table_data = [[label for _, label in cols]]
        for row in rows:
            table_data.append(
                [
                    fmt(row.get(key)) if isinstance(row.get(key), (int, float)) else str(row.get(key, ""))
                    for key, _ in cols
                ]
            )
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.14 * inch))

    def pdf_image_path(rel_path: str, max_px: tuple[int, int]) -> Path | None:
        path = output_root / rel_path
        if not path.exists():
            return None
        if not compress_images:
            return path
        out_dir = output_root / "pdf_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{slug(rel_path)}.jpg"
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            img.thumbnail(max_px, Image.Resampling.LANCZOS)
            img.save(out, quality=72, optimize=True)
        return out

    def add_image(rel_path: str, max_w: float = 7.2 * inch, max_h: float = 4.7 * inch) -> None:
        max_px = (int(max_w / inch * 145), int(max_h / inch * 145))
        path = pdf_image_path(rel_path, max_px)
        if path is None:
            return
        with Image.open(path) as img:
            w, h = img.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        story.append(RLImage(str(path), width=w * scale, height=h * scale))
        story.append(Spacer(1, 0.14 * inch))

    per_cols = [
        ("face_id", "face"),
        ("prompt", "prompt"),
        ("final_Z", "final Z"),
        ("best_Z", "best Z"),
        ("final_Z_stock_public", "final stock Z"),
        ("best_Z_stock_public", "best stock Z"),
        ("best_Z_stock_gap", "best Z gap"),
        ("final_identity_similarity_score_pct", "final edit %"),
        ("original_vs_original_edit_identity_similarity_score_pct", "orig/orig-edit %"),
        ("perturbed_best_vs_perturbed_best_edit_identity_similarity_score_pct", "pert/best-edit %"),
        ("best_input_identity_similarity_score_pct", "orig/pert %"),
        ("ssim_to_original", "SSIM"),
        ("psnr_to_original", "PSNR"),
        ("output_ssim", "edit SSIM"),
        ("max_disp_px", "max disp"),
        ("dct_gain_mean_abs", "DCT gain"),
        ("dct_energy_change", "DCT energy"),
        ("public_edits_stock_regenerated", "stock public edits"),
    ]

    story.append(Paragraph(TITLE, styles["Title"]))
    p(SUBTITLE, "Heading2")
    p(f"Author: {AUTHOR}")
    p(
        "FACE4 optimizes <b>Z = cosine_similarity</b> between ArcFace embeddings of "
        "the clean InstructPix2Pix edit and the perturbed InstructPix2Pix edit. "
        "The optimized loss is exactly <b>loss = Z</b>. InstructPix2Pix and "
        "ArcFace weights are frozen; only perturbation parameters are optimized."
    )
    p(
        "TPS, Delaunay, and rolling are coordinate perturbations. DCT is a blockwise "
        "image-frequency coefficient perturbation."
    )
    p(
        "The Z curve is the exact stock-equivalent objective used for optimization. "
        "Decorated stock-pipeline Z is reported separately as a parity measurement."
    )
    p("Run matrix and final values", "Heading2")
    add_table(data["per_run_rows"], per_cols, font_size=5)

    story.append(PageBreak())
    p("Case image strips", "Heading2")
    for run in data["runs"]:
        p(f"{run['face_id']} / {run['prompt']}", "Heading3")
        add_image(run["strip_path"], max_w=7.2 * inch, max_h=2.4 * inch)

    story.append(PageBreak())
    p("Graphs", "Heading2")
    for graph in data["graphs"]:
        p(graph["title"], "Heading3")
        add_image(graph["path"], max_w=7.1 * inch, max_h=4.2 * inch)

    doc.build(story)


def main() -> None:
    args = parse_args()
    run_root = resolve_run_root(args)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs, missing = collect_runs(run_root)
    for run in runs:
        run["strip_path"] = make_strip(run, output_root, args.compress_images)
    graphs = make_graphs(runs, output_root)
    per_run_rows, identity_rows = build_tables(runs)
    write_csv(output_root / "per_run_final_values.csv", per_run_rows)
    write_csv(output_root / "aggregate_summary.csv", per_run_rows)
    write_csv(output_root / "identity_panel_all_runs.csv", identity_rows)
    (output_root / "missing_artifacts.md").write_text(
        "# Missing artifacts\n\n" + ("\n".join(f"- {m['case']}: {m['artifact']} ({m['path']})" for m in missing) if missing else "None.\n"),
        encoding="utf-8",
    )
    (output_root / "image_index.md").write_text("\n".join(f"- {run['case']}: {run['strip_path']}" for run in runs) + "\n", encoding="utf-8")
    data = {
        "runs": runs,
        "run_root": run_root.as_posix(),
        "missing": missing,
        "graphs": graphs,
        "per_run_rows": per_run_rows,
    }
    (output_root / "report.html").write_text(build_html(data), encoding="utf-8")
    (output_root / "report.md").write_text(build_markdown(data), encoding="utf-8")
    if not args.no_pdf:
        make_pdf(data, output_root, output_root / "report.pdf", bool(args.compress_images))
    (output_root / "report_data_summary.json").write_text(
        json.dumps(
            {
                "run_root": run_root.as_posix(),
                "num_runs": len(runs),
                "num_missing": len(missing),
                "graphs": graphs,
                "compress_images": bool(args.compress_images),
                "html": "report.html",
                "markdown": "report.md",
                "pdf": None if args.no_pdf else "report.pdf",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[face-report] run root: {run_root}")
    print(f"[face-report] wrote: {output_root / 'report.html'}")
    print(f"[face-report] wrote: {output_root / 'report.md'}")
    if not args.no_pdf:
        print(f"[face-report] wrote: {output_root / 'report.pdf'}")


if __name__ == "__main__":
    main()
