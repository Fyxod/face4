"""Validate or optionally download an ArcFace iResNet-100 checkpoint."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch

from face4.models.arcface import ArcFaceIResNet100, sha256_file, write_setup_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup/validate ArcFace iResNet-100 checkpoint for face4.")
    parser.add_argument("--checkpoint-path", default="models/arcface/iresnet100.pth")
    parser.add_argument("--download-url", default=None, help="Optional documented checkpoint URL. No default download is attempted.")
    parser.add_argument("--source-name", default="user_provided_local_checkpoint")
    parser.add_argument("--report-path", default="outputs/smoke/arcface_setup_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.checkpoint_path)
    report_path = Path(args.report_path)
    try:
        if not path.exists() and args.download_url:
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[face-setup] downloading checkpoint from {args.download_url}")
            urllib.request.urlretrieve(args.download_url, path)
        if not path.exists():
            payload = {
                "status": "missing_checkpoint",
                "checkpoint_path": str(path),
                "message": "Place a pretrained ArcFace iResNet-100 checkpoint at this path or rerun with --download-url.",
            }
            write_setup_report(report_path, payload)
            raise FileNotFoundError(payload["message"])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ArcFaceIResNet100(path, device, source_url=args.download_url or args.source_name)
        payload = {
            "status": "ok",
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "source": args.download_url or args.source_name,
            "arcface": model.metadata(),
            "random_weights_used": False,
            "parameters_frozen": all(not p.requires_grad for p in model.model.parameters()),
        }
        write_setup_report(report_path, payload)
        print(f"[face-setup] checkpoint ok: {path}")
        print(f"[face-setup] sha256: {payload['checkpoint_sha256']}")
        print(f"[face-setup] report: {report_path}")
    except Exception as error:
        if not report_path.exists():
            write_setup_report(report_path, {"status": "failed", "checkpoint_path": str(path), "error": repr(error)})
        print(f"[face-setup] failed: {error!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
