#!/usr/bin/env bash
set -euo pipefail

echo "[face-install] Python:"
python -V
echo "[face-install] Environment checks:"
python - <<'PY'
import importlib
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for name in ["torchvision", "diffusers", "transformers", "deepface", "cv2"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "available"))
    except Exception as exc:
        print(name, "missing", repr(exc))
PY

echo "[face4-install] Installing FACE4 without changing the shared MAT dependency versions."
python -m pip install -e . --no-deps
echo "[face4-install] If a dependency above is missing, run: python -m pip install -r requirements.txt"
