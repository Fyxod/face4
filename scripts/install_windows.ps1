$ErrorActionPreference = "Stop"
Write-Host "[face-install] Python:"
python -V
Write-Host "[face-install] Environment checks:"
@'
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
'@ | python -

Write-Host "[face4-install] Installing FACE4 without changing existing dependency versions."
python -m pip install -e . --no-deps
Write-Host "[face4-install] If a dependency above is missing, run: python -m pip install -r requirements.txt"
