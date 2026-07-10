"""Runtime and CUDA helpers."""
from __future__ import annotations


def torch_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FACE white-box runs require CUDA. Run this in the A6000 environment.")
    return torch.device("cuda:0")


def torch_peak_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None
