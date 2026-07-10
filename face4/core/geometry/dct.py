"""Deprecated compatibility wrapper for the legacy DCT coordinate warp.

The active FACE pipeline uses `face4.core.geometry.dct_image` for a true
blockwise image-domain DCT coefficient perturbation. This module is kept
only so old imports fail less abruptly; it is not imported by the active
pipeline.
"""
from __future__ import annotations

from .dct_warp_legacy import dct_basis

__all__ = ["dct_basis"]
