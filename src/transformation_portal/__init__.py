"""
Transformation Portal - Professional Image and Video Processing Toolkit

A comprehensive suite of AI-powered tools and pipelines for luxury real estate
rendering, architectural visualization, and editorial post-production.

Key Components:
- Pipelines: High-level processing workflows (Lux Render, Depth Processing)
- Processors: Core image/video processing engines (Material Response, TIFF, Video)
- Enhancers: Specialized enhancement tools (Aerial, Board Material)
- Analyzers: Code quality and workflow analysis tools
- Rendering: Rendering workflow utilities
- Utils: Shared utilities and helpers
"""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

_PACKAGE_NAME = "transformation-portal"
_FALLBACK_VERSION = "0.1.0"


def _read_pyproject_version() -> str:
    """Resolve the package version directly from the repo source of truth."""

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def _resolve_runtime_version() -> str:
    """Prefer installed package metadata and fall back to pyproject in source trees."""

    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        try:
            return _read_pyproject_version()
        except (FileNotFoundError, KeyError, OSError, tomllib.TOMLDecodeError):
            # Last-resort fallback for source-tree execution when pyproject cannot be read.
            return _FALLBACK_VERSION


# Runtime version - pyproject.toml is the source of truth for source-tree execution.
__version__ = _resolve_runtime_version()

__author__ = "RC219805"

# Lazy imports for commonly used components
# This reduces initial import time while maintaining convenience


def _lazy_import(module_path, attr_name):
    """Lazy import helper to defer loading until needed."""

    def _loader():
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    return _loader


# Pipelines (lazy loaded)
_lux_render = None
_depth_tools = None

# Processors (lazy loaded)
_material_response = None
_video_grader = None


def get_lux_render_pipeline():
    """Get the Lux Render Pipeline (lazy loaded)."""
    global _lux_render
    if _lux_render is None:
        from .pipelines import lux_render_pipeline as _lux_render
    return _lux_render


def get_material_response():
    """Get Material Response processor (lazy loaded)."""
    global _material_response
    if _material_response is None:
        from .processors.material_response import core as _material_response
    return _material_response


# Convenience exports for backward compatibility
__all__ = [
    "__version__",
    "__author__",
    "get_lux_render_pipeline",
    "get_material_response",
]

# Expose submodules lazily for Sphinx autodoc and type checkers
# PEP 562: module __getattr__ allows lazy submodule loading
# This avoids eager importing heavy dependencies (torch, transformers, etc.)
# while maintaining transformation_portal.submodule access for docs/autocomplete
import importlib
from typing import Any

_LAZY_SUBMODULES = {
    "config_loader",
    "scene_types",
    "cli",
    "lux_depth_v3",
    "metrics",
    "enhancers",
    "rendering",
    "interfaces",
    "utils",
}


def __getattr__(name: str) -> Any:
    """Lazy-load submodules on attribute access."""
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose submodules in dir() for autocomplete/introspection."""
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))
