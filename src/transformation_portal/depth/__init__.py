"""Depth package public API.

This module exposes depth pipeline symbols lazily so importing
``transformation_portal.depth`` does not eagerly load optional ML stacks.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "1.0.0"
__author__ = "Transformation Portal"

__all__ = [
    "ArchitecturalDepthPipeline",
    "DepthAnythingV2Model",
    "DepthCache",
    "DepthAwareDofOptions",
    "DepthAwareDofResult",
    "run_depth_aware_dof",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ArchitecturalDepthPipeline": (".pipeline", "ArchitecturalDepthPipeline"),
    "DepthAnythingV2Model": (".models.depth_anything_v2", "DepthAnythingV2Model"),
    "DepthCache": (".utils.cache", "DepthCache"),
    "DepthAwareDofOptions": (".depth_aware_dof", "DepthAwareDofOptions"),
    "DepthAwareDofResult": (".depth_aware_dof", "DepthAwareDofResult"),
    "run_depth_aware_dof": (".depth_aware_dof", "run_depth_aware_dof"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = target
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
