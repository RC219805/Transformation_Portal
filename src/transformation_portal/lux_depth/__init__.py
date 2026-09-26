"""Unified LuxDepth operator API, retaining each engine's native authority.

Importing this package never imports numerical libraries or model runtimes.
Requests are aliases of the existing immutable carriers, not new wire schemas.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PhotographyRequest": ("lux_depth_v6.managed", "ManagedLuxDepthV6Request"),
    "InferenceRequest": ("lux_depth_v5.lifecycle", "LuxDepthV5Request"),
    "FinishingRequest": ("lux_depth_v6.plan", "LuxDepthV6Request"),
    "DepthProRequest": ("lux_depth_v6.depth_pro", "NativeDepthProRequest"),
    "GradeRecipe": ("lux_depth_v6.color", "GradeRecipe"),
    "RenderRecipe": ("lux_depth_v6.color", "RenderRecipe"),
    "DepthMapRecipe": ("lux_depth_v6.depth_maps", "DepthMapRecipe"),
    "SourceLimits": ("lux_depth_v6.source", "SourceLimits"),
    "OutputLimits": ("lux_depth_v6.plan", "OutputLimits"),
    "VerifiedLuxDepth": ("lux_depth.lifecycle", "VerifiedLuxDepth"),
    "prepare": ("lux_depth.lifecycle", "prepare"),
    "run": ("lux_depth.lifecycle", "run"),
    "verify": ("lux_depth.lifecycle", "verify"),
    "result_summary": ("lux_depth.lifecycle", "result_summary"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"transformation_portal.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
