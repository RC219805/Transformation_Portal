"""Depth model wrappers.

Expose model helpers lazily so importing ``transformation_portal.depth.models``
does not eagerly pull in optional ML stacks like torch-backed depth models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CoreMLDepthModel": (".coreml_wrapper", "CoreMLDepthModel"),
    "DepthAnythingV2Model": (".depth_anything_v2", "DepthAnythingV2Model"),
    "ModelBackend": (".depth_anything_v2", "ModelBackend"),
    "ModelVariant": (".depth_anything_v2", "ModelVariant"),
    # CoreML exports are implemented in the lux_depth_v3 CoreML backend.
    # Keep these names as lazy aliases to preserve the public import surface
    # without requiring a local coreml_exporter module.
    "CoreMLExporter": (
        "transformation_portal.lux_depth_v3.coreml_backend",
        "CoreMLDepthEstimator",
    ),
    "CoreMLDepthEstimator": (
        "transformation_portal.lux_depth_v3.coreml_backend",
        "CoreMLDepthEstimator",
    ),
}

__all__ = [
    "DepthAnythingV2Model",
    "ModelBackend",
    "ModelVariant",
    "CoreMLDepthModel",
    "CoreMLExporter",
    "CoreMLDepthEstimator",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
