"""Depth model wrappers.

Expose model helpers lazily so importing ``transformation_portal.depth.models``
does not eagerly pull in optional ML stacks like torch-backed depth models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


class CoreMLExporter:
    """Compatibility shim for the removed optional CoreML exporter surface.

    The historical ``coreml_exporter`` module is not shipped in this repository.
    Keep the symbol importable so existing imports fail with a clear message
    instead of silently aliasing to an unrelated estimator class.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ModuleNotFoundError(
            "CoreMLExporter is not bundled in transformation_portal.depth.models. "
            "Use transformation_portal.lux_depth_v3.coreml_backend.CoreMLDepthEstimator "
            "or an explicit coremltools export flow instead."
        )


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CoreMLDepthModel": (".coreml_wrapper", "CoreMLDepthModel"),
    "DepthAnythingV2Model": (".depth_anything_v2", "DepthAnythingV2Model"),
    "ModelBackend": (".depth_anything_v2", "ModelBackend"),
    "ModelVariant": (".depth_anything_v2", "ModelVariant"),
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
