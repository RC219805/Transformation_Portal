"""Depth-intelligence package public API.

Currently exposes the Depth Anything V2 estimator surface
(``DepthEstimator``, ``DepthConfig``, ``DepthMap``). The broader
spatial-intelligence/atmospheric/depth-guided-filter components
described in earlier docs are not yet shipped — when those modules
land they should be added to ``_LAZY_EXPORTS`` and ``__all__``.

Symbols are exported lazily via PEP 562 (mirroring
``transformation_portal.depth.__init__``) so importing this package
does not eagerly pull in ``torch``/``numpy``/``PIL`` and minimal /
wheel-smoke environments can still ``import
transformation_portal.depth_intelligence`` without the full ML stack.

Usage:
    from transformation_portal.depth_intelligence import DepthEstimator
    estimator = DepthEstimator(...)
    result = estimator.estimate(image)
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "1.0.0"

__all__ = [
    "DepthEstimator",
    "DepthConfig",
    "DepthMap",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "DepthEstimator": (".depth_estimator", "DepthEstimator"),
    "DepthConfig": (".depth_estimator", "DepthConfig"),
    "DepthMap": (".depth_estimator", "DepthMap"),
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
