"""Unified depth backend abstraction for Transformation Portal.

This module provides a unified interface for depth estimation backends
(Depth Anything V2/V3, Depth Pro) with consistent contracts, license
governance, and caching behavior.

Public API:
    - DepthBackend: Protocol for depth estimation backends
    - DepthResult: Unified result dataclass with metric depth support
    - DepthBackendRegistry: Factory for backend selection with license gates
    - LicenseType: Enum for license classification
    - LicenseRestrictionError: Exception for license violations
    - DepthCacheWriter: Enhanced caching with metadata sidecar

Example:
    >>> from transformation_portal.depth.backends import (
    ...     DepthBackendRegistry,
    ...     DepthResult,
    ... )
    >>> from transformation_portal.lux_depth_v3 import EnhanceConfig
    >>>
    >>> config = EnhanceConfig(
    ...     depth_backend="depth_pro",
    ...     non_commercial_ok=True,
    ...     accept_apple_depth_pro_research_license=True,
    ... )
    >>> registry = DepthBackendRegistry()
    >>> backend = registry.get_backend("depth_pro", config)
    >>> result = backend.compute(image)
    >>> print(f"Depth units: {result.depth_units}")

See Also:
    - ADR-019: Depth Backend Unification Architecture
    - docs/architecture/ADR-019-depth-backend-unification.md
"""

from .protocol import (
    DepthBackend,
    DepthResult,
    LicenseType,
    LicenseRestrictionError,
)
from .registry import DepthBackendRegistry
from .cache import DepthCacheWriter

__all__ = [
    "DepthBackend",
    "DepthResult",
    "DepthBackendRegistry",
    "LicenseType",
    "LicenseRestrictionError",
    "DepthCacheWriter",
]
