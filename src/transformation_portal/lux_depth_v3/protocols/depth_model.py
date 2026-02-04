"""DepthModel Protocol - Unified interface for depth model backends.

This module defines the protocol (interface) that all depth model backends
must implement. This enables:
- Hot-swappable backends without changing downstream code
- License-aware routing
- Consistent provenance tracking
- Circuit breaker patterns for fallback

Protocol Version: 1.0.0
Compatible with: v2.0.0 Golden Path

Example:
    >>> from transformation_portal.lux_depth_v3.protocols import DepthModel, BackendRole
    >>> class MyDepthBackend(DepthModel):
    ...     def load(self, device: str, weights_path: Optional[Path]) -> None:
    ...         ...
    ...     def predict(self, image: np.ndarray) -> DepthArtifact:
    ...         ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Type, runtime_checkable

import numpy as np

from ..contracts.depth_artifact import DepthArtifact, LicenseTier

logger = logging.getLogger(__name__)

# Protocol version for compatibility checks
DEPTH_MODEL_PROTOCOL_VERSION = "1.0.0"


class BackendRole(Enum):
    """Backend role classification for routing decisions.

    Roles define the intended use case for each depth backend:
    - DRAFT: Fast preview, lower quality, commercial-safe (e.g., DA2-Small)
    - PRODUCTION: Default commercial backend, balanced quality/speed
    - VIDEO: Temporal consistency optimized for video sequences
    - AUDIT: Metric depth for measurement/compliance (often research-only)
    """

    DRAFT = auto()
    PRODUCTION = auto()
    VIDEO = auto()
    AUDIT = auto()


class BackendCapability(Enum):
    """Capabilities that a depth backend may support."""

    RELATIVE_DEPTH = auto()  # Normalized depth [0, 1]
    METRIC_DEPTH = auto()  # Absolute depth in meters
    CONFIDENCE_MAP = auto()  # Per-pixel uncertainty
    VIDEO_STREAMING = auto()  # Temporal consistency / streaming
    BATCH_INFERENCE = auto()  # Multiple images at once
    INTRINSICS_ESTIMATION = auto()  # Camera intrinsics from image


@dataclass(frozen=True)
class BackendInfo:
    """Metadata describing a depth model backend.

    Attributes:
        name: Human-readable backend name
        model_id: HuggingFace model ID or internal identifier
        role: Backend role (DRAFT, PRODUCTION, VIDEO, AUDIT)
        license_tier: License classification
        capabilities: Set of supported capabilities
        min_version: Minimum required package version (if applicable)
        checkpoint_size_mb: Approximate checkpoint size in MB
        description: Human-readable description
    """

    name: str
    model_id: str
    role: BackendRole
    license_tier: LicenseTier
    capabilities: frozenset[BackendCapability] = field(default_factory=lambda: frozenset({BackendCapability.RELATIVE_DEPTH}))
    min_version: Optional[str] = None
    checkpoint_size_mb: Optional[float] = None
    description: str = ""

    def supports(self, capability: BackendCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in self.capabilities

    def is_commercial_safe(self) -> bool:
        """Check if backend is safe for commercial use."""
        return self.license_tier == LicenseTier.COMMERCIAL


@runtime_checkable
class DepthModel(Protocol):
    """Protocol defining the interface for depth model backends.

    All depth backends must implement this protocol to be used with the
    EnhanceOrchestrator. The protocol enforces:

    1. **load()**: Lazy model loading with license validation
    2. **predict()**: Single image inference returning DepthArtifact
    3. **info**: Backend metadata for routing decisions

    Optional methods:
    - **stream_video()**: Temporal-consistent video inference
    - **predict_batch()**: Batch inference for throughput

    Example Implementation:
        ```python
        class DA3ProductionBackend:
            @property
            def info(self) -> BackendInfo:
                return BackendInfo(
                    name="Depth Anything V3 Production",
                    model_id="depth-anything/DA3-Large",
                    role=BackendRole.PRODUCTION,
                    license_tier=LicenseTier.COMMERCIAL,
                )

            def load(
                self,
                device: str = "auto",
                weights_path: Optional[Path] = None,
                strict_license: bool = True,
            ) -> None:
                # Load model weights...

            def predict(self, image: np.ndarray) -> DepthArtifact:
                # Run inference and return DepthArtifact...
        ```
    """

    @property
    def info(self) -> BackendInfo:
        """Return backend metadata.

        This property provides routing information including:
        - License tier for compliance checks
        - Capabilities for feature detection
        - Role for use-case matching
        """
        ...

    def load(
        self,
        device: str = "auto",
        weights_path: Optional[Path] = None,
        strict_license: bool = True,
    ) -> None:
        """Load model weights with license validation.

        This method must be called before predict(). It handles:
        - Device selection (auto, cpu, mps, cuda)
        - Weight loading (local path or HuggingFace download)
        - License validation (fail if strict and license violated)

        Args:
            device: Target device ("auto", "cpu", "mps", "cuda")
            weights_path: Optional local path to model weights
            strict_license: If True, fail on license violations

        Raises:
            LicenseRestrictionError: If strict_license and license violated
            FileNotFoundError: If weights_path specified but not found
            RuntimeError: If model loading fails
        """
        ...

    def predict(self, image: np.ndarray) -> DepthArtifact:
        """Run depth inference on a single image.

        Args:
            image: Input RGB image as numpy array (H, W, 3), uint8 or float32

        Returns:
            DepthArtifact with depth_map and provenance

        Raises:
            RuntimeError: If model not loaded or inference fails
            ValueError: If image format is invalid
        """
        ...

    # Optional methods (check hasattr before calling)

    def stream_video(
        self,
        frames: Iterator[np.ndarray],
    ) -> Iterator[DepthArtifact]:
        """Stream temporally-consistent depth for video.

        This optional method provides frame-to-frame consistency for video
        processing. Implementations should maintain internal state for
        temporal smoothing.

        Args:
            frames: Iterator yielding RGB frames as numpy arrays

        Yields:
            DepthArtifact for each input frame

        Note:
            Check hasattr(backend, 'stream_video') before calling.
            Not all backends support video streaming.
        """
        ...

    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[DepthArtifact]:
        """Run batch inference on multiple images.

        This optional method enables throughput optimization for batch
        processing workflows.

        Args:
            images: List of RGB images as numpy arrays

        Returns:
            List of DepthArtifact, one per input image

        Note:
            Check hasattr(backend, 'predict_batch') before calling.
            Not all backends support batch inference.
        """
        ...


class DepthModelRegistry:
    """Registry for managing available depth model backends.

    The registry provides:
    - Backend registration and discovery
    - License-aware routing
    - Role-based backend selection
    - Fallback chain management

    Example:
        >>> registry = DepthModelRegistry()
        >>> registry.register(DA3ProductionBackend)
        >>> backend = registry.get_backend(
        ...     role=BackendRole.PRODUCTION,
        ...     commercial_only=True,
        ... )
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._backends: Dict[str, Type[DepthModel]] = {}
        self._instances: Dict[str, DepthModel] = {}
        self._fallback_chains: Dict[BackendRole, List[str]] = {}

    def register(
        self,
        backend_class: Type[DepthModel],
        name: Optional[str] = None,
    ) -> None:
        """Register a depth model backend.

        Args:
            backend_class: Class implementing DepthModel protocol
            name: Optional registration name (defaults to class name)

        Raises:
            TypeError: If backend_class doesn't implement DepthModel
        """
        if not isinstance(backend_class, type):
            raise TypeError("backend_class must be a class")

        # Validate protocol compliance
        # Require core DepthModel surface: info, load, predict
        instance = backend_class()
        required_attrs = ("load", "predict")
        missing_or_invalid = [
            attr for attr in required_attrs if not hasattr(instance, attr) or not callable(getattr(instance, attr, None))
        ]
        # Check for info property separately (can be property or method)
        if not hasattr(instance, "info"):
            missing_or_invalid.append("info")

        if missing_or_invalid:
            raise TypeError(
                f"{backend_class.__name__} does not implement DepthModel protocol "
                f"(missing or non-callable: {', '.join(missing_or_invalid)})"
            )

        reg_name = name or backend_class.__name__
        self._backends[reg_name] = backend_class
        logger.info("Registered depth backend: %s", reg_name)

    def list_backends(
        self,
        role: Optional[BackendRole] = None,
        commercial_only: bool = False,
    ) -> List[BackendInfo]:
        """List available backends matching criteria.

        Args:
            role: Filter by backend role
            commercial_only: If True, only return commercially-licensed backends

        Returns:
            List of BackendInfo for matching backends
        """
        results = []
        for name, cls in self._backends.items():
            instance = cls()
            info = instance.info

            if role is not None and info.role != role:
                continue

            if commercial_only and not info.is_commercial_safe():
                continue

            results.append(info)

        return results

    def get_backend(
        self,
        name: Optional[str] = None,
        role: Optional[BackendRole] = None,
        commercial_only: bool = False,
        use_cache: bool = True,
    ) -> DepthModel:
        """Get a depth model backend instance.

        Args:
            name: Specific backend name (takes precedence over role)
            role: Backend role to match
            commercial_only: If True, only return commercially-licensed backends
            use_cache: If True, return cached instance if available

        Returns:
            DepthModel instance

        Raises:
            KeyError: If no matching backend found
            ValueError: If commercial_only is True and the selected backend is not commercially licensed
        """
        # Direct name lookup
        if name is not None:
            if name not in self._backends:
                raise KeyError(f"Backend '{name}' not registered")

            if use_cache and name in self._instances:
                return self._instances[name]

            instance = self._backends[name]()

            if commercial_only and not instance.info.is_commercial_safe():
                raise ValueError(
                    f"Backend '{name}' requires non-commercial license " f"(tier: {instance.info.license_tier.value})"
                )

            if use_cache:
                self._instances[name] = instance
            return instance

        # Role-based lookup
        if role is not None:
            candidates = self.list_backends(role=role, commercial_only=commercial_only)
            if not candidates:
                raise KeyError(f"No backend found for role={role.name}, commercial_only={commercial_only}")

            # Return first matching (priority order determined by registration)
            for info in candidates:
                for reg_name, cls in self._backends.items():
                    if cls().info.model_id == info.model_id:
                        return self.get_backend(
                            name=reg_name,
                            commercial_only=commercial_only,
                            use_cache=use_cache,
                        )

        raise KeyError("Must specify either 'name' or 'role'")

    def set_fallback_chain(
        self,
        role: BackendRole,
        backend_names: List[str],
    ) -> None:
        """Set fallback chain for a backend role.

        When a backend fails, the orchestrator can fall back to alternatives
        in the specified order.

        Args:
            role: Backend role to configure
            backend_names: Ordered list of backend names to try

        Raises:
            KeyError: If any backend name not registered
        """
        for name in backend_names:
            if name not in self._backends:
                raise KeyError(f"Backend '{name}' not registered")

        self._fallback_chains[role] = backend_names
        logger.info(
            "Set fallback chain for %s: %s",
            role.name,
            " -> ".join(backend_names),
        )

    def get_fallback_chain(self, role: BackendRole) -> List[str]:
        """Get fallback chain for a backend role.

        Args:
            role: Backend role

        Returns:
            Ordered list of backend names
        """
        return self._fallback_chains.get(role, [])


# Global registry instance
_global_registry: Optional[DepthModelRegistry] = None


def get_registry() -> DepthModelRegistry:
    """Get the global depth model registry.

    Returns:
        Global DepthModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DepthModelRegistry()
    return _global_registry


def register_backend(
    name: Optional[str] = None,
) -> Callable[[Type[DepthModel]], Type[DepthModel]]:
    """Decorator to register a depth model backend.

    Example:
        @register_backend("da3_production")
        class DA3ProductionBackend:
            ...
    """

    def decorator(cls: Type[DepthModel]) -> Type[DepthModel]:
        get_registry().register(cls, name)
        return cls

    return decorator
