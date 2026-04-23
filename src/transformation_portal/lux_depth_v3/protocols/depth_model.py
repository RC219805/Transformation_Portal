"""DepthModel Protocol - Unified interface for depth model backends.

This module defines the protocol (interface) that all depth model backends
must implement. This enables:

- Hot-swappable backends without changing downstream code
- License-aware routing
- Consistent provenance tracking
- Circuit breaker patterns for fallback

The protocol version is exposed as ``DEPTH_MODEL_PROTOCOL_VERSION``.

Example
-------

.. code-block:: python

    from transformation_portal.lux_depth_v3.protocols import (
        DepthModel, BackendRole,
    )

    class MyDepthBackend(DepthModel):
        def load(self, device: str, weights_path: Optional[Path]) -> None:
            ...

        def predict(self, image: np.ndarray) -> DepthArtifact:
            ...
"""

from __future__ import annotations

import inspect
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
    capabilities: frozenset[BackendCapability] = field(
        default_factory=lambda: frozenset({BackendCapability.RELATIVE_DEPTH}),
    )
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

    Example Implementation
    ----------------------
    ::

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
    """

    @property
    def info(self) -> BackendInfo:
        """Return backend metadata.

        This property provides routing information including:
        - License tier for compliance checks
        - Capabilities for feature detection
        - Role for use-case matching
        """
        return None  # type: ignore[return-value]

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
        return None

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
        return None  # type: ignore[return-value]

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
        return None  # type: ignore[return-value]

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
        return None  # type: ignore[return-value]


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
        self._backend_info: Dict[str, BackendInfo] = {}
        self._instances: Dict[str, DepthModel] = {}
        self._fallback_chains: Dict[BackendRole, List[str]] = {}

    @staticmethod
    def _resolve_backend_info(backend_class: Type[DepthModel]) -> BackendInfo:
        """Resolve class-available backend metadata without construction."""
        backend_info = getattr(backend_class, "backend_info", None)
        if isinstance(backend_info, BackendInfo):
            return backend_info
        if callable(backend_info):
            resolved = backend_info()
            if isinstance(resolved, BackendInfo):
                return resolved

        get_backend_info = getattr(backend_class, "get_backend_info", None)
        if callable(get_backend_info):
            resolved = get_backend_info()
            if isinstance(resolved, BackendInfo):
                return resolved

        info_attr = getattr(backend_class, "info", None)
        if isinstance(info_attr, BackendInfo):
            return info_attr

        raise TypeError(
            f"{backend_class.__name__} must expose BackendInfo via a class-level "
            "'backend_info' attribute/callable or get_backend_info() method"
        )

    @staticmethod
    def _validate_backend_class_surface(backend_class: Type[DepthModel]) -> None:
        required_methods = ("load", "predict")
        missing_or_invalid = [attr for attr in required_methods if not callable(getattr(backend_class, attr, None))]
        try:
            info_member = inspect.getattr_static(backend_class, "info")
        except AttributeError:
            info_member = None
        if not isinstance(info_member, (BackendInfo, property)):
            missing_or_invalid.append("info")

        if missing_or_invalid:
            raise TypeError(
                f"{backend_class.__name__} does not implement DepthModel protocol "
                "(missing or invalid: "
                f"{', '.join(missing_or_invalid)})"
            )

    @staticmethod
    def _validate_runtime_backend_info(
        name: str,
        registered_info: BackendInfo,
        instance: DepthModel,
        commercial_only: bool,
    ) -> None:
        """Validate runtime metadata before returning a constructed backend."""
        runtime_info = getattr(instance, "info", None)
        if not isinstance(runtime_info, BackendInfo):
            raise TypeError(f"Backend '{name}' instance.info must be a BackendInfo instance")

        if registered_info.model_id != runtime_info.model_id or registered_info.license_tier != runtime_info.license_tier:
            logger.warning(
                "Registered backend info for '%s' differs from instance.info "
                "(registered model_id=%s, tier=%s; runtime model_id=%s, tier=%s)",
                name,
                registered_info.model_id,
                registered_info.license_tier.value,
                runtime_info.model_id,
                runtime_info.license_tier.value,
            )

        if commercial_only and not runtime_info.is_commercial_safe():
            raise ValueError(f"Backend '{name}' requires non-commercial license " f"(tier: {runtime_info.license_tier.value})")

    def register(
        self,
        backend_class: Type[DepthModel],
        name: Optional[str] = None,
        info: Optional[BackendInfo] = None,
    ) -> None:
        """Register a depth model backend.

        Args:
            backend_class: Class implementing DepthModel protocol
            name: Optional registration name (defaults to class name)
            info: Optional class-available backend metadata override

        Raises:
            TypeError: If backend_class doesn't implement DepthModel
        """
        if not isinstance(backend_class, type):
            raise TypeError("backend_class must be a class")

        self._validate_backend_class_surface(backend_class)
        resolved_info = info or self._resolve_backend_info(backend_class)
        if not isinstance(resolved_info, BackendInfo):
            raise TypeError("backend metadata must be a BackendInfo instance")

        reg_name = name or backend_class.__name__
        self._backends[reg_name] = backend_class
        self._backend_info[reg_name] = resolved_info
        self._instances.pop(reg_name, None)
        logger.info("Registered depth backend: %s", reg_name)

    def list_backends(
        self,
        role: Optional[BackendRole] = None,
        commercial_only: bool = False,
    ) -> List[BackendInfo]:
        """List available backends matching criteria.

        Args:
            role: Filter by backend role
            commercial_only: If True, only return
                commercially-licensed backends

        Returns:
            List of BackendInfo for matching backends
        """
        results = []
        for info in self._backend_info.values():
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
            commercial_only: If True, only return
                commercially-licensed backends
            use_cache: If True, return cached instance
                if available

        Returns:
            DepthModel instance

        Raises:
            KeyError: If no matching backend found
            ValueError: If commercial_only and the
                selected backend is not commercial
        """
        # Direct name lookup
        if name is not None:
            if name not in self._backends:
                raise KeyError(f"Backend '{name}' not registered")

            info = self._backend_info[name]
            if commercial_only and not info.is_commercial_safe():
                raise ValueError(f"Backend '{name}' requires " "non-commercial license " f"(tier: {info.license_tier.value})")

            if use_cache and name in self._instances:
                instance = self._instances[name]
                self._validate_runtime_backend_info(name, info, instance, commercial_only)
                return instance

            instance = self._backends[name]()
            self._validate_runtime_backend_info(name, info, instance, commercial_only)

            if use_cache:
                self._instances[name] = instance
            return instance

        # Role-based lookup
        if role is not None:
            candidates = self.list_backends(
                role=role,
                commercial_only=commercial_only,
            )
            if not candidates:
                raise KeyError(f"No backend found for " f"role={role.name}, " f"commercial_only={commercial_only}")

            # Return first matching (priority order determined by registration)
            failures = []
            last_error: Optional[Exception] = None
            for info in candidates:
                for reg_name, registered_info in self._backend_info.items():
                    if registered_info.model_id == info.model_id:
                        try:
                            return self.get_backend(
                                name=reg_name,
                                commercial_only=commercial_only,
                                use_cache=use_cache,
                            )
                        except Exception as exc:
                            last_error = exc
                            failures.append(f"{reg_name}: {exc}")
                            logger.warning(
                                "Skipping depth backend '%s' during role lookup: %s",
                                reg_name,
                                exc,
                            )

            if failures:
                raise RuntimeError(
                    f"No constructable backend found for role={role.name}, "
                    f"commercial_only={commercial_only}; failures: {'; '.join(failures)}"
                ) from last_error

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
    info: Optional[BackendInfo] = None,
) -> Callable[[Type[DepthModel]], Type[DepthModel]]:
    """Decorator to register a depth model backend.

    Example:
        @register_backend("da3_production")
        class DA3ProductionBackend:
            ...
    """

    def decorator(cls: Type[DepthModel]) -> Type[DepthModel]:
        get_registry().register(cls, name, info=info)
        return cls

    return decorator
