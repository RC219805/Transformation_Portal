"""Depth backend registry with license governance.

Factory for selecting depth backends by name with multi-layer license enforcement.
See ADR-019 for architectural rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from .protocol import DepthBackend, LicenseRestrictionError, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)


class DepthBackendRegistry:
    """Factory for depth backends with license governance.

    Provides centralized backend selection with:
    - Multi-layer license enforcement
    - Device auto-detection
    - Helpful error messages

    Usage:
        >>> from transformation_portal.depth.backends import DepthBackendRegistry
        >>> from transformation_portal.lux_depth_v3 import EnhanceConfig
        >>>
        >>> config = EnhanceConfig(
        ...     non_commercial_ok=True,
        ...     accept_apple_depth_pro_research_license=True,
        ... )
        >>> registry = DepthBackendRegistry()
        >>> backend = registry.get_backend("depth_pro", config)
    """

    # Registered backends (populated by register_backend)
    _backends: Dict[str, Type[DepthBackend]] = {}

    def __init__(self):
        """Initialize registry and ensure built-in backends are registered."""
        self._ensure_builtins_registered()

    def _ensure_builtins_registered(self) -> None:
        """Register built-in backends if not already registered."""
        # Lazy import to avoid circular dependencies
        try:
            from .da3 import DA3Backend

            if "da3" not in self._backends:
                self._backends["da3"] = DA3Backend
        except ImportError:
            logger.debug("DA3Backend not available (missing dependencies)")

        try:
            from .depth_pro import DepthProBackend

            if "depth_pro" not in self._backends:
                self._backends["depth_pro"] = DepthProBackend
        except ImportError:
            logger.debug("DepthProBackend not available (missing dependencies)")

        # Synthetic backend is always available (no ML dependencies)
        try:
            from .synthetic import SyntheticDepthBackend

            if "synthetic" not in self._backends:
                self._backends["synthetic"] = SyntheticDepthBackend
        except ImportError:
            logger.debug("SyntheticDepthBackend not available (unexpected)")

    @classmethod
    def register_backend(cls, backend_class: Type[DepthBackend]) -> None:
        """Register a depth backend class.

        Args:
            backend_class: Backend class implementing DepthBackend protocol.
        """
        cls._backends[backend_class.name] = backend_class
        logger.info(f"Registered depth backend: {backend_class.name}")

    def list_backends(self) -> Dict[str, Dict[str, Any]]:
        """List all registered backends with metadata.

        Returns:
            Dict mapping backend name to metadata (license_type, requires_checkpoint).
        """
        return {
            name: {
                "license_type": cls.license_type.value,
                "requires_checkpoint": cls.requires_checkpoint,
            }
            for name, cls in self._backends.items()
        }

    def get_backend_class(self, backend_id: str) -> Optional[Type[DepthBackend]]:
        """Get backend class by ID without instantiation.

        Public API for introspection/dependency checking without creating instances.

        Args:
            backend_id: Backend identifier (e.g., "da3", "depth_pro").

        Returns:
            Backend class if registered, None otherwise.
        """
        return self._backends.get(backend_id)

    def available_backend_ids(self) -> list[str]:
        """Get list of all registered backend IDs.

        Returns:
            Sorted list of backend identifiers.
        """
        return sorted(self._backends.keys())

    def has_backend(self, backend_id: str) -> bool:
        """Check if backend is registered.

        Args:
            backend_id: Backend identifier to check.

        Returns:
            True if backend is registered.
        """
        return backend_id in self._backends

    def get_backend(
        self,
        backend_name: str,
        config: Optional["EnhanceConfig"] = None,
    ) -> DepthBackend:
        """Get depth backend with license validation.

        This is Layer 2 of license enforcement (factory level).
        Layer 1 is config validation, Layer 3 is runtime in backend.compute().

        Args:
            backend_name: Backend identifier (e.g., "depth_pro").
            config: EnhanceConfig for license validation and backend configuration.

        Returns:
            Instantiated backend.

        Raises:
            ValueError: If backend_name is unknown.
            LicenseRestrictionError: If license requirements not met.
        """
        backend_cls = self._backends.get(backend_name)
        if backend_cls is None:
            available = ", ".join(sorted(self._backends.keys())) or "(none)"
            raise ValueError(f"Unknown depth backend: '{backend_name}'. " f"Available backends: {available}")

        # Layer 2: License enforcement at factory level
        self._validate_license(backend_name, backend_cls, config)

        # Instantiate backend
        if config is not None:
            return backend_cls(config)
        else:
            return backend_cls()

    def _validate_license(
        self,
        backend_name: str,
        backend_cls: Type[DepthBackend],
        config: Optional["EnhanceConfig"],
    ) -> None:
        """Validate license requirements for backend.

        Args:
            backend_name: Backend identifier.
            backend_cls: Backend class.
            config: EnhanceConfig with license flags.

        Raises:
            LicenseRestrictionError: If license requirements not met.
        """
        if backend_cls.license_type != LicenseType.RESEARCH_ONLY:
            return  # Commercial backends have no restrictions

        if config is None:
            raise LicenseRestrictionError(f"Backend '{backend_name}' requires EnhanceConfig for license validation.")

        # Check non_commercial_ok flag
        if not getattr(config, "non_commercial_ok", False):
            raise LicenseRestrictionError(
                f"Backend '{backend_name}' is research-only and requires:\n"
                f"  non_commercial_ok=True\n\n"
                f"This flag acknowledges that the model cannot be used for:\n"
                f"  - Commercial products or services\n"
                f"  - Revenue-generating applications\n"
                f"  - Paid client work\n\n"
                f"Set non_commercial_ok=True in EnhanceConfig to proceed."
            )

        # Depth Pro specific: require explicit Apple license acceptance
        if backend_name == "depth_pro":
            if not getattr(config, "accept_apple_depth_pro_research_license", False):
                raise LicenseRestrictionError(
                    "Backend 'depth_pro' requires explicit license acceptance.\n\n"
                    "Set accept_apple_depth_pro_research_license=True to acknowledge:\n"
                    "  - Apple Machine Learning Research License (AMLR)\n"
                    "  - Research and non-commercial use only\n"
                    "  - No commercial exploitation or deployment\n\n"
                    "License details: https://github.com/apple/ml-depth-pro/blob/main/LICENSE\n\n"
                    "Required config:\n"
                    "  config = EnhanceConfig(\n"
                    "      non_commercial_ok=True,\n"
                    "      accept_apple_depth_pro_research_license=True,\n"
                    "  )"
                )

        logger.info(f"License validation passed for '{backend_name}' " f"(non_commercial_ok={config.non_commercial_ok})")


# Global registry instance for convenience
_default_registry: Optional[DepthBackendRegistry] = None


def get_registry() -> DepthBackendRegistry:
    """Get the default registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DepthBackendRegistry()
    return _default_registry
