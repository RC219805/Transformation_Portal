"""Upscaler backend registry.

Factory for selecting upscaler backends with graceful fallback.
Follows the pattern established by DepthBackendRegistry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from .protocol import UpscalerBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class UpscalerRegistry:
    """Factory for upscaler backends.

    Provides centralized backend selection with:
    - Graceful fallback to bicubic if ML dependencies missing
    - Device auto-detection
    - Helpful error messages

    Usage:
        >>> from transformation_portal.upscaling import UpscalerRegistry
        >>> registry = UpscalerRegistry()
        >>> backend = registry.get("bicubic")  # Always available
        >>> backend = registry.get("realesrgan", device="cuda")  # ML backend
    """

    # Registered backends (populated by register_backend)
    _backends: Dict[str, Type[UpscalerBackend]] = {}

    def __init__(self):
        """Initialize registry and ensure built-in backends are registered."""
        self._ensure_builtins_registered()

    def _ensure_builtins_registered(self) -> None:
        """Register built-in backends if not already registered."""
        # Bicubic is always available (no ML dependencies)
        try:
            from .backends.bicubic import BicubicUpscaler

            if "bicubic" not in self._backends:
                self._backends["bicubic"] = BicubicUpscaler
        except ImportError:
            logger.warning("BicubicUpscaler not available (unexpected)")

        # Real-ESRGAN is optional (requires ML dependencies)
        try:
            from .backends.realesrgan import RealESRGANUpscaler

            if "realesrgan" not in self._backends:
                self._backends["realesrgan"] = RealESRGANUpscaler
        except ImportError:
            logger.debug("RealESRGANUpscaler not available (missing ML dependencies)")

    @classmethod
    def register_backend(cls, backend_class: Type[UpscalerBackend]) -> None:
        """Register an upscaler backend class.

        Args:
            backend_class: Backend class implementing UpscalerBackend protocol.
        """
        cls._backends[backend_class.name] = backend_class
        logger.info(f"Registered upscaler backend: {backend_class.name}")

    def list_backends(self) -> Dict[str, Dict[str, Any]]:
        """List all registered backends with metadata.

        Returns:
            Dict mapping backend name to metadata (requires_ml).
        """
        result = {}
        for name, cls in self._backends.items():
            # Instantiate temporarily to check properties
            try:
                if name == "bicubic":
                    instance = cls()
                else:
                    # ML backends might fail if deps missing, that's ok
                    try:
                        instance = cls(device="cpu")
                    except Exception:
                        # Can't instantiate, assume requires ML
                        result[name] = {"requires_ml": True}
                        continue

                result[name] = {
                    "requires_ml": instance.requires_ml,
                }
            except Exception:
                # Fallback for any instantiation errors
                result[name] = {"requires_ml": True}

        return result

    def get_backend_class(self, backend_id: str) -> Optional[Type[UpscalerBackend]]:
        """Get backend class by ID without instantiation.

        Public API for introspection/dependency checking without creating instances.

        Args:
            backend_id: Backend identifier (e.g., "bicubic", "realesrgan").

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

    def get(
        self,
        backend_name: str,
        device: str = "cpu",
        fallback_to_bicubic: bool = True,
        **kwargs,
    ) -> UpscalerBackend:
        """Get upscaler backend with optional fallback.

        Args:
            backend_name: Backend identifier (e.g., "bicubic", "realesrgan").
            device: Device to use (cpu, cuda, mps).
            fallback_to_bicubic: If True, fallback to bicubic on error.
            **kwargs: Additional backend-specific arguments (e.g., model, half_precision).

        Returns:
            Instantiated upscaler backend.

        Raises:
            ValueError: If backend_name is unknown and fallback disabled.
            ImportError: If backend requires missing ML dependencies and fallback disabled.
        """
        # Normalize backend name
        backend_name = backend_name.lower()

        # Handle "default" alias
        if backend_name == "default":
            backend_name = "bicubic"

        # Check if backend is registered
        backend_cls = self._backends.get(backend_name)
        if backend_cls is None:
            available = ", ".join(sorted(self._backends.keys())) or "(none)"
            msg = f"Unknown upscaler backend: '{backend_name}'. Available backends: {available}"

            if fallback_to_bicubic and "bicubic" in self._backends:
                logger.warning(f"{msg}. Falling back to bicubic.")
                backend_cls = self._backends["bicubic"]
                backend_name = "bicubic"
            else:
                raise ValueError(msg)

        # Try to instantiate backend
        try:
            if backend_name == "bicubic":
                # Bicubic has no __init__ parameters
                return backend_cls()
            else:
                # ML backends take device and optional kwargs
                return backend_cls(device=device, **kwargs)

        except ImportError as e:
            # ML dependencies missing
            msg = f"Backend '{backend_name}' requires ML dependencies: {e}"

            if fallback_to_bicubic and "bicubic" in self._backends:
                logger.warning(f"{msg}. Falling back to bicubic.")
                return self._backends["bicubic"]()
            else:
                raise

        except Exception as e:
            # Other errors
            msg = f"Failed to initialize backend '{backend_name}': {e}"

            if fallback_to_bicubic and "bicubic" in self._backends:
                logger.warning(f"{msg}. Falling back to bicubic.")
                return self._backends["bicubic"]()
            else:
                raise


# Global registry instance for convenience
_default_registry: Optional[UpscalerRegistry] = None


def get_registry() -> UpscalerRegistry:
    """Get the default registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = UpscalerRegistry()
    return _default_registry
