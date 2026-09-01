"""Depth backend registry with license governance.

Factory for selecting depth backends by name
with multi-layer license enforcement.
See ADR-019 for architectural rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from ...lux_depth_v3._backend_contract import normalize_backend_id
from .protocol import DepthBackend, LicenseRestrictionError, LicenseType


class UnknownDepthBackendError(ValueError):
    """Requested depth backend is not present in the registry."""


if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig
    from ...lux_depth_v3.execution_lifecycle import BackendCandidateAuthority

logger = logging.getLogger(__name__)


class DepthBackendRegistry:
    """Factory for depth backends with license governance.

    Provides centralized backend selection with:
    - Multi-layer license enforcement
    - Device auto-detection
    - Helpful error messages

    Usage:
        >>> from transformation_portal.depth.backends import (
        ...     DepthBackendRegistry,
        ... )
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

    def __init__(self) -> None:
        """Initialize registry and ensure built-in backends are registered."""
        self._ensure_builtins_registered()

    def _ensure_builtins_registered(self) -> None:
        """Register built-in backends if not already registered."""
        # Lazy import to avoid circular dependencies
        try:
            from .da2 import DA2Backend

            if "da2" not in self._backends:
                self._backends["da2"] = DA2Backend
        except ImportError:
            logger.debug("DA2Backend not available (missing dependencies)")

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
            logger.debug(
                "DepthProBackend not available" " (missing dependencies)",
            )

        # Synthetic backend is always available (no ML dependencies)
        try:
            from .synthetic import SyntheticDepthBackend

            if "synthetic" not in self._backends:
                self._backends["synthetic"] = SyntheticDepthBackend
        except ImportError:
            logger.debug("SyntheticDepthBackend not available (unexpected)")

        # DepthCrafter temporal backend (ADR-026)
        try:
            from .depthcrafter import DepthCrafterBackend

            if "depthcrafter" not in self._backends:
                self._backends["depthcrafter"] = DepthCrafterBackend
        except ImportError:
            logger.debug(
                "DepthCrafterBackend not available" " (missing dependencies)",
            )

        # Ensemble backend (ADR-026)
        try:
            from .ensemble import DepthEnsembleBackend

            if "ensemble" not in self._backends:
                self._backends["ensemble"] = DepthEnsembleBackend
        except ImportError:
            logger.debug(
                "DepthEnsembleBackend not available" " (missing dependencies)",
            )

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
            Dict mapping backend name to metadata
            (license_type, requires_checkpoint).
        """
        return {
            name: {
                "license_type": cls.license_type.value,
                "requires_checkpoint": cls.requires_checkpoint,
            }
            for name, cls in self._backends.items()
        }

    def validate_backend_request(
        self,
        backend_id: str,
        config: Optional["EnhanceConfig"],
    ) -> str:
        """Validate a requested backend without instantiating it.

        Plan-time seam (P0-1, issue #2065): applies the same registry
        membership and license checks execution applies, so a plan cannot
        advertise a backend the runtime registry would reject. Returns the
        normalized backend id.

        Raises:
            UnknownDepthBackendError: If the backend is not registered.
            LicenseRestrictionError: If license requirements are not met.
        """
        normalized = normalize_backend_id(backend_id) or ""
        backend_cls = self._backends.get(normalized)
        if backend_cls is None:
            raise UnknownDepthBackendError(
                f"Unknown depth backend {backend_id!r}. Registered backends: {sorted(self._backends.keys())}."
            )
        self._validate_license(normalized, backend_cls, config)
        return normalized

    def get_backend_class(
        self,
        backend_id: str,
    ) -> Optional[Type[DepthBackend]]:
        """Get backend class by ID without instantiation.

        Public API for introspection/dependency
        checking without creating instances.

        Args:
            backend_id: Backend identifier (e.g., "da3", "depth_pro").

        Returns:
            Backend class if registered, None otherwise.
        """
        return self._backends.get(normalize_backend_id(backend_id) or "")

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
        normalized_backend_id = normalize_backend_id(backend_id)
        return bool(normalized_backend_id and normalized_backend_id in self._backends)

    def get_backend(
        self,
        backend_name: str,
        config: Optional["EnhanceConfig"] = None,
        *,
        candidate_authority: Optional["BackendCandidateAuthority"] = None,
        canonical_plan_bytes: Optional[bytes] = None,
    ) -> DepthBackend:
        """Get depth backend with license validation.

        This is Layer 2 of license enforcement (factory level).
        Layer 1 is config validation, Layer 3 is runtime in backend.compute().

        Args:
            backend_name: Backend identifier (e.g., "depth_pro").
            config: EnhanceConfig for license
                validation and backend configuration.
            candidate_authority: Exact immutable plan candidate selected by
                the lifecycle boundary. Must be paired with canonical bytes.
            canonical_plan_bytes: Exact canonical plan bytes sent to isolated
                workers. Must be paired with candidate authority.

        Returns:
            Instantiated backend.

        Raises:
            ValueError: If backend_name is unknown.
            LicenseRestrictionError: If license requirements not met.
        """
        normalized_backend_name = normalize_backend_id(backend_name)
        backend_cls = self._backends.get(normalized_backend_name or "")
        if backend_cls is None:
            available = ", ".join(sorted(self._backends.keys())) or "(none)"
            raise ValueError(f"Unknown depth backend:" f" '{backend_name}'." f" Available backends: {available}")

        if (candidate_authority is None) != (canonical_plan_bytes is None):
            raise ValueError("candidate_authority and canonical_plan_bytes must be provided together")
        if candidate_authority is not None:
            if type(canonical_plan_bytes) is not bytes or not canonical_plan_bytes:
                raise ValueError("canonical_plan_bytes must be non-empty immutable bytes")
            if normalize_backend_id(candidate_authority.backend_id) != normalized_backend_name:
                raise ValueError(
                    "Carried backend authority does not match requested backend "
                    f"({candidate_authority.backend_id!r} != {normalized_backend_name!r})"
                )
            if candidate_authority.constituent_backend_id is None:
                if candidate_authority.candidate_id != normalized_backend_name:
                    raise ValueError("Top-level candidate authority does not match the requested backend")
            elif normalize_backend_id(candidate_authority.constituent_backend_id) != normalized_backend_name:
                raise ValueError("Constituent authority does not match the requested backend")
            from ...lux_depth_v3.execution_lifecycle import backend_candidate_authority as select_candidate_authority
            from ...lux_depth_v3.execution_lifecycle import (
                consume_lux_worker_execution_plan,
            )

            carried_plan = consume_lux_worker_execution_plan(canonical_plan_bytes)
            reselected = select_candidate_authority(
                carried_plan,
                candidate_authority.candidate_id,
                model_backend_id=candidate_authority.constituent_backend_id,
            )
            if reselected != candidate_authority:
                raise ValueError("Carried backend authority does not match the exact canonical plan bytes")

        # Layer 2: License enforcement at factory level
        self._validate_license(normalized_backend_name or backend_name, backend_cls, config)

        # Instantiate backend. The canonical-aware built-ins accept the exact
        # carrier explicitly; legacy/custom backends retain their historical
        # constructor shape and receive inert provenance attributes only.
        if candidate_authority is not None:
            if normalized_backend_name in {"da2", "da3", "depth_pro", "ensemble"}:
                canonical_backend_cls: Any = backend_cls
                return canonical_backend_cls(
                    config,
                    candidate_authority=candidate_authority,
                    canonical_plan_bytes=canonical_plan_bytes,
                )
            backend = backend_cls(config) if config is not None else backend_cls()
            setattr(backend, "_candidate_authority", candidate_authority)
            setattr(backend, "_canonical_plan_bytes", canonical_plan_bytes)
            return backend
        if config is not None:
            return backend_cls(config)
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
            raise LicenseRestrictionError(f"Backend '{backend_name}' requires" " EnhanceConfig for license" " validation.")

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
            accept_apple = getattr(
                config,
                "accept_apple_depth_pro_research_license",
                False,
            )
            if not accept_apple:
                raise LicenseRestrictionError(
                    "Backend 'depth_pro' requires"
                    " explicit license acceptance.\n\n"
                    "Set accept_apple_depth_pro"
                    "_research_license=True to"  # noqa: E501
                    " acknowledge:\n"
                    "  - Apple Machine Learning"
                    " Research License (AMLR)\n"
                    "  - Research and non-commercial use only\n"
                    "  - No commercial exploitation or deployment\n\n"
                    "License details: https://"
                    "github.com/apple/ml-depth"
                    "-pro/blob/main/LICENSE\n\n"
                    "Required config:\n"
                    "  config = EnhanceConfig(\n"
                    "      non_commercial_ok=True,\n"
                    "      accept_apple_depth_pro_research_license=True,\n"
                    "  )"
                )

        # Ensemble specific: require research tools license (ADR-026)
        if backend_name == "ensemble":
            if not getattr(config, "accept_research_tools_license", False):
                raise LicenseRestrictionError(
                    "Backend 'ensemble' requires"
                    " APEX Research Ultra license"
                    " acceptance.\n\n"
                    "Set accept_research_tools_license=True to acknowledge:\n"
                    "  - APEX Research Ultra (ADR-026) umbrella license\n"
                    "  - Research and non-commercial use only\n"
                    "  - Experimental workflow (subject to change)\n\n"
                    "This enables multi-model ensemble with:\n"
                    "  - Depth Pro (Apple AMLR)\n"
                    "  - DA3 1.1 (CC BY-NC 4.0)\n"
                    "  - DepthCrafter (Apache 2.0)\n\n"
                    "See: docs/architecture/ADR-026-apex-research-ultra.md\n\n"
                    "Required config:\n"
                    "  config = EnhanceConfig(\n"
                    "      non_commercial_ok=True,\n"
                    "      accept_research_tools_license=True,\n"
                    "      spatial_ai_linear_ingest=True,\n"
                    "  )"
                )

        logger.info(
            "License validation passed for" " '%s' (non_commercial_ok=%s)",
            backend_name,
            config.non_commercial_ok,
        )


# Global registry instance for convenience
_default_registry: Optional[DepthBackendRegistry] = None


def get_registry() -> DepthBackendRegistry:
    """Get the default registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DepthBackendRegistry()
    return _default_registry
