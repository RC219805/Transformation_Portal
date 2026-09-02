"""Pipeline coordination for lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition.

This module provides:
- Backend selection and fallback logic
- Runtime backend chain resolution
- Model ID resolution for provenance
- ExecutionPlan and BackendSelection data classes

The pipeline coordinator handles:
1. Selecting depth backends based on availability and fallback rules
2. Building ordered fallback chains for runtime resilience
3. Resolving model identifiers for provenance tracking

Usage:
    from transformation_portal.lux_depth_v3.pipeline_coordinator import (
        PipelineCoordinator,
        BackendSelection,
        ExecutionPlan,
        resolve_runtime_backend_chain,
        select_backend,
    )

    # Using PipelineCoordinator class
    coordinator = PipelineCoordinator(config, registry)
    selection = coordinator.select_backend("da3")
    plan = coordinator.plan()

    # Using standalone functions
    chain = resolve_runtime_backend_chain("da3", config)
    model_id = default_model_id_for_backend("da3", model_variant)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.execution_plan import EXECUTION_COMPLETE, CanonicalExecutionPlan
from ..core.platform_matrix import CURRENT_PLATFORM, PlatformMatrix
from ..depth.backends.protocol import LicenseRestrictionError

# Use absolute import to avoid circular dependencies
from ._backend_contract import normalize_backend_id, normalize_backend_provenance, normalize_backend_sequence
from .config import EnhanceConfig, ModelVariant
from .config_resolver import (
    apply_effective_da3_runtime_config,
    apply_effective_depth_pro_runtime_config,
    resolve_effective_depth_pro_python_executable,
)
from .manifest import BackendSelectionMetadata
from .model_resolution import BackendCapabilityError, ModelLicenseError, ModelRequest, resolve_model_contract

logger = logging.getLogger(__name__)


def _carried_execution_plan(config: EnhanceConfig) -> Optional[CanonicalExecutionPlan]:
    """Return a validated native plan carried by a projected runtime config."""

    plan = getattr(config, "execution_plan_authority", None)
    if plan is None:
        return None
    if not isinstance(plan, CanonicalExecutionPlan) or plan.configuration_completeness != EXECUTION_COMPLETE:
        raise ValueError("execution_plan_authority must be an execution-complete CanonicalExecutionPlan")
    canonical_plan_bytes = getattr(config, "execution_plan_canonical_bytes", None)
    expected_bytes = plan.to_canonical_json().encode("utf-8")
    if type(canonical_plan_bytes) is not bytes or canonical_plan_bytes != expected_bytes:
        raise ValueError("execution_plan_authority must be paired with its exact canonical bytes")
    return plan


def _carried_candidate_context(
    config: EnhanceConfig,
    backend_id: str,
) -> Optional[tuple[EnhanceConfig, Any, bytes]]:
    """Project one exact candidate and its immutable backend carrier."""

    plan = _carried_execution_plan(config)
    if plan is None:
        return None
    from .execution_lifecycle import backend_candidate_authority, runtime_config_from_execution_plan

    authority = backend_candidate_authority(plan, backend_id)
    candidate_config = runtime_config_from_execution_plan(
        plan,
        candidate_authority=authority,
    )
    canonical_plan_bytes = getattr(config, "execution_plan_canonical_bytes")
    if candidate_config.execution_plan_canonical_bytes != canonical_plan_bytes:
        raise ValueError("Candidate projection changed the exact canonical execution-plan bytes")
    return candidate_config, authority, canonical_plan_bytes


def _carried_candidate_device(config: EnhanceConfig, backend_id: str) -> Optional[str]:
    """Return the exact planned device for one carried backend candidate."""

    plan = _carried_execution_plan(config)
    if plan is None:
        return None
    from .execution_lifecycle import backend_candidate_authority

    return backend_candidate_authority(plan, backend_id).device


def _apple_silicon_depth_pro_opt_in(config: EnhanceConfig) -> bool:
    """Return True when Apple Silicon should consider the Depth Pro lane."""
    if not getattr(config, "non_commercial_ok", False):
        return False
    if not getattr(config, "accept_apple_depth_pro_research_license", False):
        return False

    return bool(
        getattr(config, "depth_pro_checkpoint_path", None)
        or getattr(config, "depth_pro_python_executable", None)
        or os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT")
        or os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON")
        or resolve_effective_depth_pro_python_executable(config)
    )


def resolve_requested_backend(
    requested: Optional[str],
    config: EnhanceConfig,
    platform: Optional[PlatformMatrix] = None,
) -> str:
    """Resolve the effective requested backend for the current platform."""
    carried_plan = _carried_execution_plan(config)
    if carried_plan is not None:
        normalized_requested = normalize_backend_id(requested)
        if normalized_requested is not None and normalized_requested != carried_plan.planned_backend:
            raise ValueError(
                f"Requested backend {normalized_requested!r} disagrees with carried plan "
                f"backend {carried_plan.planned_backend!r}"
            )
        return carried_plan.planned_backend

    normalized_requested = normalize_backend_id(requested) or normalize_backend_id(config.depth_backend)
    if normalized_requested:
        return normalized_requested

    effective_platform = platform or CURRENT_PLATFORM
    if effective_platform is not None and effective_platform.is_apple_silicon:
        if _apple_silicon_depth_pro_opt_in(config):
            return "depth_pro"
        if getattr(config, "use_coreml_backend", False):
            return "da3"

    return "da3"


@dataclass
class BackendSelection:
    """Result of backend selection process.

    Captures the selection outcome including:
    - What was requested vs what was resolved
    - Selection status and reason
    - The backend instance (if successful)
    - Any errors encountered during selection

    Attributes:
        requested_backend: Originally requested backend ID
        resolved_backend: Actually selected backend ID
        status: Selection outcome (success, fallback, synthetic_fallback, error)
        reason: Human-readable explanation of selection
        backend: The backend instance if selection succeeded
        model_id: Resolved model identifier for provenance
        device: Target device for inference
        init_errors: Dictionary of backend IDs to error messages
    """

    requested_backend: str
    resolved_backend: Optional[str]
    status: str
    reason: Optional[str] = None
    backend: Optional[Any] = None
    model_id: Optional[str] = None
    device: str = "cpu"
    init_errors: Dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return True if selection succeeded."""
        return self.status in ("success", "fallback", "synthetic_fallback")

    def to_metadata(self, attempts: Optional[List[Dict[str, Any]]] = None) -> BackendSelectionMetadata:
        """Convert to BackendSelectionMetadata for manifest storage.

        Raises:
            ValueError: If resolved_backend is None (error state)
        """
        if self.resolved_backend is None:
            raise ValueError("Cannot convert error selection to metadata: resolved_backend is None")
        return BackendSelectionMetadata(
            requested_backend=self.requested_backend,
            resolved_backend=self.resolved_backend,
            resolution_status=self.status,
            resolution_reason=self.reason,
            model_id=self.model_id or "",
            device=self.device,
            attempts=attempts or [],
        )


@dataclass
class ExecutionPlan:
    """Legacy flat execution-plan projection for pipeline processing.

    This public name is retained for import compatibility.  It is not the
    canonical ``tp.execution.plan.v1`` contract; new contract consumers must
    import ``core.execution_plan.ExecutionPlan`` (or
    ``core.CanonicalExecutionPlan``).  The live Lux executor continues to use
    this flat rollback-path projection until ADR-051's activation gate passes.

    Describes the planned execution stages and their configuration.

    Attributes:
        stages: Ordered list of stage names to execute
        backend_selection: Backend selection result
        enable_depth: Whether depth stage is enabled
        enable_v2: Whether V2 enhancement stage is enabled
        enable_pbr: Whether PBR generation is enabled
        enable_materials_v3: Whether Materials V3 is enabled
        enable_reconstruction: Whether scene reconstruction is enabled
        quality_tier: Target quality tier (standard, premium, apex)
    """

    stages: List[str]
    backend_selection: Optional[BackendSelection] = None
    enable_depth: bool = True
    enable_v2: bool = True
    enable_pbr: bool = False
    enable_materials_v3: bool = False
    enable_reconstruction: bool = False
    quality_tier: str = "standard"


# Explicit alias lets migration code name the old surface without breaking
# callers that import ``pipeline_coordinator.ExecutionPlan``.
LegacyExecutionPlan = ExecutionPlan


@dataclass
class InitializedDepthBackendState:
    """Initialized depth backend state for orchestrator startup."""

    registry: Any
    depth_backend: Any
    backend_cache: Dict[str, Any]
    init_errors: Dict[str, str]
    backend_metadata: BackendSelectionMetadata


@dataclass
class ActiveDepthState:
    """Per-image active backend state for orchestrator reporting paths."""

    backend_metadata: Optional[BackendSelectionMetadata]
    depth_attempts: List[Dict[str, Any]]
    selected_attempt_index: Optional[int]


def resolve_runtime_backend_chain(
    primary_backend_id: str,
    config: EnhanceConfig,
) -> List[str]:
    """Resolve ordered runtime fallback chain.

    Builds an ordered list of backend IDs to try during runtime,
    starting with the primary backend and falling back through
    configured alternatives.

    Args:
        primary_backend_id: Primary backend to try first
        config: EnhanceConfig with fallback configuration

    Returns:
        Ordered list of backend IDs to attempt
    """
    normalized_primary = normalize_backend_id(primary_backend_id) or "da3"
    carried_plan = _carried_execution_plan(config)
    if carried_plan is not None:
        carried_chain = list(carried_plan.candidate_fallback_chain)
        try:
            start = carried_chain.index(normalized_primary)
        except ValueError as exc:
            raise ValueError(
                f"Backend {normalized_primary!r} is absent from carried candidate chain {carried_chain!r}"
            ) from exc
        return carried_chain[start:]

    chain: List[str] = [normalized_primary]

    configured_chain = getattr(
        config,
        "depth_operational_fallback_chain",
        ("da3", "da2"),
    )
    for backend_id in normalize_backend_sequence(configured_chain):
        if backend_id and backend_id not in chain:
            chain.append(backend_id)

    allow_synthetic = bool(config.allow_synthetic_fallback) or os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1"
    if allow_synthetic and "synthetic" not in chain:
        chain.append("synthetic")

    return chain


def expected_output_depth_units_for_backend(backend_id: str) -> str:
    """Return expected output depth units for a backend.

    Args:
        backend_id: Backend identifier

    Returns:
        "meters" for metric backends, "relative" otherwise
    """
    return "meters" if normalize_backend_id(backend_id) == "depth_pro" else "relative"


def default_model_id_for_backend(
    backend_id: str,
    model_variant: Optional[ModelVariant] = None,
    config: Optional[EnhanceConfig] = None,
) -> str:
    """Return canonical backend model identifier for provenance.

    Args:
        backend_id: Backend identifier
        model_variant: Optional model variant for DA3 backend

    Returns:
        Canonical model identifier string
    """
    normalized_backend = normalize_backend_id(backend_id) or ""

    carried_plan = _carried_execution_plan(config) if config is not None else None
    if carried_plan is not None:
        from .execution_lifecycle import backend_candidate_authority

        authority = backend_candidate_authority(carried_plan, normalized_backend)
        if authority.model_contract is not None:
            model = authority.model_contract.model
            return model.repo_id or model.canonical_key

    if normalized_backend == "depth_pro":
        return "apple/ml-depth-pro"
    if normalized_backend == "da2":
        return "depth-anything/Depth-Anything-V2-Small-hf"
    if normalized_backend == "da3":
        if config is not None:
            try:
                return resolve_model_contract(
                    ModelRequest(
                        model_key=getattr(config, "model_key", None),
                        raw_model_id=getattr(config, "raw_model_id", None),
                        model_variant=model_variant or getattr(config, "model_variant", None),
                        use_coreml_backend=bool(getattr(config, "use_coreml_backend", False)),
                        non_commercial_ok=bool(getattr(config, "non_commercial_ok", False)),
                        enforce_license=False,
                    )
                ).spec.repo_id
            except Exception:
                logger.debug(
                    "Falling back to legacy DA3 model ID resolution for backend provenance",
                    exc_info=True,
                )
        if model_variant is not None:
            return model_variant.value.huggingface_id
        return ModelVariant.METRIC_LARGE.value.huggingface_id
    if normalized_backend == "depthcrafter":
        return "tencent/depthcrafter"
    if normalized_backend == "ensemble":
        return "ensemble/multi-backend"
    if normalized_backend == "synthetic":
        return "synthetic/depth-analytic-v1"

    # Fallback: prefer backend ID when non-empty, otherwise use model variant or default
    # Note: normalized_backend is empty string "" when normalize_backend_id returns None and we OR with ""
    # on line 196. An empty string is falsy, so we fall through to model_variant or default.
    if normalized_backend:
        return normalized_backend
    if model_variant is not None:
        return model_variant.value.huggingface_id
    return ModelVariant.METRIC_LARGE.value.huggingface_id


def derive_model_id_from_backend_instance(
    backend_id: str,
    backend: Optional[Any],
) -> Optional[str]:
    """Best-effort model id extraction from backend instance.

    Attempts to extract a model identifier from various backend
    attributes for provenance tracking.

    Args:
        backend_id: Backend identifier
        backend: Backend instance to inspect

    Returns:
        Model identifier if found, None otherwise
    """
    if backend is None:
        return None

    # Try direct model_id attribute
    for attr_name in ("model_id", "_model_id"):
        candidate = getattr(backend, attr_name, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Try model_variant path
    model_variant = getattr(backend, "_model_variant", None)
    if model_variant is not None:
        variant_value = getattr(model_variant, "value", None)
        hf_id = getattr(variant_value, "huggingface_id", None)
        if isinstance(hf_id, str) and hf_id.strip():
            return hf_id.strip()
        if isinstance(variant_value, str) and variant_value.strip():
            return variant_value.strip()

    # Try nested model.variant path
    backend_model = getattr(backend, "_model", None)
    model_variant = getattr(backend_model, "variant", None)
    variant_value = getattr(model_variant, "value", None)
    if isinstance(variant_value, str) and variant_value.strip():
        return variant_value.strip()

    # Special case for depth_pro
    if str(backend_id).strip().lower() == "depth_pro":
        return "apple/ml-depth-pro"

    return None


def resolve_backend_model_id(
    backend_id: str,
    *,
    result_metadata: Optional[Dict[str, Any]] = None,
    backend: Optional[Any] = None,
    model_variant: Optional[ModelVariant] = None,
    config: Optional[EnhanceConfig] = None,
) -> str:
    """Resolve stable model id for provenance and run-card semantics.

    Attempts resolution in order:
    1. Canonical override for depth_pro
    2. Explicit metadata fields
    3. Backend instance extraction
    4. Default for backend type

    Args:
        backend_id: Backend identifier
        result_metadata: Optional metadata dict with model info
        backend: Optional backend instance
        model_variant: Optional model variant for DA3

    Returns:
        Resolved model identifier string
    """
    normalized_backend = str(backend_id).strip().lower()

    # Canonical for depth_pro
    if normalized_backend == "depth_pro":
        return "apple/ml-depth-pro"

    # Try metadata fields
    metadata = result_metadata or {}
    for key in ("resolved_model_id", "requested_model_id", "model_id"):
        candidate = metadata.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Try backend instance
    from_backend = derive_model_id_from_backend_instance(backend_id, backend)
    if from_backend:
        return from_backend

    # Fall back to default
    return default_model_id_for_backend(backend_id, model_variant, config=config)


def normalize_sha256(value: Any) -> Optional[str]:
    """Normalize SHA-256 digest to lowercase hex."""
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest):
        return digest
    return None


def typed_nullary_callable(value: Any) -> Optional[Callable[[], Any]]:
    """Return a typed no-arg callable for dynamically loaded attributes."""
    if callable(value):
        return value
    return None


def resolve_backend_model_artifact(
    backend_id: str,
    *,
    result_metadata: Optional[Dict[str, Any]] = None,
    backend: Optional[Any] = None,
) -> Dict[str, Optional[str]]:
    """Resolve backend model artifact identity fields."""
    artifact_filename: Optional[str] = None
    artifact_sha256: Optional[str] = None

    if str(backend_id).strip().lower() != "depth_pro":
        return {
            "model_artifact_filename": None,
            "model_artifact_sha256": None,
        }

    metadata = result_metadata or {}
    checkpoint_meta = metadata.get("checkpoint")
    if isinstance(checkpoint_meta, dict):
        path_value = checkpoint_meta.get("path")
        if isinstance(path_value, str) and path_value.strip():
            artifact_filename = Path(path_value).name
        artifact_sha256 = normalize_sha256(checkpoint_meta.get("sha256"))

    if artifact_filename is None and backend is not None:
        checkpoint_path = getattr(backend, "_checkpoint_path", None)
        if checkpoint_path is not None:
            try:
                artifact_filename = Path(checkpoint_path).name
            except TypeError:
                artifact_filename = None

    if artifact_sha256 is None and backend is not None:
        artifact_sha256 = normalize_sha256(
            getattr(backend, "_checkpoint_hash_cached", None),
        )

    checkpoint_meta_present = isinstance(checkpoint_meta, dict)
    if artifact_sha256 is None and checkpoint_meta_present and backend is not None:
        checkpoint_hash_getter = typed_nullary_callable(
            getattr(backend, "_get_checkpoint_hash", None),
        )
        if checkpoint_hash_getter is not None:
            try:
                artifact_sha256 = normalize_sha256(
                    checkpoint_hash_getter(),
                )
            except Exception:  # pragma: no cover - best-effort provenance enrichment
                artifact_sha256 = None

    return {
        "model_artifact_filename": artifact_filename,
        "model_artifact_sha256": artifact_sha256,
    }


def extract_model_id_from_attempts(
    selected_backend: str,
    attempts: List[Dict[str, Any]],
    *,
    selected_attempt_index: Optional[int] = None,
) -> Optional[str]:
    """Extract selected backend model id from attempt history."""
    if selected_attempt_index is not None and 0 <= selected_attempt_index < len(attempts):
        selected_attempt = attempts[selected_attempt_index]
        if selected_attempt.get("backend") == selected_backend:
            selected_attempt_model = selected_attempt.get("model_id")
            if isinstance(selected_attempt_model, str) and selected_attempt_model.strip():
                return selected_attempt_model.strip()

    for attempt in attempts:
        if attempt.get("backend") != selected_backend:
            continue
        if attempt.get("status") != "success":
            continue
        attempt_model = attempt.get("model_id")
        if isinstance(attempt_model, str) and attempt_model.strip():
            return attempt_model.strip()

    return None


def extract_model_artifact_from_attempts(
    selected_backend: str,
    attempts: List[Dict[str, Any]],
    *,
    selected_attempt_index: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    """Extract selected backend model artifact identity from attempt history."""

    def _extract_from_attempt(attempt: Dict[str, Any]) -> Dict[str, Optional[str]]:
        raw_filename = attempt.get("model_artifact_filename")
        artifact_filename = raw_filename.strip() if isinstance(raw_filename, str) and raw_filename.strip() else None
        return {
            "model_artifact_filename": artifact_filename,
            "model_artifact_sha256": normalize_sha256(
                attempt.get("model_artifact_sha256"),
            ),
        }

    if selected_attempt_index is not None and 0 <= selected_attempt_index < len(attempts):
        selected_attempt = attempts[selected_attempt_index]
        if selected_attempt.get("backend") == selected_backend:
            selected_artifact = _extract_from_attempt(selected_attempt)
            if selected_artifact["model_artifact_filename"] or selected_artifact["model_artifact_sha256"]:
                return selected_artifact

    for attempt in attempts:
        if attempt.get("backend") != selected_backend:
            continue
        if attempt.get("status") != "success":
            continue
        selected_artifact = _extract_from_attempt(attempt)
        if selected_artifact["model_artifact_filename"] or selected_artifact["model_artifact_sha256"]:
            return selected_artifact

    return {
        "model_artifact_filename": None,
        "model_artifact_sha256": None,
    }


def infer_operational_error_code(
    error: Exception,
) -> str:
    """Map backend exceptions to error codes."""
    if isinstance(error, ImportError):
        return "BACKEND_IMPORT_ERROR"
    if isinstance(error, FileNotFoundError):
        return "BACKEND_RESOURCE_MISSING"
    message = str(error).lower()
    if "torch not compiled with cuda enabled" in message:
        return "CUDA_HARDCODED_IN_BACKEND"
    if "mps" in message and "not available" in message:
        return "MPS_UNAVAILABLE"
    if "cuda" in message and "not available" in message:
        return "CUDA_UNAVAILABLE"
    return "BACKEND_RUNTIME_ERROR"


def seed_depth_attempts_from_selection_fallback(
    backend_metadata: Optional[BackendSelectionMetadata],
    init_errors: Optional[Dict[str, str]],
    config: EnhanceConfig,
    model_variant: ModelVariant,
) -> List[Dict[str, Any]]:
    """Materialize backend-selection fallback into per-image attempt history."""
    effective_errors = init_errors or {}
    requested_backend = normalize_backend_id(getattr(backend_metadata, "requested_backend", None))
    resolved_backend = normalize_backend_id(getattr(backend_metadata, "resolved_backend", None))

    if (
        not requested_backend
        or not resolved_backend
        or requested_backend == resolved_backend
        or not isinstance(effective_errors, dict)
    ):
        return []

    requested_error = effective_errors.get(requested_backend)
    if not isinstance(requested_error, str) or not requested_error.strip():
        return []

    attempt: Dict[str, Any] = {
        "attempt": 0,
        "backend": requested_backend,
        "model_id": default_model_id_for_backend(requested_backend, model_variant, config=config),
        "device": config.depth_device,
        "status": "failed",
        "failure_kind": "operational",
        "error_code": infer_operational_error_code(
            RuntimeError(requested_error),
        ),
        "error_message": requested_error,
        "apex_gate_passed": False,
        "cached": False,
        "duration_s": 0.0,
        "model_artifact_filename": None,
        "model_artifact_sha256": None,
    }

    if requested_backend == "depth_pro":
        carried_plan = _carried_execution_plan(config)
        if carried_plan is not None:
            from .execution_lifecycle import backend_candidate_authority

            authority = backend_candidate_authority(carried_plan, requested_backend)
            artifact_path = None if authority.model_contract is None else authority.model_contract.artifact_path
            checkpoint_path = Path(artifact_path or "checkpoints/depth_pro.pt")
        else:
            checkpoint_path = Path(
                getattr(config, "depth_pro_checkpoint_path", None)
                or os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT")
                or "checkpoints/depth_pro.pt"
            )
        attempt["model_artifact_filename"] = checkpoint_path.name

    return [attempt]


def get_or_create_depth_backend(
    backend_id: str,
    *,
    active_backend: Optional[Any],
    backend_cache: Dict[str, Any],
    registry: Any,
    config: EnhanceConfig,
) -> Any:
    """Fetch backend from cache or registry."""
    carried_context = _carried_candidate_context(config, backend_id)
    expected_authority = None if carried_context is None else carried_context[1]
    canonical_plan_bytes = None if carried_context is None else carried_context[2]

    def _matches_carried_authority(backend: Any) -> bool:
        if carried_context is None:
            return True
        return (
            getattr(backend, "_candidate_authority", None) == expected_authority
            and getattr(backend, "_canonical_plan_bytes", None) == canonical_plan_bytes
        )

    if (
        active_backend is not None
        and getattr(
            active_backend,
            "name",
            None,
        )
        == backend_id
        and _matches_carried_authority(active_backend)
    ):
        backend_cache[backend_id] = active_backend
        return active_backend

    cached = backend_cache.get(backend_id)
    if cached is not None and _matches_carried_authority(cached):
        return cached

    if carried_context is None:
        backend = registry.get_backend(backend_id, config)
    else:
        candidate_config, authority, exact_bytes = carried_context
        backend = registry.get_backend(
            backend_id,
            candidate_config,
            candidate_authority=authority,
            canonical_plan_bytes=exact_bytes,
        )
    backend.ensure_available()
    backend_cache[backend_id] = backend
    return backend


def build_active_depth_state(
    backend_metadata: Optional[BackendSelectionMetadata],
    depth_attempts: List[Dict[str, Any]],
    selected_attempt_index: Optional[int],
) -> ActiveDepthState:
    """Build per-image active backend state for downstream reporting paths."""
    return ActiveDepthState(
        backend_metadata=backend_metadata,
        depth_attempts=list(depth_attempts),
        selected_attempt_index=selected_attempt_index,
    )


def build_backend_metadata_for_attempts(
    selected_backend: str,
    attempts: List[Dict[str, Any]],
    startup_backend_metadata: BackendSelectionMetadata,
    config: EnhanceConfig,
    resolve_model_id: Callable[..., str],
    *,
    result_metadata: Optional[Dict[str, Any]] = None,
    selected_attempt_index: Optional[int] = None,
    backend_cache: Optional[Dict[str, Any]] = None,
) -> BackendSelectionMetadata:
    """Build per-image backend selection metadata."""
    normalized_selected_backend = normalize_backend_id(selected_backend) or selected_backend
    requested = normalize_backend_provenance(
        startup_backend_metadata.requested_backend or startup_backend_metadata.resolved_backend,
    )
    resolution_status = "success" if normalized_selected_backend == requested else "fallback"
    resolution_reason: Optional[str] = None
    if resolution_status == "fallback":
        startup_reason = getattr(startup_backend_metadata, "resolution_reason", None)
        if (
            isinstance(startup_reason, str)
            and startup_reason.strip()
            and normalize_backend_provenance(startup_backend_metadata.requested_backend) == requested
            and normalize_backend_provenance(startup_backend_metadata.resolved_backend) == normalized_selected_backend
            and startup_backend_metadata.resolution_status != "success"
        ):
            resolution_reason = startup_reason
        else:
            failed = [attempt for attempt in attempts if attempt.get("status") == "failed"]
            if failed:
                first_failure = failed[0]
                failure_kind = first_failure.get(
                    "failure_kind",
                    "operational",
                )
                failure_code = first_failure.get(
                    "error_code",
                    "UNKNOWN",
                )
                resolution_reason = (
                    f"Fallback from"
                    f" '{requested}' to"
                    f" '{normalized_selected_backend}'"
                    f" after {failure_kind}"
                    f" failure ({failure_code})"
                )
            else:
                resolution_reason = f"Fallback from" f" '{requested}'" f" to '{normalized_selected_backend}'"

    model_id = extract_model_id_from_attempts(
        normalized_selected_backend,
        attempts,
        selected_attempt_index=selected_attempt_index,
    )
    if not model_id:
        effective_backend_cache = backend_cache or {}
        model_id = resolve_model_id(
            normalized_selected_backend,
            result_metadata=result_metadata,
            backend=effective_backend_cache.get(normalized_selected_backend),
        )

    resolved_device = _carried_candidate_device(config, normalized_selected_backend) or config.depth_device
    return BackendSelectionMetadata(
        requested_backend=requested,
        resolved_backend=normalized_selected_backend,
        resolution_status=resolution_status,
        resolution_reason=resolution_reason,
        model_id=str(model_id),
        device=resolved_device,
        attempts=attempts,
    )


def select_backend(
    requested: Optional[str],
    config: EnhanceConfig,
    registry: Any,
    model_variant: Optional[ModelVariant] = None,
) -> BackendSelection:
    """Select depth backend with fallback logic.

    Implements backend selection with fallback:
    1. Try requested backend
    2. Check availability (checkpoint + dependencies)
    3. Fallback through operational chain
    4. Optionally use synthetic for testing

    Args:
        requested: Requested backend ID
        config: EnhanceConfig with fallback settings
        registry: DepthBackendRegistry instance
        model_variant: Optional model variant for model ID resolution

    Returns:
        BackendSelection with result
    """
    carried_plan = _carried_execution_plan(config)
    if carried_plan is None:
        apply_effective_da3_runtime_config(config)
    explicit_backend_request = carried_plan is None and (
        normalize_backend_id(requested) is not None or normalize_backend_id(config.depth_backend) is not None
    )
    normalized_requested = resolve_requested_backend(requested, config)
    if carried_plan is None and normalized_requested == "depth_pro":
        apply_effective_depth_pro_runtime_config(config)
    strict_explicit_da3_request = explicit_backend_request and normalized_requested == "da3"

    allow_synthetic = (
        "synthetic" in carried_plan.candidate_fallback_chain
        if carried_plan is not None
        else bool(config.allow_synthetic_fallback) or os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1"
    )

    # Derive operational fallback chain from config to keep behavior consistent with
    # depth_operational_fallback_chain setting in EnhanceConfig
    operational_chain = resolve_runtime_backend_chain(normalized_requested, config)

    # Optionally extend with synthetic backend for test environments.
    # Convert to tuple for immutable concatenation, then to list for normalize_backend_sequence.
    full_chain = list(operational_chain)
    if carried_plan is None and allow_synthetic and "synthetic" not in full_chain:
        full_chain.append("synthetic")

    candidate_chain = list(normalize_backend_sequence(full_chain))

    backend = None
    resolved = None
    status = "error"
    reason = None
    init_errors: Dict[str, str] = {}
    selected_runtime_config = config

    for index, backend_id in enumerate(candidate_chain):
        try:
            carried_context = _carried_candidate_context(config, backend_id)
            if carried_context is None:
                candidate_runtime_config = config
                candidate_backend = registry.get_backend(backend_id, candidate_runtime_config)
            else:
                candidate_runtime_config, authority, canonical_plan_bytes = carried_context
                candidate_backend = registry.get_backend(
                    backend_id,
                    candidate_runtime_config,
                    candidate_authority=authority,
                    canonical_plan_bytes=canonical_plan_bytes,
                )
            candidate_backend.ensure_available()
            backend = candidate_backend
            resolved = backend_id
            selected_runtime_config = candidate_runtime_config

            if index == 0:
                status = "success"
                reason = f"{candidate_backend.name} backend ready"
            elif backend_id == "synthetic":
                status = "synthetic_fallback"
                reason = f"Test environment synthetic fallback after: {init_errors}"
            else:
                status = "fallback"
                requested_error = init_errors.get(normalized_requested, "unknown error")
                reason = f"Requested '{normalized_requested}' unavailable: {requested_error}. " f"Selected '{backend_id}'"
            break

        except (LicenseRestrictionError, ModelLicenseError, BackendCapabilityError):
            # Never bypass explicit license restrictions on requested backend
            if carried_plan is not None or index == 0:
                raise
            init_errors[backend_id] = "license_restriction"

        except ValueError:
            # A carried-plan mismatch is an authority failure, never an
            # operational reason to fall through to another candidate.
            if carried_plan is not None or index == 0:
                raise
            init_errors[backend_id] = "unknown_backend"

        except (ImportError, FileNotFoundError, RuntimeError) as backend_error:
            if strict_explicit_da3_request and index == 0 and backend_id == normalized_requested:
                raise
            init_errors[backend_id] = str(backend_error)

        except Exception as backend_error:  # pragma: no cover
            if strict_explicit_da3_request and index == 0 and backend_id == normalized_requested:
                raise
            init_errors[backend_id] = str(backend_error)

    # Resolve model ID for successful selection
    model_id = None
    if backend is not None and resolved is not None:
        model_id = resolve_backend_model_id(
            resolved,
            backend=backend,
            model_variant=model_variant,
        )

    return BackendSelection(
        requested_backend=normalized_requested,
        resolved_backend=resolved,
        status=status,
        reason=reason,
        backend=backend,
        model_id=model_id,
        device=selected_runtime_config.depth_device,
        init_errors=init_errors,
    )


def initialize_depth_backend_state(
    config: EnhanceConfig,
    model_variant: ModelVariant,
    resolve_model_id: Callable[..., str],
    *,
    registry_factory: Optional[Callable[[], Any]] = None,
) -> InitializedDepthBackendState:
    """Initialize depth backend registry, backend cache, and metadata."""
    if registry_factory is None:
        from ..depth.backends.registry import DepthBackendRegistry

        registry_factory = DepthBackendRegistry

    registry = registry_factory()
    backend_cache: Dict[str, Any] = {}
    init_errors: Dict[str, str] = {}
    carried_plan = _carried_execution_plan(config)
    allow_synthetic = (
        "synthetic" in carried_plan.candidate_fallback_chain
        if carried_plan is not None
        else bool(config.allow_synthetic_fallback) or os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1"
    )

    try:
        selection = select_backend(
            config.depth_backend,
            config,
            registry,
            model_variant,
        )
        if not selection.is_success or selection.backend is None or selection.resolved_backend is None:
            requested = resolve_requested_backend(config.depth_backend, config)
            candidate_chain = resolve_runtime_backend_chain(requested, config)
            if not allow_synthetic:
                raise RuntimeError(
                    "No depth backend"
                    " available from"
                    " candidates"
                    f" {candidate_chain}."
                    f" Errors: {selection.init_errors}."
                    " Install ML deps"
                    " (torch, transformers)"
                    " or explicitly enable"
                    " synthetic fallback for"
                    " testing (config"
                    ".allow_synthetic_fallback"
                    "=True or TP_ALLOW_"
                    "SYNTHETIC_FALLBACK=1)."
                )
            raise RuntimeError(
                "No depth backend" " available from" " candidates" f" {candidate_chain}." f" Errors: {selection.init_errors}"
            )

        init_errors = dict(selection.init_errors or {})
        backend_cache[selection.resolved_backend] = selection.backend
        backend_metadata = BackendSelectionMetadata(
            requested_backend=selection.requested_backend,
            resolved_backend=selection.resolved_backend,
            resolution_status=selection.status,
            resolution_reason=selection.reason,
            model_id=resolve_model_id(
                selection.resolved_backend,
                backend=selection.backend,
            ),
            device=selection.device,
            attempts=[],
        )

        logger.info(
            "Depth backend:" " requested=%s" " resolved=%s device=%s",
            selection.requested_backend,
            selection.resolved_backend,
            selection.device,
        )
        return InitializedDepthBackendState(
            registry=registry,
            depth_backend=selection.backend,
            backend_cache=backend_cache,
            init_errors=init_errors,
            backend_metadata=backend_metadata,
        )

    except LicenseRestrictionError as e:
        logger.error(f"License restriction: {e}")
        raise
    except Exception as e:
        logger.error(f"Backend initialization failed: {e}")
        raise


class PipelineCoordinator:
    """Pipeline coordination and stage planning.

    Provides a unified interface for:
    - Backend selection with fallback
    - Execution plan generation
    - Runtime backend chain resolution
    - Model ID resolution

    This class is the primary interface for pipeline coordination
    per ADR-043.

    Example:
        coordinator = PipelineCoordinator(config, registry)

        # Select backend
        selection = coordinator.select_backend("da3")
        if selection.is_success:
            backend = selection.backend

        # Create execution plan
        plan = coordinator.plan(resolved_config)
        for stage in plan.stages:
            execute_stage(stage)
    """

    def __init__(
        self,
        config: EnhanceConfig,
        registry: Optional[Any] = None,
        model_variant: Optional[ModelVariant] = None,
    ) -> None:
        """Initialize pipeline coordinator.

        Args:
            config: EnhanceConfig instance
            registry: Optional DepthBackendRegistry (created if not provided)
            model_variant: Optional resolved model variant
        """
        self._config = config
        self._registry = registry
        self._model_variant = model_variant or config.model_variant
        self._backend_cache: Dict[str, Any] = {}
        self._current_selection: Optional[BackendSelection] = None

    @property
    def config(self) -> EnhanceConfig:
        """Return the configuration."""
        return self._config

    def select_backend(
        self,
        requested: Optional[str] = None,
    ) -> BackendSelection:
        """Select depth backend with fallback logic.

        Args:
            requested: Optional backend ID to request (defaults to config value)

        Returns:
            BackendSelection with result
        """
        if self._registry is None:
            from ..depth.backends.registry import DepthBackendRegistry

            self._registry = DepthBackendRegistry()

        selection = select_backend(
            requested,
            self._config,
            self._registry,
            self._model_variant,
        )

        if selection.is_success and selection.resolved_backend:
            self._backend_cache[selection.resolved_backend] = selection.backend

        self._current_selection = selection
        return selection

    def get_or_create_backend(self, backend_id: str) -> Optional[Any]:
        """Get cached backend or create new one.

        Args:
            backend_id: Backend identifier

        Returns:
            Backend instance or None if unavailable
        """
        if backend_id in self._backend_cache:
            return self._backend_cache[backend_id]

        if self._registry is None:
            from ..depth.backends.registry import DepthBackendRegistry

            self._registry = DepthBackendRegistry()

        try:
            return get_or_create_depth_backend(
                backend_id,
                active_backend=None,
                backend_cache=self._backend_cache,
                registry=self._registry,
                config=self._config,
            )
        except Exception as e:
            if getattr(self._config, "execution_plan_authority", None) is not None:
                raise
            logger.debug("Failed to create backend %s: %s", backend_id, e)
            return None

    def resolve_runtime_chain(
        self,
        primary_backend_id: Optional[str] = None,
    ) -> List[str]:
        """Resolve ordered runtime fallback chain.

        Args:
            primary_backend_id: Primary backend (defaults to current selection)

        Returns:
            Ordered list of backend IDs
        """
        backend_id = primary_backend_id
        if backend_id is None and self._current_selection:
            backend_id = self._current_selection.resolved_backend
        if backend_id is None:
            backend_id = resolve_requested_backend(None, self._config)

        return resolve_runtime_backend_chain(backend_id, self._config)

    def plan(self, enable_depth: bool = True) -> ExecutionPlan:
        """Create execution plan based on configuration.

        Args:
            enable_depth: Whether to include depth stage

        Returns:
            ExecutionPlan describing stages to execute
        """
        stages: List[str] = []

        # Always include preprocessing
        stages.append("preprocess")

        # Depth stage
        if enable_depth:
            stages.append("depth")

        # PBR generation (depends on depth). This historical flat projection
        # remains unchanged for compatibility; canonical ordering is owned by
        # ResolvedInvocation and the tp.execution.plan.v1 adapter.
        if self._config.generate_pbr and enable_depth:
            stages.append("pbr")

        # Materials V3 (depends on depth)
        if self._config.enable_materials_v3 and enable_depth:
            stages.append("materials_v3")

        # V2 enhancement
        if self._config.enable_v2 and self._config.v2_preset is not None:
            stages.append("v2")

        # Scene reconstruction
        if self._config.enable_reconstruction:
            stages.append("reconstruction")

        # Always include postprocess/output
        stages.append("output")

        return ExecutionPlan(
            stages=stages,
            backend_selection=self._current_selection,
            enable_depth=enable_depth,
            enable_v2=self._config.enable_v2,
            enable_pbr=self._config.generate_pbr,
            enable_materials_v3=self._config.enable_materials_v3,
            enable_reconstruction=self._config.enable_reconstruction,
            quality_tier=self._config.quality_tier,
        )

    def resolve_model_id(
        self,
        backend_id: str,
        *,
        result_metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ) -> str:
        """Resolve model ID for a backend.

        Args:
            backend_id: Backend identifier
            result_metadata: Optional metadata with model info
            backend: Optional backend instance

        Returns:
            Resolved model identifier
        """
        return resolve_backend_model_id(
            backend_id,
            result_metadata=result_metadata,
            backend=backend,
            model_variant=self._model_variant,
        )

    def default_model_id(self, backend_id: str) -> str:
        """Get default model ID for a backend.

        Args:
            backend_id: Backend identifier

        Returns:
            Default model identifier
        """
        return default_model_id_for_backend(backend_id, self._model_variant)

    @staticmethod
    def expected_depth_units(backend_id: str) -> str:
        """Get expected output depth units for a backend.

        Args:
            backend_id: Backend identifier

        Returns:
            "meters" or "relative"
        """
        return expected_output_depth_units_for_backend(backend_id)
