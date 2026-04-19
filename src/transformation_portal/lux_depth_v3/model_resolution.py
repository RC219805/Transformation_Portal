"""Central Lux Depth V3 model resolver."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from transformation_portal.core.security.model_lock import resolve_model_lock_revision

from .model_registry import (
    DEFAULT_MODEL_KEY,
    AcceleratorKind,
    BackendKind,
    UsageClass,
    get_model_spec,
    legacy_model_variant_warning,
    resolve_legacy_model_variant_key,
    resolve_registry_key,
)


class ModelResolutionError(RuntimeError):
    """Base error for model resolution failures."""


class UnknownModelError(ModelResolutionError):
    """Raised for selectors absent from the registry."""


class ModelLicenseError(ModelResolutionError):
    """Raised when a model selection violates the license policy."""


class BackendCapabilityError(ModelResolutionError):
    """Raised when the requested accelerator/backend is unsupported."""


class DeprecatedModelSelectorWarning(DeprecationWarning):
    """Warning emitted for legacy compatibility selectors."""


@dataclass(frozen=True)
class ModelRequest:
    """Inputs to final model selection."""

    model_key: Optional[str] = None
    raw_model_id: Optional[str] = None
    model_variant: Optional[Any] = None
    use_coreml_backend: bool = False
    non_commercial_ok: bool = False
    enforce_license: bool = True
    strict_model_lock: Optional[bool] = None
    manifest_path: Optional[Path] = None


@dataclass(frozen=True)
class ResolvedModel:
    """Resolved execution contract for a model selection."""

    requested_selector: str
    canonical_key: str
    spec: Any
    revision: Optional[str]
    fallback_chain: Tuple[str, ...]
    accelerator_kind: AcceleratorKind
    legacy_model_variant_name: Optional[str] = None


def _resolve_selector(
    request: ModelRequest,
) -> tuple[str, str, Optional[str]]:
    if request.model_key:
        key = resolve_registry_key(request.model_key)
        if key is None:
            raise UnknownModelError(f"Unknown model selector: {request.model_key}")
        return request.model_key, key, None

    if request.raw_model_id:
        key = resolve_registry_key(request.raw_model_id)
        if key is None:
            raise UnknownModelError(
                f"Unsupported model repo ID: {request.raw_model_id}. "
                "Only registry-approved models are allowed in this release."
            )
        return request.raw_model_id, key, None

    if request.model_variant is not None:
        key = resolve_legacy_model_variant_key(request.model_variant)
        if key is None:
            raise UnknownModelError(f"Unknown model variant: {request.model_variant!r}")
        warning_message = legacy_model_variant_warning(request.model_variant)
        if warning_message:
            warnings.warn(
                warning_message,
                DeprecatedModelSelectorWarning,
                stacklevel=3,
            )
        variant_name = getattr(request.model_variant, "name", None)
        selector = variant_name if isinstance(variant_name, str) else str(request.model_variant)
        return selector, key, variant_name if isinstance(variant_name, str) else None

    return "da3", DEFAULT_MODEL_KEY, None


def _resolve_accelerator(spec: Any, use_coreml_backend: bool) -> AcceleratorKind:
    if not use_coreml_backend:
        return AcceleratorKind.NONE
    if not spec.supports_coreml:
        raise BackendCapabilityError(
            f"CoreML backend is not supported for {spec.repo_id}. "
            "Use a registry-listed published CoreML artifact or disable CoreML."
        )
    if spec.backend_kind != BackendKind.COREML_PUBLISHED:
        raise BackendCapabilityError(
            f"Invalid registry state for {spec.repo_id}: CoreML support requires " "backend_kind=coreml_published."
        )
    return AcceleratorKind.COREML


def _enforce_license_policy(spec: Any, non_commercial_ok: bool) -> None:
    if spec.usage_class == UsageClass.UNKNOWN:
        raise ModelLicenseError(f"Model {spec.repo_id} has unknown usage policy and is not allowed.")
    if spec.requires_non_commercial_ok and not non_commercial_ok:
        raise ModelLicenseError(
            f"Selected model {spec.repo_id} uses {spec.license_id} and is restricted "
            "to non-commercial use. Re-run with --non-commercial-ok for permitted "
            "research/non-commercial use, or select model_key='da3-metric'."
        )


def resolve_model_contract(request: ModelRequest) -> ResolvedModel:
    """Resolve a model selection to the final execution contract."""
    selector, canonical_key, legacy_variant_name = _resolve_selector(request)
    spec = get_model_spec(canonical_key)
    revision = resolve_model_lock_revision(
        spec.repo_id,
        requested_revision=None,
        strict=request.strict_model_lock,
        manifest_path=request.manifest_path,
        context="lux_depth_v3 model resolution",
    )
    if request.enforce_license:
        _enforce_license_policy(spec, request.non_commercial_ok)
    accelerator_kind = _resolve_accelerator(spec, request.use_coreml_backend)
    return ResolvedModel(
        requested_selector=selector,
        canonical_key=canonical_key,
        spec=spec,
        revision=revision,
        fallback_chain=(),
        accelerator_kind=accelerator_kind,
        legacy_model_variant_name=legacy_variant_name,
    )
