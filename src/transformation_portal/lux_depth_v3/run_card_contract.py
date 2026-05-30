"""Shared run-card versioning and path normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, Optional

from transformation_portal.schemas.run_card import (
    RUN_CARD_SCHEMA_URIS,
    SUPPORTED_RUN_CARD_VERSIONS,
    get_run_card_schema_uri,
    normalize_run_card_version,
)

from .path_aliasing import normalize_lexical_path, relative_to_path_alias


class RunCardPathValidationError(ValueError):
    """Raised when a run-card relative path is empty or unsafe."""


def infer_run_card_version(payload: Mapping[str, Any]) -> str:
    """Infer the run-card version with legacy compatibility."""
    explicit_version = payload.get("run_card_version")
    if explicit_version is not None:
        return normalize_run_card_version(explicit_version)
    return "v2" if "artifact_tree" in payload else "v1"


def with_inferred_run_card_version(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow payload copy with run_card_version materialized."""
    version = infer_run_card_version(payload)
    hydrated = dict(payload)
    hydrated.setdefault("run_card_version", version)
    return hydrated


def get_run_card_schema_uri_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical schema URI for the given payload."""
    return get_run_card_schema_uri(infer_run_card_version(payload))


def normalize_run_card_relative_path(relative_path: Any) -> str:
    """Normalize a run-card relative path and reject unsafe forms."""
    raw = str(relative_path or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw or "\\" in raw:
        raise RunCardPathValidationError(f"artifact relative_path is invalid: {relative_path!r}")

    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        raise RunCardPathValidationError(f"artifact relative_path must not be absolute: {relative_path}")

    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise RunCardPathValidationError(f"artifact relative_path is invalid: {relative_path!r}")
    if any(part == ".." for part in candidate.parts):
        raise RunCardPathValidationError(f"artifact relative_path must not contain traversal segments: {relative_path}")
    return normalized


def render_run_card_output_relative_path(path_value: Any, output_root: Any) -> Optional[str]:
    """Render an output-root-relative path while tolerating alias-equivalent roots."""
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        return PurePosixPath(relative_to_path_alias(path_value, output_root)).as_posix()
    except ValueError:
        return PurePosixPath(normalize_lexical_path(path_value).name).as_posix()


def build_runtime_licensing_manifest(
    *,
    model_contract: Optional[Mapping[str, Any]] = None,
    config: Any = None,
) -> dict[str, Any]:
    """Build machine-readable runtime licensing evidence for run cards/manifests."""
    models: list[dict[str, Any]] = []
    if isinstance(model_contract, Mapping):
        model_id = str(
            model_contract.get("resolved_repo_id")
            or model_contract.get("canonical_model_key")
            or model_contract.get("requested_model_selector")
            or ""
        ).strip()
        license_id = str(model_contract.get("license_id") or "unknown").strip() or "unknown"
        runtime_role = str(model_contract.get("backend_kind") or "depth").strip() or "depth"
        usage_class = str(model_contract.get("usage_class") or "").strip()
        requires_non_commercial_ok = bool(model_contract.get("requires_non_commercial_ok", False))
        if model_id:
            models.append(
                {
                    "id": model_id,
                    "license": license_id,
                    "runtime_role": runtime_role,
                    "usage_class": usage_class or None,
                    "requires_non_commercial_ok": requires_non_commercial_ok,
                }
            )
    else:
        requires_non_commercial_ok = False

    non_commercial_ok = bool(getattr(config, "non_commercial_ok", False))
    apple_research_ack = bool(getattr(config, "accept_apple_depth_pro_research_license", False))
    research_tools_ack = bool(getattr(config, "accept_research_tools_license", False))
    model_requires_research = any(
        model.get("requires_non_commercial_ok") or "non_commercial" in str(model.get("usage_class") or "") for model in models
    )
    research_acknowledgement_required = bool(model_requires_research or apple_research_ack or research_tools_ack)
    non_commercial_active = bool(non_commercial_ok and (model_requires_research or apple_research_ack or research_tools_ack))
    software_license_tier = (
        "research_or_non_commercial" if research_acknowledgement_required or non_commercial_active else "commercial"
    )

    return {
        "schema_version": "1.0",
        "software_license_tier": software_license_tier,
        "models": models,
        "non_commercial_active": non_commercial_active,
        "research_acknowledgement_required": research_acknowledgement_required,
    }


__all__ = [
    "RUN_CARD_SCHEMA_URIS",
    "SUPPORTED_RUN_CARD_VERSIONS",
    "build_runtime_licensing_manifest",
    "RunCardPathValidationError",
    "get_run_card_schema_uri",
    "get_run_card_schema_uri_for_payload",
    "infer_run_card_version",
    "normalize_run_card_relative_path",
    "normalize_run_card_version",
    "render_run_card_output_relative_path",
    "with_inferred_run_card_version",
]
