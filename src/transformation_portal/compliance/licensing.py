"""Licensing compliance enforcement and validation.

Provides decorators and validators to ensure non-commercial and
research-only backends are used only with explicit authorization and
attested source metadata.
"""

import functools
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, TypeVar, cast

import yaml

from transformation_portal.attestation.materials_policy import (
    ALLOWED_MATERIAL_BACKEND_PATHS,
    VALID_MATERIAL_BACKENDS,
    find_unknown_material_backend_schema_locations,
)
from transformation_portal.attestation.materials_policy import looks_like_material_preset as _looks_like_material_preset
from transformation_portal.attestation.materials_policy import (
    material_preset_family_error,
)
from transformation_portal.attestation.materials_policy import normalize_material_backend as _normalize_material_backend
from transformation_portal.attestation.model_lock_manifest import load_model_lock_manifest as _shared_load_model_lock_manifest
from transformation_portal.attestation.model_lock_manifest import model_lock_manifest_path as _shared_model_lock_manifest_path
from transformation_portal.attestation.model_lock_manifest import repo_root as _shared_repo_root
from transformation_portal.preset_governance import is_placeholder_string


class LicenseRestrictionError(Exception):
    """Raised when a licensing restriction is violated.

    This exception covers commercial-use restrictions, research-only backend
    gates, and missing/insufficient attestation metadata for materials
    backends that require explicit governance approval.
    """

    pass


F = TypeVar("F", bound=Callable[..., Any])

RESEARCH_ONLY_MATERIAL_BACKENDS = frozenset({"nvdiffrec", "material_gan"})
RESEARCH_ALLOWED_MATERIAL_TIERS = frozenset({"dev", "experimental", "research", "apex_research", "apex_research_ultra"})
UNATTESTED_ALLOWED_MATERIAL_TIERS = frozenset({"dev", "experimental", "apex_research_ultra"})
FLOATING_REVISIONS = frozenset({"main", "master", "latest", "head", "tip", "default"})
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VERIFY_RUNTIME_BYTES_ENV_VAR = "TP_VERIFY_MATERIAL_RUNTIME_BYTES"
NON_COMMERCIAL_HF_MODEL_IDS = frozenset(
    {
        "depth-anything/da3-giant-1.1",
        "depth-anything/da3nested-giant-large-1.1",
    }
)
ACCEPTED_NON_COMMERCIAL_LICENSE_MARKERS = frozenset({"non_commercial", "research_only"})


def require_non_commercial(reason: str = "") -> Callable[[F], F]:
    """Decorator enforcing non-commercial usage authorization.

    This decorator ensures that functions using non-commercial models
    (e.g., DA3 Giant / Nested 1.1 variants) only execute when the caller has explicitly set
    `non_commercial_ok=True` in their configuration.

    Args:
        reason: Human-readable explanation of the licensing restriction
                (e.g., "DA3 Giant 1.1 uses CC BY-NC 4.0")

    Raises:
        LicenseRestrictionError: If the configuration does not have
                                `non_commercial_ok=True`

    Example:
        ```python
        @require_non_commercial(reason="DA3 Giant 1.1 uses CC BY-NC 4.0")
        def load_da3_1_1_preset(config: EnhanceConfig):
            ...
        ```
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config from first positional arg or 'config' kwarg
            config = None
            if args and hasattr(args[0], "non_commercial_ok"):
                config = args[0]
            elif "config" in kwargs and hasattr(kwargs["config"], "non_commercial_ok"):
                config = kwargs["config"]

            if config is None:
                raise TypeError(
                    f"@require_non_commercial decorator on {func.__name__} "
                    "expects first argument or 'config' kwarg to have "
                    "'non_commercial_ok' attribute"
                )

            if not config.non_commercial_ok:
                raise LicenseRestrictionError(
                    f"Function '{func.__name__}' requires "
                    "non_commercial_ok=True in your EnhanceConfig.\n"
                    f"Reason: {reason}\n"
                    "This model uses CC BY-NC 4.0 (non-commercial research only).\n"
                    "For commercial applications, use a commercially-licensed "
                    "depth model instead."
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_non_commercial_preset(preset_dict: Dict[str, Any]) -> bool:
    """Validate that non-commercial presets have required licensing markers.

    Checks if a preset dictionary (from YAML) contains an exact Hugging Face
    model ID known to be non-commercial. If so, ensures the preset has an
    explicit `license_restriction` acknowledgement marker of either
    `non_commercial` or `research_only`.

    Args:
        preset_dict: Dictionary loaded from a preset YAML file
                    (should have 'model' key with 'hf_id')

    Returns:
        True if preset is valid (either commercial or properly marked non-commercial)

    Raises:
        ValueError: If the preset root is not a mapping.
        LicenseRestrictionError: If preset uses non-commercial model
                                without proper marker
    """
    if not isinstance(preset_dict, dict):
        raise ValueError(f"Preset must be a mapping (dict), got {type(preset_dict).__name__}.")

    model = preset_dict.get("model", {})
    hf_id = model.get("hf_id", "")

    normalized_hf_id = hf_id.strip().lower() if isinstance(hf_id, str) else ""

    if normalized_hf_id in NON_COMMERCIAL_HF_MODEL_IDS:
        # Verify marker exists
        license_restriction = preset_dict.get("license_restriction")
        if license_restriction not in ACCEPTED_NON_COMMERCIAL_LICENSE_MARKERS:
            raise LicenseRestrictionError(
                f"Preset uses non-commercial model (hf_id={hf_id}) "
                "but lacks license_restriction='non_commercial' or 'research_only' marker.\n"
                "Please add one of these markers to acknowledge CC BY-NC 4.0 restrictions."
            )

    return True


def _looks_placeholder(value: Any) -> bool:
    """Return True when a preset field is clearly unresolved."""
    return is_placeholder_string(value, treat_empty_as_placeholder=True)


def _has_pinned_repo_revision(model_dict: Dict[str, Any]) -> bool:
    """Return True when repo_id/revision look pinned and non-placeholder."""
    repo_id = model_dict.get("repo_id")
    revision = model_dict.get("revision")
    normalized_revision = revision.strip().lower() if isinstance(revision, str) else None

    if not isinstance(repo_id, str) or not repo_id.strip() or _looks_placeholder(repo_id):
        return False
    if not isinstance(revision, str) or not revision.strip() or _looks_placeholder(revision):
        return False

    if normalized_revision in FLOATING_REVISIONS or not _HEX40_RE.fullmatch(normalized_revision):
        return False

    return not all(ch == normalized_revision[0] for ch in normalized_revision)


def _is_valid_sha256_digest(expected_sha256: Any) -> bool:
    """Return True when the supplied SHA-256 digest is pinned and non-placeholder."""
    if not isinstance(expected_sha256, str) or _looks_placeholder(expected_sha256):
        return False

    normalized = expected_sha256.strip().lower()
    if not _HEX64_RE.fullmatch(normalized):
        return False

    return not all(ch == normalized[0] for ch in normalized)


def _has_attested_checkpoint(model_dict: Dict[str, Any]) -> bool:
    """Return True when a local checkpoint has an expected SHA-256."""
    checkpoint = model_dict.get("checkpoint")
    expected_sha256 = model_dict.get("expected_sha256")

    if not isinstance(checkpoint, str) or not checkpoint.strip() or _looks_placeholder(checkpoint):
        return False
    return _is_valid_sha256_digest(expected_sha256)


def _has_attested_material_source(model_dict: Dict[str, Any]) -> bool:
    """Return True when model metadata identifies a pinned material source."""
    return _has_pinned_repo_revision(model_dict) or _has_attested_checkpoint(model_dict)


def _load_model_lock_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the model lock manifest used for materials attestation checks."""
    return _shared_load_model_lock_manifest(path)


def _manifest_location_label(path: Optional[Path]) -> str:
    """Return the resolved model-lock manifest path used for validation messages."""
    return str(_shared_model_lock_manifest_path(path))


def _parse_bool_env(raw: Optional[str]) -> bool:
    """Parse a permissive boolean environment variable value."""
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_verify_runtime_bytes(verify_runtime_bytes: Optional[bool]) -> bool:
    """Resolve whether local checkpoint bytes should be hashed during validation."""
    if verify_runtime_bytes is not None:
        return verify_runtime_bytes
    return _parse_bool_env(os.getenv(_VERIFY_RUNTIME_BYTES_ENV_VAR))


def _compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a local file using streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_path_within_allowed_roots(path: Path, allowed_roots: list[Path]) -> Path:
    """Ensure a resolved path remains within one of the explicitly allowed roots."""
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    formatted_roots = ", ".join(str(root) for root in allowed_roots)
    raise LicenseRestrictionError(f"Checkpoint path '{resolved}' is outside allowed roots: {formatted_roots}.")


def _resolve_material_checkpoint_path(checkpoint: str, *, preset_path: Optional[Path]) -> Path:
    """Resolve a checkpoint path relative to the preset file or repo root.

    Relative paths are only allowed to resolve within the preset directory or
    repository root so user-controlled checkpoint values cannot traverse the
    filesystem and trigger arbitrary file hashing.
    """
    candidate = Path(checkpoint)
    repo_root = _shared_repo_root().resolve()
    allowed_roots: list[Path] = []
    if preset_path is not None:
        allowed_roots.append(preset_path.parent.resolve())
    if repo_root not in allowed_roots:
        allowed_roots.append(repo_root)

    if candidate.is_absolute():
        return _ensure_path_within_allowed_roots(candidate, allowed_roots)

    candidate_paths: list[Path] = []
    if preset_path is not None:
        candidate_paths.append(preset_path.parent / candidate)
    candidate_paths.append(repo_root / candidate)

    first_valid: Optional[Path] = None
    for candidate_path in candidate_paths:
        try:
            resolved = _ensure_path_within_allowed_roots(candidate_path, allowed_roots)
        except LicenseRestrictionError:
            continue
        if first_valid is None:
            first_valid = resolved
        if resolved.exists():
            return resolved

    if first_valid is not None:
        return first_valid

    return _ensure_path_within_allowed_roots(candidate_paths[0], allowed_roots)


def _extract_materials_governance_overrides(
    preset_dict: Dict[str, Any],
    *,
    allow_research_materials: Optional[bool],
    allow_unattested_materials: Optional[bool],
) -> dict[str, bool]:
    """Resolve materials governance overrides from explicit args and preset metadata."""
    governance = preset_dict.get("governance", {})
    materials_governance = governance.get("materials", {}) if isinstance(governance, dict) else {}
    materials_cfg = preset_dict.get("materials", {})
    pipeline_cfg = preset_dict.get("pipeline", {})
    pipeline_materials = pipeline_cfg.get("materials", {}) if isinstance(pipeline_cfg, dict) else {}

    sources = {
        "allow_research_materials": [
            allow_research_materials,
            materials_governance.get("allow_research_materials") if isinstance(materials_governance, dict) else None,
            materials_cfg.get("allow_research_materials") if isinstance(materials_cfg, dict) else None,
            pipeline_materials.get("allow_research_materials") if isinstance(pipeline_materials, dict) else None,
        ],
        "allow_unattested_materials": [
            allow_unattested_materials,
            materials_governance.get("allow_unattested_materials") if isinstance(materials_governance, dict) else None,
            materials_cfg.get("allow_unattested_materials") if isinstance(materials_cfg, dict) else None,
            pipeline_materials.get("allow_unattested_materials") if isinstance(pipeline_materials, dict) else None,
        ],
    }

    resolved: dict[str, bool] = {}
    for key, candidates in sources.items():
        provided_values = [value for value in candidates if value is not None]
        if any(not isinstance(value, bool) for value in provided_values):
            bad_value = next(value for value in provided_values if not isinstance(value, bool))
            raise ValueError(f"{key} must be a boolean when provided, got {type(bad_value).__name__}.")

        explicit_override = candidates[0]
        if explicit_override is not None:
            resolved[key] = cast(bool, explicit_override)
            continue

        fallback_values = [cast(bool, value) for value in candidates[1:] if value is not None]
        if not fallback_values:
            resolved[key] = False
            continue

        unique_values = set(fallback_values)
        if len(unique_values) > 1:
            raise ValueError(
                f"Conflicting non-None values for {key}; ensure a single boolean value across "
                "governance/materials/pipeline configuration."
            )

        resolved[key] = fallback_values[0]

    return resolved


def _iter_material_backend_specs(
    preset_dict: Dict[str, Any], preset_path: Optional[Path]
) -> list[tuple[str, str, Dict[str, Any]]]:
    """Collect material backend declarations embedded in a preset."""
    specs: list[tuple[str, str, Dict[str, Any]]] = []

    materials_cfg = preset_dict.get("materials")
    if isinstance(materials_cfg, dict):
        backend = _normalize_material_backend(materials_cfg.get("backend"))
        if backend in VALID_MATERIAL_BACKENDS:
            model_cfg = materials_cfg.get("model")
            specs.append(("materials.backend", backend, model_cfg if isinstance(model_cfg, dict) else {}))

    pipeline_cfg = preset_dict.get("pipeline")
    if isinstance(pipeline_cfg, dict):
        pipeline_materials = pipeline_cfg.get("materials")
        if isinstance(pipeline_materials, dict):
            backend = _normalize_material_backend(pipeline_materials.get("backend"))
            if backend in VALID_MATERIAL_BACKENDS:
                model_cfg = pipeline_materials.get("model")
                specs.append(("pipeline.materials.backend", backend, model_cfg if isinstance(model_cfg, dict) else {}))

    if not _looks_like_material_preset(preset_dict, preset_path):
        return specs

    backend_cfg = preset_dict.get("backend")
    if isinstance(backend_cfg, dict):
        backend = _normalize_material_backend(backend_cfg.get("type"))
        if backend in VALID_MATERIAL_BACKENDS:
            model_cfg = backend_cfg.get("model")
            specs.append(("backend.type", backend, model_cfg if isinstance(model_cfg, dict) else {}))
            return specs

    model_cfg = preset_dict.get("model")
    if isinstance(model_cfg, dict):
        backend = _normalize_material_backend(model_cfg.get("backend"))
        if backend in VALID_MATERIAL_BACKENDS:
            specs.append(("model.backend", backend, model_cfg))

    return specs


def _get_manifest_repo_entry(manifest: Dict[str, Any], repo_id: str) -> Optional[Dict[str, Any]]:
    """Return a manifest repository entry for the given repo_id."""
    repositories = manifest.get("repositories", {})
    if not isinstance(repositories, dict):
        return None
    entry = repositories.get(repo_id)
    return entry if isinstance(entry, dict) else None


def _manifest_entry_allows_materials(entry: Dict[str, Any]) -> bool:
    """Return True when the manifest entry is explicitly approved for materials."""
    owner = entry.get("owner")
    if not isinstance(owner, str):
        return False
    return "materials" in owner.lower()


def _validate_repo_source_against_manifest(
    *,
    backend: str,
    source_path: str,
    model_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    manifest_location: str,
) -> None:
    """Ensure repo-backed materials sources match an approved manifest entry."""
    if not _has_pinned_repo_revision(model_dict):
        return

    repo_id = cast(str, model_dict["repo_id"]).strip()
    revision = cast(str, model_dict["revision"]).strip().lower()
    entry = _get_manifest_repo_entry(manifest, repo_id)
    if entry is None:
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} references repo_id='{repo_id}', "
            f"which is not approved in {manifest_location}."
        )

    if not _manifest_entry_allows_materials(entry):
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} references repo_id='{repo_id}', "
            "but that manifest entry is not owned/approved for materials use."
        )

    manifest_revision = entry.get("revision")
    normalized_manifest_revision = manifest_revision.strip().lower() if isinstance(manifest_revision, str) else None
    if normalized_manifest_revision != revision:
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} must use the exact approved revision from "
            f"{manifest_location} (preset={revision}, manifest={normalized_manifest_revision})."
        )


def _get_material_artifact_attestation_entry(manifest: Dict[str, Any], backend: str) -> Optional[Dict[str, Any]]:
    """Return the manifest artifact-attestation entry for a materials backend."""
    artifact_attestation = manifest.get("artifact_attestation", {})
    if not isinstance(artifact_attestation, dict):
        return None
    materials_attestation = artifact_attestation.get("materials")
    if not isinstance(materials_attestation, dict):
        return None
    entry = materials_attestation.get(backend)
    return entry if isinstance(entry, dict) else None


def _validate_checkpoint_source_against_manifest(
    *,
    backend: str,
    source_path: str,
    model_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    manifest_location: str,
) -> None:
    """Ensure checkpoint-backed materials sources match manifest artifact attestation."""
    if not _has_attested_checkpoint(model_dict):
        return

    checkpoint = cast(str, model_dict["checkpoint"]).strip()
    expected_sha256 = cast(str, model_dict["expected_sha256"]).strip().lower()
    entry = _get_material_artifact_attestation_entry(manifest, backend)
    if entry is None:
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} uses checkpoint='{checkpoint}', "
            f"but {manifest_location} has no artifact_attestation.materials entry for it."
        )

    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list):
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} requires a valid artifact_attestation.materials entry "
            f"in {manifest_location}."
        )

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        filename = artifact.get("filename")
        manifest_sha256 = artifact.get("sha256")
        if filename == checkpoint and isinstance(manifest_sha256, str) and manifest_sha256.strip().lower() == expected_sha256:
            return

    raise LicenseRestrictionError(
        f"Materials backend '{backend}' in {source_path} must match an approved checkpoint+sha256 entry "
        f"in {manifest_location}."
    )


def _verify_runtime_checkpoint_bytes(
    *,
    backend: str,
    source_path: str,
    model_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    preset_path: Optional[Path],
    manifest_location: str,
) -> None:
    """Verify local checkpoint bytes when the checkpoint is present on disk at runtime."""
    if not _has_attested_checkpoint(model_dict):
        return

    checkpoint = cast(str, model_dict["checkpoint"]).strip()
    expected_sha256 = cast(str, model_dict["expected_sha256"]).strip().lower()
    checkpoint_path = _resolve_material_checkpoint_path(checkpoint, preset_path=preset_path)
    if not checkpoint_path.exists():
        return

    actual_sha256 = _compute_file_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} has checkpoint bytes that do not match "
            f"expected_sha256 for '{checkpoint_path}'."
        )

    entry = _get_material_artifact_attestation_entry(manifest, backend)
    if entry is None:
        return

    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list):
        return

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        filename = artifact.get("filename")
        manifest_sha256 = artifact.get("sha256")
        if filename == checkpoint and isinstance(manifest_sha256, str):
            normalized_manifest_sha = manifest_sha256.strip().lower()
            if normalized_manifest_sha != actual_sha256:
                raise LicenseRestrictionError(
                    f"Materials backend '{backend}' in {source_path} has checkpoint bytes that do not match "
                    f"{manifest_location} artifact attestation."
                )
            return


def validate_materials_preset(
    preset_dict: Dict[str, Any],
    *,
    preset_path: Optional[Path] = None,
    allow_research_materials: Optional[bool] = None,
    allow_unattested_materials: Optional[bool] = None,
    manifest_path: Optional[Path] = None,
    verify_runtime_bytes: Optional[bool] = None,
) -> bool:
    """Validate materials backend tier, licensing, and attestation policy."""
    if not isinstance(preset_dict, dict):
        raise ValueError(f"Preset must be a mapping (dict), got {type(preset_dict).__name__}.")

    family_error = material_preset_family_error(preset_dict, preset_path)
    if family_error is not None:
        raise LicenseRestrictionError(family_error)

    unknown_paths = find_unknown_material_backend_schema_locations(preset_dict, preset_path)
    if unknown_paths:
        raise LicenseRestrictionError(
            "Materials backend declarations must use approved schema locations. "
            f"Unknown paths: {unknown_paths}. Allowed paths: {sorted(ALLOWED_MATERIAL_BACKEND_PATHS)}."
        )

    overrides = _extract_materials_governance_overrides(
        preset_dict,
        allow_research_materials=allow_research_materials,
        allow_unattested_materials=allow_unattested_materials,
    )
    tier = str(preset_dict.get("tier", "")).strip().lower()
    license_restriction = preset_dict.get("license_restriction")
    specs = _iter_material_backend_specs(preset_dict, preset_path)
    if not specs:
        return True

    manifest: Optional[Dict[str, Any]] = None
    manifest_location = _manifest_location_label(manifest_path)
    verify_runtime_bytes_enabled = _should_verify_runtime_bytes(verify_runtime_bytes)

    def _manifest() -> Dict[str, Any]:
        nonlocal manifest
        if manifest is None:
            try:
                manifest = _load_model_lock_manifest(manifest_path)
            except (FileNotFoundError, ValueError) as exc:
                raise LicenseRestrictionError(
                    "Materials licensing validation requires a valid model lock manifest. "
                    f"Resolved manifest path: {manifest_location}. "
                    f"Set {_VERIFY_RUNTIME_BYTES_ENV_VAR}=1 only when runtime byte hashing is desired, "
                    "and set TP_MODEL_LOCK_MANIFEST or pass manifest_path= to load_and_validate_preset()."
                ) from exc
        return manifest

    for source_path, backend, model_dict in specs:
        if backend == "heuristic":
            continue

        if backend in RESEARCH_ONLY_MATERIAL_BACKENDS:
            if license_restriction != "research_only":
                raise LicenseRestrictionError(
                    f"Materials backend '{backend}' in {source_path} requires " "license_restriction='research_only'."
                )

            if tier not in RESEARCH_ALLOWED_MATERIAL_TIERS:
                raise LicenseRestrictionError(
                    f"Materials backend '{backend}' is research-only and cannot be used in tier='{tier}'. "
                    f"Allowed tiers: {sorted(RESEARCH_ALLOWED_MATERIAL_TIERS)}."
                )

            if not overrides["allow_research_materials"]:
                raise LicenseRestrictionError(
                    f"Materials backend '{backend}' is research-only. "
                    "Reload this preset with allow_research_materials=True or set "
                    "governance.materials.allow_research_materials=true to acknowledge the restriction."
                )

        if _has_pinned_repo_revision(model_dict):
            _validate_repo_source_against_manifest(
                backend=backend,
                source_path=source_path,
                model_dict=model_dict,
                manifest=_manifest(),
                manifest_location=manifest_location,
            )
            continue

        if _has_attested_checkpoint(model_dict):
            _validate_checkpoint_source_against_manifest(
                backend=backend,
                source_path=source_path,
                model_dict=model_dict,
                manifest=_manifest(),
                manifest_location=manifest_location,
            )
            if verify_runtime_bytes_enabled:
                _verify_runtime_checkpoint_bytes(
                    backend=backend,
                    source_path=source_path,
                    model_dict=model_dict,
                    manifest=_manifest(),
                    preset_path=preset_path,
                    manifest_location=manifest_location,
                )
            continue

        if overrides["allow_unattested_materials"] and tier in UNATTESTED_ALLOWED_MATERIAL_TIERS:
            continue

        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} lacks an attested source tuple "
            "(repo_id+revision or checkpoint+expected_sha256). "
            "Only dev/experimental presets may bypass this with allow_unattested_materials=True."
        )

    return True


_EXTENDS_KEY = "extends"


def _deep_merge_preset(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge child into parent. Child wins on conflicts; lists replaced."""
    merged: Dict[str, Any] = dict(parent)
    for key, child_value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            merged[key] = _deep_merge_preset(parent_value, child_value)
        else:
            merged[key] = child_value
    return merged


def _resolve_extends_target(extends_value: str, child_path: Path) -> Path:
    """Resolve an ``extends:`` reference to a concrete preset file path.

    Search order: child preset's directory, then ``config/presets/`` under the
    repo root. Accepts bare names (``apex_research``) or filenames
    (``apex_research.yaml`` / ``apex_research.yml``).

    A match that resolves to the child file itself is skipped — a preset cannot
    extend itself. This lets ``config/presets/experimental/foo.yaml`` declare
    ``extends: foo`` and have it resolve to the sibling ``config/presets/foo.yaml``.
    """
    candidate = Path(extends_value)
    if candidate.suffix not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")

    child_resolved = child_path.resolve()
    search_dirs = [child_path.parent, _shared_repo_root() / "config" / "presets"]
    for directory in search_dirs:
        resolved = (directory / candidate).resolve()
        if resolved == child_resolved:
            continue
        # Require an actual preset file, not a directory or symlink to a dir.
        if resolved.is_file():
            return resolved
    raise LicenseRestrictionError(
        f"`extends: {extends_value}` declared in {child_path} could not be resolved "
        f"to a preset file. Searched: {[str(d) for d in search_dirs]}"
    )


def _load_with_extends(preset_path: Path, _visited: Optional[Set[Path]] = None) -> Dict[str, Any]:
    """Load a preset and recursively merge any ``extends:`` parents.

    Returns the merged dict with the ``extends`` key stripped. Detects cycles
    in the inheritance chain and raises ``LicenseRestrictionError``.
    """
    visited: Set[Path] = set(_visited or ())
    resolved = preset_path.resolve()
    if resolved in visited:
        raise LicenseRestrictionError(
            f"Cycle detected in preset `extends:` chain at {resolved}. " f"Visited: {sorted(str(p) for p in visited)}"
        )
    visited.add(resolved)

    with preset_path.open(encoding="utf-8") as f:
        # YAML_GOVERNANCE_AUTHORITY: shared preset loader for config/presets/** and preset-like runtime entrypoints.
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Preset must be a mapping (dict), got {type(data).__name__}.")

    parent_ref = data.pop(_EXTENDS_KEY, None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise LicenseRestrictionError(f"`extends:` in {preset_path} must be a non-empty string, got {parent_ref!r}.")

    parent_path = _resolve_extends_target(parent_ref, preset_path)
    parent_data = _load_with_extends(parent_path, visited)
    return _deep_merge_preset(parent_data, data)


def load_and_validate_preset(
    preset_path: Path,
    *,
    allow_research_materials: Optional[bool] = None,
    allow_unattested_materials: Optional[bool] = None,
    manifest_path: Optional[Path] = None,
    verify_runtime_bytes: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load a preset YAML file and validate licensing compliance.

    If the preset declares ``extends: <name>``, the named parent preset is
    resolved (child directory first, then ``config/presets/``), recursively
    loaded, and deep-merged with the child overriding the parent. The
    ``extends`` key is stripped before validation and from the return value.

    Args:
        preset_path: Path to preset YAML file
        allow_research_materials: Explicit opt-in required for research-only
            materials backends such as NVDIFFREC and MaterialGAN.
        allow_unattested_materials: Allow unresolved material source tuples in
            dev/experimental presets only.
        manifest_path: Optional override for the model lock manifest path.
        verify_runtime_bytes: If True, verify on-disk checkpoint bytes when
            the referenced materials runtime artifacts are present locally.
            If omitted, this remains disabled unless
            ``TP_VERIFY_MATERIAL_RUNTIME_BYTES=1`` is set.

    Returns:
        Loaded preset dictionary (with any ``extends:`` chain resolved)

    Raises:
        FileNotFoundError: If preset file does not exist
        ValueError: If the loaded preset root is not a mapping.
        yaml.YAMLError: If YAML is malformed
        LicenseRestrictionError: If licensing markers are missing or the
            ``extends:`` chain is unresolvable or cyclic.
    """
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    preset = _load_with_extends(preset_path)

    validate_non_commercial_preset(preset)
    validate_materials_preset(
        preset,
        preset_path=preset_path,
        allow_research_materials=allow_research_materials,
        allow_unattested_materials=allow_unattested_materials,
        manifest_path=manifest_path,
        verify_runtime_bytes=verify_runtime_bytes,
    )
    return preset
