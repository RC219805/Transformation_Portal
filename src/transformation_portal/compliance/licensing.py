"""Licensing compliance enforcement and validation.

Provides decorators and validators to ensure non-commercial and
research-only backends are used only with explicit authorization and
attested source metadata.
"""

import functools
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import yaml

from transformation_portal.spatial_ai.materials.contracts import VALID_MATERIAL_BACKENDS


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
UNATTESTED_ALLOWED_MATERIAL_TIERS = frozenset({"dev", "experimental"})
FLOATING_REVISIONS = frozenset({"main", "master", "latest", "head", "tip", "default"})
PLACEHOLDER_MARKERS = ("NEEDS_VERIFICATION", "PLACEHOLDER", "PENDING", "TODO", "TBD", "UPDATE_WHEN")
MATERIAL_BACKEND_ALIASES = {"materialgan": "material_gan"}
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def require_non_commercial(reason: str = "") -> Callable[[F], F]:
    """Decorator enforcing non-commercial usage authorization.

    This decorator ensures that functions using non-commercial models
    (e.g., DA3 1.1) only execute when the caller has explicitly set
    `non_commercial_ok=True` in their configuration.

    Args:
        reason: Human-readable explanation of the licensing restriction
                (e.g., "DA3 1.1 uses CC BY-NC 4.0 models")

    Raises:
        LicenseRestrictionError: If the configuration does not have
                                `non_commercial_ok=True`

    Example:
        ```python
        @require_non_commercial(reason="DA3 1.1 uses CC BY-NC 4.0 models")
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

    Checks if a preset dictionary (from YAML) contains a known non-commercial
    model. If so, ensures the preset has explicit `license_restriction: non_commercial`
    marker.

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

    # Check for known non-commercial models
    non_commercial_identifiers = [
        "DA3-Large-1.1",
        "DA3-Base-1.1",
        "DA3-Small-1.1",
        "DA3NESTED-GIANT-LARGE-1.1",
    ]

    is_non_commercial_model = any(identifier in hf_id for identifier in non_commercial_identifiers)

    if is_non_commercial_model:
        # Verify marker exists
        license_restriction = preset_dict.get("license_restriction")
        if license_restriction != "non_commercial":
            raise LicenseRestrictionError(
                f"Preset uses non-commercial model (hf_id={hf_id}) "
                "but lacks license_restriction='non_commercial' marker.\n"
                "Please add this marker to acknowledge CC BY-NC 4.0 restrictions."
            )

    return True


def _normalize_material_backend(backend: Any) -> Optional[str]:
    """Normalize backend identifiers used in materials presets."""
    if not isinstance(backend, str):
        return None

    normalized = backend.strip().lower()
    return MATERIAL_BACKEND_ALIASES.get(normalized, normalized)


def _looks_placeholder(value: Any) -> bool:
    """Return True when a preset field is clearly unresolved."""
    if not isinstance(value, str):
        return False

    normalized = value.strip()
    if not normalized:
        return True

    upper = normalized.upper()
    return any(marker in upper for marker in PLACEHOLDER_MARKERS)


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


def _looks_like_material_preset(preset_dict: Dict[str, Any], preset_path: Optional[Path]) -> bool:
    """Heuristically detect the dedicated material PBR preset family."""
    if preset_path is not None and "material_pbr" in preset_path.as_posix().lower():
        return True

    name = preset_dict.get("name")
    return isinstance(name, str) and "pbr material" in name.lower()


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


def validate_materials_preset(
    preset_dict: Dict[str, Any],
    *,
    preset_path: Optional[Path] = None,
    allow_research_materials: bool = False,
    allow_unattested_materials: bool = False,
) -> bool:
    """Validate materials backend tier, licensing, and attestation policy."""
    if not isinstance(preset_dict, dict):
        raise ValueError(f"Preset must be a mapping (dict), got {type(preset_dict).__name__}.")

    tier = str(preset_dict.get("tier", "")).strip().lower()
    license_restriction = preset_dict.get("license_restriction")

    for source_path, backend, model_dict in _iter_material_backend_specs(preset_dict, preset_path):
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

            if not allow_research_materials:
                raise LicenseRestrictionError(
                    f"Materials backend '{backend}' is research-only. "
                    "Reload this preset with allow_research_materials=True to acknowledge the restriction."
                )

        if _has_attested_material_source(model_dict):
            continue

        if allow_unattested_materials and tier in UNATTESTED_ALLOWED_MATERIAL_TIERS:
            continue

        raise LicenseRestrictionError(
            f"Materials backend '{backend}' in {source_path} lacks an attested source tuple "
            "(repo_id+revision or checkpoint+expected_sha256). "
            "Only dev/experimental presets may bypass this with allow_unattested_materials=True."
        )

    return True


def load_and_validate_preset(
    preset_path: Path,
    *,
    allow_research_materials: bool = False,
    allow_unattested_materials: bool = False,
) -> Dict[str, Any]:
    """Load a preset YAML file and validate licensing compliance.

    Args:
        preset_path: Path to preset YAML file
        allow_research_materials: Explicit opt-in required for research-only
            materials backends such as NVDIFFREC and MaterialGAN.
        allow_unattested_materials: Allow unresolved material source tuples in
            dev/experimental presets only.

    Returns:
        Loaded preset dictionary

    Raises:
        FileNotFoundError: If preset file does not exist
        ValueError: If the loaded preset root is not a mapping.
        yaml.YAMLError: If YAML is malformed
        LicenseRestrictionError: If licensing markers are missing
    """
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    with open(preset_path) as f:
        preset = yaml.safe_load(f)

    if not isinstance(preset, dict):
        raise ValueError(f"Preset must be a mapping (dict), got {type(preset).__name__}.")

    validate_non_commercial_preset(preset)
    validate_materials_preset(
        preset,
        preset_path=preset_path,
        allow_research_materials=allow_research_materials,
        allow_unattested_materials=allow_unattested_materials,
    )
    return preset
