"""Lightweight shared materials governance policy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from transformation_portal.preset_governance import (
    MATERIALS_PBR_PRESET_FAMILY,
    is_material_preset_family,
    normalize_preset_family,
)

VALID_MATERIAL_BACKENDS = ("pbr_fusion", "nvdiffrec", "material_gan", "heuristic")
MATERIAL_BACKEND_ALIASES = {"materialgan": "material_gan"}
ALLOWED_MATERIAL_BACKEND_PATHS = frozenset(
    {
        "materials.backend",
        "pipeline.materials.backend",
        "backend.type",
        "model.backend",
    }
)


def _has_nested_materials_sections(payload: dict[str, Any]) -> bool:
    """Return True when a payload embeds a materials stage/config section."""
    if isinstance(payload.get("materials"), dict):
        return True

    pipeline_cfg = payload.get("pipeline")
    return isinstance(pipeline_cfg, dict) and isinstance(pipeline_cfg.get("materials"), dict)


def normalize_material_backend(value: Any) -> str | None:
    """Normalize backend identifiers used in materials configs and presets."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    return MATERIAL_BACKEND_ALIASES.get(normalized, normalized)


def _looks_like_top_level_material_preset(payload: dict[str, Any], preset_path: Optional[Path]) -> bool:
    """Return True when top-level metadata indicates a dedicated Material PBR preset."""
    if "pbr" in payload:
        return True

    if preset_path is not None and "material_pbr" in preset_path.as_posix().lower():
        return True

    name = payload.get("name")
    return isinstance(name, str) and "pbr material" in name.lower()


def looks_like_material_preset(payload: dict[str, Any], preset_path: Optional[Path]) -> bool:
    """Heuristically detect a materials-oriented preset/config payload."""
    family = normalize_preset_family(payload.get("preset_family"))
    if is_material_preset_family(family):
        return True

    if _has_nested_materials_sections(payload):
        return True

    if uses_top_level_material_preset_schema(payload) and _looks_like_top_level_material_preset(payload, preset_path):
        return True

    return False


def uses_top_level_material_preset_schema(payload: dict[str, Any]) -> bool:
    """Return True when a payload uses the legacy top-level materials preset schema."""
    backend_cfg = payload.get("backend")
    if isinstance(backend_cfg, dict):
        backend = normalize_material_backend(backend_cfg.get("type"))
        if backend in VALID_MATERIAL_BACKENDS:
            return True

    model_cfg = payload.get("model")
    if isinstance(model_cfg, dict):
        backend = normalize_material_backend(model_cfg.get("backend"))
        if backend in VALID_MATERIAL_BACKENDS:
            return True

    return False


def missing_material_preset_family_marker(payload: dict[str, Any], preset_path: Optional[Path]) -> bool:
    """Return True when a top-level materials preset omits the explicit family marker."""
    return (
        uses_top_level_material_preset_schema(payload)
        and _looks_like_top_level_material_preset(payload, preset_path)
        and not is_material_preset_family(payload.get("preset_family"))
    )


def material_preset_family_error(payload: dict[str, Any], preset_path: Optional[Path]) -> str | None:
    """Return a stable validation error for missing/incorrect materials preset family markers."""
    if not missing_material_preset_family_marker(payload, preset_path):
        return None

    actual_family = payload.get("preset_family")
    actual_suffix = "" if actual_family is None else f", got {actual_family!r}"
    return (
        "Top-level materials presets must declare "
        f"preset_family='{MATERIALS_PBR_PRESET_FAMILY}' to avoid schema drift across backend/model forms"
        f"{actual_suffix}."
    )


def iter_material_backend_declaration_paths(payload: dict[str, Any], preset_path: Optional[Path]) -> list[str]:
    """Return all material backend declaration paths inside a payload."""
    paths: list[str] = []

    def _walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = path + (str(key),)
                normalized = normalize_material_backend(value) if key in {"backend", "type"} else None
                if normalized in VALID_MATERIAL_BACKENDS:
                    paths.append(".".join(next_path))
                _walk(value, next_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, path + (str(index),))

    if isinstance(payload.get("materials"), dict):
        _walk(payload["materials"], ("materials",))

    pipeline_cfg = payload.get("pipeline")
    if isinstance(pipeline_cfg, dict) and isinstance(pipeline_cfg.get("materials"), dict):
        _walk(pipeline_cfg["materials"], ("pipeline", "materials"))

    if looks_like_material_preset(payload, preset_path):
        backend_cfg = payload.get("backend")
        if isinstance(backend_cfg, dict):
            _walk(backend_cfg, ("backend",))
        model_cfg = payload.get("model")
        if isinstance(model_cfg, dict):
            _walk(model_cfg, ("model",))

    return sorted(set(paths))


def find_unknown_material_backend_schema_locations(payload: dict[str, Any], preset_path: Optional[Path]) -> list[str]:
    """Return backend declaration paths outside the approved materials schema."""
    return [
        path
        for path in iter_material_backend_declaration_paths(payload, preset_path)
        if path not in ALLOWED_MATERIAL_BACKEND_PATHS
    ]
