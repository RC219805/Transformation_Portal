"""Lightweight shared materials governance policy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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


def normalize_material_backend(value: Any) -> str | None:
    """Normalize backend identifiers used in materials configs and presets."""
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    return MATERIAL_BACKEND_ALIASES.get(normalized, normalized)


def looks_like_material_preset(payload: dict[str, Any], preset_path: Optional[Path]) -> bool:
    """Heuristically detect a materials-oriented preset/config payload."""
    if preset_path is not None and "material_pbr" in preset_path.as_posix().lower():
        return True

    name = payload.get("name")
    if isinstance(name, str) and "pbr material" in name.lower():
        return True

    pipeline_cfg = payload.get("pipeline")
    if isinstance(pipeline_cfg, dict) and isinstance(pipeline_cfg.get("materials"), dict):
        return True

    return isinstance(payload.get("materials"), dict)


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
