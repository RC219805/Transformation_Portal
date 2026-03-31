#!/usr/bin/env python3
"""Validate approved schema locations for materials backend declarations.

This guard prevents materials backend configuration from drifting into ad hoc
YAML locations that the runtime governance validators would miss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_MATERIAL_BACKENDS = frozenset({"pbr_fusion", "nvdiffrec", "material_gan", "materialgan", "heuristic"})
ALLOWED_MATERIAL_BACKEND_PATHS = frozenset(
    {
        "materials.backend",
        "pipeline.materials.backend",
        "backend.type",
        "model.backend",
    }
)


def find_preset_files() -> list[Path]:
    """Return all checked-in preset YAML files."""
    return sorted(Path("config/presets").glob("**/*.yaml"))


def _normalize_backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "materialgan":
        return "material_gan"
    return normalized


def _looks_like_material_preset(payload: dict[str, Any], preset_path: Path) -> bool:
    if "material_pbr" in preset_path.as_posix().lower():
        return True

    name = payload.get("name")
    if isinstance(name, str) and "pbr material" in name.lower():
        return True

    pipeline_cfg = payload.get("pipeline")
    if isinstance(pipeline_cfg, dict) and isinstance(pipeline_cfg.get("materials"), dict):
        return True

    return isinstance(payload.get("materials"), dict)


def iter_material_backend_declaration_paths(payload: dict[str, Any], preset_path: Path) -> list[str]:
    """Return all material backend declaration paths inside a preset."""
    paths: list[str] = []

    def _walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = path + (str(key),)
                if key in {"backend", "type"} and _normalize_backend(value) in VALID_MATERIAL_BACKENDS:
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

    if _looks_like_material_preset(payload, preset_path):
        backend_cfg = payload.get("backend")
        if isinstance(backend_cfg, dict):
            _walk(backend_cfg, ("backend",))
        model_cfg = payload.get("model")
        if isinstance(model_cfg, dict):
            _walk(model_cfg, ("model",))

    return sorted(set(paths))


def check_preset(preset_path: Path) -> list[str]:
    """Return schema issues for a preset file."""
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        return [f"{preset_path}: preset root must be a mapping"]

    if not _looks_like_material_preset(payload, preset_path):
        return []

    issues: list[str] = []
    for path in iter_material_backend_declaration_paths(payload, preset_path):
        if path not in ALLOWED_MATERIAL_BACKEND_PATHS:
            issues.append(
                f"{preset_path}: materials backend declaration at '{path}' is not allowed "
                f"(allowed: {sorted(ALLOWED_MATERIAL_BACKEND_PATHS)})"
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate materials backend declaration schema locations")
    parser.add_argument("paths", nargs="*", help="Optional preset paths to validate instead of config/presets/**/*.yaml")
    args = parser.parse_args()

    preset_files = [Path(path) for path in args.paths] if args.paths else find_preset_files()
    issues: list[str] = []
    for preset_path in preset_files:
        issues.extend(check_preset(preset_path))

    if not issues:
        print("✅ Materials preset schema locations are valid")
        return 0

    print("❌ Materials preset schema violations detected:")
    for issue in issues:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
