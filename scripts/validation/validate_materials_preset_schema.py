#!/usr/bin/env python3
"""Validate approved schema locations for materials backend declarations.

This guard prevents materials backend configuration from drifting into ad hoc
YAML locations that the runtime governance validators would miss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformation_portal.attestation.materials_policy import (  # noqa: E402
    ALLOWED_MATERIAL_BACKEND_PATHS,
    find_unknown_material_backend_schema_locations,
    looks_like_material_preset,
    material_preset_family_error,
)


def find_preset_files() -> list[Path]:
    """Return all checked-in preset YAML files."""
    return sorted((PROJECT_ROOT / "config/presets").glob("**/*.yaml"))


def check_preset(preset_path: Path) -> list[str]:
    """Return schema issues for a preset file."""
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        return [f"{preset_path}: preset root must be a mapping"]

    issues: list[str] = []
    family_error = material_preset_family_error(payload, preset_path)
    if family_error is not None:
        issues.append(f"{preset_path}: {family_error}")

    if not looks_like_material_preset(payload, preset_path):
        return issues

    for path in find_unknown_material_backend_schema_locations(payload, preset_path):
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
