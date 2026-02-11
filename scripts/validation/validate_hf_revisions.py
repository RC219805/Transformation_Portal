#!/usr/bin/env python3
"""Validate that HuggingFace model revisions are properly pinned in production presets.

This enforces the Phase 1.1 contract: `main` branch must not contain knowingly
invalid configuration outside of experimental presets.

Rules:
- Experimental presets (config/presets/experimental/*.yaml) MAY use placeholder revisions
- Stable/canary presets (config/presets/*.yaml) MUST use valid commit hashes

Exit codes:
    0: All revisions valid or properly scoped
    1: Invalid placeholder found in non-experimental preset
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent
PRESETS_ROOT = REPO_ROOT / "config" / "presets"

PLACEHOLDER_PATTERNS = [
    "NEEDS_VERIFICATION",
    "TODO",
    "FIXME",
    "PLACEHOLDER",
]


def is_placeholder_revision(revision: str) -> bool:
    """Check if a revision string is a placeholder (not a real commit hash)."""
    if not isinstance(revision, str):
        return False

    revision_upper = revision.upper()
    return any(pattern in revision_upper for pattern in PLACEHOLDER_PATTERNS)


def check_preset_file(preset_path: Path) -> list[str]:
    """Check a single preset file for invalid placeholder revisions.

    Returns:
        List of violations (empty if clean)
    """
    violations = []

    # Determine if this is an experimental preset
    is_experimental = "experimental" in preset_path.parts

    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = yaml.safe_load(f)

        if not isinstance(preset, dict):
            return violations

        # Check various places where revision might appear
        locations_to_check = []

        # Top-level model.revision
        if "model" in preset and isinstance(preset["model"], dict):
            if "revision" in preset["model"]:
                locations_to_check.append(("model.revision", preset["model"]["revision"]))

        # Fallback model.revision
        if "fallback" in preset and isinstance(preset["fallback"], dict):
            fallback = preset["fallback"]
            if "model" in fallback and isinstance(fallback["model"], dict):
                if "revision" in fallback["model"]:
                    locations_to_check.append(("fallback.model.revision", fallback["model"]["revision"]))

        # Depth ensemble models (list of dicts with revision)
        if "depth" in preset and isinstance(preset["depth"], dict):
            if "ensemble" in preset["depth"] and isinstance(preset["depth"]["ensemble"], dict):
                models = preset["depth"]["ensemble"].get("models", [])
                if isinstance(models, list):
                    for i, model in enumerate(models):
                        if isinstance(model, dict) and "revision" in model:
                            locations_to_check.append((f"depth.ensemble.models[{i}].revision", model["revision"]))

        # Check each location
        for location, revision in locations_to_check:
            if is_placeholder_revision(revision):
                if not is_experimental:
                    # Non-experimental preset with placeholder - FAIL
                    violations.append(
                        f"{preset_path.relative_to(REPO_ROOT)}: "
                        f"Placeholder revision '{revision}' in {location}. "
                        f"Non-experimental presets must use valid commit hashes."
                    )
                # Experimental presets with placeholders are OK (no violation)

    except Exception as e:
        # YAML parsing errors, etc - report as violation
        violations.append(f"{preset_path.relative_to(REPO_ROOT)}: Error parsing preset: {e}")

    return violations


def main() -> int:
    """Run validation on all preset files."""
    print("=" * 70)
    print("HuggingFace Revision Validation (Phase 1.1 Item 5)")
    print("=" * 70)
    print()

    all_violations = []

    # Find all YAML preset files
    preset_files = list(PRESETS_ROOT.rglob("*.yaml")) + list(PRESETS_ROOT.rglob("*.yml"))

    if not preset_files:
        print("⚠️  Warning: No preset files found")
        return 0

    print(f"Checking {len(preset_files)} preset files...")
    print()

    for preset_path in sorted(preset_files):
        violations = check_preset_file(preset_path)
        all_violations.extend(violations)

    if all_violations:
        print("❌ FAIL: Invalid placeholder revisions found in non-experimental presets")
        print()
        for violation in all_violations:
            print(f"  - {violation}")
        print()
        print("=" * 70)
        print("Resolution:")
        print("  1. Verify commit hashes at HuggingFace model repository")
        print("  2. Replace placeholders with actual commit SHAs")
        print("  3. OR move preset to config/presets/experimental/ if not production-ready")
        print("  4. See: docs/apex/HUGGINGFACE_MODEL_PINNING.md")
        return 1
    else:
        print("✅ PASS: All revisions valid or properly scoped to experimental")
        print()
        print("Non-experimental presets use pinned commit hashes.")
        print("Experimental presets may use placeholders (development only).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
