#!/usr/bin/env python3
"""Validate license compliance of presets and models.

This script checks that non-commercial models (e.g., DA3 1.1) are properly
marked with license_restriction metadata in their preset YAML files.

Usage:
    python -m transformation_portal.compliance.validate_licenses --check-presets config/presets/
"""

import argparse
import sys
from pathlib import Path
from typing import List

import yaml

# Non-commercial model identifiers
NON_COMMERCIAL_IDENTIFIERS = [
    "DA3-Large-1.1",
    "DA3-Base-1.1",
    "DA3-Small-1.1",
    "DA3NESTED-GIANT-LARGE-1.1",
]


def validate_preset_file(preset_path: Path) -> tuple[bool, List[str]]:
    """Validate a single preset file for licensing compliance.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    try:
        with open(preset_path) as f:
            preset = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML error: {e}"]
    except Exception as e:
        return False, [f"Read error: {e}"]

    if not preset:
        return True, []

    model = preset.get("model", {})
    hf_id = model.get("hf_id", "")

    # Check if this is a known non-commercial model
    is_non_commercial = any(identifier in hf_id for identifier in NON_COMMERCIAL_IDENTIFIERS)

    if is_non_commercial:
        # Verify it has the required marker
        license_restriction = preset.get("license_restriction")
        if license_restriction != "non_commercial":
            issues.append(f"Non-commercial model ({hf_id}) missing " "license_restriction='non_commercial' marker")

    return len(issues) == 0, issues


def main() -> int:
    """Validate license compliance of presets.

    Returns:
        Exit code: 0 if all presets are compliant, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate license compliance of presets")
    parser.add_argument("--check-presets", type=Path, help="Directory containing preset YAML files")
    args = parser.parse_args()

    if not args.check_presets:
        print("Usage: python -m transformation_portal.compliance.validate_licenses " "--check-presets config/presets/")
        return 1

    if not args.check_presets.exists():
        print(f"Error: Directory not found: {args.check_presets}")
        return 1

    preset_files = list(args.check_presets.glob("*.yaml"))
    if not preset_files:
        print(f"No YAML preset files found in {args.check_presets}")
        return 0

    all_valid = True
    for preset_file in sorted(preset_files):
        is_valid, issues = validate_preset_file(preset_file)

        if not is_valid:
            print(f"❌ {preset_file.name}")
            for issue in issues:
                print(f"   - {issue}")
            all_valid = False
        else:
            if issues:
                print(f"⚠️  {preset_file.name}")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print(f"✓ {preset_file.name}")

    if all_valid:
        print("\n✅ All presets are compliant")
        return 0
    else:
        print("\n❌ Some presets have compliance issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
