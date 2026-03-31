#!/usr/bin/env python3
"""Validate preset licensing and governance compliance.

This script routes preset files through the shared compliance loader so
non-commercial depth gates and materials governance rules stay consistent
with the runtime pipeline.

Usage:
    python -m transformation_portal.compliance.validate_licenses --check-presets config/presets/
"""

import argparse
import sys
from pathlib import Path
from typing import List

import yaml

from transformation_portal.compliance import LicenseRestrictionError, load_and_validate_preset


def validate_preset_file(preset_path: Path) -> tuple[bool, List[str]]:
    """Validate a single preset file for licensing and governance compliance.

    Returns:
        (is_valid, list_of_issues)
    """
    try:
        load_and_validate_preset(preset_path)
    except yaml.YAMLError as e:
        return False, [f"YAML error: {e}"]
    except LicenseRestrictionError as e:
        return False, [str(e)]
    except (FileNotFoundError, ValueError) as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Read error: {e}"]

    return True, []


def main() -> int:
    """Validate license compliance of presets.

    Returns:
        Exit code: 0 if all presets are compliant, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate license compliance of presets")
    parser.add_argument("--check-presets", type=Path, help="Directory containing preset YAML files (scanned recursively)")
    args = parser.parse_args()

    if not args.check_presets:
        print("Usage: python -m transformation_portal.compliance.validate_licenses " "--check-presets config/presets/")
        return 1

    if not args.check_presets.exists():
        print(f"Error: Directory not found: {args.check_presets}")
        return 1

    preset_files = [path for path in args.check_presets.rglob("*.yaml") if path.is_file()]
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
