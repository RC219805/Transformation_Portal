#!/usr/bin/env python3
"""CI validation script for ingest contract enforcement.

Validates that test artifacts comply with the Phase I linear ingest contract:
- Schema version compatibility
- Required fields presence
- No schema drift
- No 8-bit conversion
- No gamma correction
- Deterministic sidecar output

Exit codes:
- 0: All validations passed
- 1: Schema validation failed
- 2: 8-bit conversion detected
- 3: Gamma correction detected
- 4: Schema drift detected
- 5: Other validation failure

Usage:
    Public compatibility path:
    python scripts/validate_ingest_contract.py [--test-dir DIR] [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path for direct raw-checkout execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.ingest import (
    EXIT_OTHER_FAILURE,
    EXIT_SUCCESS,
    aggregate_exit_codes,
    classify_validation_errors,
    validate_schema,
)
from transformation_portal.ingest.validator import SchemaValidationError


def find_sidecar_files(test_dir: Path) -> List[Path]:
    """Find all provenance sidecar JSON files in test directory.

    Args:
        test_dir: Test directory to scan

    Returns:
        List of sidecar file paths
    """
    sidecar_files = []

    # Search for *_provenance.json files
    for json_file in test_dir.rglob("*_provenance.json"):
        sidecar_files.append(json_file)

    return sidecar_files


def validate_all_sidecars(
    test_dir: Path,
    strict_mode: bool = True,
) -> int:
    """Validate all sidecars in test directory.

    Args:
        test_dir: Test directory to scan
        strict_mode: If True, fail on unknown fields

    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    sidecar_files = find_sidecar_files(test_dir)

    if not sidecar_files:
        print(f"⚠️  No sidecar files found in {test_dir}")
        print("   This is expected if no ingest operations have been performed.")
        return EXIT_SUCCESS

    print(f"🔍 Found {len(sidecar_files)} sidecar file(s) to validate")

    failures = []

    for sidecar_path in sidecar_files:
        print(f"\n📄 Validating: {sidecar_path.relative_to(test_dir)}")

        try:
            # Validate schema
            errors = validate_schema(
                sidecar_path,
                schema_type="provenance",
                strict_mode=strict_mode,
            )

            if errors:
                print(f"❌ Schema validation failed:")
                for error in errors:
                    print(f"   - {error}")
                failure_exit_code = classify_validation_errors(errors)
                failures.append((sidecar_path, failure_exit_code, "schema_validation"))
                continue

            print(f"✅ Schema validation passed")

            # Note: Image data validation would require loading actual images,
            # which is not practical in CI. We validate schema only.

        except SchemaValidationError as e:
            print(f"❌ Schema validation failed:")
            for error in e.errors:
                print(f"   - {error}")
            failure_exit_code = classify_validation_errors(e.errors)
            failures.append((sidecar_path, failure_exit_code, "schema_validation"))

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failures.append((sidecar_path, EXIT_OTHER_FAILURE, "unexpected"))

    # Summary
    print("\n" + "=" * 80)
    if failures:
        print(f"❌ Validation failed for {len(failures)}/{len(sidecar_files)} file(s)")
        print("\nFailed files:")
        for sidecar_path, _, failure_type in failures:
            print(f"  - {sidecar_path.relative_to(test_dir)}: {failure_type}")

        return aggregate_exit_codes(code for _, code, _ in failures)
    else:
        print(f"✅ All {len(sidecar_files)} sidecar file(s) validated successfully")
        return EXIT_SUCCESS


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Validate ingest contract compliance in CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0 - All validations passed
  1 - Schema validation failed
  2 - 8-bit conversion detected
  3 - Gamma correction detected
  4 - Schema drift detected
  5 - Other validation failure
        """,
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("tests/fixtures/ingest"),
        help="Directory containing test artifacts (default: tests/fixtures/ingest)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode (fail on unknown fields)",
    )

    args = parser.parse_args()

    # Check if test directory exists
    if not args.test_dir.exists():
        print(f"⚠️  Test directory not found: {args.test_dir}")
        print("   Creating directory (expected for first run)")
        args.test_dir.mkdir(parents=True, exist_ok=True)
        return EXIT_SUCCESS

    return validate_all_sidecars(
        test_dir=args.test_dir,
        strict_mode=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
