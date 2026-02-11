#!/usr/bin/env python3
"""Verify pipeline isolation boundaries (ADR-023 enforcement).

This script ensures that the spatial_ai and lux_depth_v3 pipelines
maintain complete isolation in RAW decode logic, preventing silent
cross-contamination between rendering and training data paths.

Usage:
    python scripts/security/verify_pipeline_isolation.py

Exit codes:
    0: All isolation boundaries intact
    1: Isolation violation detected (CI should fail)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "transformation_portal"


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively."""
    return list(directory.rglob("*.py"))


def check_imports(filepath: Path, forbidden_patterns: List[str]) -> List[str]:
    """Check if file contains forbidden import patterns.

    Args:
        filepath: Path to Python file
        forbidden_patterns: List of import patterns to reject

    Returns:
        List of violations (empty if none found)
    """
    violations = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for pattern in forbidden_patterns:
            if pattern in content:
                violations.append(f"{filepath.relative_to(REPO_ROOT)}: {pattern}")

    except Exception as e:
        # Non-blocking: warn but don't fail on read errors
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

    return violations


def verify_no_spatial_imports_in_lux_depth() -> Tuple[bool, List[str]]:
    """Verify lux_depth_v3 does not import spatial_ai modules."""
    lux_depth_dir = SRC_ROOT / "lux_depth_v3"

    if not lux_depth_dir.exists():
        print(f"Warning: {lux_depth_dir} not found, skipping check", file=sys.stderr)
        return True, []

    forbidden_patterns = [
        "from transformation_portal.spatial_ai",
        "import transformation_portal.spatial_ai",
        "from ..spatial_ai",
        "from ...spatial_ai",  # 3-dot relative imports (fixed: no space)
        "from ....spatial_ai",  # 4-dot relative imports
    ]

    violations = []
    for filepath in find_python_files(lux_depth_dir):
        violations.extend(check_imports(filepath, forbidden_patterns))

    return len(violations) == 0, violations


def verify_no_lux_depth_decode_imports_in_spatial() -> Tuple[bool, List[str]]:
    """Verify spatial_ai.ingest does not import lux_depth_v3 decode logic."""
    spatial_ingest_dir = SRC_ROOT / "spatial_ai" / "ingest"

    if not spatial_ingest_dir.exists():
        # Not yet created (Phase I not started), skip check
        print(f"Info: {spatial_ingest_dir} not found, skipping check", file=sys.stderr)
        return True, []

    forbidden_patterns = [
        "from transformation_portal.lux_depth_v3.raw_loader",
        "import transformation_portal.lux_depth_v3.raw_loader",
        "from ..lux_depth_v3.raw_loader",
        "from ...lux_depth_v3.raw_loader",  # relative imports
        # Also block other lux_depth_v3 internal imports (only contracts allowed)
        "from transformation_portal.lux_depth_v3.preprocessing",
        "from transformation_portal.lux_depth_v3.postprocessing",
    ]

    violations = []
    for filepath in find_python_files(spatial_ingest_dir):
        violations.extend(check_imports(filepath, forbidden_patterns))

    return len(violations) == 0, violations


def verify_allowed_shared_utilities() -> Tuple[bool, List[str]]:
    """Verify shared utilities are metadata-only (no pixel decode logic).

    This check is informational (warns but doesn't fail) until utils/raw_metadata.py
    exists and needs enforcement.
    """
    utils_raw_metadata = SRC_ROOT / "utils" / "raw_metadata.py"

    if not utils_raw_metadata.exists():
        # Not yet created, skip
        return True, []

    # If created, warn if it contains pixel decode keywords
    # (Architect review required before this becomes enforced)
    warning_patterns = [
        "postprocess",  # LibRaw postprocessing
        "dcraw_process",
        "gamma",
        "white_balance",
        "demosaic",
    ]

    warnings = []
    for pattern in warning_patterns:
        violations = check_imports(utils_raw_metadata, [pattern])
        if violations:
            warnings.append(f"Warning: {utils_raw_metadata} may contain pixel decode logic (review required)")
            break

    return True, warnings  # Don't fail, just warn


def main() -> int:
    """Run all isolation checks."""
    print("=" * 70)
    print("ADR-023: Pipeline Isolation Verification")
    print("=" * 70)
    print()

    all_passed = True

    # Check 1: lux_depth_v3 isolation
    print("Check 1: lux_depth_v3 must not import spatial_ai...")
    passed, violations = verify_no_spatial_imports_in_lux_depth()
    if passed:
        print("✅ PASS: lux_depth_v3 isolation intact")
    else:
        print("❌ FAIL: lux_depth_v3 imports spatial_ai (ADR-023 violation)")
        for v in violations:
            print(f"  - {v}")
        all_passed = False
    print()

    # Check 2: spatial_ai.ingest isolation
    print("Check 2: spatial_ai.ingest must not import lux_depth_v3 decode logic...")
    passed, violations = verify_no_lux_depth_decode_imports_in_spatial()
    if passed:
        print("✅ PASS: spatial_ai.ingest isolation intact")
    else:
        print("❌ FAIL: spatial_ai.ingest imports lux_depth_v3 decode (ADR-023 violation)")
        for v in violations:
            print(f"  - {v}")
        all_passed = False
    print()

    # Check 3: Shared utilities (informational)
    print("Check 3: Shared utilities must be metadata-only (informational)...")
    passed, warnings = verify_allowed_shared_utilities()
    if warnings:
        for w in warnings:
            print(f"⚠️  {w}")
    else:
        print("✅ INFO: No shared utilities found or all metadata-only")
    print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("✅ All pipeline isolation checks passed")
        print()
        print("ADR-023 enforcement: COMPLIANT")
        return 0
    else:
        print("❌ Pipeline isolation violations detected")
        print()
        print("ADR-023 enforcement: VIOLATION")
        print()
        print("Remediation:")
        print("  1. Remove cross-pipeline imports")
        print("  2. Duplicate code if necessary (duplication > contamination)")
        print("  3. Use utils/raw_metadata.py for metadata-only sharing")
        print("  4. See: docs/architecture/ADR-023-spatial-ai-ingest-isolation.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
