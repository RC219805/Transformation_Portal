#!/usr/bin/env python3
"""Verify pipeline isolation boundaries (ADR-023 enforcement).

This script ensures that the spatial_ai and lux_depth_v3 pipelines
maintain complete isolation in RAW decode logic, preventing silent
cross-contamination between rendering and training data paths.

Uses AST-based import parsing for precision (ignores comments/docstrings).

Usage:
    python scripts/security/verify_pipeline_isolation.py

Exit codes:
    0: All isolation boundaries intact
    1: Isolation violation detected (CI should fail)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "transformation_portal"


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory recursively."""
    return list(directory.rglob("*.py"))


def check_imports_ast(filepath: Path, forbidden_modules: List[str]) -> List[str]:
    """Check if file contains forbidden imports using AST parsing.

    This is more precise than string matching - only catches actual imports,
    not comments or docstrings mentioning the module.

    Args:
        filepath: Path to Python file
        forbidden_modules: List of module patterns to reject (e.g., "spatial_ai", "lux_depth_v3.raw_loader")

    Returns:
        List of violations with line numbers (empty if none found)
    """
    violations = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))

        for node in ast.walk(tree):
            # Check "from X import Y" statements
            if isinstance(node, ast.ImportFrom):
                # For relative imports, node.module contains the part after the dots
                # For absolute imports, node.module is the full path
                module_name = node.module or ""

                # Check against forbidden patterns
                for forbidden in forbidden_modules:
                    # Match if:
                    # 1. Module exactly equals forbidden (e.g., "spatial_ai")
                    # 2. Module starts with forbidden as prefix (e.g., "spatial_ai.ingest")
                    # 3. Forbidden appears as a path component (e.g., "transformation_portal.spatial_ai")
                    is_match = (
                        module_name == forbidden
                        or module_name.startswith(forbidden + ".")
                        or ("." + forbidden + ".") in ("." + module_name + ".")
                    )

                    if is_match:
                        # Build display string for violation
                        if node.level > 0:
                            rel_path = ["."] * node.level
                            if module_name:
                                rel_path.append(module_name)
                            display_path = "".join(rel_path)
                        else:
                            display_path = module_name

                        import_stmt = ast.unparse(node) if hasattr(ast, "unparse") else f"from {display_path} import ..."
                        # Handle paths both inside and outside repo
                        try:
                            file_display = str(filepath.relative_to(REPO_ROOT))
                        except ValueError:
                            file_display = str(filepath)
                        violations.append(f"{file_display}:{node.lineno}: {import_stmt} (matches forbidden: {forbidden})")

            # Check "import X" statements (less common but possible)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        # Match using same logic as ImportFrom
                        is_match = (
                            alias.name == forbidden
                            or alias.name.startswith(forbidden + ".")
                            or ("." + forbidden + ".") in ("." + alias.name + ".")
                        )

                        if is_match:
                            import_stmt = ast.unparse(node) if hasattr(ast, "unparse") else f"import {alias.name}"
                            try:
                                file_display = str(filepath.relative_to(REPO_ROOT))
                            except ValueError:
                                file_display = str(filepath)
                            violations.append(f"{file_display}:{node.lineno}: {import_stmt} (matches forbidden: {forbidden})")

    except SyntaxError as e:
        # Syntax errors are real problems - report them
        try:
            file_display = str(filepath.relative_to(REPO_ROOT))
        except ValueError:
            file_display = str(filepath)
        violations.append(f"{file_display}:{e.lineno}: SyntaxError: {e.msg}")
    except Exception as e:
        # Other errors (e.g., encoding): warn but don't fail
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)

    return violations


def verify_no_spatial_imports_in_lux_depth() -> Tuple[bool, List[str]]:
    """Verify lux_depth_v3 does not import spatial_ai modules."""
    lux_depth_dir = SRC_ROOT / "lux_depth_v3"

    if not lux_depth_dir.exists():
        print(f"Warning: {lux_depth_dir} not found, skipping check", file=sys.stderr)
        return True, []

    # Forbidden module patterns (will match in import paths)
    forbidden_modules = [
        "spatial_ai",  # Catches both absolute and relative
    ]

    violations = []
    for filepath in find_python_files(lux_depth_dir):
        violations.extend(check_imports_ast(filepath, forbidden_modules))

    return len(violations) == 0, violations


def verify_no_lux_depth_decode_imports_in_spatial() -> Tuple[bool, List[str]]:
    """Verify spatial_ai.ingest does not import lux_depth_v3 decode logic."""
    spatial_ingest_dir = SRC_ROOT / "spatial_ai" / "ingest"

    if not spatial_ingest_dir.exists():
        # Not yet created (Phase I not started), skip check
        print(f"Info: {spatial_ingest_dir} not found, skipping check", file=sys.stderr)
        return True, []

    # Forbidden module patterns
    forbidden_modules = [
        "lux_depth_v3.raw_loader",
        "lux_depth_v3.preprocessing",
        "lux_depth_v3.postprocessing",
    ]

    violations = []
    for filepath in find_python_files(spatial_ingest_dir):
        violations.extend(check_imports_ast(filepath, forbidden_modules))

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
    # For this check, we still use string matching since we're looking for
    # function/variable names, not imports
    warning_patterns = [
        "postprocess",  # LibRaw postprocessing
        "dcraw_process",
        "gamma",
        "white_balance",
        "demosaic",
    ]

    warnings = []
    try:
        with open(utils_raw_metadata, "r", encoding="utf-8") as f:
            content = f.read()

        for pattern in warning_patterns:
            if pattern in content:
                warnings.append(f"Warning: {utils_raw_metadata} may contain pixel decode logic (review required)")
                break
    except Exception as e:
        print(f"Warning: Could not read {utils_raw_metadata}: {e}", file=sys.stderr)

    return True, warnings  # Don't fail, just warn


def main() -> int:
    """Run all isolation checks."""
    print("=" * 70)
    print("ADR-023: Pipeline Isolation Verification (AST-based)")
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
