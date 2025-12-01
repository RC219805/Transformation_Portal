#!/usr/bin/env python3
"""
Audit all Python files for TIFF saving methods.
Identifies any files still using PIL for TIFF saving (should use tifffile instead).
"""

import re
from pathlib import Path


def audit_file(filepath):
    """Check a Python file for TIFF saving methods."""
    with open(filepath, 'r') as f:
        content = f.read()

    issues = []

    # Check for PIL TIFF saves
    if re.search(r'\.save\([^)]*["\']tiff["\']', content, re.IGNORECASE):
        issues.append("Uses PIL .save() for TIFF")

    if re.search(r'\.save\([^)]*\.tiff?["\']', content, re.IGNORECASE):
        issues.append("Uses PIL .save() with .tif extension")

    # Check for good patterns
    has_tifffile_import = 'import tifffile' in content
    has_tifffile_write = 'tifffile.imwrite' in content or 'tifffile.imsave' in content
    has_correct_function = 'save_16bit_tiff_tifffile' in content

    return {
        'issues': issues,
        'has_tifffile': has_tifffile_import,
        'uses_tifffile': has_tifffile_write,
        'uses_correct_func': has_correct_function,
        'is_good': (has_tifffile_write or has_correct_function) and not issues
    }


def main():
    repo_root = Path.cwd()

    print("=" * 80)
    print("TIFF Saving Methods Audit")
    print("=" * 80)

    # Find all Python files (excluding tests, deprecated, venv)
    python_files = []
    for pattern in ['*.py', '*/*.py', '*/*/*.py']:
        for f in repo_root.glob(pattern):
            # Skip test files, deprecated, venv, .git
            if any(skip in str(f) for skip in ['test_', 'deprecated', '.venv', 'venv', '.git', '__pycache__']):
                continue
            python_files.append(f)

    # Audit each file
    results = {}
    for filepath in sorted(python_files):
        rel_path = filepath.relative_to(repo_root)
        result = audit_file(filepath)

        # Only report files that do TIFF operations
        if result['issues'] or result['uses_tifffile'] or result['uses_correct_func']:
            results[str(rel_path)] = result

    # Report results
    print("\n📊 Files with TIFF Operations:\n")

    good_files = []
    bad_files = []

    for filepath, result in results.items():
        if result['is_good']:
            good_files.append((filepath, result))
        elif result['issues']:
            bad_files.append((filepath, result))

    # Good files
    if good_files:
        print("✅ Files using OPTIMAL method (tifffile):")
        for filepath, result in good_files:
            markers = []
            if result['uses_tifffile']:
                markers.append("tifffile.imwrite")
            if result['uses_correct_func']:
                markers.append("save_16bit_tiff_tifffile")
            print(f"   ✓ {filepath:60s} ({', '.join(markers)})")

    # Bad files
    if bad_files:
        print("\n⚠️  Files using PROBLEMATIC method (PIL):")
        for filepath, result in bad_files:
            print(f"   ⚠️  {filepath:60s}")
            for issue in result['issues']:
                print(f"       → {issue}")
    else:
        print("\n✅ No files found using problematic PIL TIFF saving")

    print("\n" + "=" * 80)
    print(f"Summary: {len(good_files)} optimal, {len(bad_files)} need attention")
    print("=" * 80)

    if bad_files:
        print("\n⚠️  Recommendation: Update problematic files to use tifffile.imwrite()")
        return 1
    else:
        print("\n✅ All TIFF-related code uses optimal saving methods")
        return 0


if __name__ == "__main__":
    exit(main())
