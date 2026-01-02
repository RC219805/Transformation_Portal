"""
Static validation tests that don't require dependencies.
Tests code structure, configuration values, and documentation.
"""

import os
import re
from pathlib import Path


def test_module_structure():
    """Verify expected module structure exists."""
    print("=== Module Structure Test ===\n")

    required_files = [
        "config.py",
        "da3_integration.py",
        "da3_wrapper.py",
        "metric_depth.py",
        "license.py",
        "cli.py",
        "inference.py",
        "enhance/__init__.py",
        "enhance/orchestrator.py",
        "enhance/depth_writer.py",
        "enhance/manifest.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
    ]

    missing = []
    found = []

    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path}")
            found.append(file_path)
        else:
            print(f"✗ {file_path} - MISSING")
            missing.append(file_path)

    print(f"\n{len(found)}/{len(required_files)} required files found")
    return len(missing) == 0


def test_config_structure():
    """Verify config.py has expected model variants."""
    print("\n=== Config Structure Test ===\n")

    with open("config.py", "r") as f:
        config_content = f.read()

    # Check for model variants
    expected_variants = [
        "DA3_BASE",
        "DA3_SMALL",
        "DA3_LARGE_V1_1",
        "DA3_GIANT_V1_1",
        "DA3_NESTED_GIANT_LARGE_V1_1",
        "DA3_METRIC_LARGE",
        "DA3_MONO_LARGE",
    ]

    found_variants = []
    missing_variants = []

    for variant in expected_variants:
        if variant in config_content:
            print(f"✓ {variant}")
            found_variants.append(variant)
        else:
            print(f"✗ {variant} - MISSING")
            missing_variants.append(variant)

    print(f"\n{len(found_variants)}/{len(expected_variants)} model variants defined")

    # Check for ModelLicense enum
    if "class ModelLicense" in config_content:
        print("✓ ModelLicense enum defined")
    else:
        print("✗ ModelLicense enum - MISSING")
        return False

    return len(missing_variants) == 0


def test_metric_depth_structure():
    """Verify metric_depth.py structure."""
    print("\n=== Metric Depth Structure Test ===\n")

    with open("metric_depth.py", "r") as f:
        content = f.read()

    expected_classes = [
        "MetricDepthConverter",
        "MetricDepthResult",
    ]

    all_found = True
    for class_name in expected_classes:
        if f"class {class_name}" in content:
            print(f"✓ {class_name} class defined")
        else:
            print(f"✗ {class_name} class - MISSING")
            all_found = False

    # Check for DA3METRIC-LARGE support
    if "DA3METRIC-LARGE" in content:
        print("✓ DA3METRIC-LARGE support")
    else:
        print("✗ DA3METRIC-LARGE support - MISSING")
        all_found = False

    return all_found


def test_license_validation_structure():
    """Verify license.py structure."""
    print("\n=== License Validation Structure Test ===\n")

    with open("license.py", "r") as f:
        content = f.read()

    if "class LicenseValidator" in content:
        print("✓ LicenseValidator class defined")
    else:
        print("✗ LicenseValidator class - MISSING")
        return False

    # Check for commercial use validation (method name is check_commercial_use)
    if "check_commercial_use" in content:
        print("✓ check_commercial_use method")
    else:
        print("✗ check_commercial_use method - MISSING")
        return False

    if "get_commercial_alternative" in content:
        print("✓ get_commercial_alternative method")

    return True


def test_orchestrator_structure():
    """Verify enhance/orchestrator.py structure."""
    print("\n=== Orchestrator Structure Test ===\n")

    orch_path = Path("enhance/orchestrator.py")
    if not orch_path.exists():
        print("✗ enhance/orchestrator.py - FILE MISSING")
        return False

    with open(orch_path, "r") as f:
        content = f.read()

    # Check for EnhanceOrchestrator class
    if "class EnhanceOrchestrator" not in content:
        print("✗ EnhanceOrchestrator class - MISSING")
        return False

    print("✓ EnhanceOrchestrator class defined")

    # Check for main processing methods
    expected_methods = [
        "enhance_image",
        "enhance_batch",
    ]

    all_found = True
    for method in expected_methods:
        if f"def {method}" in content:
            print(f"✓ {method} method defined")
        else:
            print(f"✗ {method} method - MISSING")
            all_found = False

    return all_found


def test_documentation():
    """Verify key documentation exists."""
    print("\n=== Documentation Test ===\n")

    docs = [
        "README.md",
        "QUICK_START.md",
        "INTEGRATION_GUIDE.md",
        "SECURITY.md",
    ]

    all_found = True
    for doc in docs:
        if Path(doc).exists():
            size_kb = Path(doc).stat().st_size / 1024
            print(f"✓ {doc} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {doc} - MISSING")
            all_found = False

    return all_found


def test_cli_structure():
    """Verify CLI entry points."""
    print("\n=== CLI Structure Test ===\n")

    with open("cli.py", "r") as f:
        content = f.read()

    # Check for main commands
    if "def enhance(" in content:
        print("✓ enhance command defined")
    else:
        print("✗ enhance command - MISSING")
        return False

    if "def infer(" in content:
        print("✓ infer command defined")
    else:
        print("✗ infer command - MISSING")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("LUX DEPTH V3 - Static Validation Tests")
    print("(No external dependencies required)")
    print("=" * 60 + "\n")

    results = {
        "Module Structure": test_module_structure(),
        "Config Structure": test_config_structure(),
        "Metric Depth": test_metric_depth_structure(),
        "License Validation": test_license_validation_structure(),
        "Orchestrator": test_orchestrator_structure(),
        "CLI Structure": test_cli_structure(),
        "Documentation": test_documentation(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All static validation tests PASSED")
        print("  Code structure is correct - ready for dependency installation")
        exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("  Review failures before proceeding")
        exit(1)
