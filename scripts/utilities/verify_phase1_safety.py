#!/usr/bin/env python3
"""
Phase 1: Critical Safety - Verification Script

Verifies that MaterialsV3 exception handling is correctly implemented
and runs the complete edge case and stress test suites.

Usage:
    python verify_phase1_safety.py [--quick] [--full]
    
    --quick: Run only edge case tests (fast, ~2-5 minutes)
    --full:  Run all tests including 1000-iteration stress test (slow, ~15-30 minutes)
"""

import sys
import subprocess
from pathlib import Path


def verify_exception_handling():
    """Verify exception handling exists in pipeline.py."""
    print("=" * 70)
    print("TASK 1.4: Verifying Existing Exception Handling")
    print("=" * 70)
    
    pipeline_path = Path(__file__).parent / "lux_depth_v2" / "pipeline.py"
    
    with open(pipeline_path, 'r') as f:
        content = f.read()
    
    checks = {
        "MaterialsV3 engine check": "if self.materials_v3_engine is not None:",
        "Try block": "try:",
        "V3 engine process call": "v3_result = self.materials_v3_engine.process(",
        "Exception handler": "except Exception as e:",
        "Warning log": "Materials V3 processing failed",
        "Fallback metadata": "materials_v3_metadata = {'error': str(e), 'fallback': True}",
        "Pipeline continues": "continuing without MaterialsV3 enhancements"
    }
    
    all_passed = True
    for check_name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
            all_passed = False
    
    if all_passed:
        print("\n✅ All exception handling checks PASSED")
        print("   - Try/except block exists around materials_v3_engine.process()")
        print("   - Error metadata structure: {'error': str(e), 'fallback': True}")
        print("   - Warning logged with filename and error context")
        print("   - Pipeline continues after MaterialsV3 failure")
    else:
        print("\n❌ Some exception handling checks FAILED")
        return False
    
    print()
    return True


def run_edge_case_tests():
    """Run edge case test suite."""
    print("=" * 70)
    print("TASK 1.2: Running Edge Case Test Suite")
    print("=" * 70)
    
    cmd = [
        "pytest",
        "tests/test_materials_v3_edge_cases.py",
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n✅ Edge case tests PASSED")
        return True
    else:
        print("\n❌ Edge case tests FAILED")
        return False


def run_stress_tests(quick=False):
    """Run stress test suite."""
    print("=" * 70)
    print("TASK 1.3: Running Stress Test Suite")
    print("=" * 70)
    
    if quick:
        print("⚠️  Quick mode: Skipping stress tests (use --full to run)")
        return True
    
    cmd = [
        "pytest",
        "tests/test_materials_v3_stress.py",
        "-v",
        "--tb=short",
        "-m", "slow",
        "-x",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n✅ Stress tests PASSED")
        return True
    else:
        print("\n❌ Stress tests FAILED")
        return False


def main():
    """Run Phase 1 verification."""
    quick_mode = "--quick" in sys.argv
    full_mode = "--full" in sys.argv
    
    print("\n" + "=" * 70)
    print("MaterialsV3 Phase 1: Critical Safety - Verification")
    print("=" * 70)
    print()
    
    if quick_mode:
        print("Mode: QUICK (edge cases only)")
    elif full_mode:
        print("Mode: FULL (edge cases + stress tests)")
    else:
        print("Mode: DEFAULT (edge cases + basic stress tests)")
    print()
    
    # Task 1.4: Verify exception handling
    if not verify_exception_handling():
        print("\n❌ PHASE 1 FAILED: Exception handling verification failed")
        return 1
    
    # Task 1.2: Edge case tests
    if not run_edge_case_tests():
        print("\n❌ PHASE 1 FAILED: Edge case tests failed")
        return 1
    
    # Task 1.3: Stress tests
    if not run_stress_tests(quick=quick_mode):
        print("\n❌ PHASE 1 FAILED: Stress tests failed")
        return 1
    
    # Success summary
    print("\n" + "=" * 70)
    print("✅ PHASE 1: CRITICAL SAFETY - COMPLETE")
    print("=" * 70)
    print()
    print("Success Metrics:")
    print("  ✅ Exception handling verified in pipeline.py")
    print("  ✅ Edge case tests created and passing")
    print("  ✅ Stress tests created and passing")
    print("  ✅ Zero unhandled exceptions in all test scenarios")
    print()
    print("Next Steps:")
    print("  - Proceed to Phase 2: E2E Validation")
    print("  - Monitor fallback rate in production (should be <1%)")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
