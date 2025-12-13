#!/usr/bin/env python3
"""
Pre-flight check for Stage 6 A/B test with boundary metrics.

Validates all prerequisites before running the A/B test.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add lux_depth_v2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_benchmark_images() -> bool:
    """Check that all benchmark images exist."""
    print("📂 Checking benchmark images...")
    
    benchmark_paths = [
        "assets/phase2_bench/750Picacho_Kitchen_Ultimate.tif",
        "assets/phase2_bench/750Picacho_Pool_Ultimate.tif",
        "assets/phase2_bench/750Picacho_PrimaryBedroom_Ultimate.tif",
        "assets/phase2_bench/750Picacho_PrimaryBathroom_Ultimate.tif",
        "assets/phase2_bench/750Picacho_Aerial_Ultimate.tif",
    ]
    
    all_exist = True
    for p in benchmark_paths:
        path = Path(p)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {path.name}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_efficientsam_model() -> bool:
    """Check that EfficientSAM model is available."""
    print("\n🤖 Checking EfficientSAM model...")
    
    try:
        from lux_depth_v2.backends.efficientsam_backend import EfficientSAMBackend
        
        backend = EfficientSAMBackend(model_name="efficientsam_s", lazy_load=True)
        available = backend.available
        
        if available:
            print("   ✅ EfficientSAM model available")
            return True
        else:
            print("   ❌ EfficientSAM model NOT available")
            print("      Run: python -m lux_depth_v2.cli --download-efficientsam --efficientsam-model efficientsam_s")
            return False
    except Exception as exc:
        print(f"   ❌ EfficientSAM check failed: {exc}")
        return False


def check_dependencies() -> bool:
    """Check core dependencies."""
    print("\n📦 Checking dependencies...")
    
    deps = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "scipy": "SciPy",
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} NOT FOUND")
            all_ok = False
    
    return all_ok


def check_boundary_metrics_module() -> bool:
    """Check that boundary metrics module exists."""
    print("\n📐 Checking boundary metrics module...")
    
    try:
        from lux_depth_v2.metrics.boundary_metrics import compute_full_boundary_metrics
        print("   ✅ Boundary metrics module available")
        return True
    except ImportError as exc:
        print(f"   ❌ Boundary metrics module NOT FOUND: {exc}")
        return False


def check_output_dir() -> bool:
    """Check output directory is writable."""
    print("\n💾 Checking output directory...")
    
    output_dir = Path("outputs/stage6_ab_boundary_metrics")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to write a test file
        test_file = output_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        print(f"   ✅ Output directory writable: {output_dir}")
        return True
    except Exception as exc:
        print(f"   ❌ Output directory not writable: {exc}")
        return False


def main() -> int:
    """Run all pre-flight checks."""
    print("="*60)
    print("STAGE 6 A/B PRE-FLIGHT CHECK")
    print("="*60)
    
    checks = [
        ("Benchmark images", check_benchmark_images),
        ("Dependencies", check_dependencies),
        ("Boundary metrics", check_boundary_metrics_module),
        ("EfficientSAM model", check_efficientsam_model),
        ("Output directory", check_output_dir),
    ]
    
    results = []
    for name, check_fn in checks:
        result = check_fn()
        results.append((name, result))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print()
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run Stage 6 A/B test")
        print()
        print("Run:")
        print("  python scripts/stage6_ab_with_boundary_metrics_FIXED.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before running A/B test")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
