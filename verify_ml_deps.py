#!/usr/bin/env python3
"""
APEX ML Dependencies Verification Script
Checks all required ML dependencies and backends are properly installed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_dependencies():
    """Verify all ML dependencies are installed."""
    print("=" * 80)
    print("  APEX ML STACK VERIFICATION")
    print("=" * 80)
    print()

    dependencies = {
        "Core": [
            ("numpy", "NumPy", "1.26.4"),
            ("PIL", "Pillow", "11.3.0"),
            ("cv2", "OpenCV", "4.13.0"),
        ],
        "ML": [
            ("torch", "PyTorch", "2.10.0"),
            ("transformers", "Transformers", "4.57.6"),
            ("diffusers", "Diffusers", "0.36.0"),
        ],
        "Depth": [
            ("depth_pro", "Apple Depth Pro", "0.1"),
        ],
    }

    all_ok = True

    for category, deps in dependencies.items():
        print(f"{category} Libraries:")
        for module_name, display_name, expected_version in deps:
            try:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "unknown")
                status = "✅" if version >= expected_version or version == "unknown" else "⚠️"
                print(f"  {status} {display_name:20s} - v{version}")
            except ImportError:
                print(f"  ❌ {display_name:20s} - NOT INSTALLED")
                all_ok = False
        print()

    # Check backends
    print("Transformation Portal Backends:")
    try:
        from transformation_portal.depth.backends.synthetic import SyntheticDepthBackend

        print("  ✅ SyntheticDepthBackend - Available")
    except ImportError:
        print("  ❌ SyntheticDepthBackend - FAILED")
        all_ok = False

    try:
        from transformation_portal.depth.backends.depth_pro import DepthProBackend

        print("  ✅ DepthProBackend       - Available")
    except ImportError:
        print("  ❌ DepthProBackend       - FAILED")
        all_ok = False

    print()

    # Check device
    import torch

    print("Hardware Acceleration:")
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  {'✅' if mps else '❌'} MPS (Apple Silicon)")
    print(f"  {'✅' if torch.cuda.is_available() else '❌'} CUDA (NVIDIA)")
    print(f"  ✅ CPU (always available)")
    print()

    if all_ok:
        print("=" * 80)
        print("  ✅ ALL DEPENDENCIES VERIFIED - READY FOR APEX PROCESSING")
        print("=" * 80)
        return 0
    else:
        print("=" * 80)
        print("  ❌ SOME DEPENDENCIES MISSING - INSTALL REQUIRED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(check_dependencies())
