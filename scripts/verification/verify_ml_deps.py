#!/usr/bin/env python3
"""
APEX ML Dependencies Verification Script
Checks all required ML dependencies and backends are properly installed.
"""

import argparse
import sys
from pathlib import Path


def _seed_repo_root_for_imports() -> None:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".github" / "workflows").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_seed_repo_root_for_imports()

from scripts.lib.repo_root import RepoRootError, resolve_repo_root


def _bootstrap_paths(repo_override: str | None = None) -> Path:
    """Resolve repository root and ensure local imports use repo + src paths."""
    repo_path = Path(repo_override).expanduser() if repo_override else None
    repo_root = resolve_repo_root(start=Path(__file__), repo=repo_path)

    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


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
            ("torch", "PyTorch", "2.8.0"),
            ("torchvision", "TorchVision", "0.23.0"),
            ("transformers", "Transformers", "4.57.6"),
            ("diffusers", "Diffusers", "0.38.0"),
        ],
        "Depth": [
            ("depth_pro", "Apple Depth Pro", "0.1"),
        ],
    }

    all_ok = True

    # Import packaging for version comparisons
    try:
        from packaging import version
    except ImportError:
        print("  ⚠️  packaging module not available - using string comparisons")
        version = None

    for category, deps in dependencies.items():
        print(f"{category} Libraries:")
        for module_name, display_name, expected_version in deps:
            try:
                mod = __import__(module_name)
                mod_version = getattr(mod, "__version__", "unknown")

                # Use proper version comparison if packaging available
                if version and mod_version != "unknown":
                    try:
                        v_installed = version.parse(mod_version)
                        v_expected = version.parse(expected_version)
                        status = "✅" if v_installed >= v_expected else "⚠️"
                    except Exception:
                        # Fallback to string comparison
                        status = "✅" if mod_version >= expected_version else "⚠️"
                else:
                    # Fallback for unknown versions
                    status = "✅" if mod_version == "unknown" or mod_version >= expected_version else "⚠️"

                print(f"  {status} {display_name:20s} - v{mod_version}")
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

    # Check device - wrap torch import
    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False
        torch = None

    print("Hardware Acceleration:")
    if torch_available:
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"  {'✅' if mps else '❌'} MPS (Apple Silicon)")
        print(f"  {'✅' if torch.cuda.is_available() else '❌'} CUDA (NVIDIA)")
        print(f"  ✅ CPU (always available)")
    else:
        print("  ❌ PyTorch not available - cannot check backends")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify APEX ML dependencies and backends.")
    parser.add_argument("--repo", help="Explicit repository root path override.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        _bootstrap_paths(repo_override=args.repo)
    except RepoRootError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    sys.exit(check_dependencies())
