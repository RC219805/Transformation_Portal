#!/usr/bin/env python3
"""Pre-cache all DA3 model variants."""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v3.model_cache import ModelCacheManager


def main():
    parser = argparse.ArgumentParser(description="Pre-cache DA3 models for offline use")
    parser.add_argument(
        "--set", default="production", choices=["essential", "production", "benchmark", "all"], help="Model set to download"
    )
    parser.add_argument("--cache-dir", type=Path, help="Custom cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")

    args = parser.parse_args()

    print("=" * 70)
    print("DA3 Model Pre-caching")
    print("=" * 70)

    manager = ModelCacheManager(cache_dir=args.cache_dir)

    print(f"Model Set: {args.set}")
    print(f"Cache Dir: {manager.cache_dir}")
    print(f"Verify: {not args.no_verify}")
    print("=" * 70)

    # Download
    results = manager.download_models(model_set=args.set, force=args.force, verify=not args.no_verify)

    # Summary
    stats = manager.get_cache_stats()

    print("\n" + "=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Models Downloaded: {len(results)}")
    print(f"Total Size: {stats['total_size_gb']:.2f} GB")
    print(f"Cache Location: {stats['cache_dir']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
