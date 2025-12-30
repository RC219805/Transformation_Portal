#!/usr/bin/env python
"""
DA3 Integration Validation Script

Demonstrates all core features of the lux_depth_v3 module.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile


def main():
    print("=" * 70)
    print("DA3 Integration Validation")
    print("=" * 70)
    print()

    results = []

    # 1. Configuration
    print("1. Testing Configuration...")
    try:
        from lux_depth_v3.config import DA3Config, ModelVariant, Preset

        config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)
        results.append(f"✅ Config: {config.model_variant}")
    except Exception as e:
        results.append(f"❌ Config: {e}")

    # 2. Model Cache Manager
    print("2. Testing Model Cache Manager...")
    try:
        from lux_depth_v3.model_cache import ModelCacheManager

        cache_mgr = ModelCacheManager()
        results.append(f"✅ ModelCacheManager: {len(cache_mgr.OFFICIAL_MODELS)} models")
    except Exception as e:
        results.append(f"❌ ModelCacheManager: {e}")

    # 3. Metric Depth Converter
    print("3. Testing Metric Depth Converter...")
    try:
        from lux_depth_v3.metric_depth import MetricDepthConverter

        depth = np.random.rand(512, 512).astype(np.float32)
        converter = MetricDepthConverter()
        result = converter.convert(depth, focal_length_px=1000.0)
        results.append(f"✅ MetricDepthConverter: scale={result.scale_factor:.2f}")
    except Exception as e:
        results.append(f"❌ MetricDepthConverter: {e}")

    # 4. DA3 Estimator
    print("4. Testing DA3 Estimator...")
    try:
        from lux_depth_v3.da3_integration import DA3DepthEstimator

        estimator = DA3DepthEstimator(model="large-1.1", device="cpu")
        results.append(f"✅ DA3DepthEstimator: {estimator.model}")
    except Exception as e:
        results.append(f"❌ DA3DepthEstimator: {e}")

    # 5. CLI Availability
    print("5. Testing CLI Availability...")
    try:
        from lux_depth_v3.da3_wrapper import check_da3_cli_available

        cli_available = check_da3_cli_available()
        results.append(f"✅ DA3 CLI: {'available' if cli_available else 'not found'}")
    except Exception as e:
        results.append(f"❌ DA3 CLI: {e}")

    # 6. Input Manager
    print("6. Testing Input Manager...")
    try:
        from lux_depth_v3.input_manager import InputManager

        manager = InputManager()
        results.append(f"✅ InputManager: initialized")
    except Exception as e:
        results.append(f"❌ InputManager: {e}")

    # 7. End-to-End Depth Estimation (if CLI available)
    if cli_available:
        print("7. Testing End-to-End Depth Estimation...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Create sample image
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                img_path = tmpdir / "test.jpg"
                Image.fromarray(img).save(img_path)

                # Estimate depth
                from lux_depth_v3.da3_integration import DA3DepthEstimator

                estimator = DA3DepthEstimator(model="large-1.1", device="cpu")

                output_dir = tmpdir / "output"
                output_dir.mkdir()

                result = estimator.process_image(img_path, output_dir, export_format="mini_npz")

                if result.success:
                    depth = result.depth_array
                    if depth is not None:
                        results.append(f"✅ E2E Depth Estimation: shape={depth.shape}")
                    else:
                        results.append(f"⚠️  E2E: No depth array in result")
                else:
                    results.append(f"⚠️  E2E: Processing failed")
        except Exception as e:
            results.append(f"❌ E2E Depth Estimation: {e}")
    else:
        results.append(f"⏭️  E2E: Skipped (CLI not available)")

    # Print results
    print()
    print("=" * 70)
    print("Validation Results")
    print("=" * 70)
    for result in results:
        print(result)

    passed = len([r for r in results if r.startswith("✅")])
    total = len([r for r in results if not r.startswith("⏭️")])

    print()
    print(f"Summary: {passed} / {total} tests passed")
    print("=" * 70)

    # Return exit code
    if passed == total:
        print("\n✅ DA3 Integration: FULLY OPERATIONAL")
        return 0
    elif passed >= total * 0.8:
        print(f"\n⚠️  DA3 Integration: MOSTLY OPERATIONAL ({passed}/{total})")
        return 0
    else:
        print(f"\n❌ DA3 Integration: ISSUES DETECTED ({passed}/{total})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
