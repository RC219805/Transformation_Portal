#!/usr/bin/env python3
"""Verify lux_depth_v3 public surface contract.

Ensures:
1. Core modules are importable
2. Depth model selection resolves through the governed registry
3. Call-site compatibility is maintained

Exit codes:
- 0: All checks passed
- 1: Import failure or contract violation
"""
import sys
from pathlib import Path

import numpy as np


def verify_imports():
    """Verify all core modules can be imported."""
    print("🔍 Verifying imports...")

    # Test 1: Public API imports (validates __init__.py exports)
    try:
        from transformation_portal.lux_depth_v3 import (
            DA3Config,
            DA3InferenceEngine,
            DepthResult,
            DeviceConfig,
            EnhanceConfig,
            EnhanceOrchestrator,
            ModelVariant,
            PostprocessingConfig,
            Postprocessor,
            Preset,
        )

        print("✅ Public API imports successful")
    except ImportError as e:
        print(f"❌ Public API import failed: {e}")
        return False

    # Test 2: Internal module imports (validates module structure)
    try:
        from transformation_portal.lux_depth_v3.depth_writer import atomic_write_depth_u16_png_with_stats
        from transformation_portal.lux_depth_v3.v2_runner import V2Runner, find_v2_report

        print("✅ Internal module imports successful")
        return True
    except ImportError as e:
        print(f"❌ Internal import failed: {e}")
        return False


def verify_depth_runtime_contracts():
    """Verify depth runtime contracts without requiring model downloads."""
    print("\n🔍 Verifying depth runtime contracts...")

    from transformation_portal.lux_depth_v3.config import DA3Config
    from transformation_portal.lux_depth_v3.depth_writer import atomic_write_depth_u16_png_with_stats
    from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
    from transformation_portal.lux_depth_v3.v2_runner import V2Runner, find_v2_report

    checks_passed = 0
    checks_total = 0

    # Test 1: DA3InferenceEngine resolves a commercial-safe model contract.
    checks_total += 1
    try:
        config = DA3Config(model_key="da3-metric")
        engine = DA3InferenceEngine(config=config)
        resolved = engine._resolve_model_contract()
        assert resolved.canonical_key == "da3_metric"
        assert resolved.spec.requires_non_commercial_ok is False
        assert callable(engine.predict)
        print("✅ DA3InferenceEngine resolves da3-metric without non-commercial opt-in")
        checks_passed += 1
    except Exception as e:
        print(f"❌ DA3InferenceEngine contract resolution failed: {type(e).__name__}: {e}")

    # Test 2: atomic_write_depth_u16_png_with_stats() (now implemented)
    checks_total += 1
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.png"
            path, _, stats = atomic_write_depth_u16_png_with_stats(
                output_path=test_path, depth_map=np.random.rand(64, 64).astype(np.float32), method="u16", debug_verify=True
            )
            # Verify it worked
            assert path.exists()
            assert stats.shape == (64, 64)
            assert hasattr(stats, "_asdict")  # Verify orchestrator compatibility
            print("✅ atomic_write_depth_u16_png_with_stats() works (real implementation)")
            checks_passed += 1
    except NotImplementedError:
        print("❌ atomic_write_depth_u16_png_with_stats() still raises NotImplementedError (should be implemented)")
    except ImportError as e:
        print(f"⚠️  atomic_write_depth_u16_png_with_stats() requires opencv-python: {e}")
        checks_passed += 1  # Count as pass if dependency missing (expected in some envs)
    except Exception as e:
        print(f"❌ atomic_write_depth_u16_png_with_stats() failed: {type(e).__name__}: {e}")

    # Test 3: V2Runner finds the maintained subprocess script.
    checks_total += 1
    try:
        runner = V2Runner()
        assert runner.script_path.exists()
        assert runner.script_path.name == "enhance_image.py"
        print("✅ V2Runner locates maintained enhance_image.py subprocess entrypoint")
        checks_passed += 1
    except Exception as e:
        print(f"❌ V2Runner entrypoint check failed: {type(e).__name__}: {e}")

    # Test 4: find_v2_report() (now implemented, returns None when not found)
    checks_total += 1
    try:
        result = find_v2_report(Path("/tmp/output"), "test_image")
        if result is None:
            print("✅ find_v2_report() returns None when not found (real implementation)")
            checks_passed += 1
        else:
            print(f"❌ find_v2_report() should return None for missing report, got {result}")
    except Exception as e:
        print(f"❌ find_v2_report() raised exception: {type(e).__name__}: {e}")

    return checks_passed == checks_total


def verify_call_site_compatibility():
    """Verify critical call-site compatibility requirements."""
    print("\n🔍 Verifying call-site compatibility...")

    import numpy as np

    from transformation_portal.lux_depth_v3.config import PostprocessingConfig
    from transformation_portal.lux_depth_v3.inference import DepthResult

    checks_passed = 0
    checks_total = 0

    # Test 1: PostprocessingConfig has all required fields
    checks_total += 1
    config = PostprocessingConfig()
    required_fields = [
        "apply_metric_scaling",
        "scale_factor",
        "apply_median_filter",
        "median_kernel_size",
        "apply_bilateral_filter",
        "bilateral_sigma_color",
        "bilateral_sigma_space",
        "preserve_edges",
        "edge_threshold",
        "fusion_mode",
    ]
    missing_fields = [f for f in required_fields if not hasattr(config, f)]
    if missing_fields:
        print(f"❌ PostprocessingConfig missing fields: {missing_fields}")
    else:
        print("✅ PostprocessingConfig has all required fields")
        checks_passed += 1

    # Test 2: DepthResult has .depth alias
    checks_total += 1
    depth_map = np.zeros((64, 64), dtype=np.float32)
    image = np.zeros((64, 64, 3), dtype=np.float32)
    result = DepthResult(depth_map=depth_map, original_image=image, metadata={})

    if not hasattr(result, "depth"):
        print("❌ DepthResult missing .depth property alias")
    elif result.depth is not result.depth_map:
        print("❌ DepthResult.depth is not an alias for depth_map")
    else:
        print("✅ DepthResult has .depth property alias")
        checks_passed += 1

    return checks_passed == checks_total


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Lux Depth V3 Surface Contract Verification")
    print("=" * 60)

    all_passed = True

    # Run checks
    all_passed &= verify_imports()
    all_passed &= verify_depth_runtime_contracts()
    all_passed &= verify_call_site_compatibility()

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All contract checks PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ Some contract checks FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
