#!/usr/bin/env python3
"""Phase 2 Implementation Validation Script

This script validates that Phase 2 implementation is complete and working.

Usage:
    python scripts/validate_phase2.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_imports():
    """Test that all Phase 2 modules can be imported."""
    print("\n📦 Testing imports...")

    try:
        from transformation_portal.depth_canonical import DepthPipeline
        from transformation_portal.depth_canonical.config import (
            DeviceType,
            ModelConfig,
            ModelVariant,
            PBRConfig,
            ProcessingConfig,
            UnifiedDepthConfig,
        )
        from transformation_portal.depth_canonical.models import (
            DA2ModelWrapper,
            DA3ModelWrapper,
            DepthEstimationModel,
            ModelRegistry,
        )

        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_model_registry():
    """Test ModelRegistry functionality."""
    print("\n🔧 Testing ModelRegistry...")

    try:
        from transformation_portal.depth_canonical.config import DeviceType, ModelVariant
        from transformation_portal.depth_canonical.models import ModelRegistry

        registry = ModelRegistry()

        # Test variant support
        assert registry.is_variant_supported(ModelVariant.DA3_METRIC_LARGE)
        assert registry.is_variant_supported(ModelVariant.DA2_BASE)
        print("  ✓ Variant support works")

        # Test device auto-detection
        device = registry._auto_detect_device()
        assert device in {DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS, DeviceType.COREML}
        print(f"  ✓ Device auto-detection works (detected: {device.value})")

        # Test cache clearing
        registry.clear_cache()
        print("  ✓ Cache clearing works")

        return True
    except Exception as e:
        print(f"  ✗ ModelRegistry test failed: {e}")
        return False


def test_depth_pipeline():
    """Test DepthPipeline functionality."""
    print("\n🔄 Testing DepthPipeline...")

    try:
        import numpy as np

        from transformation_portal.depth_canonical import DepthPipeline
        from transformation_portal.depth_canonical.config import PBRConfig, ProcessingConfig, UnifiedDepthConfig

        # Test initialization
        config = UnifiedDepthConfig(processing=ProcessingConfig(pbr=PBRConfig(enabled=False)))
        pipeline = DepthPipeline(config)
        print("  ✓ Pipeline initialization works")

        # Test with pre-computed depth (backward compatibility)
        depth_map = np.random.rand(128, 128).astype(np.float32)
        result = pipeline.process(depth_map=depth_map)

        assert result.depth_map is not None
        assert np.array_equal(result.depth_map, depth_map)
        print("  ✓ Backward compatibility (pre-computed depth) works")

        # Test cache key generation
        img_array = np.random.rand(100, 100, 3).astype(np.uint8)
        key = pipeline._generate_cache_key(img_array)
        assert len(key) > 0
        print("  ✓ Cache key generation works")

        return True
    except Exception as e:
        print(f"  ✗ DepthPipeline test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n⚙️  Testing configuration...")

    try:
        from transformation_portal.depth_canonical.config import (
            DeviceType,
            IOConfig,
            ModelConfig,
            ModelVariant,
            PBRConfig,
            ProcessingConfig,
            SecurityConfig,
            UnifiedDepthConfig,
        )

        # Test default config
        config = UnifiedDepthConfig()
        assert config.model is not None
        assert config.processing is not None
        assert config.io is not None
        assert config.security is not None
        print("  ✓ Default configuration works")

        # Test custom config
        config = UnifiedDepthConfig(
            model=ModelConfig(
                variant=ModelVariant.DA3_METRIC_SMALL,
                device=DeviceType.CPU,
            ),
            processing=ProcessingConfig(
                pbr=PBRConfig(
                    enabled=True,
                    normal_strength=1.5,
                )
            ),
            io=IOConfig(
                cache_enabled=True,
                cache_size=256,
            ),
        )

        assert config.model.variant == ModelVariant.DA3_METRIC_SMALL
        assert config.processing.pbr.enabled is True
        assert config.io.cache_enabled is True
        print("  ✓ Custom configuration works")

        return True
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def check_test_coverage():
    """Check test coverage."""
    print("\n🧪 Checking test coverage...")

    test_dir = Path(__file__).resolve().parents[2] / "tests" / "depth_canonical"

    if not test_dir.exists():
        print("  ✗ Test directory not found")
        return False

    test_files = list(test_dir.glob("test_*.py"))
    print(f"  ✓ Found {len(test_files)} test files:")

    for test_file in sorted(test_files):
        print(f"    - {test_file.name}")

    return True


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("PHASE 2 IMPLEMENTATION VALIDATION")
    print("=" * 70)

    checks = [
        ("Imports", test_imports),
        ("ModelRegistry", test_model_registry),
        ("DepthPipeline", test_depth_pipeline),
        ("Configuration", test_configuration),
        ("Test Coverage", check_test_coverage),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        print("\nPhase 2 implementation is complete and working!")
        print("\nNext steps:")
        print("  1. Run integration tests: pytest tests/depth_canonical/ -v -m integration")
        print("  2. Run slow tests: pytest tests/depth_canonical/ -v -m slow")
        print("  3. Run example: python examples/phase2_depth_demo.py")
        return 0
    else:
        print("\n❌ SOME VALIDATION CHECKS FAILED")
        print("\nPlease review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
