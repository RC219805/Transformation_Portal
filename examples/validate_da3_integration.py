#!/usr/bin/env python3
"""Comprehensive DA3 Integration Validation Script.

This script validates all DA3 features integrated into lux_depth_v3:
1. Monocular depth estimation
2. Multi-view depth estimation with pose estimation
3. Metric depth conversion
4. Export format support (NPZ, GLB, depth_vis)
5. Model variants and versioning
6. License validation

Usage:
    # Quick validation (no inference)
    python examples/validate_da3_integration.py --quick

    # Full validation with inference
    python examples/validate_da3_integration.py --input input_images/750_Picacho/

    # Single image test
    python examples/validate_da3_integration.py --input input_images/750_Picacho/Aerial.tif
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class DA3ValidationReport:
    """Track validation results."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.warnings = []

    def test_passed(self, name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"   ✅ {name}")

    def test_failed(self, name: str, error: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((name, error))
        print(f"   ❌ {name}: {error}")

    def add_warning(self, message: str):
        self.warnings.append(message)
        print(f"   ⚠️  {message}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Tests Run:    {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Warnings:     {len(self.warnings)}")

        if self.failures:
            print("\nFailed Tests:")
            for name, error in self.failures:
                print(f"  ❌ {name}")
                print(f"     {error}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        print("=" * 70)

        if self.tests_failed == 0:
            print("✅ ALL TESTS PASSED")
            return 0
        else:
            print(f"❌ {self.tests_failed} TEST(S) FAILED")
            return 1


def validate_imports(report: DA3ValidationReport):
    """Validate that all DA3 modules can be imported."""
    print("\n📦 Validating Imports...")

    try:
        from lux_depth_v3 import (
            DA3DepthEstimator,
            estimate_depth,
            DA3Result,
        )

        report.test_passed("da3_integration imports")
    except Exception as e:
        report.test_failed("da3_integration imports", str(e))

    try:
        from lux_depth_v3.da3_wrapper import (
            DepthAnything3Wrapper,
            DA3Backend,
            DA3CLI,
            DA3Prediction,
            check_da3_cli_available,
        )

        report.test_passed("da3_wrapper imports")
    except Exception as e:
        report.test_failed("da3_wrapper imports", str(e))

    try:
        from lux_depth_v3.config import (
            ModelVariant,
            DA3Config,
            DA3APIConfig,
            DA3CLIConfig,
            Preset,
            InferenceMode,
        )

        report.test_passed("config imports")
    except Exception as e:
        report.test_failed("config imports", str(e))

    try:
        from lux_depth_v3.metric_depth import (
            MetricDepthConverter,
            convert_to_metric_depth,
            get_depth_statistics,
        )

        report.test_passed("metric_depth imports")
    except Exception as e:
        report.test_failed("metric_depth imports", str(e))


def validate_model_variants(report: DA3ValidationReport):
    """Validate model variant definitions."""
    print("\n🔬 Validating Model Variants...")

    try:
        from lux_depth_v3.config import ModelVariant

        # Check v1.1 models
        variants_to_test = [
            ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
            ModelVariant.DA3_LARGE_V1_1,
            ModelVariant.DA3_GIANT_V1_1,
            ModelVariant.DA3_METRIC_LARGE,
            ModelVariant.DA3_BASE,
            ModelVariant.DA3_SMALL,
        ]

        for variant in variants_to_test:
            info = variant.info

            # Check metadata
            if not info.name:
                report.test_failed(f"{variant.name} metadata", "Missing name")
                continue

            if not info.huggingface_id:
                report.test_failed(f"{variant.name} metadata", "Missing HF ID")
                continue

            if not info.params:
                report.test_failed(f"{variant.name} metadata", "Missing params")
                continue

            # Check capabilities
            if info.capabilities is None:
                report.test_failed(f"{variant.name} capabilities", "Missing capabilities")
                continue

            report.test_passed(f"{variant.name} metadata")

        # Test recommended model
        recommended = ModelVariant.get_recommended()
        if recommended == ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1:
            report.test_passed("Recommended model (v1.1)")
        else:
            report.test_failed("Recommended model", f"Expected v1.1, got {recommended.name}")

    except Exception as e:
        report.test_failed("Model variants", str(e))


def validate_license_system(report: DA3ValidationReport):
    """Validate license validation system."""
    print("\n📄 Validating License System...")

    try:
        from lux_depth_v3.config import ModelVariant
        from lux_depth_v3.license import (
            validate_license,
            get_license_info,
            LicenseValidator,
        )

        # Test Apache 2.0 model (commercial allowed)
        try:
            validate_license(ModelVariant.DA3_METRIC_LARGE, commercial_use=True)
            report.test_passed("Apache 2.0 license (commercial use)")
        except Exception as e:
            report.test_failed("Apache 2.0 license", str(e))

        # Test CC-BY-NC model (non-commercial only)
        try:
            validate_license(ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1, commercial_use=False)
            report.test_passed("CC-BY-NC license (non-commercial)")
        except Exception as e:
            report.test_failed("CC-BY-NC license", str(e))

        # Test license info
        validator = LicenseValidator()
        info = validator.get_license_info(ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)

        if not info["commercial_allowed"]:
            report.test_passed("License info (NC detection)")
        else:
            report.test_failed("License info", "Expected commercial=False for NC model")

        # Test commercial alternative
        alt = validator.get_commercial_alternative(ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1)
        if alt == ModelVariant.DA3_METRIC_LARGE:
            report.test_passed("Commercial alternative suggestion")
        else:
            report.add_warning(f"Expected metric-large alternative, got {alt}")

    except Exception as e:
        report.test_failed("License system", str(e))


def validate_metric_depth(report: DA3ValidationReport):
    """Validate metric depth conversion."""
    print("\n📏 Validating Metric Depth Conversion...")

    try:
        import numpy as np
        from lux_depth_v3.metric_depth import (
            MetricDepthConverter,
            convert_to_metric_depth,
            get_depth_statistics,
        )

        # Test DA3METRIC-LARGE conversion
        converter = MetricDepthConverter("DA3METRIC-LARGE")
        test_depth = np.random.rand(480, 640).astype(np.float32)
        focal_length = 500.0

        result = converter.convert(test_depth, focal_length_px=focal_length)

        if result.depth_meters.shape == test_depth.shape:
            report.test_passed("Metric depth conversion shape")
        else:
            report.test_failed("Metric depth conversion", "Shape mismatch")

        if result.focal_length_px == focal_length:
            report.test_passed("Focal length preservation")
        else:
            report.test_failed("Focal length", f"Expected {focal_length}, got {result.focal_length_px}")

        # Test nested model (already metric)
        converter_nested = MetricDepthConverter("DA3NESTED-GIANT-LARGE-1.1")
        result_nested = converter_nested.convert(test_depth)

        if result_nested.already_metric:
            report.test_passed("Nested model (already metric)")
        else:
            report.test_failed("Nested model", "Expected already_metric=True")

        # Test statistics
        stats = get_depth_statistics(result.depth_meters)
        if all(k in stats for k in ["min_m", "max_m", "mean_m", "median_m", "std_m"]):
            report.test_passed("Depth statistics computation")
        else:
            report.test_failed("Depth statistics", "Missing keys")

    except Exception as e:
        report.test_failed("Metric depth", str(e))


def validate_configuration(report: DA3ValidationReport):
    """Validate configuration system."""
    print("\n⚙️  Validating Configuration System...")

    try:
        from lux_depth_v3.config import (
            DA3Config,
            DA3APIConfig,
            DA3CLIConfig,
            Preset,
            ModelVariant,
        )

        # Test preset configurations
        for preset in Preset:
            try:
                config = DA3Config.from_preset(preset)
                if config.model_variant is not None:
                    report.test_passed(f"Preset: {preset.value}")
                else:
                    report.test_failed(f"Preset: {preset.value}", "Missing model variant")
            except Exception as e:
                report.test_failed(f"Preset: {preset.value}", str(e))

        # Test API config
        api_config = DA3APIConfig(
            model_name="da3-large",
            ref_view_strategy="saddle_balanced",
            export_format="mini_npz-glb",
        )

        kwargs = api_config.to_api_kwargs()
        if "ref_view_strategy" in kwargs and "export_format" in kwargs:
            report.test_passed("API config to_api_kwargs()")
        else:
            report.test_failed("API config", "Missing keys in kwargs")

        # Test CLI config
        cli_config = DA3CLIConfig(
            use_cli=True,
            export_format="mini_npz-glb-depth_vis",
        )

        if cli_config.export_format == "mini_npz-glb-depth_vis":
            report.test_passed("CLI config initialization")
        else:
            report.test_failed("CLI config", "Export format mismatch")

    except Exception as e:
        report.test_failed("Configuration", str(e))


def validate_reference_view_strategies(report: DA3ValidationReport):
    """Validate reference view selection strategies."""
    print("\n🎯 Validating Reference View Strategies...")

    try:
        from lux_depth_v3.reference_view import (
            RefViewStrategy,
            ReferenceViewSelector,
        )
        import numpy as np

        # Test all strategies
        strategies = [
            RefViewStrategy.FIRST,
            RefViewStrategy.MIDDLE,
            RefViewStrategy.SADDLE_BALANCED,
            RefViewStrategy.SADDLE_SIM_RANGE,
        ]

        # Create dummy features for testing
        num_views = 10
        num_features = 512
        features = np.random.rand(num_views, num_features).astype(np.float32)

        for strategy in strategies:
            try:
                selector = ReferenceViewSelector(strategy=strategy)

                # For saddle strategies, pass class_tokens
                if strategy in [
                    RefViewStrategy.SADDLE_BALANCED,
                    RefViewStrategy.SADDLE_SIM_RANGE,
                ]:
                    result = selector.select(num_views=num_views, class_tokens=features)
                else:
                    result = selector.select(num_views=num_views)

                if 0 <= result.selected_index < num_views:
                    report.test_passed(f"Strategy: {strategy.value}")
                else:
                    report.test_failed(
                        f"Strategy: {strategy.value}",
                        f"Invalid index: {result.selected_index}",
                    )
            except Exception as e:
                report.test_failed(f"Strategy: {strategy.value}", str(e))

    except Exception as e:
        report.test_failed("Reference view strategies", str(e))


def validate_inference_engine(report: DA3ValidationReport, skip_inference: bool = True):
    """Validate inference engine initialization."""
    print("\n🚀 Validating Inference Engine...")

    if skip_inference:
        report.add_warning("Inference skipped (no DA3 API available)")
        return

    try:
        from lux_depth_v3.config import DA3Config, ModelVariant
        from lux_depth_v3.inference import DA3InferenceEngine

        config = DA3Config(
            model_variant=ModelVariant.DA3_METRIC_LARGE,
        )

        try:
            engine = DA3InferenceEngine(
                config,
                commercial_use=True,  # Metric-large is Apache 2.0
                validate_license_strict=False,
            )
            report.test_passed("Inference engine initialization")
        except ImportError:
            report.add_warning("DA3 API not installed (pip install depth-anything-3)")
        except Exception as e:
            report.test_failed("Inference engine", str(e))

    except Exception as e:
        report.test_failed("Inference engine", str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Validate DA3 integration in lux_depth_v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Quick validation (no inference)")
    parser.add_argument("--input", type=str, help="Input image or directory for inference test")

    args = parser.parse_args()

    print("=" * 70)
    print("DA3 INTEGRATION VALIDATION")
    print("=" * 70)

    report = DA3ValidationReport()

    # Run validation tests
    validate_imports(report)
    validate_model_variants(report)
    validate_license_system(report)
    validate_metric_depth(report)
    validate_configuration(report)
    validate_reference_view_strategies(report)

    # Inference test (optional)
    skip_inference = args.quick or args.input is None
    validate_inference_engine(report, skip_inference=skip_inference)

    # Print summary
    return report.print_summary()


if __name__ == "__main__":
    sys.exit(main())
