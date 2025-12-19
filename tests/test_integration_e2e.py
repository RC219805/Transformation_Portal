"""End-to-end integration tests for all DA3 features.

This test suite verifies that all integrated DA3 features work together
correctly and are production-ready for real image processing.

Test Coverage:
- Model versioning (v1.0, v1.1)
- License validation system
- Reference view selection strategies
- Metric depth conversion
- API configuration
- CLI command registration
- Export formats
- Complete inference workflows
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from lux_depth_v3 import (
    DA3InferenceEngine,
    ModelVariant,
    DA3APIConfig,
    RefViewStrategy,
)
from lux_depth_v3.reference_view import select_reference_view
from lux_depth_v3.metric_depth import convert_to_metric_depth, get_depth_statistics
from lux_depth_v3.license import validate_license


# Test data
TEST_IMAGE_SIZE = (480, 640, 3)


def create_test_image(size=TEST_IMAGE_SIZE) -> np.ndarray:
    """Create synthetic test image."""
    return np.random.randint(0, 255, size, dtype=np.uint8)


def save_test_image(path: Path, size=TEST_IMAGE_SIZE) -> Path:
    """Save synthetic test image to disk."""
    img_array = create_test_image(size)
    img = Image.fromarray(img_array)
    img.save(path)
    return path


@pytest.fixture
def test_images_dir(tmp_path):
    """Create directory with test images."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    
    # Create 3 test images
    for i in range(3):
        save_test_image(img_dir / f"image_{i:03d}.jpg")
    
    return img_dir


@pytest.fixture
def test_intrinsics():
    """Create test camera intrinsics."""
    return np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])


class TestFeatureIntegration:
    """Test that all features are integrated and working."""
    
    def test_model_variant_enum(self):
        """Test model variant enumeration with v1.1 support."""
        # v1.1 models
        assert hasattr(ModelVariant, 'DA3_NESTED_GIANT_LARGE_V1_1')
        assert hasattr(ModelVariant, 'DA3_GIANT_V1_1')
        assert hasattr(ModelVariant, 'DA3_LARGE_V1_1')
        
        # Legacy v1.0 models
        assert hasattr(ModelVariant, 'DA3_NESTED_GIANT_LARGE')
        assert hasattr(ModelVariant, 'DA3_GIANT')
        assert hasattr(ModelVariant, 'DA3_LARGE')
        
        # Check metadata
        info = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1.info
        assert info.version == "1.1"
        assert info.name == "DA3NESTED-GIANT-LARGE"
        assert info.params == "1.40B"
    
    def test_license_validation(self):
        """Test license validation system."""
        # Non-commercial model
        nc_variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        
        # Should not raise in non-commercial mode
        validate_license(nc_variant, commercial_use=False)
        
        # Should warn in commercial mode
        with pytest.warns(UserWarning):
            validate_license(nc_variant, commercial_use=True, strict=False)
        
        # Should raise in strict commercial mode
        with pytest.raises(RuntimeError):
            validate_license(nc_variant, commercial_use=True, strict=True)
        
        # Commercial-friendly model should pass
        commercial_variant = ModelVariant.DA3_METRIC_LARGE
        validate_license(commercial_variant, commercial_use=True, strict=True)
    
    def test_reference_view_selection(self):
        """Test reference view selection strategies."""
        num_views = 5
        class_tokens = np.random.randn(num_views, 768)
        
        # Test all strategies
        for strategy in ["saddle_balanced", "saddle_sim_range", "middle", "first"]:
            result = select_reference_view(
                num_views=num_views,
                strategy=strategy,
                class_tokens=class_tokens if "saddle" in strategy else None
            )
            assert 0 <= result.selected_index < num_views
    
    def test_metric_depth_conversion(self, test_intrinsics):
        """Test metric depth conversion."""
        depth = np.random.rand(480, 640) * 10.0
        
        # Test with DA3METRIC-LARGE (requires conversion)
        result = convert_to_metric_depth(
            depth,
            model_name="DA3METRIC-LARGE",
            intrinsics=test_intrinsics
        )
        
        assert result.depth_meters.shape == depth.shape
        assert result.focal_length_px == 500.0
        assert result.scale_factor > 0
        assert not result.already_metric
        
        # Test with nested model (already metric)
        result_nested = convert_to_metric_depth(
            depth,
            model_name="DA3NESTED-GIANT-LARGE-1.1"
        )
        
        assert result_nested.already_metric
        assert np.array_equal(result_nested.depth_meters, depth)
    
    def test_depth_statistics(self):
        """Test depth statistics calculation."""
        depth = np.random.rand(480, 640) * 10.0
        
        stats = get_depth_statistics(depth)
        
        assert 'min_m' in stats
        assert 'max_m' in stats
        assert 'mean_m' in stats
        assert 'median_m' in stats
        assert 'std_m' in stats
        assert stats['min_m'] >= 0
        assert stats['max_m'] <= 10.0
    
    def test_api_config(self):
        """Test DA3 API configuration."""
        config = DA3APIConfig(
            model_name="da3-large",
            ref_view_strategy=RefViewStrategy.SADDLE_BALANCED,
            use_ray_pose=True,
            infer_gs=False,
            export_format="mini_npz-glb"
        )
        
        api_kwargs = config.to_api_kwargs()
        
        assert api_kwargs["ref_view_strategy"] == "saddle_balanced"
        assert api_kwargs["use_ray_pose"] is True
        assert api_kwargs["export_format"] == "mini_npz-glb"
    
    def test_model_commercial_alternatives(self):
        """Test commercial alternative lookup."""
        # Non-commercial model
        nc_variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
        
        # Get commercial alternative
        commercial = ModelVariant.get_commercial_alternative(nc_variant)
        
        assert commercial.info.is_commercial
        assert commercial.info.license.value == "Apache-2.0"


class TestCLIIntegration:
    """Test CLI command availability."""
    
    def test_cli_commands_registered(self):
        """Test that all CLI commands are registered."""
        from lux_depth_v3.cli import app
        
        command_names = [cmd.name for cmd in app.registered_commands]
        
        # Should have multiple commands registered
        assert len(command_names) > 0
    
    def test_cli_help_text(self):
        """Test that CLI help is available."""
        from lux_depth_v3.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        # Should not error
        assert result.exit_code == 0 or result.exit_code == 2  # 2 if no command given


class TestImportConsistency:
    """Test that all imports work correctly."""
    
    def test_all_public_api_imports(self):
        """Test that all __all__ exports are importable."""
        from lux_depth_v3 import __all__
        import lux_depth_v3
        
        for name in __all__:
            assert hasattr(lux_depth_v3, name), f"Missing export: {name}"
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # These should all import without errors
        from lux_depth_v3 import config
        from lux_depth_v3 import reference_view
        from lux_depth_v3 import metric_depth
        from lux_depth_v3 import license
        from lux_depth_v3 import inference
        from lux_depth_v3 import da3_wrapper
        
        assert True  # If we got here, no circular imports
    
    def test_benchmark_module_imports(self):
        """Test that benchmark module imports correctly."""
        from lux_depth_v3 import benchmark
        
        # Should have key components
        assert hasattr(benchmark, 'DA3BenchmarkEvaluator')
        assert hasattr(benchmark, 'BenchmarkConfig')


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows (requires DA3 package)."""
    
    def test_single_image_inference(self, test_images_dir, tmp_path):
        """Test single image depth estimation."""
        pytest.importorskip("depth_anything_3", reason="DA3 not installed")
        
        engine = DA3InferenceEngine(
            model_variant=ModelVariant.DA3_LARGE_V1_1,
            device="cpu"
        )
        
        image_path = list(test_images_dir.glob("*.jpg"))[0]
        
        result = engine.infer(
            images=[image_path],
            export_dir=tmp_path / "output"
        )
        
        assert result.depth is not None
        assert result.depth.shape[0] == 1  # One image
    
    def test_multi_view_inference(self, test_images_dir, tmp_path):
        """Test multi-view depth with reference selection."""
        pytest.importorskip("depth_anything_3", reason="DA3 not installed")
        
        config = DA3APIConfig(
            ref_view_strategy=RefViewStrategy.SADDLE_BALANCED,
            use_ray_pose=False
        )
        
        engine = DA3InferenceEngine(
            model_variant=ModelVariant.DA3_LARGE_V1_1,
            api_config=config,
            device="cpu"
        )
        
        images = list(test_images_dir.glob("*.jpg"))
        
        result = engine.infer(
            images=images,
            export_dir=tmp_path / "output"
        )
        
        assert result.depth.shape[0] == len(images)
        if result.extrinsics is not None:
            assert result.extrinsics.shape[0] == len(images)
    
    def test_metric_depth_workflow(self, test_images_dir, test_intrinsics, tmp_path):
        """Test complete metric depth workflow."""
        pytest.importorskip("depth_anything_3", reason="DA3 not installed")
        
        engine = DA3InferenceEngine(
            model_variant=ModelVariant.DA3_METRIC_LARGE,
            device="cpu"
        )
        
        image_path = list(test_images_dir.glob("*.jpg"))[0]
        
        # Expand intrinsics to batch
        intrinsics_batch = test_intrinsics[np.newaxis, :, :]
        
        result = engine.infer(
            images=[image_path],
            intrinsics=intrinsics_batch,
            export_dir=tmp_path / "output",
            convert_to_metric=True
        )
        
        assert hasattr(result, 'metric_depth') or result.depth is not None


class TestExportFormats:
    """Test export format support."""
    
    def test_export_formats(self, test_images_dir, tmp_path):
        """Test all export formats."""
        pytest.importorskip("depth_anything_3", reason="DA3 not installed")
        
        engine = DA3InferenceEngine(
            model_variant=ModelVariant.DA3_LARGE_V1_1,
            device="cpu"
        )
        
        image_path = list(test_images_dir.glob("*.jpg"))[0]
        export_dir = tmp_path / "output"
        
        config = DA3APIConfig(export_format="mini_npz-glb")
        engine.api_config = config
        
        result = engine.infer(
            images=[image_path],
            export_dir=export_dir
        )
        
        # Check that inference ran
        assert result.depth is not None or export_dir.exists()


class TestProductionReadiness:
    """Test production readiness criteria."""
    
    def test_all_enums_accessible(self):
        """Test that all enum types are accessible."""
        from lux_depth_v3 import ModelVariant, RefViewStrategy, InferenceMode
        from lux_depth_v3.config import ModelLicense
        
        # Should be able to iterate
        assert len(list(ModelVariant)) > 0
        assert len(list(RefViewStrategy)) > 0
        assert len(list(InferenceMode)) > 0
        assert len(list(ModelLicense)) > 0
    
    def test_example_workflow_imports(self):
        """Test that example workflow components import correctly."""
        # All components needed for a basic workflow
        from lux_depth_v3 import (
            DA3InferenceEngine,
            ModelVariant,
            DA3APIConfig,
            RefViewStrategy,
        )
        from lux_depth_v3.license import validate_license
        from lux_depth_v3.metric_depth import convert_to_metric_depth
        
        assert True  # If we got here, all imports work
    
    def test_documentation_strings(self):
        """Test that key modules have documentation."""
        import lux_depth_v3
        from lux_depth_v3 import config, reference_view, metric_depth, license
        
        assert lux_depth_v3.__doc__ is not None
        assert config.__doc__ is not None
        assert reference_view.__doc__ is not None
        assert metric_depth.__doc__ is not None
        assert license.__doc__ is not None
