"""Import smoke tests for lux_depth_v3 package.

Verifies that all critical modules and types can be imported without errors.
These are stub implementations that will be fully developed later.
"""
import pytest
import sys
from pathlib import Path


def test_import_config():
    """Test that config module imports successfully."""
    from transformation_portal.lux_depth_v3.config import (
        DA3Config,
        ModelVariant,
        Preset,
        EnhanceConfig,
        PostprocessingConfig,
        DeviceConfig,
    )

    # Verify enums have expected values
    assert ModelVariant.METRIC_LARGE is not None
    assert ModelVariant.METRIC_BASE is not None
    assert ModelVariant.METRIC_SMALL is not None

    assert Preset.ARCHITECTURAL_INTERIOR is not None
    assert Preset.DEFAULT is not None

    # Verify basic instantiation works
    config = DA3Config()
    assert config.model_variant == ModelVariant.METRIC_LARGE

    enhance_config = EnhanceConfig()
    assert enhance_config is not None


def test_import_inference():
    """Test that inference module imports successfully."""
    from transformation_portal.lux_depth_v3.inference import (
        DA3InferenceEngine,
        DepthResult,
    )
    from transformation_portal.lux_depth_v3.config import DA3Config

    # Verify basic instantiation works
    config = DA3Config()
    engine = DA3InferenceEngine(config=config)
    assert engine is not None
    assert engine.config == config


def test_import_input_manager():
    """Test that input_manager module imports successfully."""
    from transformation_portal.lux_depth_v3.input_manager import ImageInput

    # Verify basic instantiation works
    input_img = ImageInput(path=Path("/tmp/test.jpg"))
    assert input_img.path == Path("/tmp/test.jpg")
    assert isinstance(input_img.path, Path)


def test_import_depth_writer():
    """Test that depth_writer module imports successfully."""
    from transformation_portal.lux_depth_v3.depth_writer import (
        atomic_write_depth_u16_png_with_stats,
        read_depth_u16_png,
    )

    # Functions should exist and be callable
    assert callable(atomic_write_depth_u16_png_with_stats)
    assert callable(read_depth_u16_png)


def test_import_v2_runner():
    """Test that v2_runner module imports successfully."""
    from transformation_portal.lux_depth_v3.v2_runner import (
        V2Runner,
        find_v2_report,
    )

    # Verify basic instantiation works
    runner = V2Runner()
    assert runner is not None

    # Function should exist and be callable
    assert callable(find_v2_report)


def test_import_security():
    """Test that security module imports successfully."""
    from transformation_portal.lux_depth_v3.security import (
        HashMode,
        sanitize_file_stem,
        sanitize_path_component_nonlossy,
        validate_device_spec,
        validate_quantization_method,
        validate_depth_fallback,
    )

    # Verify enum has expected values
    assert HashMode.ALWAYS is not None
    assert HashMode.IF_MANIFEST_EXISTS is not None
    assert HashMode.NEVER is not None

    # Test basic sanitization
    assert sanitize_file_stem("test_file") == "test_file"
    assert sanitize_file_stem("test/file") == "test_file"
    assert sanitize_path_component_nonlossy("valid_path") == "valid_path"

    # Test validation functions
    assert validate_device_spec("cpu") == "cpu"
    assert validate_device_spec("cuda") == "cuda"
    assert validate_quantization_method("none") == "none"
    assert validate_depth_fallback(None) is None


def test_import_manifest():
    """Test that manifest module imports successfully."""
    from transformation_portal.lux_depth_v3.manifest import (
        CombinedManifest,
        ConfigFingerprint,
        InputMetadata,
        DepthMetadata,
        V2Metadata,
        TimingMetadata,
        ReproMetadata,
        BatchManifest,
        compute_file_sha256,
        get_git_revision,
        capture_environment,
    )

    # Verify basic instantiation works
    manifest = CombinedManifest()
    assert manifest is not None

    fingerprint = ConfigFingerprint(
        model_variant="test",
        depth_quantization="none",
        depth_device="cpu",
    )
    assert fingerprint is not None

    # Test methods exist
    depth_fp = fingerprint.depth_only()
    assert depth_fp.model_variant == "test"

    v2_fp = fingerprint.v2_only()
    assert v2_fp.model_variant == ""

    # Verify functions are callable
    assert callable(compute_file_sha256)
    assert callable(get_git_revision)
    assert callable(capture_environment)


def test_import_batch_stats():
    """Test that batch_stats module imports successfully."""
    from transformation_portal.lux_depth_v3.batch_stats import (
        compute_batch_runtime_stats,
    )

    # Test basic functionality
    stats = compute_batch_runtime_stats([1.0, 2.0, 3.0])
    assert stats['count'] == 3
    assert stats['total'] == 6.0
    assert stats['mean'] == 2.0
    assert stats['min'] == 1.0
    assert stats['max'] == 3.0
    assert stats['median'] == 2.0

    # Test empty list
    empty_stats = compute_batch_runtime_stats([])
    assert empty_stats['count'] == 0


def test_import_preprocessing():
    """Test that preprocessing module imports successfully."""
    from transformation_portal.lux_depth_v3.preprocessing import (
        normalize_exif_orientation,
        validate_depth_image_alignment,
    )

    # Functions should exist and be callable
    assert callable(normalize_exif_orientation)
    assert callable(validate_depth_image_alignment)


def test_import_orchestrator():
    """Test that orchestrator module imports successfully."""
    from transformation_portal.lux_depth_v3.orchestrator import (
        EnhanceOrchestrator,
        make_output_key,
    )

    # Verify classes and functions exist
    assert EnhanceOrchestrator is not None
    assert callable(make_output_key)


def test_import_postprocessing():
    """Test that postprocessing module imports successfully."""
    from transformation_portal.lux_depth_v3.postprocessing import (
        Postprocessor,
    )
    from transformation_portal.lux_depth_v3.config import PostprocessingConfig

    # Verify basic instantiation works
    config = PostprocessingConfig()
    processor = Postprocessor(config)
    assert processor is not None
    assert processor.config == config


def test_all_imports_together():
    """Test that all modules can be imported together without conflicts."""
    from transformation_portal.lux_depth_v3 import (
        orchestrator,
        postprocessing,
    )
    from transformation_portal.lux_depth_v3 import config
    from transformation_portal.lux_depth_v3 import inference
    from transformation_portal.lux_depth_v3 import input_manager
    from transformation_portal.lux_depth_v3 import depth_writer
    from transformation_portal.lux_depth_v3 import v2_runner
    from transformation_portal.lux_depth_v3 import security
    from transformation_portal.lux_depth_v3 import manifest
    from transformation_portal.lux_depth_v3 import batch_stats
    from transformation_portal.lux_depth_v3 import preprocessing

    # All modules should be importable
    assert orchestrator is not None
    assert postprocessing is not None
    assert config is not None
    assert inference is not None
    assert input_manager is not None
    assert depth_writer is not None
    assert v2_runner is not None
    assert security is not None
    assert manifest is not None
    assert batch_stats is not None
    assert preprocessing is not None


def test_stub_not_implemented_errors():
    """Test that stub implementations raise NotImplementedError with clear messages."""
    from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
    from transformation_portal.lux_depth_v3.config import DA3Config
    from transformation_portal.lux_depth_v3.depth_writer import atomic_write_depth_u16_png_with_stats
    import numpy as np

    # Test inference engine raises NotImplementedError
    engine = DA3InferenceEngine(config=DA3Config())

    with pytest.raises(NotImplementedError) as exc_info:
        engine.infer(np.zeros((100, 100, 3)))

    assert "stub" in str(exc_info.value).lower()
    assert "full implementation pending" in str(exc_info.value).lower()

    # Test depth writer raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        atomic_write_depth_u16_png_with_stats(
            np.zeros((100, 100)),
            Path("/tmp/test.png")
        )

    assert "stub" in str(exc_info.value).lower()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])


# Intentional failure tests - verify stubs fail gracefully
def test_da3_predict_fails_intentionally():
    """Test that DA3InferenceEngine.predict() raises NotImplementedError, not TypeError."""
    import numpy as np
    from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
    from transformation_portal.lux_depth_v3.config import DA3Config

    config = DA3Config()
    engine = DA3InferenceEngine(config=config)

    # Should raise NotImplementedError (intentional stub), not TypeError/AttributeError
    with pytest.raises(NotImplementedError, match="stub"):
        engine.predict(np.zeros((64, 64, 3), dtype=np.float32))


def test_depth_writer_fails_intentionally():
    """Test that atomic_write_depth_u16_png_with_stats raises NotImplementedError."""
    import numpy as np
    from pathlib import Path
    from transformation_portal.lux_depth_v3.depth_writer import atomic_write_depth_u16_png_with_stats

    depth_map = np.zeros((64, 64), dtype=np.float32)
    output_path = Path("/tmp/test_depth.png")

    # Should raise NotImplementedError with call-site compatible signature
    with pytest.raises(NotImplementedError, match="stub"):
        atomic_write_depth_u16_png_with_stats(
            output_path=output_path,
            depth_map=depth_map,
            method="u16",
            debug_verify=False
        )


def test_v2_runner_fails_intentionally():
    """Test that V2Runner.run() raises NotImplementedError with correct signature."""
    from pathlib import Path
    from transformation_portal.lux_depth_v3.v2_runner import V2Runner

    runner = V2Runner()

    # Should raise NotImplementedError with orchestrator-compatible signature
    with pytest.raises(NotImplementedError, match="stub"):
        runner.run(
            input_path=Path("/tmp/input.png"),
            depth_dir=Path("/tmp/depth"),
            output_dir=Path("/tmp/output"),
            preset="default",
            device="cpu",
            upscaler_backend="default",
            log_file=Path("/tmp/v2.log"),
            timeout=300.0
        )


def test_postprocessor_config_has_required_fields():
    """Test that PostprocessingConfig has all fields Postprocessor accesses."""
    from transformation_portal.lux_depth_v3.config import PostprocessingConfig

    config = PostprocessingConfig()

    # These fields are accessed by postprocessing.py - must not raise AttributeError
    assert hasattr(config, 'apply_metric_scaling')
    assert hasattr(config, 'scale_factor')
    assert hasattr(config, 'apply_median_filter')
    assert hasattr(config, 'median_kernel_size')
    assert hasattr(config, 'apply_bilateral_filter')
    assert hasattr(config, 'bilateral_sigma_color')
    assert hasattr(config, 'bilateral_sigma_space')
    assert hasattr(config, 'preserve_edges')
    assert hasattr(config, 'edge_threshold')
    assert hasattr(config, 'fusion_mode')
    assert hasattr(config, 'refinement')


def test_depth_result_has_depth_alias():
    """Test that DepthResult.depth property works (orchestrator uses .depth, not .depth_map)."""
    import numpy as np
    from transformation_portal.lux_depth_v3.inference import DepthResult

    depth_map = np.zeros((64, 64), dtype=np.float32)
    image = np.zeros((64, 64, 3), dtype=np.float32)

    result = DepthResult(depth_map=depth_map, original_image=image, metadata={})

    # Should not raise AttributeError - orchestrator accesses .depth
    assert result.depth is not None
    assert result.depth is result.depth_map  # Verify it's an alias
