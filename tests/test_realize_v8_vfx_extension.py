#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for realize_v8_unified_cli_extension.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skip(reason="realize_v8 modules not yet migrated to src package")

# Import modules under test
try:
    from scripts.utilities.realize_v8_unified import (
        PRESETS,
        Preset,
        _image_to_float_array,
        _open_any,
        _save_with_meta,
        enhance,
    )
    from scripts.utilities.realize_v8_unified_cli_extension import (
        VFX_PRESETS,
        _process_single_image_vfx,
        apply_color_grade_zones,
        apply_depth_bloom,
        apply_depth_fog,
        apply_depth_of_field,
        apply_lut_with_depth,
        batch_process_vfx,
        enhance_with_vfx,
        estimate_depth_fast,
    )
except ImportError:
    pass


# ==================== Fixtures ====================

@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    rng = np.random.default_rng(42)  # Use isolated RNG for reproducibility
    arr = rng.random((100, 100, 3)).astype(np.float32)
    return Image.fromarray((arr * 255).astype(np.uint8))


@pytest.fixture
def sample_array():
    """Create a sample numpy array."""
    rng = np.random.default_rng(42)  # Use isolated RNG for reproducibility
    return rng.random((100, 100, 3)).astype(np.float32)


@pytest.fixture
def sample_depth():
    """Create a sample depth map."""
    h, w = 100, 100
    y = np.linspace(0, 1, h)
    depth = np.tile(y[:, None], (1, w))
    return depth.astype(np.float32)


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        sample_image.save(f.name)
        yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)


# ==================== Test realize_v8_unified ====================

class TestRealizeV8Unified:
    """Test base realize_v8_unified functionality."""

    def test_presets_exist(self):
        """Test that presets are defined."""
        assert len(PRESETS) > 0
        assert "signature_estate" in PRESETS
        assert "signature_estate_agx" in PRESETS

    def test_preset_structure(self):
        """Test preset dataclass structure."""
        preset = PRESETS["signature_estate"]
        assert isinstance(preset, Preset)
        assert hasattr(preset, 'name')
        assert hasattr(preset, 'exposure')
        assert hasattr(preset, 'contrast')

    def test_image_to_float_array(self, sample_image):
        """Test image conversion to float array."""
        arr = _image_to_float_array(sample_image)
        assert arr.dtype == np.float32
        assert arr.shape == (100, 100, 3)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_enhance_basic(self, sample_array):
        """Test basic enhancement."""
        preview, working, metrics = enhance(
            sample_array,
            exposure=0.1,
            contrast=1.1,
            saturation=1.05
        )

        assert isinstance(preview, Image.Image)
        assert working.shape == sample_array.shape
        assert 'total_time_ms' in metrics
        assert metrics['exposure'] == 0.1

    def test_enhance_with_preset(self, sample_image):
        """Test enhancement with preset."""
        preset_params = PRESETS["signature_estate"].to_dict()
        preview, working, metrics = enhance(sample_image, **preset_params)

        assert isinstance(preview, Image.Image)
        assert working.shape == (100, 100, 3)
        assert metrics['contrast'] == preset_params['contrast']

    def test_open_any(self, temp_image_file):
        """Test opening image file."""
        img, meta = _open_any(temp_image_file)

        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'
        assert 'format' in meta
        assert 'size' in meta

    def test_save_with_meta(self, sample_image, sample_array):
        """Test saving with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jpg"
            meta = {'format': 'JPEG', 'mode': 'RGB', 'size': (100, 100), 'info': {}}

            _save_with_meta(sample_image, sample_array, output_path, meta, out_bitdepth=8)

            assert output_path.exists()


# ==================== Test VFX Extension ====================

class TestVFXExtension:
    """Test VFX extension functionality."""

    def test_vfx_presets_exist(self):
        """Test that VFX presets are defined."""
        assert len(VFX_PRESETS) > 0
        assert "subtle_estate" in VFX_PRESETS
        assert "montecito_golden" in VFX_PRESETS
        assert "cinematic_fog" in VFX_PRESETS
        assert "dramatic_dof" in VFX_PRESETS  # Correct name is dramatic_dof not dramatic_do

    def test_vfx_preset_structure(self):
        """Test VFX preset structure."""
        preset = VFX_PRESETS["subtle_estate"]
        assert "description" in preset
        assert "bloom_intensity" in preset
        assert "material_boost" in preset

    def test_estimate_depth_fast(self, sample_array):
        """Test depth estimation (mock mode)."""
        depth = estimate_depth_fast(sample_array)

        assert depth.dtype == np.float32
        assert depth.shape == (100, 100)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_apply_depth_bloom(self, sample_array, sample_depth):
        """Test depth-aware bloom effect."""
        result = apply_depth_bloom(sample_array, sample_depth, intensity=0.2, radius=10)

        assert result.shape == sample_array.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_apply_depth_fog(self, sample_array, sample_depth):
        """Test depth-aware fog effect."""
        result = apply_depth_fog(
            sample_array,
            sample_depth,
            fog_color=(0.8, 0.85, 0.9),
            density=0.3
        )

        assert result.shape == sample_array.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_apply_depth_of_field(self, sample_array, sample_depth):
        """Test depth of field effect."""
        result = apply_depth_of_field(
            sample_array,
            sample_depth,
            focus_depth=0.35,
            blur_strength=5.0
        )

        assert result.shape == sample_array.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_apply_color_grade_zones(self, sample_array, sample_depth):
        """Test depth-based color grading."""
        result = apply_color_grade_zones(
            sample_array,
            sample_depth,
            near_color=(1.05, 1.0, 0.95),
            far_color=(0.8, 0.9, 1.0)
        )

        assert result.shape == sample_array.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_apply_lut_with_depth_missing_file(self, sample_array, sample_depth):
        """Test LUT application with missing file."""
        result = apply_lut_with_depth(
            sample_array,
            Path("nonexistent.cube"),
            sample_depth
        )

        # Should return original image when LUT missing
        np.testing.assert_array_equal(result, sample_array)

    def test_enhance_with_vfx_basic(self, sample_image):
        """Test complete VFX enhancement pipeline."""
        result = enhance_with_vfx(
            sample_image,
            base_preset="signature_estate",
            vfx_preset="subtle_estate",
            material_response=False,
            save_depth=True
        )

        assert "image" in result
        assert "array" in result
        assert "depth" in result
        assert "metrics" in result

        assert isinstance(result["image"], Image.Image)
        assert result["array"].shape == (100, 100, 3)
        assert result["depth"] is not None
        assert result["depth"].shape == (100, 100)

        assert "total_ms" in result["metrics"]
        assert "depth_estimation_ms" in result["metrics"]
        assert "vfx_ms" in result["metrics"]

    def test_enhance_with_vfx_all_presets(self, sample_image):
        """Test VFX enhancement with all presets."""
        for vfx_preset in VFX_PRESETS:
            result = enhance_with_vfx(
                sample_image,
                base_preset="natural",
                vfx_preset=vfx_preset,
                material_response=False,
                save_depth=False
            )

            assert result["image"] is not None
            assert result["array"].shape == (100, 100, 3)
            assert result["metrics"]["total_ms"] > 0

    def test_enhance_with_vfx_material_response(self, sample_image):
        """Test VFX with material response enabled."""
        result = enhance_with_vfx(
            sample_image,
            base_preset="signature_estate",
            vfx_preset="subtle_estate",
            material_response=True,
            save_depth=False
        )

        assert result["image"] is not None
        # Material response should add timing metric
        assert "material_response_ms" in result["metrics"]

    def test_enhance_with_vfx_no_depth_save(self, sample_image):
        """Test VFX without saving depth."""
        result = enhance_with_vfx(
            sample_image,
            base_preset="signature_estate",
            vfx_preset="subtle_estate",
            save_depth=False
        )

        assert result["depth"] is None

    def test_enhance_with_vfx_from_array(self, sample_array):
        """Test VFX enhancement from numpy array."""
        result = enhance_with_vfx(
            sample_array,
            base_preset="natural",
            vfx_preset="subtle_estate",
            save_depth=False
        )

        assert result["image"] is not None
        assert result["array"].shape == sample_array.shape

    def test_enhance_with_vfx_from_path(self, temp_image_file):
        """Test VFX enhancement from file path."""
        result = enhance_with_vfx(
            temp_image_file,
            base_preset="natural",
            vfx_preset="subtle_estate",
            save_depth=False
        )

        assert result["image"] is not None
        assert result["array"].shape == (100, 100, 3)


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_pipeline_single_image(self, temp_image_file):
        """Test full pipeline on single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jpg"

            # Open image
            img, meta = _open_any(temp_image_file)

            # Process with VFX
            result = enhance_with_vfx(
                img,
                base_preset="signature_estate_agx",
                vfx_preset="montecito_golden",
                material_response=True,
                save_depth=True
            )

            # Save
            _save_with_meta(
                result["image"],
                result["array"],
                output_path,
                meta,
                out_bitdepth=8
            )

            assert output_path.exists()

            # Verify depth was generated
            assert result["depth"] is not None
            assert result["depth"].shape == (100, 100)

    def test_preset_combinations(self, sample_image):
        """Test various preset combinations."""
        base_presets = ["signature_estate", "natural"]
        vfx_presets = ["subtle_estate", "cinematic_fog"]

        for base in base_presets:
            for vfx in vfx_presets:
                result = enhance_with_vfx(
                    sample_image,
                    base_preset=base,
                    vfx_preset=vfx,
                    material_response=False,
                    save_depth=False
                )

                assert result["image"] is not None
                assert result["metrics"]["total_ms"] > 0


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and timing tests."""

    def test_timing_metrics_present(self, sample_image):
        """Test that all timing metrics are present."""
        result = enhance_with_vfx(
            sample_image,
            base_preset="natural",
            vfx_preset="subtle_estate",
            material_response=True,
            save_depth=True
        )

        metrics = result["metrics"]
        assert "total_time_ms" in metrics
        assert "depth_estimation_ms" in metrics
        assert "vfx_ms" in metrics
        assert "material_response_ms" in metrics
        assert "total_ms" in metrics

    def test_minimal_processing_time(self, sample_array):
        """Test that processing completes in reasonable time."""
        import time

        start = time.perf_counter()
        result = enhance_with_vfx(
            sample_array,
            base_preset="natural",
            vfx_preset="subtle_estate",
            save_depth=False
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Should complete in under 5 seconds for small test image
        assert elapsed < 5000
        assert result["metrics"]["total_ms"] > 0


# ==================== Batch Processing Tests ====================

class TestBatchProcessing:
    """Tests for batch processing functionality."""

    @pytest.fixture
    def temp_input_dir(self, sample_image):
        """Create a temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()

            # Create 4 test images
            for i in range(4):
                img_path = input_dir / f"test_{i}.jpg"
                sample_image.save(img_path)

            yield input_dir

    def test_batch_process_sequential(self, temp_input_dir):
        """Test sequential batch processing (jobs=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = batch_process_vfx(
                input_dir=temp_input_dir,
                output_dir=output_dir,
                base_preset="signature_estate",
                vfx_preset="subtle_estate",
                material_response=False,
                pattern="*.jpg",
                jobs=1,
                out_bitdepth=8
            )

            # Verify results
            assert len(results) == 4
            assert all(r[1] for r in results)  # All succeeded
            assert all(r[2] is not None for r in results)  # All have timing

            # Verify output files exist
            output_files = list(output_dir.glob("*.jpg"))
            assert len(output_files) == 4

    def test_batch_process_parallel(self, temp_input_dir):
        """Test parallel batch processing (jobs>1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = batch_process_vfx(
                input_dir=temp_input_dir,
                output_dir=output_dir,
                base_preset="signature_estate",
                vfx_preset="subtle_estate",
                material_response=False,
                pattern="*.jpg",
                jobs=2,
                out_bitdepth=8
            )

            # Verify results
            assert len(results) == 4
            assert all(r[1] for r in results)  # All succeeded
            assert all(r[2] is not None for r in results)  # All have timing

            # Verify output files exist
            output_files = list(output_dir.glob("*.jpg"))
            assert len(output_files) == 4

    def test_batch_process_empty_directory(self):
        """Test batch processing with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "empty"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()

            results = batch_process_vfx(
                input_dir=input_dir,
                output_dir=output_dir,
                pattern="*.jpg",
                jobs=1
            )

            # Should return empty list for empty directory
            assert results == []

    def test_batch_process_returns_list(self, temp_input_dir):
        """Test that batch processing returns a list of results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = batch_process_vfx(
                input_dir=temp_input_dir,
                output_dir=output_dir,
                jobs=1
            )

            assert isinstance(results, list)
            for result in results:
                assert len(result) == 4
                path, success, proc_time, error = result
                assert isinstance(path, Path)
                assert isinstance(success, bool)
                if success:
                    assert isinstance(proc_time, int)
                    assert error is None
                else:
                    assert error is not None

    def test_process_single_image_vfx(self, temp_input_dir):
        """Test the single image processing helper function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_dir.mkdir(exist_ok=True)

            img_path = list(temp_input_dir.glob("*.jpg"))[0]

            result = _process_single_image_vfx(
                img_path=img_path,
                output_dir=output_dir,
                base_preset="signature_estate",
                vfx_preset="subtle_estate",
                material_response=False,
                out_bitdepth=8
            )

            path, success, proc_time, error = result
            assert success is True
            assert proc_time is not None
            assert error is None

            # Verify output file exists
            output_files = list(output_dir.glob("*.jpg"))
            assert len(output_files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
