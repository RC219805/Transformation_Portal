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

# Import modules under test
from realize_v8_unified import (
    enhance,
    _open_any,
    _save_with_meta,
    _image_to_float_array,
    PRESETS,
    Preset,
)

from realize_v8_unified_cli_extension import (
    VFX_PRESETS,
    estimate_depth_fast,
    apply_depth_bloom,
    apply_depth_fog,
    apply_depth_of_field,
    apply_lut_with_depth,
    apply_color_grade_zones,
    enhance_with_vfx,
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    arr = np.random.rand(100, 100, 3).astype(np.float32)
    return Image.fromarray((arr * 255).astype(np.uint8))


@pytest.fixture
def sample_array():
    """Create a sample numpy array."""
    return np.random.rand(100, 100, 3).astype(np.float32)


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
        assert "dramatic_dof" in VFX_PRESETS
    
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
        for vfx_preset in VFX_PRESETS.keys():
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
