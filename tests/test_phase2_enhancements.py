#!/usr/bin/env python3
"""
Test Suite for Phase 2 Enhancements
====================================
Comprehensive tests for material detection, depth-aware LUT, performance profiler,
and exposure fusion.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import time
import json

# Import Phase 2 modules
from tools.material_detector import (
    MaterialDetector, MaterialType, MaterialConfidence, MaterialDetectionResult
)
from tools.depth_aware_lut import (
    DepthAwareLUT, DepthAwareLUTConfig, ZoneLUTConfig, DepthZone, LUTReader
)
from utils.performance_profiler import (
    PerformanceProfiler, StageMetrics, SystemSnapshot
)
from utils.exposure_fusion import ExposureFusion, ExposureTarget


# ============================================================================
# Material Detector Tests
# ============================================================================

class TestMaterialDetector:
    """Tests for material detection with confidence scores."""
    
    @pytest.fixture
    def synthetic_wood_image(self, tmp_path):
        """Create synthetic wood-colored image."""
        from PIL import Image
        
        # Brown/tan colors typical of wood
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[..., 0] = 139  # R
        img[..., 1] = 90   # G
        img[..., 2] = 43   # B
        
        # Add some texture
        noise = np.random.randint(-20, 20, (100, 100, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        img_path = tmp_path / "wood_test.png"
        Image.fromarray(img).save(img_path)
        return img_path
    
    @pytest.fixture
    def synthetic_metal_image(self, tmp_path):
        """Create synthetic metallic image."""
        from PIL import Image
        
        # Gray with high value (metallic)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[...] = 180  # Light gray
        
        # Add specular highlights
        img[20:30, 20:30] = 240  # Bright spot
        
        img_path = tmp_path / "metal_test.png"
        Image.fromarray(img).save(img_path)
        return img_path
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = MaterialDetector(min_confidence=0.3)
        assert detector.min_confidence == 0.3
        assert len(detector.material_profiles) == 8  # 8 material types
    
    def test_rgb_to_hsv_conversion(self):
        """Test RGB to HSV conversion."""
        detector = MaterialDetector()
        
        # Red
        rgb = np.array([[[1.0, 0.0, 0.0]]])
        hsv = detector._rgb_to_hsv(rgb)
        assert hsv[0, 0, 0] == 0.0  # Hue = 0 for red
        assert hsv[0, 0, 1] == 1.0  # Full saturation
        assert hsv[0, 0, 2] == 1.0  # Full value
        
        # Gray
        rgb = np.array([[[0.5, 0.5, 0.5]]])
        hsv = detector._rgb_to_hsv(rgb)
        assert hsv[0, 0, 1] == 0.0  # No saturation
        assert hsv[0, 0, 2] == 0.5  # Mid value
    
    def test_texture_strength_computation(self):
        """Test texture strength calculation."""
        detector = MaterialDetector()
        
        # Smooth image (low texture)
        smooth = np.ones((50, 50, 3)) * 0.5
        texture = detector._compute_texture_strength(smooth)
        assert texture.max() < 0.01  # Very low texture
        
        # Textured image (high texture)
        textured = np.random.rand(50, 50, 3)
        texture = detector._compute_texture_strength(textured)
        assert texture.max() > 0.1  # Noticeable texture
    
    def test_specular_map_computation(self):
        """Test specular highlight detection."""
        detector = MaterialDetector()
        
        # Image with bright highlight
        img = np.ones((50, 50, 3)) * 0.3
        img[20:30, 20:30] = 1.0  # Bright region
        
        specular = detector._compute_specular_map(img)
        
        # Highlight region should have high specular
        assert specular[25, 25] > 0.8
        # Dark region should have low specular
        assert specular[5, 5] < 0.5
    
    @pytest.mark.slow
    def test_detect_wood_image(self, synthetic_wood_image):
        """Test detection on synthetic wood image."""
        detector = MaterialDetector(min_confidence=0.1)
        result = detector.detect(synthetic_wood_image)
        
        assert isinstance(result, MaterialDetectionResult)
        assert result.image_path == synthetic_wood_image
        assert len(result.confidence_maps) == 8
        
        # Wood should be detected with reasonable confidence
        if MaterialType.WOOD in result.materials:
            wood_conf = result.materials[MaterialType.WOOD]
            assert wood_conf.percentage > 10  # At least 10% coverage
            assert wood_conf.mean_confidence > 0.1
    
    @pytest.mark.slow
    def test_generate_heatmap(self, synthetic_wood_image, tmp_path):
        """Test heatmap generation."""
        detector = MaterialDetector()
        result = detector.detect(synthetic_wood_image)
        
        output_path = tmp_path / "heatmap.png"
        detector.generate_heatmap(result, MaterialType.WOOD, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_generate_report(self, synthetic_wood_image, tmp_path):
        """Test JSON report generation."""
        detector = MaterialDetector()
        result = detector.detect(synthetic_wood_image)
        
        report_path = tmp_path / "report.json"
        detector.generate_report(result, report_path)
        
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert 'image_path' in report
        assert 'materials' in report
        assert 'dominant_material' in report
        assert 'processing_time_seconds' in report


# ============================================================================
# Depth-Aware LUT Tests
# ============================================================================

class TestDepthAwareLUT:
    """Tests for depth-aware LUT application."""
    
    @pytest.fixture
    def sample_lut_file(self, tmp_path):
        """Create a sample .cube LUT file."""
        lut_path = tmp_path / "test.cube"
        
        # Create simple identity LUT (3x3x3)
        with open(lut_path, 'w') as f:
            f.write("TITLE \"Test LUT\"\n")
            f.write("LUT_3D_SIZE 3\n\n")
            
            for b in [0.0, 0.5, 1.0]:
                for g in [0.0, 0.5, 1.0]:
                    for r in [0.0, 0.5, 1.0]:
                        f.write(f"{r} {g} {b}\n")
        
        return lut_path
    
    @pytest.fixture
    def sample_depth_map(self):
        """Create sample depth map."""
        # Gradient from top (far) to bottom (near)
        depth = np.linspace(1.0, 0.0, 100)[:, np.newaxis]
        depth = np.repeat(depth, 100, axis=1)
        return depth
    
    def test_lut_reader_cube_format(self, sample_lut_file):
        """Test reading .cube LUT files."""
        lut, size = LUTReader.read_cube_lut(sample_lut_file)
        
        assert size == 3
        assert lut.shape == (3, 3, 3, 3)
        assert lut.dtype == np.float32
    
    def test_lut_application(self, sample_lut_file):
        """Test LUT application with trilinear interpolation."""
        lut, size = LUTReader.read_cube_lut(sample_lut_file)
        
        # Test image
        img = np.random.rand(50, 50, 3).astype(np.float32)
        
        result = LUTReader.apply_lut(img, lut, size)
        
        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_zone_mask_creation(self, sample_depth_map):
        """Test depth zone mask creation."""
        config = DepthAwareLUTConfig(
            zone_configs={},
            depth_falloff=2.0
        )
        
        # Create dummy processor to test mask creation
        processor = DepthAwareLUT.__new__(DepthAwareLUT)
        processor.config = config
        
        masks = processor._create_zone_masks(sample_depth_map)
        
        assert len(masks) == 3
        assert DepthZone.FOREGROUND in masks
        assert DepthZone.MIDGROUND in masks
        assert DepthZone.BACKGROUND in masks
        
        # Masks should sum to 1.0
        total = sum(masks.values())
        np.testing.assert_allclose(total, 1.0, rtol=1e-5)
        
        # Foreground should be strong at bottom (near)
        assert masks[DepthZone.FOREGROUND][-1, 50] > 0.5
        
        # Background should be strong at top (far)
        assert masks[DepthZone.BACKGROUND][0, 50] > 0.5
    
    def test_color_temperature_shift(self):
        """Test color temperature adjustment."""
        processor = DepthAwareLUT.__new__(DepthAwareLUT)
        
        img = np.ones((50, 50, 3), dtype=np.float32) * 0.5
        
        # Warm shift
        warmed = processor._apply_color_temp(img, 500)
        assert warmed[..., 0].mean() > img[..., 0].mean()  # More red
        assert warmed[..., 2].mean() < img[..., 2].mean()  # Less blue
        
        # Cool shift
        cooled = processor._apply_color_temp(img, -500)
        assert cooled[..., 2].mean() > img[..., 2].mean()  # More blue
        assert cooled[..., 0].mean() < img[..., 0].mean()  # Less red
    
    @pytest.mark.slow
    def test_depth_aware_lut_application(self, sample_lut_file, sample_depth_map):
        """Test full depth-aware LUT application."""
        config = DepthAwareLUTConfig(
            zone_configs={
                DepthZone.FOREGROUND: ZoneLUTConfig(
                    zone=DepthZone.FOREGROUND,
                    lut_path=sample_lut_file,
                    strength=0.8
                ),
                DepthZone.BACKGROUND: ZoneLUTConfig(
                    zone=DepthZone.BACKGROUND,
                    lut_path=sample_lut_file,
                    strength=0.5
                )
            },
            atmospheric_strength=0.3
        )
        
        processor = DepthAwareLUT(config)
        
        img = np.random.rand(100, 100, 3).astype(np.float32)
        result = processor.apply(img, sample_depth_map)
        
        assert result.shape == img.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ============================================================================
# Performance Profiler Tests
# ============================================================================

class TestPerformanceProfiler:
    """Tests for performance profiling."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(session_id="test_session")
        
        assert profiler.session_id == "test_session"
        assert len(profiler.stages) == 0
        assert len(profiler.snapshots) == 1  # Baseline snapshot
    
    def test_stage_profiling(self):
        """Test profiling individual stages."""
        profiler = PerformanceProfiler()
        
        with profiler.stage('test_stage', items=10):
            # Simulate work
            time.sleep(0.1)
            data = np.random.rand(1000, 1000)  # Allocate memory
        
        assert len(profiler.stages) == 1
        stage = profiler.stages[0]
        
        assert stage.name == 'test_stage'
        assert stage.duration >= 0.1
        assert stage.items_processed == 10
        assert stage.throughput is not None
        assert stage.throughput <= 100  # 10 items in 0.1s = 100/s max
    
    def test_multiple_stages(self):
        """Test profiling multiple sequential stages."""
        profiler = PerformanceProfiler()
        
        with profiler.stage('stage1', items=5):
            time.sleep(0.05)
        
        with profiler.stage('stage2', items=10):
            time.sleep(0.1)
        
        with profiler.stage('stage3', items=3):
            time.sleep(0.03)
        
        assert len(profiler.stages) == 3
        assert profiler.stages[0].name == 'stage1'
        assert profiler.stages[1].name == 'stage2'
        assert profiler.stages[2].name == 'stage3'
    
    def test_peak_memory_tracking(self):
        """Test peak memory tracking within stage."""
        profiler = PerformanceProfiler()
        
        with profiler.stage('memory_test'):
            data1 = np.random.rand(1000, 1000)
            profiler.update_peak_memory()
            
            data2 = np.random.rand(1000, 1000)
            profiler.update_peak_memory()
        
        stage = profiler.stages[0]
        assert stage.memory_peak >= stage.memory_start
    
    def test_report_generation(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler(session_id="report_test")
        
        with profiler.stage('stage1', items=5):
            time.sleep(0.05)
        
        with profiler.stage('stage2', items=10):
            time.sleep(0.1)
        
        report = profiler.generate_report()
        
        assert report.session_id == "report_test"
        assert report.total_duration >= 0.15
        assert len(report.stages) == 2
        assert 'total_stages' in report.summary
        assert 'total_items_processed' in report.summary
        assert report.summary['total_items_processed'] == 15
    
    def test_bottleneck_identification(self):
        """Test automatic bottleneck identification."""
        profiler = PerformanceProfiler()
        
        # Create a bottleneck stage (much slower than others)
        with profiler.stage('fast_stage', items=10):
            time.sleep(0.05)
        
        with profiler.stage('slow_stage', items=5):
            time.sleep(0.3)  # Bottleneck
        
        with profiler.stage('another_fast', items=10):
            time.sleep(0.05)
        
        report = profiler.generate_report()
        
        assert len(report.bottlenecks) > 0
        assert any('slow_stage' in b for b in report.bottlenecks)
    
    def test_report_save_load(self, tmp_path):
        """Test saving and loading reports."""
        profiler = PerformanceProfiler()
        
        with profiler.stage('test', items=5):
            time.sleep(0.05)
        
        report = profiler.generate_report()
        
        report_path = tmp_path / "performance.json"
        profiler.save_report(report, report_path)
        
        assert report_path.exists()
        
        with open(report_path) as f:
            data = json.load(f)
        
        assert 'session_id' in data
        assert 'stages' in data
        assert len(data['stages']) == 1


# ============================================================================
# Exposure Fusion Tests
# ============================================================================

class TestExposureFusion:
    """Tests for multi-exposure fusion."""
    
    @pytest.fixture
    def hdr_image(self):
        """Create synthetic HDR image (linear RGB)."""
        # Image with wide dynamic range
        img = np.random.rand(100, 100, 3).astype(np.float32)
        img = img * 10.0  # Extend range beyond [0, 1]
        return img
    
    def test_fusion_initialization(self):
        """Test fusion processor initialization."""
        fusion = ExposureFusion()
        assert fusion is not None
    
    def test_tone_mapping_reinhard(self):
        """Test Reinhard tone mapping."""
        fusion = ExposureFusion()
        
        # HDR input
        hdr = np.array([[[0.5, 1.0, 5.0]]]).astype(np.float32)
        
        tone_mapped = fusion._tone_map(hdr, method='reinhard')
        
        assert tone_mapped.min() >= 0.0
        assert tone_mapped.max() <= 1.0
    
    def test_bracket_extraction(self, hdr_image):
        """Test exposure bracket extraction."""
        fusion = ExposureFusion()
        
        brackets = fusion.extract_brackets(hdr_image, num_brackets=3, ev_range=2.0)
        
        assert len(brackets) == 3
        
        # Check EV values
        evs = [b[0] for b in brackets]
        assert evs[0] < 0  # Underexposed
        assert evs[1] == 0  # Neutral
        assert evs[2] > 0  # Overexposed
        
        # Check images
        for ev, img in brackets:
            assert img.shape == hdr_image.shape
            assert img.min() >= 0.0
            assert img.max() <= 1.0
    
    def test_weighted_average_fusion(self, hdr_image):
        """Test weighted average fusion."""
        fusion = ExposureFusion()
        
        brackets = fusion.extract_brackets(hdr_image, num_brackets=3, ev_range=2.0)
        bracket_images = [b[1] for b in brackets]
        
        fused = fusion._weighted_average_fusion(bracket_images)
        
        assert fused.shape == hdr_image.shape
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0
    
    @pytest.mark.slow
    def test_laplacian_pyramid_fusion(self, hdr_image):
        """Test Laplacian pyramid fusion."""
        fusion = ExposureFusion()
        
        brackets = fusion.extract_brackets(hdr_image, num_brackets=3, ev_range=2.0)
        bracket_images = [b[1] for b in brackets]
        
        fused = fusion._laplacian_pyramid_fusion(bracket_images)
        
        assert fused.shape == hdr_image.shape
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0
    
    def test_gaussian_pyramid(self):
        """Test Gaussian pyramid construction."""
        fusion = ExposureFusion()
        
        img = np.random.rand(64, 64, 3).astype(np.float32)
        pyramid = fusion._build_gaussian_pyramid(img, levels=4)
        
        assert len(pyramid) == 4
        assert pyramid[0].shape == (64, 64, 3)
        assert pyramid[1].shape == (32, 32, 3)
        assert pyramid[2].shape == (16, 16, 3)
        assert pyramid[3].shape == (8, 8, 3)
    
    def test_variant_generation(self, hdr_image):
        """Test exposure-optimized variant generation."""
        fusion = ExposureFusion()
        
        variants = fusion.generate_variants(hdr_image)
        
        assert len(variants) == 3  # Web, print, social
        
        targets = [v.target for v in variants]
        assert ExposureTarget.WEB in targets
        assert ExposureTarget.PRINT in targets
        assert ExposureTarget.SOCIAL in targets
        
        for variant in variants:
            assert variant.image.shape == hdr_image.shape
            assert variant.image.min() >= 0.0
            assert variant.image.max() <= 1.0
            assert variant.description != ""
    
    def test_web_variant_characteristics(self, hdr_image):
        """Test web variant has correct characteristics."""
        fusion = ExposureFusion()
        variants = fusion.generate_variants(hdr_image)
        
        web_variant = next(v for v in variants if v.target == ExposureTarget.WEB)
        
        # Should be slightly underexposed
        assert web_variant.exposure_ev < 0
    
    def test_social_variant_characteristics(self, hdr_image):
        """Test social media variant has correct characteristics."""
        fusion = ExposureFusion()
        variants = fusion.generate_variants(hdr_image)
        
        social_variant = next(v for v in variants if v.target == ExposureTarget.SOCIAL)
        
        # Should be slightly overexposed
        assert social_variant.exposure_ev > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 features."""
    
    @pytest.mark.slow
    def test_material_detection_with_profiling(self, tmp_path):
        """Test material detection with performance profiling."""
        from PIL import Image
        
        # Create test image
        img_path = tmp_path / "test.png"
        img = np.random.rand(200, 200, 3) * 255
        Image.fromarray(img.astype(np.uint8)).save(img_path)
        
        # Profile detection
        profiler = PerformanceProfiler(session_id="material_detection_test")
        detector = MaterialDetector()
        
        with profiler.stage('detection', items=1):
            result = detector.detect(img_path)
        
        report = profiler.generate_report()
        
        assert len(report.stages) == 1
        assert report.stages[0].name == 'detection'
        assert isinstance(result, MaterialDetectionResult)
    
    @pytest.mark.slow
    def test_exposure_fusion_with_profiling(self, tmp_path):
        """Test exposure fusion with performance profiling."""
        profiler = PerformanceProfiler(session_id="fusion_test")
        fusion = ExposureFusion()
        
        hdr_image = np.random.rand(100, 100, 3).astype(np.float32) * 5.0
        
        with profiler.stage('bracket_extraction', items=3):
            brackets = fusion.extract_brackets(hdr_image, num_brackets=3)
        
        with profiler.stage('fusion', items=1):
            bracket_images = [b[1] for b in brackets]
            fused = fusion.fuse_exposures(bracket_images)
        
        report = profiler.generate_report()
        
        assert len(report.stages) == 2
        assert report.summary['total_items_processed'] == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
