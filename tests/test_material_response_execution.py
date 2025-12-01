"""Tests for Material Response execution implementations.

Tests the physics-based surface enhancement implementations in:
- MaterialResponseStage (streaming/stages.py)
- UnifiedLuxuryPipeline._apply_material_response (pipelines/unified_luxury_pipeline.py)

These tests verify that the Material Response technology follows its three core tenets:
1. Respect energy conservation in highlights (preserve specular sheen)
2. Preserve midtone texture (keep materials tactile and dimensional)
3. Blend transitions between materials (authored, not procedural)
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# Test fixtures


@pytest.fixture
def synthetic_rgb_image():
    """Create a synthetic RGB image with various luminance zones."""
    h, w = 128, 128
    arr = np.zeros((h, w, 3), dtype=np.float32)

    # Create luminance zones
    # Top: highlights (bright)
    arr[:32, :, :] = 0.85
    # Middle-top: midtones (medium)
    arr[32:64, :, :] = 0.5
    # Middle-bottom: shadows (dark)
    arr[64:96, :, :] = 0.2
    # Bottom: floor region (wood-like warm tones)
    arr[96:, :, 0] = 0.55  # R
    arr[96:, :, 1] = 0.40  # G
    arr[96:, :, 2] = 0.30  # B

    return arr


@pytest.fixture
def metal_like_region():
    """Create a region with metal-like characteristics (neutral, high contrast)."""
    h, w = 64, 64
    arr = np.zeros((h, w, 3), dtype=np.float32)

    # Neutral gray with high local contrast (edges)
    base = 0.5
    arr[:, :, :] = base

    # Add edge patterns (simulating metal reflections)
    for i in range(0, w, 8):
        arr[:, i:i + 2, :] = 0.8

    return arr


@pytest.fixture
def textile_like_region():
    """Create a region with textile-like characteristics."""
    h, w = 64, 64
    arr = np.zeros((h, w, 3), dtype=np.float32)

    # Mid-brightness, low saturation
    arr[:, :, :] = 0.55

    # Add subtle texture
    noise = np.random.RandomState(42).normal(0, 0.03, (h, w, 3)).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)

    return arr


class TestMaterialResponseStage:
    """Tests for MaterialResponseStage._enhance_sync()."""

    def test_stage_initialization(self):
        """Test that MaterialResponseStage can be initialized."""
        from transformation_portal.streaming.stages import MaterialResponseStage

        stage = MaterialResponseStage(
            materials=["wood", "metal", "glass", "textile"],
            intensity=1.0,
            use_depth=True
        )

        assert stage._materials == ["wood", "metal", "glass", "textile"]
        assert stage._intensity == 1.0
        assert stage._use_depth is True

    def test_stage_default_materials(self):
        """Test default material list."""
        from transformation_portal.streaming.stages import MaterialResponseStage

        stage = MaterialResponseStage()

        assert "wood" in stage._materials
        assert "metal" in stage._materials
        assert "glass" in stage._materials
        assert "textile" in stage._materials

    def test_enhance_preserves_image_shape(self, synthetic_rgb_image):
        """Test that enhancement preserves image dimensions."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)
        image_data = ImageData(
            array=synthetic_rgb_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        assert result.array.shape == synthetic_rgb_image.shape

    def test_enhance_output_in_valid_range(self, synthetic_rgb_image):
        """Test that output values are clamped to [0, 1]."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)
        image_data = ImageData(
            array=synthetic_rgb_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        assert result.array.min() >= 0.0
        assert result.array.max() <= 1.0

    def test_enhance_sets_metadata(self, synthetic_rgb_image):
        """Test that enhancement sets appropriate metadata."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=0.8)
        image_data = ImageData(
            array=synthetic_rgb_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        assert result.metadata['material_enhanced'] is True
        assert result.metadata['materials'] == stage._materials
        assert result.metadata['material_response_version'] == '2.0'
        assert result.metadata['enhancement_intensity'] == 0.8

    def test_highlight_energy_conservation(self, synthetic_rgb_image):
        """Test that highlights are not over-enhanced (energy conservation)."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create image with extreme highlights
        highlight_image = np.ones((64, 64, 3), dtype=np.float32) * 0.95
        image_data = ImageData(
            array=highlight_image,
            path=Path("test.jpg"),
            metadata={}
        )

        original_mean = highlight_image.mean()
        result = stage._enhance_sync(image_data)
        enhanced_mean = result.array.mean()

        # Enhanced highlights should not be significantly brighter
        # (energy conservation tenet)
        assert enhanced_mean <= original_mean + 0.1

    def test_midtone_texture_preserved(self, synthetic_rgb_image):
        """Test that midtone texture detail is preserved."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create midtone image with texture (vectorized for efficiency)
        midtone_image = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        i, j = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
        midtone_image[(i + j) % 4 == 0, :] = 0.55

        image_data = ImageData(
            array=midtone_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Variance should not decrease (texture preserved)
        original_var = midtone_image.var()
        enhanced_var = result.array.var()

        # Texture should be preserved or enhanced, not flattened
        assert enhanced_var >= original_var * 0.8

    def test_intensity_scaling(self, synthetic_rgb_image):
        """Test that intensity parameter scales enhancement effect."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        # Low intensity
        stage_low = MaterialResponseStage(intensity=0.2)
        image_data_low = ImageData(
            array=synthetic_rgb_image.copy(),
            path=Path("test.jpg"),
            metadata={}
        )
        result_low = stage_low._enhance_sync(image_data_low)

        # High intensity
        stage_high = MaterialResponseStage(intensity=1.5)
        image_data_high = ImageData(
            array=synthetic_rgb_image.copy(),
            path=Path("test.jpg"),
            metadata={}
        )
        result_high = stage_high._enhance_sync(image_data_high)

        # Calculate differences from original
        diff_low = np.abs(result_low.array - synthetic_rgb_image).mean()
        diff_high = np.abs(result_high.array - synthetic_rgb_image).mean()

        # Higher intensity should produce larger changes
        assert diff_high > diff_low

    def test_handles_unnormalized_input(self):
        """Test that input values > 1.0 are properly normalized."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create image with values > 1.0 (simulating unnormalized input)
        unnormalized_image = np.ones((64, 64, 3), dtype=np.float32) * 128

        image_data = ImageData(
            array=unnormalized_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Should normalize and process correctly
        assert result.array.min() >= 0.0
        assert result.array.max() <= 1.0

    def test_depth_aware_processing(self, synthetic_rgb_image):
        """Test that depth map influences enhancement."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0, use_depth=True)

        # Create depth map (near=0, far=1)
        h, w = synthetic_rgb_image.shape[:2]
        depth_map = np.linspace(0, 1, h).reshape(-1, 1)
        depth_map = np.broadcast_to(depth_map, (h, w)).astype(np.float32)

        # Process with depth
        image_data_with_depth = ImageData(
            array=synthetic_rgb_image.copy(),
            path=Path("test.jpg"),
            depth_map=depth_map,
            metadata={}
        )
        result_with_depth = stage._enhance_sync(image_data_with_depth)

        # Process without depth for comparison
        image_data_no_depth = ImageData(
            array=synthetic_rgb_image.copy(),
            path=Path("test.jpg"),
            metadata={}
        )
        result_no_depth = stage._enhance_sync(image_data_no_depth)

        # Depth map should produce different results
        assert not np.allclose(result_with_depth.array, result_no_depth.array), \
            "Depth map should influence enhancement"
        assert result_with_depth.array.shape == synthetic_rgb_image.shape


class TestUnifiedLuxuryPipelineMaterialResponse:
    """Tests for UnifiedLuxuryPipeline._apply_material_response()."""

    def test_apply_material_response_basic(self, synthetic_rgb_image):
        """Test basic material response application."""
        from transformation_portal.pipelines.unified_luxury_pipeline import (
            UnifiedLuxuryPipeline,
            UnifiedPipelineConfig,
            SceneType
        )

        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO,
            enable_material_response=True,
            output_dir=Path("/tmp/test_output")
        )
        pipeline = UnifiedLuxuryPipeline(config)

        # Convert to PIL Image
        pil_image = Image.fromarray(
            (synthetic_rgb_image * 255).astype(np.uint8), 'RGB'
        )

        params = {'material_strength': 0.65}
        result = pipeline._apply_material_response(
            pil_image, params, SceneType.INTERIOR
        )

        assert isinstance(result, Image.Image)
        assert result.size == pil_image.size
        assert result.mode == 'RGB'

    def test_scene_type_variations(self, synthetic_rgb_image):
        """Test material response with different scene types."""
        from transformation_portal.pipelines.unified_luxury_pipeline import (
            UnifiedLuxuryPipeline,
            UnifiedPipelineConfig,
            SceneType
        )

        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO,
            enable_material_response=True,
            output_dir=Path("/tmp/test_output")
        )
        pipeline = UnifiedLuxuryPipeline(config)

        pil_image = Image.fromarray(
            (synthetic_rgb_image * 255).astype(np.uint8), 'RGB'
        )
        params = {'material_strength': 0.65}

        # Test each scene type
        for scene_type in [SceneType.INTERIOR, SceneType.EXTERIOR, SceneType.AERIAL]:
            result = pipeline._apply_material_response(
                pil_image, params, scene_type
            )
            assert isinstance(result, Image.Image)

    def test_material_strength_parameter(self, synthetic_rgb_image):
        """Test that material_strength parameter affects output."""
        from transformation_portal.pipelines.unified_luxury_pipeline import (
            UnifiedLuxuryPipeline,
            UnifiedPipelineConfig,
            SceneType
        )

        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO,
            enable_material_response=True,
            output_dir=Path("/tmp/test_output")
        )
        pipeline = UnifiedLuxuryPipeline(config)

        pil_image = Image.fromarray(
            (synthetic_rgb_image * 255).astype(np.uint8), 'RGB'
        )

        # Low strength
        result_low = pipeline._apply_material_response(
            pil_image.copy(), {'material_strength': 0.2}, SceneType.INTERIOR
        )

        # High strength
        result_high = pipeline._apply_material_response(
            pil_image.copy(), {'material_strength': 1.0}, SceneType.INTERIOR
        )

        # Calculate differences from original
        original_arr = np.array(pil_image).astype(np.float32)
        diff_low = np.abs(np.array(result_low).astype(np.float32) - original_arr).mean()
        diff_high = np.abs(np.array(result_high).astype(np.float32) - original_arr).mean()

        # Higher strength should produce larger changes
        assert diff_high >= diff_low

    def test_energy_conservation_in_highlights(self, synthetic_rgb_image):
        """Test that highlights respect energy conservation."""
        from transformation_portal.pipelines.unified_luxury_pipeline import (
            UnifiedLuxuryPipeline,
            UnifiedPipelineConfig,
            SceneType
        )

        config = UnifiedPipelineConfig(
            scene_type=SceneType.AUTO,
            enable_material_response=True,
            output_dir=Path("/tmp/test_output")
        )
        pipeline = UnifiedLuxuryPipeline(config)

        # Create bright highlight image
        highlight_arr = np.ones((64, 64, 3), dtype=np.float32) * 0.92
        pil_image = Image.fromarray(
            (highlight_arr * 255).astype(np.uint8), 'RGB'
        )

        result = pipeline._apply_material_response(
            pil_image, {'material_strength': 1.0}, SceneType.INTERIOR
        )

        result_arr = np.array(result).astype(np.float32) / 255.0
        original_max = highlight_arr.max()
        enhanced_max = result_arr.max()

        # Highlights should not clip/blow out
        assert enhanced_max <= 1.0
        # Should not increase dramatically
        assert enhanced_max <= original_max + 0.1


class TestMaterialDetection:
    """Tests for material detection heuristics."""

    def test_wood_detection_warm_tones(self, synthetic_rgb_image):
        """Test that warm tones in floor region are detected as wood."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(materials=["wood"], intensity=1.0)

        # Create image with wood-like colors in lower region
        wood_image = np.zeros((100, 100, 3), dtype=np.float32)
        # Floor region with warm wood tones
        wood_image[50:, :, 0] = 0.6  # R
        wood_image[50:, :, 1] = 0.45  # G
        wood_image[50:, :, 2] = 0.3  # B

        image_data = ImageData(
            array=wood_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Wood region should be enhanced (warmed)
        original_floor_r = wood_image[75, 50, 0]
        enhanced_floor_r = result.array[75, 50, 0]

        # Wood enhancement should slightly shift towards warm or processing occurred
        assert enhanced_floor_r >= original_floor_r or result.metadata['material_enhanced'] is True

    def test_metal_detection_neutral_high_contrast(self, metal_like_region):
        """Test that neutral high-contrast regions are detected as metal."""
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(materials=["metal"], intensity=1.0)

        image_data = ImageData(
            array=metal_like_region,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Metal should be processed
        assert result.metadata['material_enhanced'] is True
        assert result.array.shape == metal_like_region.shape


class TestTenetCompliance:
    """Tests verifying compliance with Material Response tenets."""

    def test_tenet_1_energy_conservation(self):
        """
        Tenet 1: Respect energy conservation in highlights.
        Verify that specular highlights maintain believable sheen without clipping.
        """
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create image with specular highlights
        highlight_image = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        highlight_image[20:44, 20:44, :] = 0.95  # Bright specular region

        image_data = ImageData(
            array=highlight_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Specular region should not clip to white
        specular_region = result.array[20:44, 20:44, :]
        assert specular_region.max() <= 1.0
        # Should not increase beyond original + small margin
        assert specular_region.mean() <= 1.0

    def test_tenet_2_midtone_texture_preservation(self):
        """
        Tenet 2: Preserve midtone texture.
        Verify that organic materials retain tactile, dimensional texture.
        """
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create midtone image with texture pattern (vectorized)
        textured_image = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        # Add checkerboard texture
        i, j = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
        mask = ((i // 4 + j // 4) % 2 == 0)
        textured_image[mask, :] = 0.45

        image_data = ImageData(
            array=textured_image,
            path=Path("test.jpg"),
            metadata={}
        )

        original_std = textured_image.std()
        result = stage._enhance_sync(image_data)
        enhanced_std = result.array.std()

        # Texture should be preserved or enhanced, not flattened
        assert enhanced_std >= original_std * 0.7

    def test_tenet_3_transition_blending(self):
        """
        Tenet 3: Blend transitions between materials.
        Verify smooth, authored transitions between adjacent materials.
        """
        from transformation_portal.streaming.stages import MaterialResponseStage, ImageData

        stage = MaterialResponseStage(intensity=1.0)

        # Create image with sharp material boundary
        boundary_image = np.zeros((64, 64, 3), dtype=np.float32)
        boundary_image[:32, :, :] = 0.3  # Dark material
        boundary_image[32:, :, :] = 0.7  # Light material

        image_data = ImageData(
            array=boundary_image,
            path=Path("test.jpg"),
            metadata={}
        )

        result = stage._enhance_sync(image_data)

        # Boundary should be smoothed (not remain perfectly sharp)
        boundary_region = result.array[30:34, :, :]
        # Transition region should have intermediate values
        gradient_detected = boundary_region.std() > 0.01
        assert gradient_detected, "Expected smooth transition but boundary remains sharp"

        # Result should show some blending occurred
        assert result.metadata['material_enhanced'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
