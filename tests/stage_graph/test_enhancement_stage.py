"""Tests for EnhancementStage defensive shape handling and core functionality.

Tests cover:
1. Depth map shape mismatch handling (resizing)
2. Missing input handling (graceful failures)
3. Material mask processing
4. Cache key generation
5. Bit depth preservation (8-bit vs 16-bit)
6. Clarity enhancement
7. Tone mapping with various depth distributions
"""

import numpy as np
import pytest

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage

pytestmark = pytest.mark.unit


class TestEnhancementStageShapeHandling:
    """Tests for defensive shape handling."""

    def test_resizes_mismatched_depth_map(self) -> None:
        """Depth map should be resized when it does not match image dimensions."""
        image = np.random.randint(0, 256, (64, 96, 3), dtype=np.uint8)
        depth_map = np.linspace(0.0, 1.0, 72 * 104, dtype=np.float32).reshape(72, 104)

        stage = EnhancementStage(
            enhancement_strength=0.7,
            clarity_strength=0.0,
            material_strength=0.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "depth_map": depth_map,
            }
        )

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.artifacts["enhanced_image"].shape == image.shape
        assert result.metadata["has_depth"] is True

    def test_handles_matching_depth_map(self) -> None:
        """Depth map with matching dimensions should not require resizing."""
        image = np.random.randint(0, 256, (64, 96, 3), dtype=np.uint8)
        depth_map = np.linspace(0.0, 1.0, 64 * 96, dtype=np.float32).reshape(64, 96)

        stage = EnhancementStage(
            enhancement_strength=0.7,
            clarity_strength=0.0,
            material_strength=0.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "depth_map": depth_map,
            }
        )

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.artifacts["enhanced_image"].shape == image.shape


class TestEnhancementStageMissingInputs:
    """Tests for missing input handling."""

    def test_fails_with_missing_image(self) -> None:
        """Stage should fail gracefully when image is missing."""
        stage = EnhancementStage()
        context = StageContext(artifacts={})

        result = stage.compute(context)

        assert result.status == StageStatus.FAILED
        assert "Missing 'image' artifact" in result.error

    def test_succeeds_without_depth_map(self) -> None:
        """Stage should succeed without depth map (soft dependency)."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage(enhancement_strength=0.5)
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["has_depth"] is False

    def test_succeeds_without_material_masks(self) -> None:
        """Stage should succeed without material masks (soft dependency)."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage(material_strength=0.5)
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["has_materials"] is False


class TestEnhancementStageMaterialProcessing:
    """Tests for material mask processing."""

    def test_applies_wood_material_enhancement(self) -> None:
        """Wood material mask should boost warmth."""
        image = np.full((32, 32, 3), 0.5, dtype=np.float32)
        wood_mask = np.ones((32, 32), dtype=np.float32)

        stage = EnhancementStage(
            enhancement_strength=0.0,
            clarity_strength=0.0,
            material_strength=1.0,
        )
        context = StageContext(
            artifacts={
                "image": (image * 255).astype(np.uint8),
                "material_masks": {"wood": wood_mask},
            }
        )

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert result.metadata["has_materials"] is True
        assert "wood" in result.artifacts["enhancement_metadata"]["materials_applied"]

    def test_applies_metal_material_enhancement(self) -> None:
        """Metal material mask should boost contrast."""
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        metal_mask = np.ones((32, 32), dtype=np.float32)

        stage = EnhancementStage(
            enhancement_strength=0.0,
            clarity_strength=0.0,
            material_strength=1.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "material_masks": {"metal": metal_mask},
            }
        )

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        assert "metal" in result.artifacts["enhancement_metadata"]["materials_applied"]

    def test_empty_mask_does_not_modify_image(self) -> None:
        """Empty material mask (all zeros) should not modify the image pixels.

        The mask key is still reported in materials_applied (it was provided),
        but the actual enhancement is skipped because mask.max() == 0.
        """
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        empty_mask = np.zeros((32, 32), dtype=np.float32)

        stage = EnhancementStage(
            enhancement_strength=0.0,  # Disable other enhancements
            clarity_strength=0.0,
            material_strength=1.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "material_masks": {"wood": empty_mask},
            }
        )

        result = stage.compute(context)

        assert result.status == StageStatus.COMPLETED
        # Mask key is reported (it was provided)
        assert "wood" in result.artifacts["enhancement_metadata"]["materials_applied"]
        # Image should be unchanged since empty mask causes skip
        assert np.array_equal(result.artifacts["enhanced_image"], image)


class TestEnhancementStageCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_changes_with_image(self) -> None:
        """Cache key should change when image changes."""
        # Use deterministic fixtures to avoid flaky tests from random collisions
        image1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        image2 = image1.copy()
        image2[0, 0, 0] = 200  # Single pixel difference guarantees different hash

        stage = EnhancementStage()
        context1 = StageContext(artifacts={"image": image1})
        context2 = StageContext(artifacts={"image": image2})

        key1 = stage.get_cache_key(context1)
        key2 = stage.get_cache_key(context2)

        assert key1 != key2

    def test_cache_key_changes_with_depth_map(self) -> None:
        """Cache key should change when depth map changes."""
        # Use deterministic fixtures to avoid flaky tests from random collisions
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        depth1 = np.full((32, 32), 0.5, dtype=np.float32)
        depth2 = depth1.copy()
        depth2[0, 0] = 1.0  # Single pixel difference guarantees different hash

        stage = EnhancementStage()
        context1 = StageContext(artifacts={"image": image, "depth_map": depth1})
        context2 = StageContext(artifacts={"image": image, "depth_map": depth2})

        key1 = stage.get_cache_key(context1)
        key2 = stage.get_cache_key(context2)

        assert key1 != key2

    def test_cache_key_handles_missing_image(self) -> None:
        """Cache key should return placeholder for missing image."""
        stage = EnhancementStage()
        context = StageContext(artifacts={})

        key = stage.get_cache_key(context)

        assert key == "no_image"

    def test_cache_key_deterministic(self) -> None:
        """Same inputs should produce same cache key."""
        # Use local RNG to avoid leaking state to other tests
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage(enhancement_strength=0.5, clarity_strength=0.3)
        context = StageContext(artifacts={"image": image})

        key1 = stage.get_cache_key(context)
        key2 = stage.get_cache_key(context)

        assert key1 == key2


class TestEnhancementStageBitDepth:
    """Tests for bit depth preservation."""

    def test_8bit_input_produces_8bit_output(self) -> None:
        """8-bit input should produce 8-bit output by default."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage()
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)

        assert result.artifacts["enhanced_image"].dtype == np.uint8

    def test_16bit_input_with_16bit_output_dtype(self) -> None:
        """16-bit input with output_dtype=uint16 should produce 16-bit output."""
        image = np.random.randint(0, 65536, (32, 32, 3), dtype=np.uint16)

        stage = EnhancementStage(output_dtype=np.dtype("uint16"))
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)

        assert result.artifacts["enhanced_image"].dtype == np.uint16

    def test_16bit_range_preserved(self) -> None:
        """16-bit input range should be preserved correctly."""
        # Create image with full 16-bit range
        image = np.full((32, 32, 3), 32768, dtype=np.uint16)

        stage = EnhancementStage(
            enhancement_strength=0.0,  # No modification
            clarity_strength=0.0,
            material_strength=0.0,
            output_dtype=np.dtype("uint16"),
        )
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)
        enhanced = result.artifacts["enhanced_image"]

        # With no enhancement, values should be close to original
        assert np.allclose(enhanced, image, rtol=0.01)


class TestEnhancementStageClarityEnhancement:
    """Tests for clarity enhancement."""

    def test_clarity_enhancement_modifies_image(self) -> None:
        """Clarity enhancement should modify the image."""
        image = np.random.randint(64, 192, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage(
            enhancement_strength=0.0,
            clarity_strength=1.0,
            material_strength=0.0,
        )
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)
        enhanced = result.artifacts["enhanced_image"]

        # Image should be modified
        assert not np.array_equal(enhanced, image)

    def test_zero_clarity_preserves_image(self) -> None:
        """Zero clarity strength should not apply clarity enhancement."""
        image = np.full((32, 32, 3), 128, dtype=np.uint8)

        stage = EnhancementStage(
            enhancement_strength=0.0,
            clarity_strength=0.0,
            material_strength=0.0,
        )
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)
        enhanced = result.artifacts["enhanced_image"]

        # With no enhancements, output should equal input
        assert np.array_equal(enhanced, image)


class TestEnhancementStageToneMapping:
    """Tests for depth-aware tone mapping."""

    def test_tone_mapping_with_depth_map(self) -> None:
        """Tone mapping should modify image based on depth."""
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        # Gradient depth map
        depth_map = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)

        stage = EnhancementStage(
            enhancement_strength=1.0,
            clarity_strength=0.0,
            material_strength=0.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "depth_map": depth_map,
            }
        )

        result = stage.compute(context)
        enhanced = result.artifacts["enhanced_image"]

        # Image should be modified by tone mapping
        assert not np.array_equal(enhanced, image)

    def test_tone_mapping_disabled_with_zero_strength(self) -> None:
        """Tone mapping should not be applied with zero enhancement strength."""
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        depth_map = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32).astype(np.float32)

        stage = EnhancementStage(
            enhancement_strength=0.0,
            clarity_strength=0.0,
            material_strength=0.0,
        )
        context = StageContext(
            artifacts={
                "image": image,
                "depth_map": depth_map,
            }
        )

        result = stage.compute(context)
        enhanced = result.artifacts["enhanced_image"]

        # With no enhancements, output should equal input
        assert np.array_equal(enhanced, image)


class TestEnhancementStageMetadata:
    """Tests for metadata and enhancement metadata."""

    def test_enhancement_metadata_structure(self) -> None:
        """Enhancement metadata should have expected structure."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage(
            enhancement_strength=0.7,
            clarity_strength=0.5,
            material_strength=0.6,
        )
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)
        metadata = result.artifacts["enhancement_metadata"]

        assert metadata["enhancement_strength"] == 0.7
        assert metadata["clarity_strength"] == 0.5
        assert metadata["material_strength"] == 0.6
        assert "materials_applied" in metadata

    def test_duration_recorded(self) -> None:
        """Processing duration should be recorded."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

        stage = EnhancementStage()
        context = StageContext(artifacts={"image": image})

        result = stage.compute(context)

        assert result.duration_ms is not None
        assert result.duration_ms >= 0
        assert result.metadata["processing_ms"] >= 0
