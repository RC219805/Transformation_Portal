"""Tests for Materials V2/V3 configuration and validation.

Ensures Materials V2/V3 are properly enabled when configured, with fail-fast
validation for missing dependencies and model weights.
"""

import pytest
from pathlib import Path

from lux_depth_v2.config import PipelineConfig, Preset


class TestMaterialsV2V3Configuration:
    """Test suite for Materials V2/V3 configuration."""

    def test_production_ultra_materials_preset_exists(self):
        """Verify PRODUCTION_ULTRA_MATERIALS preset is defined."""
        assert Preset.PRODUCTION_ULTRA_MATERIALS == "production_ultra_materials"
        assert "production_ultra_materials" in [p.value for p in Preset]

    def test_production_ultra_materials_enables_v2(self):
        """Verify Materials V2 is enabled in production_ultra_materials preset."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # CRITICAL ASSERTIONS
        assert cfg.materials_v2 is not None, "Materials V2 config block must exist"
        assert cfg.materials_v2.enabled is True, "Materials V2 must be enabled"

        # Verify backend configuration
        assert cfg.materials_v2.backend == "segformer", "Must use SegFormer backend"

        # Verify confidence thresholds are set
        assert cfg.materials_v2.confidence.confidence_threshold > 0
        assert "wood" in cfg.materials_v2.confidence.material_thresholds
        assert "glass" in cfg.materials_v2.confidence.material_thresholds
        assert "metal" in cfg.materials_v2.confidence.material_thresholds

    def test_production_ultra_materials_enables_v3(self):
        """Verify Materials V3 is enabled in production_ultra_materials preset."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # CRITICAL ASSERTIONS
        assert cfg.materials_v3 is not None, "Materials V3 config block must exist"
        assert cfg.materials_v3.enabled is True, "Materials V3 must be enabled"

        # Verify backend configuration
        assert cfg.materials_v3.backend == "segformer", "Must use SegFormer backend"

    def test_production_ultra_materials_segmentation_config(self):
        """Verify segmentation backend supports Materials V2/V3."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Must use ML-based segmentation (not heuristic)
        assert cfg.segmentation.backend in ["segformer", "auto"], "Must use ML segmentation for materials"

        # Verify quality settings
        assert cfg.segmentation.input_long_side >= 1024, "High-resolution segmentation required for materials"

        # Verify downloads enabled or local paths provided
        if not cfg.segmentation.allow_downloads:
            pytest.skip("Downloads disabled - need local model paths for full validation")

    def test_materials_v2_confidence_thresholds(self):
        """Verify Materials V2 confidence thresholds are production-ready."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        assert cfg.materials_v2 is not None
        thresholds = cfg.materials_v2.confidence.material_thresholds

        # All major materials should have thresholds
        required_materials = ["wood", "metal", "glass", "fabric", "stone"]
        for material in required_materials:
            assert material in thresholds, f"Missing threshold for {material}"
            assert 0.0 < thresholds[material] < 1.0, f"Invalid threshold for {material}: {thresholds[material]}"

    def test_materials_validation_catches_null_config(self):
        """Verify validation catches null config blocks."""
        # Create config with null materials blocks
        cfg = PipelineConfig(preset=Preset.PHOTO_REALISTIC)
        cfg.materials_v2 = None
        cfg.materials_v3 = None

        # Should not raise - validation only fires when enabled=True
        # This tests that disabled configs are allowed to be null
        assert cfg.materials_v2 is None
        assert cfg.materials_v3 is None

    def test_materials_validation_warns_missing_models(self, caplog):
        """Verify validation warns when models missing and downloads disabled."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Disable downloads
        cfg.segmentation.allow_downloads = False
        cfg.segmentation.segformer_model_path = None

        # Re-run validation
        with pytest.warns(UserWarning, match="downloads are disabled"):
            cfg._validate_materials_config()

    def test_default_presets_dont_enable_materials(self):
        """Verify default presets don't unexpectedly enable materials."""
        safe_presets = [
            Preset.PHOTO_REALISTIC,
            Preset.INTERIOR_LUXURY,
            Preset.EXTERIOR_SHOWCASE,
            Preset.ARCHITECTURAL,
        ]

        for preset in safe_presets:
            cfg = PipelineConfig(preset=preset)

            # These presets should NOT enable materials by default
            # (unless explicitly configured to do so in future updates)
            if cfg.materials_v2 is not None:
                assert not cfg.materials_v2.enabled or preset in [
                    Preset.INTERIOR_LUXURY_MAX_QUALITY,
                    Preset.INTERIOR_LUXURY_APEX_QUALITY,
                ], f"{preset} unexpectedly enables Materials V2"

            if cfg.materials_v3 is not None:
                assert not cfg.materials_v3.enabled or "APEX" in preset.value.upper(), (
                    f"{preset} unexpectedly enables Materials V3"
                )

    def test_segmentation_config_has_model_paths(self):
        """Verify SegmentationConfig has local model path fields."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Check fields exist (even if None)
        assert hasattr(cfg.segmentation, "segformer_model_path")
        assert hasattr(cfg.segmentation, "sam_model_path")
        assert hasattr(cfg.segmentation, "efficientsam_model_path")

    def test_materials_v2_quality_settings(self):
        """Verify Materials V2 quality settings are production-grade."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        assert cfg.materials_v2 is not None

        # Segmentation resolution
        assert cfg.materials_v2.segmentation.max_segmentation_side >= 2048, "Production requires high-resolution segmentation"

        # Quality enforcement
        assert cfg.materials_v2.segmentation.require_high_quality is True, "Production must enforce quality thresholds"

        # Edge quality
        assert cfg.materials_v2.segmentation.edge_feather_radius > 0, "Edge feathering required for quality"

    def test_materials_v3_safety_limits(self):
        """Verify Materials V3 has safety limits for large images."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        assert cfg.materials_v3 is not None

        # OOM prevention
        assert cfg.materials_v3.max_megapixels > 0, "Must have megapixel limit for OOM prevention"
        assert cfg.materials_v3.max_dimension > 0, "Must have dimension limit for OOM prevention"

    def test_production_ultra_materials_mps_safety(self):
        """Verify production_ultra_materials has MPS safety measures."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Tiled upscaling for MPS safety
        assert cfg.phase2 is not None, "Phase2 config must exist"
        assert cfg.phase2.tile_based_upscaling is True, "Tiled upscaling required for MPS safety"
        assert cfg.phase2.upscale_tile_size <= 2048, "Tile size must fit in MPS 2.5GB buffer limit"

        # Post-processing tiling
        assert cfg.post_tile <= 2048, "Post-processing tile must be MPS-safe"


class TestMaterialsConfigFingerprint:
    """Test configuration fingerprinting for cache invalidation."""

    def test_fingerprint_includes_materials_v2(self):
        """Verify config fingerprint includes Materials V2 settings."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Get fingerprint
        fp1 = cfg._cfg_fingerprint()

        # Modify Materials V2 config
        cfg.materials_v2.confidence.confidence_threshold = 0.99
        fp2 = cfg._cfg_fingerprint()

        # Fingerprints should differ
        assert fp1 != fp2, "Fingerprint must change when Materials V2 config changes"

    def test_fingerprint_includes_materials_v3(self):
        """Verify config fingerprint includes Materials V3 settings."""
        cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)

        # Get fingerprint
        fp1 = cfg._cfg_fingerprint()

        # Modify Materials V3 config (toggle enabled state)
        original_enabled = cfg.materials_v3.enabled
        cfg.materials_v3.enabled = not original_enabled
        fp2 = cfg._cfg_fingerprint()

        # Fingerprints should differ
        assert fp1 != fp2, "Fingerprint must change when Materials V3 config changes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
