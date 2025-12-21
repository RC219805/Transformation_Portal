"""Regression tests for preset configuration defaults.

Ensures that preset configurations remain stable and don't silently
change critical defaults (e.g., edge refinement opt-in status).
"""
import pytest
from lux_depth_v2.config import PipelineConfig, Preset


class TestPresetRegressions:
    """Ensure preset defaults remain stable across refactors."""
    
    def test_edge_refinement_opt_in_default(self):
        """REGRESSION: Edge refinement must be opt-in by default.
        
        Critical safety check - prevents accidental enablement that could:
        - Increase processing time unexpectedly
        - Introduce visual artifacts before validation complete
        - Break existing client workflows
        """
        for preset in Preset:
            config = PipelineConfig(preset=preset)
            assert config.enable_edge_refinement is False, (
                f"Preset {preset.value} unexpectedly enabled edge refinement by default. "
                "Edge refinement must be opt-in via --edge-refinement CLI flag."
            )
    
    def test_refinement_preset_default(self):
        """REGRESSION: Default refinement preset must be 'balanced'.
        
        When edge refinement IS enabled, the default preset should be
        the balanced middle-ground option, not aggressive or subtle.
        """
        config = PipelineConfig()
        assert config.refinement_preset == "balanced", (
            f"Default refinement preset changed to '{config.refinement_preset}'. "
            "Expected 'balanced' as safe middle-ground default."
        )
    
    @pytest.mark.parametrize("preset", list(Preset))
    def test_preset_determinism(self, preset):
        """REGRESSION: Preset configs must be deterministic.
        
        Same preset should produce identical configuration every time.
        Non-determinism breaks reproducibility and client expectations.
        """
        config1 = PipelineConfig(preset=preset)
        config2 = PipelineConfig(preset=preset)
        
        # Critical fields must match exactly
        assert config1.material_strength == config2.material_strength, (
            f"Preset {preset.value} is non-deterministic (material_strength)"
        )
        assert config1.enable_edge_refinement == config2.enable_edge_refinement, (
            f"Preset {preset.value} is non-deterministic (enable_edge_refinement)"
        )
        assert config1.detail_strength == config2.detail_strength, (
            f"Preset {preset.value} is non-deterministic (detail_strength)"
        )
    
    def test_no_preset_enables_edge_by_default(self):
        """CRITICAL: No preset should enable edge refinement without explicit flag.
        
        This is the primary safety gate. Even if a preset is marked as
        "high quality" or "apex", edge refinement should remain opt-in
        until validation completes (Week 2-3).
        """
        enabled_presets = []
        for preset in Preset:
            config = PipelineConfig(preset=preset)
            if config.enable_edge_refinement:
                enabled_presets.append(preset.value)
        
        assert len(enabled_presets) == 0, (
            f"Presets {enabled_presets} enable edge refinement by default. "
            "This violates feature freeze policy and validation gate. "
            "Edge refinement must be CLI opt-in only until Week 2-3 validation completes."
        )


class TestFeatureFreezeCompliance:
    """Ensure feature freeze constraints are maintained."""
    
    def test_no_new_experimental_presets_during_freeze(self):
        """FREEZE: No new experimental presets during freeze period.
        
        Freeze period: Dec 20, 2025 - Jan 10, 2026
        Only bug fixes and validation allowed.
        """
        # Known presets as of Dec 20, 2025
        expected_presets = {
            "photo_realistic",
            "interior_luxury",
            "interior_luxury_max_quality",
            "interior_luxury_apex_quality",
            "interior_luxury_apex_quality_efficientsam",
            "interior_luxury_apex_quality_materials_v3_glass",
            "interior_luxury_apex_quality_materials_v3_glass_validate",
            "interior_luxury_apex_quality_materials_v3_stone",
            "interior_luxury_apex_quality_materials_v3_stone_validate",
            "exterior_showcase",
            "exterior_pool_apex_quality",
            "exterior_pool_apex_quality_efficientsam",
            "architectural",
            "archival_quality",
        }
        
        actual_presets = {p.value for p in Preset}
        
        assert actual_presets == expected_presets, (
            f"Preset inventory changed during feature freeze. "
            f"Added: {actual_presets - expected_presets}, "
            f"Removed: {expected_presets - actual_presets}. "
            "New presets are blocked until Jan 10, 2026."
        )
