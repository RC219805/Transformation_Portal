"""Unit tests for lux_depth_v2 preset_registry module.

Tests for preset governance, discovery, and completeness validation.
"""

import pytest

from lux_depth_v2.config import Preset
from lux_depth_v2.preset_registry import (
    PresetRegistry,
    PRESET_REGISTRY,
    get_registry,
    list_presets,
    get_preset,
    validate_preset,
)


class TestPresetRegistryCompleteness:
    """Tests for preset registry completeness and drift detection."""

    def test_all_enum_presets_have_registry_entries(self):
        """Verify all Preset enum values have corresponding registry entries."""
        enum_presets = set(p.value for p in Preset)
        registry_presets = set(PRESET_REGISTRY.keys())

        missing = enum_presets - registry_presets

        assert len(missing) == 0, (
            f"Preset registry is incomplete. Missing entries for: {sorted(missing)}. "
            f"All Preset enum values must have corresponding PRESET_REGISTRY entries."
        )

    def test_no_extra_registry_entries(self):
        """Verify no extra entries in registry that don't correspond to enum."""
        enum_presets = set(p.value for p in Preset)
        registry_presets = set(PRESET_REGISTRY.keys())

        extra = registry_presets - enum_presets

        assert len(extra) == 0, f"Preset registry has extra entries not in Preset enum: {sorted(extra)}"

    def test_registry_initialization_with_complete_data(self):
        """Test that PresetRegistry initializes successfully with complete data."""
        # This should not raise an error
        registry = PresetRegistry()
        assert registry is not None
        assert len(registry.presets) > 0

    def test_preset_count_matches_enum(self):
        """Verify registry has same count as Preset enum."""
        enum_count = len(list(Preset))
        registry_count = len(PRESET_REGISTRY)

        assert registry_count == enum_count, f"Preset count mismatch: enum has {enum_count}, registry has {registry_count}"


class TestPresetRegistryFiltering:
    """Tests for preset filtering functionality."""

    def test_list_all_presets(self):
        """Test listing all presets without filter."""
        registry = PresetRegistry()
        all_presets = registry.list_presets()

        assert len(all_presets) > 0
        assert all(hasattr(p, "name") for p in all_presets)
        assert all(hasattr(p, "stability") for p in all_presets)

    def test_list_stable_presets(self):
        """Test filtering for stable presets only."""
        registry = PresetRegistry()
        stable_presets = registry.list_presets(stability_filter="stable")

        assert len(stable_presets) > 0
        assert all(p.stability == "stable" for p in stable_presets)

    def test_list_canary_presets(self):
        """Test filtering for canary presets."""
        registry = PresetRegistry()
        canary_presets = registry.list_presets(stability_filter="canary")

        # Should have at least some canary presets
        assert all(p.stability == "canary" for p in canary_presets)

    def test_list_experimental_presets(self):
        """Test filtering for experimental presets."""
        registry = PresetRegistry()
        experimental_presets = registry.list_presets(stability_filter="experimental")

        # Should return only experimental presets (or empty list)
        assert all(p.stability == "experimental" for p in experimental_presets)

    def test_get_stable_presets_helper(self):
        """Test convenience method for getting stable presets."""
        registry = PresetRegistry()
        stable_presets = registry.get_stable_presets()

        assert all(p.stability == "stable" for p in stable_presets)

    def test_get_canary_presets_helper(self):
        """Test convenience method for getting canary presets."""
        registry = PresetRegistry()
        canary_presets = registry.get_canary_presets()

        assert all(p.stability == "canary" for p in canary_presets)


class TestPresetRetrieval:
    """Tests for retrieving individual preset metadata."""

    def test_get_preset_valid(self):
        """Test getting a valid preset."""
        registry = PresetRegistry()
        preset = registry.get_preset("photo_realistic")

        assert preset is not None
        assert preset.name == "photo_realistic"
        assert preset.display_name == "Photo Realistic"

    def test_get_preset_invalid(self):
        """Test getting a non-existent preset."""
        registry = PresetRegistry()
        preset = registry.get_preset("nonexistent_preset")

        assert preset is None

    def test_validate_preset_valid(self):
        """Test validating a valid preset name."""
        registry = PresetRegistry()
        assert registry.validate_preset("interior_luxury") is True

    def test_validate_preset_invalid(self):
        """Test validating an invalid preset name."""
        registry = PresetRegistry()
        assert registry.validate_preset("invalid_preset") is False


class TestPresetQualityTiers:
    """Tests for quality tier organization."""

    def test_get_by_quality_tier_standard(self):
        """Test getting presets by standard quality tier."""
        registry = PresetRegistry()
        standard_presets = registry.get_by_quality_tier("standard")

        assert len(standard_presets) > 0
        assert all(p.quality_tier == "standard" for p in standard_presets)

    def test_get_by_quality_tier_max(self):
        """Test getting presets by max quality tier."""
        registry = PresetRegistry()
        max_presets = registry.get_by_quality_tier("max")

        assert len(max_presets) > 0
        assert all(p.quality_tier == "max" for p in max_presets)

    def test_get_by_quality_tier_apex(self):
        """Test getting presets by apex quality tier."""
        registry = PresetRegistry()
        apex_presets = registry.get_by_quality_tier("apex")

        assert len(apex_presets) > 0
        assert all(p.quality_tier == "apex" for p in apex_presets)


class TestPresetMetadataFields:
    """Tests for preset metadata field requirements."""

    def test_all_presets_have_required_fields(self):
        """Verify all presets have required metadata fields."""
        required_fields = [
            "name",
            "display_name",
            "description",
            "intended_use",
            "quality_tier",
            "stability",
        ]

        for preset_name, metadata in PRESET_REGISTRY.items():
            for field in required_fields:
                assert hasattr(metadata, field), f"Preset '{preset_name}' missing required field '{field}'"
                assert getattr(metadata, field) is not None, f"Preset '{preset_name}' has None value for '{field}'"

    def test_all_presets_have_valid_quality_tiers(self):
        """Verify all presets use valid quality tier values."""
        valid_tiers = {"standard", "max", "apex"}

        for preset_name, metadata in PRESET_REGISTRY.items():
            assert metadata.quality_tier in valid_tiers, (
                f"Preset '{preset_name}' has invalid quality_tier: '{metadata.quality_tier}'. Must be one of {valid_tiers}"
            )

    def test_all_presets_have_valid_stability(self):
        """Verify all presets use valid stability values."""
        valid_stability = {"stable", "canary", "experimental"}

        for preset_name, metadata in PRESET_REGISTRY.items():
            assert metadata.stability in valid_stability, (
                f"Preset '{preset_name}' has invalid stability: '{metadata.stability}'. Must be one of {valid_stability}"
            )


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_registry_singleton(self):
        """Test get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        # Should be the same instance (singleton pattern)
        assert registry1 is registry2

    def test_list_presets_convenience(self):
        """Test list_presets convenience function."""
        presets = list_presets()
        assert len(presets) > 0

    def test_list_presets_with_filter(self):
        """Test list_presets with stability filter."""
        stable = list_presets(stability_filter="stable")
        assert all(p.stability == "stable" for p in stable)

    def test_get_preset_convenience(self):
        """Test get_preset convenience function."""
        preset = get_preset("photo_realistic")
        assert preset is not None
        assert preset.name == "photo_realistic"

    def test_validate_preset_convenience(self):
        """Test validate_preset convenience function."""
        assert validate_preset("interior_luxury") is True
        assert validate_preset("invalid") is False


class TestPresetFormatting:
    """Tests for preset formatting utilities."""

    def test_format_preset_list_basic(self):
        """Test basic preset list formatting."""
        registry = PresetRegistry()
        presets = registry.list_presets()[:3]  # Just test a few
        formatted = registry.format_preset_list(presets, show_details=False)

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_preset_list_with_details(self):
        """Test preset list formatting with details."""
        registry = PresetRegistry()
        presets = registry.list_presets()[:2]
        formatted = registry.format_preset_list(presets, show_details=True)

        assert isinstance(formatted, str)
        # Should include description when details are shown
        assert "description" in formatted.lower() or len(formatted) > 100

    def test_format_preset_detail(self):
        """Test detailed preset formatting."""
        registry = PresetRegistry()
        preset = registry.get_preset("photo_realistic")
        formatted = registry.format_preset_detail(preset)

        assert isinstance(formatted, str)
        assert "Photo Realistic" in formatted
        assert "Name:" in formatted
        assert "Status:" in formatted

    def test_format_empty_list(self):
        """Test formatting empty preset list."""
        registry = PresetRegistry()
        formatted = registry.format_preset_list([], show_details=False)

        assert formatted == "No presets found."


class TestSpecificPresets:
    """Tests for specific important presets."""

    def test_photo_realistic_preset_exists(self):
        """Verify photo_realistic preset exists and is stable."""
        registry = PresetRegistry()
        preset = registry.get_preset("photo_realistic")

        assert preset is not None
        assert preset.stability == "stable"

    def test_interior_luxury_preset_exists(self):
        """Verify interior_luxury preset exists and is stable."""
        registry = PresetRegistry()
        preset = registry.get_preset("interior_luxury")

        assert preset is not None
        assert preset.stability == "stable"

    def test_ci_baseline_preset_exists(self):
        """Verify ci_baseline preset exists for CI/CD."""
        registry = PresetRegistry()
        preset = registry.get_preset("ci_baseline")

        assert preset is not None
        assert preset.stability == "stable"
        assert preset.quality_tier == "standard"

    def test_validation_presets_marked_experimental(self):
        """Verify validation-only presets are marked experimental."""
        registry = PresetRegistry()

        validation_preset_names = [
            "interior_luxury_apex_quality_materials_v3_glass_validate",
            "interior_luxury_apex_quality_materials_v3_stone_validate",
        ]

        for name in validation_preset_names:
            preset = registry.get_preset(name)
            assert preset is not None, f"Validation preset '{name}' not found"
            assert preset.stability == "experimental", f"Validation preset '{name}' should be marked experimental"
