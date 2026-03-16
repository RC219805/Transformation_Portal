"""Semantic tests for PBR parameter effects.

These tests verify that PBR strength parameters produce measurably different outputs,
not just that they don't crash. This catches the "strength applied before normalization"
bug where parameters had no effect except at value 0.
"""

import numpy as np
import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.pbr import PBRConfig, generate_pbr_maps


class TestRoughnessStrengthEffect:
    """Test that roughness_strength parameter has measurable effect."""

    def test_roughness_strength_produces_different_outputs(self):
        """CRITICAL: roughness_strength must produce different outputs.

        This test catches the bug where strength is applied BEFORE normalization,
        making it effectively a no-op except at 0.
        """
        # Create a depth map with varied surface detail
        np.random.seed(42)
        depth = np.random.rand(256, 256).astype(np.float32)

        # Add some structured features (depth variation creates roughness)
        x, y = np.meshgrid(np.linspace(0, 10, 256), np.linspace(0, 10, 256))
        depth = depth + 0.3 * np.sin(x) * np.cos(y)
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]

        # Generate roughness maps with different strength values
        config_weak = PBRConfig(roughness_strength=0.5, roughness_blur_radius=3)
        config_default = PBRConfig(roughness_strength=1.0, roughness_blur_radius=3)
        config_strong = PBRConfig(roughness_strength=2.0, roughness_blur_radius=3)

        _, roughness_weak, _ = generate_pbr_maps(depth, config_weak)
        _, roughness_default, _ = generate_pbr_maps(depth, config_default)
        _, roughness_strong, _ = generate_pbr_maps(depth, config_strong)

        # Convert to float for comparison
        roughness_weak_f = roughness_weak.astype(np.float32) / 255.0
        roughness_default_f = roughness_default.astype(np.float32) / 255.0
        roughness_strong_f = roughness_strong.astype(np.float32) / 255.0

        # CRITICAL ASSERTIONS: Different strength values must produce different outputs
        # Use L2 distance to measure difference
        l2_weak_vs_default = np.linalg.norm(roughness_weak_f - roughness_default_f)
        l2_strong_vs_default = np.linalg.norm(roughness_strong_f - roughness_default_f)

        # Differences should be significant (threshold based on image size and value range)
        # For 256x256 image, meaningful difference should be > 1.0
        assert l2_weak_vs_default > 1.0, (
            f"roughness_strength=0.5 vs 1.0 produces too similar outputs (L2={l2_weak_vs_default:.2f}). "
            "Strength parameter may not be working correctly."
        )
        assert l2_strong_vs_default > 1.0, (
            f"roughness_strength=2.0 vs 1.0 produces too similar outputs (L2={l2_strong_vs_default:.2f}). "
            "Strength parameter may not be working correctly."
        )

        # Additionally, verify the expected monotonic behavior:
        # Higher strength should increase roughness values (make them more pronounced)
        mean_weak = roughness_weak_f.mean()
        mean_strong = roughness_strong_f.mean()

        # With power curve (strength applied as power after normalization),
        # we use power(x, 1/strength), so:
        # - strength=2.0 means power(x, 0.5) = sqrt(x), which spreads values UP (increases mean)
        # - strength=0.5 means power(x, 2.0) = x^2, which concentrates values DOWN (decreases mean)
        assert mean_strong > mean_weak, (
            f"Expected strength=2.0 (mean={mean_strong:.3f}) to have higher mean than "
            f"strength=0.5 (mean={mean_weak:.3f}) due to power curve behavior"
        )

    def test_roughness_strength_zero_produces_constant(self):
        """Special case: roughness_strength=0 should produce minimal/constant roughness."""
        # Create varied depth map
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        config_zero = PBRConfig(roughness_strength=0.0, roughness_blur_radius=3)
        _, roughness_zero, _ = generate_pbr_maps(depth, config_zero)

        roughness_f = roughness_zero.astype(np.float32) / 255.0

        # With strength=0, the detail is zeroed out before normalization,
        # so the normalized result should be all zeros
        # (normalization of a constant field returns all zeros)
        assert roughness_f.max() == 0.0, "roughness_strength=0 should produce all zeros"
        assert roughness_f.std() == 0.0, "roughness_strength=0 should have zero variance"

    def test_roughness_strength_values_in_expected_range(self):
        """Verify different strength values produce outputs in valid range."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        for strength in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
            config = PBRConfig(roughness_strength=strength, roughness_blur_radius=3)
            _, roughness, _ = generate_pbr_maps(depth, config)

            # All outputs must be valid uint8
            assert roughness.dtype == np.uint8
            assert roughness.min() >= 0
            assert roughness.max() <= 255

    def test_roughness_strength_rejects_negative_values(self):
        """Verify that negative roughness_strength values raise ValueError."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        with pytest.raises(ValueError, match="roughness_strength must be non-negative"):
            config = PBRConfig(roughness_strength=-0.5, roughness_blur_radius=3)
            generate_pbr_maps(depth, config)


class TestAOStrengthEffect:
    """Test that ao_strength parameter has measurable effect."""

    def test_ao_strength_produces_different_outputs(self):
        """CRITICAL: ao_strength must produce different outputs.

        This test catches the bug where strength is applied BEFORE normalization,
        making it effectively a no-op except at 0.
        """
        # Create a depth map with varied geometry (creates occlusion gradients)
        np.random.seed(42)
        depth = np.random.rand(256, 256).astype(np.float32)

        # Add geometric features that create strong gradients
        x, y = np.meshgrid(np.linspace(0, 8, 256), np.linspace(0, 8, 256))
        depth = depth + 0.4 * np.sin(x * 2) * np.cos(y * 2)
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        # Generate AO maps with different strength values
        config_weak = PBRConfig(ao_strength=0.5, ao_blur_radius=5, ao_bias=0.5)
        config_default = PBRConfig(ao_strength=1.0, ao_blur_radius=5, ao_bias=0.5)
        config_strong = PBRConfig(ao_strength=2.0, ao_blur_radius=5, ao_bias=0.5)

        _, _, ao_weak = generate_pbr_maps(depth, config_weak)
        _, _, ao_default = generate_pbr_maps(depth, config_default)
        _, _, ao_strong = generate_pbr_maps(depth, config_strong)

        # Convert to float for comparison
        ao_weak_f = ao_weak.astype(np.float32) / 255.0
        ao_default_f = ao_default.astype(np.float32) / 255.0
        ao_strong_f = ao_strong.astype(np.float32) / 255.0

        # CRITICAL ASSERTIONS: Different strength values must produce different outputs
        l2_weak_vs_default = np.linalg.norm(ao_weak_f - ao_default_f)
        l2_strong_vs_default = np.linalg.norm(ao_strong_f - ao_default_f)

        # Differences should be significant
        assert l2_weak_vs_default > 1.0, (
            f"ao_strength=0.5 vs 1.0 produces too similar outputs (L2={l2_weak_vs_default:.2f}). "
            "Strength parameter may not be working correctly."
        )
        assert l2_strong_vs_default > 1.0, (
            f"ao_strength=2.0 vs 1.0 produces too similar outputs (L2={l2_strong_vs_default:.2f}). "
            "Strength parameter may not be working correctly."
        )

        # Verify expected monotonic behavior:
        # Higher ao_strength should create darker AO (lower mean)
        # Because AO = 1 - occlusion, and higher strength increases occlusion
        mean_weak = ao_weak_f.mean()
        mean_strong = ao_strong_f.mean()

        assert mean_strong < mean_weak, (
            f"Expected ao_strength=2.0 (mean={mean_strong:.3f}) to have darker AO (lower mean) than "
            f"ao_strength=0.5 (mean={mean_weak:.3f})"
        )

    def test_ao_strength_zero_produces_constant(self):
        """Special case: ao_strength=0 should produce minimal occlusion (bright AO)."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        config_zero = PBRConfig(ao_strength=0.0, ao_blur_radius=5, ao_bias=0.5)
        _, _, ao_zero = generate_pbr_maps(depth, config_zero)

        ao_f = ao_zero.astype(np.float32) / 255.0

        # With strength=0, occlusion is zeroed after normalization,
        # so AO = 1 - 0*strength = 1 (with bias adjustment)
        # After bias: AO = (1.0) * (1 - 0.5) + 0.5 = 1.0
        assert ao_f.min() > 0.99, "ao_strength=0 should produce very bright AO (minimal occlusion)"
        assert ao_f.std() == 0.0, "ao_strength=0 should have zero variance"

    def test_ao_strength_clipping_behavior(self):
        """Verify ao_strength > 1.0 doesn't break due to clipping."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        # Very high strength should be clipped appropriately
        config_extreme = PBRConfig(ao_strength=10.0, ao_blur_radius=5, ao_bias=0.0)
        _, _, ao_extreme = generate_pbr_maps(depth, config_extreme)

        # Should still be valid uint8
        assert ao_extreme.dtype == np.uint8
        assert ao_extreme.min() >= 0
        assert ao_extreme.max() <= 255

        # With very high strength and zero bias, should see significant darkening
        ao_f = ao_extreme.astype(np.float32) / 255.0
        # Expect mean to be much lower than neutral (0.5)
        assert ao_f.mean() < 0.4, "Very high ao_strength should create dark AO"

    def test_ao_strength_values_in_expected_range(self):
        """Verify different strength values produce outputs in valid range."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        for strength in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0]:
            config = PBRConfig(ao_strength=strength, ao_blur_radius=5, ao_bias=0.5)
            _, _, ao = generate_pbr_maps(depth, config)

            # All outputs must be valid uint8
            assert ao.dtype == np.uint8
            assert ao.min() >= 0
            assert ao.max() <= 255

    def test_ao_strength_rejects_negative_values(self):
        """Verify that negative ao_strength values raise ValueError."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        with pytest.raises(ValueError, match="ao_strength must be non-negative"):
            config = PBRConfig(ao_strength=-0.5, ao_blur_radius=5, ao_bias=0.5)
            generate_pbr_maps(depth, config)


class TestPBRParameterIndependence:
    """Test that different PBR parameters are independent."""

    def test_normal_strength_does_not_affect_roughness(self):
        """Verify normal_strength doesn't affect roughness map."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        config_weak_normal = PBRConfig(normal_strength=0.5, roughness_strength=1.0)
        config_strong_normal = PBRConfig(normal_strength=2.0, roughness_strength=1.0)

        _, roughness_weak_normal, _ = generate_pbr_maps(depth, config_weak_normal)
        _, roughness_strong_normal, _ = generate_pbr_maps(depth, config_strong_normal)

        # Roughness maps should be identical
        np.testing.assert_array_equal(
            roughness_weak_normal, roughness_strong_normal, err_msg="normal_strength should not affect roughness map"
        )

    def test_normal_strength_does_not_affect_ao(self):
        """Verify normal_strength doesn't affect AO map (validates decoupling fix)."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        config_weak_normal = PBRConfig(normal_strength=0.5, ao_strength=1.0)
        config_strong_normal = PBRConfig(normal_strength=2.0, ao_strength=1.0)

        _, _, ao_weak_normal = generate_pbr_maps(depth, config_weak_normal)
        _, _, ao_strong_normal = generate_pbr_maps(depth, config_strong_normal)

        # AO maps should be identical (tests the UNSCALED gradients fix)
        np.testing.assert_array_equal(
            ao_weak_normal,
            ao_strong_normal,
            err_msg="normal_strength should not affect AO map (validates gradient decoupling)",
        )

    def test_roughness_strength_does_not_affect_ao(self):
        """Verify roughness_strength doesn't affect AO map."""
        np.random.seed(42)
        depth = np.random.rand(128, 128).astype(np.float32)

        config_weak_roughness = PBRConfig(roughness_strength=0.5, ao_strength=1.0)
        config_strong_roughness = PBRConfig(roughness_strength=2.0, ao_strength=1.0)

        _, _, ao_weak_roughness = generate_pbr_maps(depth, config_weak_roughness)
        _, _, ao_strong_roughness = generate_pbr_maps(depth, config_strong_roughness)

        # AO maps should be identical
        np.testing.assert_array_equal(
            ao_weak_roughness, ao_strong_roughness, err_msg="roughness_strength should not affect AO map"
        )
