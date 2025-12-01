#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Stable Diffusion dimension validation (Issue #3)."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import typer

    from transformation_portal.pipelines.lux_render_pipeline import validate_sd_dimensions
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    pytest.skip("Dependencies not available", allow_module_level=True)


class TestDimensionValidation:
    """Test dimension validation for SD 1.5 compatibility."""

    def test_valid_standard_dimensions(self):
        """Test that standard SD dimensions pass validation."""
        valid_dims = [
            (512, 512),
            (768, 512),
            (512, 768),
            (768, 768),
            (1024, 768),
            (1024, 1024),
        ]

        for width, height in valid_dims:
            result_w, result_h = validate_sd_dimensions(width, height, auto_correct=False)
            assert result_w == width, f"Width mismatch for {width}×{height}"
            assert result_h == height, f"Height mismatch for {width}×{height}"

    def test_invalid_dimensions_raise_error(self):
        """Test that invalid dimensions raise error when auto_correct=False."""
        invalid_dims = [
            (1024, 770),  # Not multiple of 64
            (800, 600),   # Not multiple of 64
            (1000, 1000),  # Not multiple of 64
        ]

        for width, height in invalid_dims:
            with pytest.raises(typer.BadParameter) as exc_info:
                validate_sd_dimensions(width, height, auto_correct=False)

            error_msg = str(exc_info.value)
            assert "multiples of 64" in error_msg.lower()
            assert str(width) in error_msg
            assert str(height) in error_msg

    def test_auto_correction(self, capsys):
        """Test automatic dimension correction."""
        test_cases = [
            ((1024, 770), (1024, 768)),  # Round down to 768
            ((800, 600), (768, 576)),    # Round down to nearest 64
            ((1000, 1000), (960, 960)),  # Round down both
        ]

        for (input_w, input_h), (expected_w, expected_h) in test_cases:
            result_w, result_h = validate_sd_dimensions(input_w, input_h, auto_correct=True)
            assert result_w == expected_w, f"Width correction failed for {input_w}×{input_h}"
            assert result_h == expected_h, f"Height correction failed for {input_w}×{input_h}"

            # Check that warning was printed
            captured = capsys.readouterr()
            assert "Corrected dimensions" in captured.err

    def test_minimum_dimensions(self):
        """Test that dimensions below minimum are corrected to 512."""
        small_dims = [
            (256, 256),
            (400, 400),
        ]

        for width, height in small_dims:
            result_w, result_h = validate_sd_dimensions(width, height, auto_correct=True)
            assert result_w >= 512, f"Width should be at least 512, got {result_w}"
            assert result_h >= 512, f"Height should be at least 512, got {result_h}"

    def test_large_dimensions_warning(self, capsys):
        """Test that very large dimensions trigger a warning."""
        large_dims = [
            (2048, 2048),
            (1536, 1536),
        ]

        for width, height in large_dims:
            validate_sd_dimensions(width, height, auto_correct=False)
            captured = capsys.readouterr()
            assert "VRAM" in captured.err or "memory" in captured.err.lower()

    def test_edge_cases(self):
        """Test edge cases for dimension validation."""
        # Minimum valid dimensions
        result_w, result_h = validate_sd_dimensions(512, 512, auto_correct=False)
        assert (result_w, result_h) == (512, 512)

        # Large but valid dimensions (both must be multiples of 64)
        result_w, result_h = validate_sd_dimensions(1920, 1024, auto_correct=False)
        assert (result_w, result_h) == (1920, 1024)

        # Perfect multiples of 64
        for dim in [64, 128, 192, 256, 320, 384, 448, 512]:
            result_w, result_h = validate_sd_dimensions(dim, dim, auto_correct=True)
            if dim >= 512:
                assert (result_w, result_h) == (dim, dim)
            else:
                # Should be corrected to minimum 512
                assert result_w >= 512 and result_h >= 512


class TestDimensionValidationIntegration:
    """Integration tests for dimension validation in the pipeline."""

    def test_dimension_validation_error_message(self):
        """Test that error messages are helpful."""
        try:
            validate_sd_dimensions(1024, 770, auto_correct=False)
            pytest.fail("Should have raised BadParameter")
        except typer.BadParameter as e:
            error_msg = str(e)
            # Check for helpful information in error message
            assert "multiples of 64" in error_msg.lower()
            assert "recommended" in error_msg.lower() or "512" in error_msg
            assert "1024" in error_msg and "770" in error_msg

    def test_realistic_workflow_dimensions(self):
        """Test dimensions from realistic workflows."""
        # Common aspect ratios that should work
        working_dims = [
            (768, 512),   # 3:2 landscape (common)
            (512, 768),   # 2:3 portrait (common)
            (1024, 576),  # 16:9 widescreen
            (1024, 768),  # 4:3 standard
        ]

        for width, height in working_dims:
            result_w, result_h = validate_sd_dimensions(width, height, auto_correct=False)
            assert result_w == width
            assert result_h == height

    def test_problematic_dimensions_from_bug_report(self):
        """Test specific dimensions that caused issues in bug report."""
        # From bug report: 1024×768 caused "tensor size (128) vs (88)" error
        # This was likely 1024×770 or similar, not a perfect multiple

        # These should work without issues
        assert validate_sd_dimensions(1024, 768) == (1024, 768)
        assert validate_sd_dimensions(768, 512) == (768, 512)

        # These should auto-correct with warning
        # 1024×770 was mentioned in bug report
        result = validate_sd_dimensions(1024, 770, auto_correct=True)
        assert result == (1024, 768)  # Should round down to 768

        # 1024×616 was also mentioned
        result = validate_sd_dimensions(1024, 616, auto_correct=True)
        assert result == (1024, 576)  # Should round down to 576


# Property-based tests using hypothesis (if available)
try:
    from hypothesis import given
    from hypothesis import strategies as st

    @given(
        width=st.integers(min_value=64, max_value=2048),
        height=st.integers(min_value=64, max_value=2048)
    )
    def test_dimension_validation_always_returns_valid(width, height):
        """Property test: validation always returns multiples of 64."""
        result_w, result_h = validate_sd_dimensions(width, height, auto_correct=True)

        # Result should always be multiples of 64
        assert result_w % 64 == 0, f"Width {result_w} is not a multiple of 64"
        assert result_h % 64 == 0, f"Height {result_h} is not a multiple of 64"

        # Result should be at least 512 (minimum SD dimension)
        assert result_w >= 512, f"Width {result_w} is below minimum"
        assert result_h >= 512, f"Height {result_h} is below minimum"

        # Result should not be larger than input (except when enforcing minimum)
        # When input < 512, result will be 512 (enforced minimum)
        if width >= 512:
            assert result_w <= width, f"Width increased from {width} to {result_w}"
        if height >= 512:
            assert result_h <= height, f"Height increased from {height} to {result_h}"

except ImportError:
    # hypothesis not available, skip property tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
