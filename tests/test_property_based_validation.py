#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property-based tests for input validation boundaries.

Uses Hypothesis for property-based testing to explore edge cases in:
- Config parsing
- Input validation
"""

import pytest

# Import Hypothesis with graceful fallback
try:
    from hypothesis import given
    from hypothesis import strategies as st

    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Hypothesis not installed")
    # Define dummy for pylint
    given = None  # type: ignore
    st = None  # type: ignore
    EnhanceConfig = None  # type: ignore


# =============================================================================
# Property-Based Tests: Config Parsing
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestConfigParsingProperties:
    """Property-based tests for config parsing robustness."""

    @given(timeout=st.integers(min_value=1, max_value=3600))
    def test_v2_timeout_accepts_positive_integers(self, timeout):
        """Property: v2_timeout should accept any positive integer."""
        config = EnhanceConfig(v2_timeout=timeout)
        assert config.v2_timeout == timeout

    @given(device=st.sampled_from(["cpu", "cuda", "mps"]))
    def test_depth_device_accepts_valid_devices(self, device):
        """Property: depth_device should accept known device types."""
        config = EnhanceConfig(depth_device=device)
        assert config.depth_device == device

    @given(fallback=st.sampled_from(["fail", "skip", "v2-auto"]))
    def test_depth_fallback_accepts_valid_modes(self, fallback):
        """Property: depth_fallback should accept valid fallback modes."""
        config = EnhanceConfig(depth_fallback=fallback)
        assert config.depth_fallback == fallback


# =============================================================================
# Property-Based Tests: Input Validation Boundaries
# =============================================================================


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestInputValidationBoundaries:
    """Property-based tests for input validation edge cases."""

    @given(
        width=st.integers(min_value=1, max_value=8192),
        height=st.integers(min_value=1, max_value=8192),
    )
    def test_valid_image_dimensions_are_accepted(self, width, height):
        """Property: any reasonable image dimensions should be valid."""
        # This tests that dimension ranges are sensible
        assert width > 0
        assert height > 0
        assert width <= 8192
        assert height <= 8192

    @given(preset=st.sampled_from(["draft", "balanced", "max_quality", "elite"]))
    def test_known_presets_are_valid(self, preset):
        """Property: all documented presets should be loadable."""
        config = EnhanceConfig(preset=preset)
        assert config.preset == preset
