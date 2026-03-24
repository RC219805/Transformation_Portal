"""Tests for helpers inside :mod:`transformation_portal.processors.material_response`."""

import pytest

from transformation_portal.processors.material_response.core import MaterialResponseValidator

pytestmark = pytest.mark.unit


class TestMaterialResponseValidator:
    """Tests for MaterialResponseValidator class."""

    def test_specular_preservation_returns_unity_when_reference_energy_is_zero(self) -> None:
        """Regression test ensuring zero-reference energies return a neutral ratio."""
        validator = MaterialResponseValidator()
        before = [[0.0, 0.0], [0.0, 0.0]]
        after = [[0.0, 1.0], [2.0, 3.0]]

        result = validator.measure_specular_preservation(before, after)

        assert result == pytest.approx(1.0)

    def test_specular_preservation_returns_ratio_for_nonzero_energy(self) -> None:
        """Test that specular preservation calculates energy ratio correctly."""
        validator = MaterialResponseValidator()
        # Create simple 4x4 arrays with some high-frequency content
        before = [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
        ]
        after = [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 1.0, 2.0, 1.0],
        ]

        result = validator.measure_specular_preservation(before, after)

        # Identical inputs should give ratio of 1.0
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_specular_preservation_with_different_arrays(self) -> None:
        """Test that specular preservation calculates correct ratio for different inputs."""
        validator = MaterialResponseValidator()
        # Before: alternating pattern (high frequency)
        before = [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
        # After: scaled version (doubled high frequency energy)
        after = [
            [0.0, 2.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 0.0],
        ]

        result = validator.measure_specular_preservation(before, after)

        # After has 2x the amplitude, so 4x the energy
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_specular_preservation_rejects_mismatched_shapes(self) -> None:
        """Test that specular preservation raises ValueError for mismatched shapes."""
        validator = MaterialResponseValidator()
        before = [[1.0, 2.0], [3.0, 4.0]]
        after = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        with pytest.raises(ValueError, match="same shape"):
            validator.measure_specular_preservation(before, after)
