"""Tests for helpers inside :mod:`material_response`."""

import pytest

pytestmark = pytest.mark.skip(reason="material_response module not yet migrated to src package")

try:
    from scripts.utilities.material_response import MaterialResponseValidator
except ImportError:
    pass


def test_specular_preservation_returns_unity_when_reference_energy_is_zero() -> None:
    """Regression test ensuring zero-reference energies return a neutral ratio."""

    validator = MaterialResponseValidator()
    before = [[0.0, 0.0], [0.0, 0.0]]
    after = [[0.0, 1.0], [2.0, 3.0]]

    result = validator.measure_specular_preservation(before, after)

    assert result == pytest.approx(1.0)
