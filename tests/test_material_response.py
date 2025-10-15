"""Tests for helpers inside :mod:`material_response`."""

import pytest

from material_response import MaterialResponseValidator, _clamp


def test_clamp_raises_when_minimum_exceeds_maximum() -> None:
    """Ensure invalid clamp bounds raise a :class:`ValueError`."""

    with pytest.raises(ValueError):
        _clamp(0.5, minimum=0.8, maximum=0.2)


def test_specular_preservation_returns_unity_when_reference_energy_is_zero() -> None:
    """Regression test ensuring zero-reference energies return a neutral ratio."""

    validator = MaterialResponseValidator()
    before = [[0.0, 0.0], [0.0, 0.0]]
    after = [[0.0, 1.0], [2.0, 3.0]]

    result = validator.measure_specular_preservation(before, after)

    assert result == pytest.approx(1.0)
