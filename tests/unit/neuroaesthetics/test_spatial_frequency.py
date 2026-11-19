"""Regression tests for neuroaesthetics.spatial_frequency.SpatialFrequencyAnalyzer (M-28).

The band energies must describe image structure, not exposure: the DC term
(image mean) belongs to no band, and the band radius is measured in cycles per
pixel so that the bands are isotropic for non-square images.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from transformation_portal.neuroaesthetics.spatial_frequency import SpatialFrequencyAnalyzer  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture
def analyzer() -> SpatialFrequencyAnalyzer:
    return SpatialFrequencyAnalyzer()


def test_white_noise_is_dominated_by_high_frequencies(analyzer):
    noise = np.random.default_rng(0).integers(0, 256, (256, 256)).astype(np.uint8)
    result = analyzer.analyze(noise)
    assert result.hsf_energy > 0.7
    assert result.lsf_energy < 0.1
    assert result.visual_comfort_score < 0.5
    assert not any("Excellent" in rec for rec in result.recommendations)


def test_exposure_shift_does_not_change_band_energies(analyzer):
    noise = np.random.default_rng(1).integers(0, 64, (128, 128)).astype(np.uint8)
    dark = analyzer.analyze(noise)
    bright = analyzer.analyze((noise + 190).astype(np.uint8))
    assert dark.lsf_energy == pytest.approx(bright.lsf_energy, abs=1e-9)
    assert dark.msf_energy == pytest.approx(bright.msf_energy, abs=1e-9)
    assert dark.hsf_energy == pytest.approx(bright.hsf_energy, abs=1e-9)


def test_bands_are_isotropic_for_non_square_images(analyzer):
    h, w = 128, 1024
    yy, xx = np.mgrid[:h, :w]
    freq = 0.125  # cycles per pixel along either axis
    along_x = analyzer.analyze(np.sin(2 * np.pi * freq * xx))
    along_y = analyzer.analyze(np.sin(2 * np.pi * freq * yy))
    assert along_x.msf_energy == pytest.approx(1.0, abs=1e-6)
    assert along_y.msf_energy == pytest.approx(1.0, abs=1e-6)


def test_constant_image_has_no_band_energy(analyzer):
    result = analyzer.analyze(np.full((64, 64), 200, dtype=np.uint8))
    assert (result.lsf_energy, result.msf_energy, result.hsf_energy) == (0, 0, 0)


@pytest.mark.parametrize("shape", [(63, 97), (101, 37), (480, 640)])
@pytest.mark.parametrize("value", [1, 200, 255])
def test_flat_image_of_any_size_has_no_band_energy(analyzer, shape, value):
    # Without the DC term, a flat non-power-of-two image leaves only FFT
    # rounding residue; it must not be normalised into arbitrary band shares.
    result = analyzer.analyze(np.full(shape, value, dtype=np.uint8))
    assert (result.lsf_energy, result.msf_energy, result.hsf_energy) == (0, 0, 0)
    assert not any("Excellent" in rec for rec in result.recommendations)


def test_tiny_real_structure_is_still_measured(analyzer):
    image = np.full((480, 640), 200, dtype=np.uint8)
    image[0, 0] = 201
    result = analyzer.analyze(image)
    assert result.lsf_energy + result.msf_energy + result.hsf_energy == pytest.approx(1.0)
