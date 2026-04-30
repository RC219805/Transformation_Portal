"""Unit tests for atmosphere.sky_blending.

Covers MuLawToneMapper (pure computation), LightingProfile / CorrectionSuggestion
dataclasses, SkyBlender construction, and the smart_render pipeline smoke test
(procedural sky, no pretrained ML model required).
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# MuLawToneMapper
# ---------------------------------------------------------------------------


class TestMuLawToneMapper:
    def test_output_shape_matches_input(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper

        mapper = MuLawToneMapper()
        hdr = np.random.random((50, 50, 3)).astype(np.float32) * 10.0
        assert mapper.process(hdr).shape == hdr.shape

    def test_output_clipped_to_unit_range(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper

        mapper = MuLawToneMapper()
        hdr = np.ones((10, 10, 3), dtype=np.float32) * 1000.0
        result = mapper.process(hdr)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= -1e-6

    def test_zero_input_returns_near_zero(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper

        result = MuLawToneMapper().process(np.zeros((10, 10, 3), dtype=np.float32))
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_custom_mu_stored(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper

        assert MuLawToneMapper(mu=1000.0).mu == pytest.approx(1000.0)

    def test_returns_float_array(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper

        result = MuLawToneMapper().process(np.ones((5, 5, 3), dtype=np.float32))
        assert np.issubdtype(result.dtype, np.floating)


# ---------------------------------------------------------------------------
# LightingProfile dataclass
# ---------------------------------------------------------------------------


class TestLightingProfileDataclass:
    def test_fields_stored(self):
        from transformation_portal.atmosphere.sky_blending import LightingProfile

        lp = LightingProfile(azimuth=270.0, elevation=20.0, confidence=0.85)
        assert lp.azimuth == pytest.approx(270.0)
        assert lp.elevation == pytest.approx(20.0)
        assert lp.confidence == pytest.approx(0.85)

    def test_zero_confidence_valid(self):
        from transformation_portal.atmosphere.sky_blending import LightingProfile

        lp = LightingProfile(azimuth=0.0, elevation=0.0, confidence=0.0)
        assert lp.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CorrectionSuggestion dataclass
# ---------------------------------------------------------------------------


class TestCorrectionSuggestionDataclass:
    def test_fields_stored(self):
        from transformation_portal.atmosphere.sky_blending import CorrectionSuggestion
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        sp = SkyParameters(sun_azimuth=90.0, sun_elevation=30.0, turbidity=2.0)
        suggestion = CorrectionSuggestion(
            original_request_azimuth=180.0,
            measured_source_azimuth=90.0,
            confidence=0.75,
            suggested_params=sp,
            message="Azimuth corrected",
        )
        assert suggestion.original_request_azimuth == pytest.approx(180.0)
        assert suggestion.measured_source_azimuth == pytest.approx(90.0)
        assert suggestion.confidence == pytest.approx(0.75)
        assert suggestion.message == "Azimuth corrected"


# ---------------------------------------------------------------------------
# SkyBlender construction
# ---------------------------------------------------------------------------


class TestSkyBlenderConstruction:
    def test_construct_default(self):
        from transformation_portal.atmosphere.sky_blending import SkyBlender

        blender = SkyBlender(device="cpu")
        assert blender is not None

    def test_skygan_attribute_set(self):
        from transformation_portal.atmosphere.sky_blending import SkyBlender

        blender = SkyBlender(device="cpu")
        assert blender.skygan is not None

    def test_atmosphere_attribute_set(self):
        from transformation_portal.atmosphere.sky_blending import SkyBlender

        blender = SkyBlender(device="cpu")
        assert blender.atmosphere is not None

    def test_tone_mapper_attribute_set(self):
        from transformation_portal.atmosphere.sky_blending import MuLawToneMapper, SkyBlender

        blender = SkyBlender(device="cpu")
        assert isinstance(blender.tone_mapper, MuLawToneMapper)


# ---------------------------------------------------------------------------
# SkyBlender.smart_render() — smoke test (procedural, no pretrained model)
# ---------------------------------------------------------------------------


class TestSkyBlenderSmartRender:
    def test_output_shape_matches_source(self):
        """End-to-end smoke test: procedural sky generation, luma depth, cv2 blending."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters
        from transformation_portal.atmosphere.sky_blending import SkyBlender
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        source = np.full((60, 60, 3), 100, dtype=np.uint8)
        sky_params = SkyParameters(sun_azimuth=180.0, sun_elevation=30.0, turbidity=2.0)
        atmo_params = AtmosphericParameters()

        blender = SkyBlender(skygan=None, atmosphere=None, device="cpu")
        rendered, suggestion = blender.smart_render(
            source,
            sky_params,
            atmo_params,
            auto_correct=False,
            strict_physics=False,
            random_seed=42,
        )

        assert rendered.shape == source.shape

    def test_returns_correction_suggestion(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters
        from transformation_portal.atmosphere.sky_blending import CorrectionSuggestion, SkyBlender
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        source = np.full((40, 40, 3), 80, dtype=np.uint8)
        sky_params = SkyParameters(sun_azimuth=90.0, sun_elevation=45.0)
        atmo_params = AtmosphericParameters()

        blender = SkyBlender(device="cpu")
        _, suggestion = blender.smart_render(
            source,
            sky_params,
            atmo_params,
            auto_correct=False,
            strict_physics=False,
            random_seed=0,
        )

        assert isinstance(suggestion, CorrectionSuggestion)

    def test_output_is_uint8(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters
        from transformation_portal.atmosphere.sky_blending import SkyBlender
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        source = np.full((40, 40, 3), 150, dtype=np.uint8)
        blender = SkyBlender(device="cpu")
        rendered, _ = blender.smart_render(
            source,
            SkyParameters(),
            AtmosphericParameters(),
            auto_correct=False,
            random_seed=1,
        )
        assert rendered.dtype == np.uint8
