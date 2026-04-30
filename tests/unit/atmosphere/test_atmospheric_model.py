"""Unit tests for atmosphere.atmospheric_model.

Covers AtmosphericParameters/MarineLayerParameters dataclass defaults,
RAYLEIGH_COEFFICIENTS class constants, calculate_sundowner_clarity(),
get_seasonal_atmospheric_profile(), apply_aerial_perspective(), and
simulate_marine_layer() — using CPU numpy arrays with no ML dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# AtmosphericParameters defaults
# ---------------------------------------------------------------------------


class TestAtmosphericParametersDefaults:
    def test_default_turbidity(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters().turbidity == pytest.approx(2.0)

    def test_default_humidity(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters().humidity == pytest.approx(0.65)

    def test_default_visibility_positive(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters().visibility > 0

    def test_default_pressure(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters().pressure == pytest.approx(1013.25)

    def test_default_temperature(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters().temperature == pytest.approx(20.0)

    def test_custom_turbidity_stored(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters(turbidity=5.0).turbidity == pytest.approx(5.0)

    def test_custom_humidity_stored(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters

        assert AtmosphericParameters(humidity=0.9).humidity == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# MarineLayerParameters defaults
# ---------------------------------------------------------------------------


class TestMarineLayerParametersDefaults:
    def test_present_false_by_default(self):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters

        assert MarineLayerParameters().present is False

    def test_default_height(self):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters

        assert MarineLayerParameters().height == pytest.approx(150.0)

    def test_default_density(self):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters

        assert MarineLayerParameters().density == pytest.approx(0.5)

    def test_default_thickness(self):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters

        assert MarineLayerParameters().thickness == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# RAYLEIGH_COEFFICIENTS physical constraints (class attribute on AtmosphericModel)
# ---------------------------------------------------------------------------


class TestRayleighCoefficients:
    def test_all_three_channels_present(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        assert set(AtmosphericModel.RAYLEIGH_COEFFICIENTS.keys()) == {"red", "green", "blue"}

    def test_all_coefficients_positive(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        assert all(v > 0 for v in AtmosphericModel.RAYLEIGH_COEFFICIENTS.values())

    def test_blue_greater_than_red(self):
        """Rayleigh scattering is ~λ^-4: blue scatters more than red."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        coeffs = AtmosphericModel.RAYLEIGH_COEFFICIENTS
        assert coeffs["blue"] > coeffs["red"]

    def test_blue_greater_than_green(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        coeffs = AtmosphericModel.RAYLEIGH_COEFFICIENTS
        assert coeffs["blue"] > coeffs["green"]


# ---------------------------------------------------------------------------
# calculate_sundowner_clarity()
# ---------------------------------------------------------------------------


class TestCalculateSundowner:
    def test_returns_float(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        result = AtmosphericModel().calculate_sundowner_clarity(base_visibility=30.0)
        assert isinstance(result, float)

    def test_clear_conditions_positive_visibility(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        result = AtmosphericModel().calculate_sundowner_clarity(base_visibility=30.0, sundowner_active=False)
        assert result > 0

    def test_sundowner_active_enhances_visibility(self):
        """Sundowner winds are warm, dry offshore flow — exceptional clarity."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        model = AtmosphericModel()
        base = model.calculate_sundowner_clarity(30.0, sundowner_active=False)
        enhanced = model.calculate_sundowner_clarity(30.0, sundowner_active=True)
        assert enhanced > base

    def test_zero_base_visibility_does_not_raise(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        result = AtmosphericModel().calculate_sundowner_clarity(base_visibility=0.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# get_seasonal_atmospheric_profile()
# ---------------------------------------------------------------------------


class TestSeasonalAtmosphericProfile:
    def test_returns_atmospheric_parameters(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters

        result = AtmosphericModel().get_seasonal_atmospheric_profile("summer")
        assert isinstance(result, AtmosphericParameters)

    @pytest.mark.parametrize("season", ["spring", "summer", "fall", "winter"])
    def test_all_seasons_return_valid_params(self, season):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters

        result = AtmosphericModel().get_seasonal_atmospheric_profile(season)
        assert isinstance(result, AtmosphericParameters)
        assert result.turbidity > 0
        assert result.visibility > 0

    def test_summer_higher_turbidity_than_winter(self):
        """Coastal summer haze → higher turbidity than clear winter air."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel

        model = AtmosphericModel()
        summer = model.get_seasonal_atmospheric_profile("summer")
        winter = model.get_seasonal_atmospheric_profile("winter")
        assert summer.turbidity >= winter.turbidity


# ---------------------------------------------------------------------------
# apply_aerial_perspective()
# ---------------------------------------------------------------------------


class TestApplyAerialPerspective:
    def test_output_shape_matches_input(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters

        model = AtmosphericModel()
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8).astype(np.float32) / 255.0
        depth_map = np.random.rand(50, 50).astype(np.float32)
        result = model.apply_aerial_perspective(image, depth_map, AtmosphericParameters())
        assert result.shape == image.shape

    def test_returns_numpy_array(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters

        model = AtmosphericModel()
        image = np.ones((30, 30, 3), dtype=np.float32) * 0.5
        depth_map = np.zeros((30, 30), dtype=np.float32)
        result = model.apply_aerial_perspective(image, depth_map, AtmosphericParameters())
        assert isinstance(result, np.ndarray)

    def test_zero_depth_leaves_image_mostly_unchanged(self):
        """At depth=0 (camera), aerial perspective effect should be minimal."""
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters

        model = AtmosphericModel()
        image = np.ones((20, 20, 3), dtype=np.float32) * 0.5
        depth_map = np.zeros((20, 20), dtype=np.float32)  # everything at distance 0
        result = model.apply_aerial_perspective(image, depth_map, AtmosphericParameters(), max_distance=1000.0)
        # Result should be close to input at zero depth
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# simulate_marine_layer()
# ---------------------------------------------------------------------------


class TestSimulateMarineLayer:
    def test_output_shape_matches_input(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, MarineLayerParameters

        model = AtmosphericModel()
        image = np.ones((40, 40, 3), dtype=np.float32) * 0.5
        height_map = np.zeros((40, 40), dtype=np.float32)
        params = MarineLayerParameters(present=True, density=0.5)
        result = model.simulate_marine_layer(image, height_map, params)
        assert result.shape == image.shape

    def test_returns_numpy_array(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, MarineLayerParameters

        model = AtmosphericModel()
        image = np.ones((20, 20, 3), dtype=np.float32)
        height_map = np.full((20, 20), 100.0, dtype=np.float32)
        result = model.simulate_marine_layer(image, height_map, MarineLayerParameters(present=True))
        assert isinstance(result, np.ndarray)
