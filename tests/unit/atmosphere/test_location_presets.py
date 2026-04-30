"""Unit tests for atmosphere.location_presets.LocationPresets.

Covers list_locations(), get_atmospheric_parameters(), get_marine_layer_parameters(),
get_sky_parameters(), and get_golden_hour_parameters() — all four built-in
locations, all four seasons, no ML model downloads required.
"""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

pytestmark = [pytest.mark.unit]

_KNOWN_LOCATIONS = ["montecito", "santa_barbara", "hope_ranch", "riviera"]
_SEASONS = ["spring", "summer", "fall", "winter"]


# ---------------------------------------------------------------------------
# list_locations()
# ---------------------------------------------------------------------------


class TestListLocations:
    def test_returns_dict(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        assert isinstance(LocationPresets().list_locations(), dict)

    def test_montecito_present(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        assert "montecito" in LocationPresets().list_locations()

    def test_all_four_locations_present(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        locations = LocationPresets().list_locations()
        for name in _KNOWN_LOCATIONS:
            assert name in locations

    def test_location_profiles_have_name_attribute(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        for name, profile in LocationPresets().list_locations().items():
            assert hasattr(profile, "name"), f"{name} profile missing .name"

    def test_location_profiles_have_coordinates(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        for name, profile in LocationPresets().list_locations().items():
            assert hasattr(profile, "latitude"), f"{name} profile missing .latitude"
            assert hasattr(profile, "longitude"), f"{name} profile missing .longitude"


# ---------------------------------------------------------------------------
# get_atmospheric_parameters()
# ---------------------------------------------------------------------------


class TestGetAtmosphericParameters:
    def test_returns_atmospheric_parameters(self):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters
        from transformation_portal.atmosphere.location_presets import LocationPresets

        assert isinstance(LocationPresets().get_atmospheric_parameters(), AtmosphericParameters)

    @pytest.mark.parametrize("location", _KNOWN_LOCATIONS)
    def test_all_locations_return_valid_turbidity(self, location):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_atmospheric_parameters(location=location)
        assert 1.0 <= result.turbidity <= 10.0

    @pytest.mark.parametrize("season", _SEASONS)
    def test_all_seasons_return_atmospheric_parameters(self, season):
        from transformation_portal.atmosphere.atmospheric_model import AtmosphericParameters
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_atmospheric_parameters(season=season)
        assert isinstance(result, AtmosphericParameters)

    def test_visibility_positive(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_atmospheric_parameters(location="montecito")
        assert result.visibility > 0

    def test_humidity_in_valid_range(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_atmospheric_parameters(location="montecito")
        assert 0.0 <= result.humidity <= 1.0


# ---------------------------------------------------------------------------
# get_marine_layer_parameters()
# ---------------------------------------------------------------------------


class TestGetMarineLayerParameters:
    def test_returns_marine_layer_parameters(self):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters
        from transformation_portal.atmosphere.location_presets import LocationPresets

        assert isinstance(LocationPresets().get_marine_layer_parameters(), MarineLayerParameters)

    def test_has_present_attribute(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_marine_layer_parameters()
        assert hasattr(result, "present")

    def test_has_density_attribute(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_marine_layer_parameters()
        assert hasattr(result, "density")

    @pytest.mark.parametrize("season", _SEASONS)
    def test_all_seasons_return_parameters(self, season):
        from transformation_portal.atmosphere.atmospheric_model import MarineLayerParameters
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_marine_layer_parameters(season=season)
        assert isinstance(result, MarineLayerParameters)


# ---------------------------------------------------------------------------
# get_sky_parameters()
# ---------------------------------------------------------------------------


class TestGetSkyParameters:
    def test_returns_sky_parameters(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        result = LocationPresets().get_sky_parameters()
        assert isinstance(result, SkyParameters)

    def test_turbidity_positive(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_sky_parameters()
        assert result.turbidity > 0

    def test_sun_elevation_is_float(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_sky_parameters()
        assert isinstance(result.sun_elevation, float)

    def test_sun_azimuth_in_valid_range(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_sky_parameters()
        assert 0.0 <= result.sun_azimuth < 360.0


# ---------------------------------------------------------------------------
# get_golden_hour_parameters()
# ---------------------------------------------------------------------------


class TestGetGoldenHourParameters:
    def test_returns_sky_parameters(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        result = LocationPresets().get_golden_hour_parameters()
        assert isinstance(result, SkyParameters)

    def test_sunset_sun_elevation_near_horizon(self):
        """At golden hour the sun is at low elevation (< 30°)."""
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_golden_hour_parameters(time="sunset")
        assert result.sun_elevation < 30.0

    def test_sunrise_sun_elevation_near_horizon(self):
        from transformation_portal.atmosphere.location_presets import LocationPresets

        result = LocationPresets().get_golden_hour_parameters(time="sunrise")
        assert result.sun_elevation < 30.0

    @pytest.mark.parametrize("location", _KNOWN_LOCATIONS)
    def test_all_locations_return_valid_params(self, location):
        from transformation_portal.atmosphere.location_presets import LocationPresets
        from transformation_portal.atmosphere.skygan_generator import SkyParameters

        result = LocationPresets().get_golden_hour_parameters(location=location)
        assert isinstance(result, SkyParameters)
