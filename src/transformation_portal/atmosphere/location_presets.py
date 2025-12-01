"""Location-specific atmospheric presets.

Predefined atmospheric parameters for luxury real estate markets:
- Montecito/Santa Barbara (34.4°N, -119.7°W)
- Other coastal California locations
- Seasonal and time-of-day variations

Each preset includes:
- Sun path calculations
- Atmospheric conditions
- Marine layer characteristics
- Seasonal variations
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from transformation_portal.atmosphere.skygan_generator import SkyParameters
from transformation_portal.atmosphere.atmospheric_model import (
    AtmosphericParameters,
    MarineLayerParameters
)


logger = logging.getLogger(__name__)


@dataclass
class LocationProfile:
    """Complete location atmospheric profile.

    Attributes:
        name: Location name
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        elevation: Elevation in meters above sea level
        timezone: IANA timezone identifier
        description: Location description
    """
    name: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    description: str


class LocationPresets:
    """Predefined location atmospheric profiles.

    Example:
        >>> presets = LocationPresets()
        >>> sky_params = presets.get_sky_parameters(
        ...     location="montecito",
        ...     time_of_day=17.5,  # 5:30 PM
        ...     season="fall",
        ...     condition="sundowner"
        ... )
        >>> atmo_params = presets.get_atmospheric_parameters(
        ...     location="montecito",
        ...     season="fall"
        ... )
    """

    # Location profiles
    LOCATIONS = {
        "montecito": LocationProfile(
            name="Montecito",
            latitude=34.4,
            longitude=-119.7,
            elevation=50.0,  # Average elevation
            timezone="America/Los_Angeles",
            description="Coastal luxury enclave with Mediterranean climate"
        ),
        "santa_barbara": LocationProfile(
            name="Santa Barbara",
            latitude=34.42,
            longitude=-119.70,
            elevation=10.0,
            timezone="America/Los_Angeles",
            description="Coastal city with marine influence"
        ),
        "hope_ranch": LocationProfile(
            name="Hope Ranch",
            latitude=34.43,
            longitude=-119.76,
            elevation=30.0,
            timezone="America/Los_Angeles",
            description="Coastal community west of Santa Barbara"
        ),
        "riviera": LocationProfile(
            name="Santa Barbara Riviera",
            latitude=34.44,
            longitude=-119.68,
            elevation=150.0,  # Hillside
            timezone="America/Los_Angeles",
            description="Hillside community with ocean views"
        )
    }

    def __init__(self):
        """Initialize location presets."""
        logger.info("LocationPresets initialized")

    def get_sky_parameters(
        self,
        location: str = "montecito",
        time_of_day: Optional[float] = None,
        date: Optional[str] = None,
        season: Optional[str] = None,
        condition: str = "clear"
    ) -> SkyParameters:
        """Get sky generation parameters for location.

        Args:
            location: Location identifier
            time_of_day: Time in hours (0-24), calculates sun position
            date: Date string "YYYY-MM-DD" for accurate sun path
            season: Season name (overrides date-based calculation)
            condition: Weather condition ("clear", "sundowner", "marine_layer", "hazy")

        Returns:
            SkyParameters for generation
        """
        profile = self.LOCATIONS.get(location, self.LOCATIONS["montecito"])

        # Calculate sun position
        if time_of_day is not None:
            sun_azimuth, sun_elevation = self._calculate_sun_position(
                profile.latitude,
                profile.longitude,
                time_of_day,
                date,
                season
            )
        else:
            # Default: mid-afternoon
            sun_azimuth = 220.0  # Southwest
            sun_elevation = 45.0

        # Get atmospheric conditions
        cloud_coverage, haze_density, turbidity = self._get_condition_parameters(
            condition,
            season or "fall"
        )

        return SkyParameters(
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            cloud_coverage=cloud_coverage,
            haze_density=haze_density,
            turbidity=turbidity,
            time_of_day=time_of_day,
            date=date,
            latitude=profile.latitude,
            longitude=profile.longitude
        )

    def get_atmospheric_parameters(
        self,
        location: str = "montecito",
        season: str = "fall",
        condition: str = "clear"
    ) -> AtmosphericParameters:
        """Get atmospheric parameters for location.

        Args:
            location: Location identifier
            season: Season ("spring", "summer", "fall", "winter")
            condition: Atmospheric condition

        Returns:
            AtmosphericParameters
        """
        _profile = self.LOCATIONS.get(location, self.LOCATIONS["montecito"])  # noqa: F841

        # Base parameters for coastal location
        base_params = AtmosphericParameters(
            turbidity=2.0,
            humidity=0.65,
            aerosol_density=0.12,
            marine_influence=0.6,
            visibility=30.0,
            temperature=20.0,
            pressure=1013.25
        )

        # Seasonal adjustments
        seasonal_adjustments = {
            "spring": {"humidity": 0.68, "visibility": 25.0, "turbidity": 2.5},
            "summer": {"humidity": 0.75, "visibility": 20.0, "turbidity": 3.0},
            "fall": {"humidity": 0.50, "visibility": 40.0, "turbidity": 1.5},
            "winter": {"humidity": 0.62, "visibility": 35.0, "turbidity": 1.8}
        }

        if season in seasonal_adjustments:
            for key, value in seasonal_adjustments[season].items():
                setattr(base_params, key, value)

        # Condition adjustments
        if condition == "sundowner":
            # Exceptional clarity
            base_params.turbidity = 1.3
            base_params.visibility = 50.0
            base_params.humidity = 0.40
            base_params.marine_influence = 0.3

        elif condition == "marine_layer":
            # Marine layer morning
            base_params.turbidity = 3.5
            base_params.visibility = 15.0
            base_params.humidity = 0.85
            base_params.marine_influence = 0.9

        elif condition == "hazy":
            base_params.turbidity = 4.0
            base_params.visibility = 12.0

        return base_params

    def get_marine_layer_parameters(
        self,
        season: str = "summer",
        time_of_day: float = 8.0
    ) -> MarineLayerParameters:
        """Get marine layer parameters.

        Marine layer is most common in summer (June gloom) and
        typically burns off by mid-morning.

        Args:
            season: Season
            time_of_day: Time in hours (0-24)

        Returns:
            MarineLayerParameters
        """
        # Marine layer probability by season
        season_probability = {
            "spring": 0.6,
            "summer": 0.7,  # Peak June gloom
            "fall": 0.3,
            "winter": 0.4
        }

        prob = season_probability.get(season, 0.4)

        # Marine layer typically present in early morning, burns off by 10-11 AM
        if time_of_day < 10.0:
            present = np.random.random() < prob
            # Denser in early morning
            density = max(0.3, 0.8 - (time_of_day - 6.0) / 4.0 * 0.5)
        elif time_of_day < 12.0:
            # Burning off
            present = np.random.random() < prob * 0.3
            density = 0.3
        else:
            # Usually clear by afternoon
            present = False
            density = 0.0

        return MarineLayerParameters(
            present=present,
            height=150.0,  # ~500 feet typical
            density=density if present else 0.0,
            thickness=100.0
        )

    def _calculate_sun_position(
        self,
        latitude: float,
        longitude: float,
        time_of_day: float,
        date: Optional[str] = None,
        season: Optional[str] = None
    ) -> Tuple[float, float]:
        """Calculate sun azimuth and elevation.

        Simplified sun position calculation. For production,
        use a library like pysolar or pvlib for accurate calculations.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            time_of_day: Time in hours (0-24)
            date: Date string
            season: Season (overrides date)

        Returns:
            Tuple of (azimuth, elevation) in degrees
        """
        # Simplified calculation based on time of day
        # Real implementation would use full solar position algorithm

        # Determine day of year for seasonal variation
        if season:
            season_day = {"spring": 80, "summer": 172, "fall": 266, "winter": 355}
            day_of_year = season_day.get(season, 266)  # Default to fall
        elif date:
            dt = datetime.fromisoformat(date)
            day_of_year = dt.timetuple().tm_yday
        else:
            day_of_year = 266  # Fall equinox

        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.deg2rad((360 / 365) * (day_of_year - 81)))

        # Hour angle
        hour_angle = (time_of_day - 12) * 15  # 15 degrees per hour

        # Solar elevation (simplified)
        lat_rad = np.deg2rad(latitude)
        decl_rad = np.deg2rad(declination)
        hour_rad = np.deg2rad(hour_angle)

        sin_elevation = (
            np.sin(lat_rad) * np.sin(decl_rad) +
            np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad)
        )
        elevation = np.rad2deg(np.arcsin(np.clip(sin_elevation, -1, 1)))

        # Solar azimuth (simplified)
        # Azimuth measured from North (0°=N, 90°=E, 180°=S, 270°=W)
        if time_of_day < 12:
            # Morning: East to South
            azimuth = 90 + (12 - time_of_day) / 12 * 90
        else:
            # Afternoon: South to West
            azimuth = 180 + (time_of_day - 12) / 12 * 90

        # Adjust for latitude (northern hemisphere)
        if latitude > 0:
            # Sun arcs through southern sky
            azimuth = min(270, max(90, azimuth))

        return float(azimuth), float(max(0, elevation))

    def _get_condition_parameters(
        self,
        condition: str,
        season: str
    ) -> Tuple[float, float, float]:
        """Get cloud coverage, haze density, turbidity for condition.

        Args:
            condition: Weather condition
            season: Season

        Returns:
            Tuple of (cloud_coverage, haze_density, turbidity)
        """
        conditions = {
            "clear": (0.1, 0.1, 2.0),
            "sundowner": (0.0, 0.05, 1.3),  # Exceptional clarity
            "marine_layer": (0.8, 0.4, 3.5),
            "partly_cloudy": (0.4, 0.15, 2.2),
            "hazy": (0.2, 0.5, 4.0),
            "overcast": (0.95, 0.3, 3.0)
        }

        # Summer typically hazier (June gloom)
        base = conditions.get(condition, conditions["clear"])

        if season == "summer":
            return (base[0], base[1] * 1.3, base[2] * 1.2)
        elif season == "fall":
            # Clearer in fall
            return (base[0] * 0.7, base[1] * 0.7, base[2] * 0.8)

        return base

    def get_golden_hour_parameters(
        self,
        location: str = "montecito",
        season: str = "fall",
        time: str = "sunset"  # "sunrise" or "sunset"
    ) -> SkyParameters:
        """Get parameters for golden hour photography.

        Args:
            location: Location
            season: Season
            time: "sunrise" or "sunset"

        Returns:
            SkyParameters optimized for golden hour
        """
        profile = self.LOCATIONS.get(location, self.LOCATIONS["montecito"])

        # Golden hour times vary by season
        golden_hour_times = {
            "spring": {"sunrise": 6.5, "sunset": 18.5},
            "summer": {"sunrise": 6.0, "sunset": 19.5},
            "fall": {"sunrise": 7.0, "sunset": 17.5},
            "winter": {"sunrise": 7.5, "sunset": 16.5}
        }

        time_of_day = golden_hour_times[season][time]

        # Golden hour has low sun elevation
        sun_elevation = 10.0 if time == "sunrise" else 8.0
        sun_azimuth = 90.0 if time == "sunrise" else 270.0  # East/West

        return SkyParameters(
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            cloud_coverage=0.2,  # Light clouds enhance golden hour
            haze_density=0.15,  # Slight haze for warm glow
            turbidity=2.0,
            time_of_day=time_of_day,
            latitude=profile.latitude,
            longitude=profile.longitude
        )

    def list_locations(self) -> Dict[str, LocationProfile]:
        """Get all available location profiles.

        Returns:
            Dictionary of location profiles
        """
        return self.LOCATIONS.copy()

    def __repr__(self) -> str:
        return f"LocationPresets(locations={list(self.LOCATIONS.keys())})"
