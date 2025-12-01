"""Atmospheric scattering and environmental modeling.

Implements physics-based atmospheric effects:
- Rayleigh scattering (molecular atmosphere)
- Mie scattering (aerosols, haze)
- Aerial perspective (depth-based atmospheric effects)
- Marine layer simulation
- Coastal atmosphere characteristics

For Montecito/Santa Barbara:
- Marine aerosol concentrations (ocean spray, salt)
- Humidity effects (typical 65% RH)
- Sundowner wind clarity (exceptional visibility)
- Seasonal atmospheric variations
"""

import logging
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AtmosphericParameters:
    """Atmospheric condition parameters.

    Attributes:
        turbidity: Atmospheric turbidity (1=very clear, 10=hazy)
        humidity: Relative humidity (0-1)
        aerosol_density: Aerosol concentration (0-1)
        marine_influence: Coastal/marine atmosphere influence (0-1)
        visibility: Visibility distance in km
        temperature: Air temperature in Celsius
        pressure: Atmospheric pressure in hPa
    """
    turbidity: float = 2.0
    humidity: float = 0.65  # 65% typical for coastal
    aerosol_density: float = 0.12  # Coastal scattering coefficient
    marine_influence: float = 0.6  # Coastal location
    visibility: float = 30.0  # km
    temperature: float = 20.0  # Celsius
    pressure: float = 1013.25  # hPa (sea level)


@dataclass
class MarineLayerParameters:
    """Marine layer characteristics.

    Attributes:
        present: Whether marine layer is present
        height: Marine layer top height in meters
        density: Marine layer density (0-1)
        thickness: Layer thickness in meters
    """
    present: bool = False
    height: float = 150.0  # meters (~500 feet typical)
    density: float = 0.5
    thickness: float = 100.0  # meters


class AtmosphericModel:
    """Physics-based atmospheric effects model.

    Provides atmospheric scattering, aerial perspective, and
    location-specific environmental effects.

    Example:
        >>> model = AtmosphericModel()
        >>> params = AtmosphericParameters(
        ...     turbidity=1.5,  # Very clear (Sundowner conditions)
        ...     humidity=0.45,
        ...     visibility=50.0
        ... )
        >>> depth_map = load_depth_map("image_depth.png")
        >>> atmospheric_effect = model.apply_aerial_perspective(
        ...     image, depth_map, params
        ... )
    """

    # Wavelength-dependent Rayleigh scattering coefficients
    RAYLEIGH_COEFFICIENTS = {
        'red': 0.058,      # 650nm
        'green': 0.135,    # 510nm
        'blue': 0.331      # 450nm
    }

    def __init__(self):
        """Initialize atmospheric model."""
        logger.info("AtmosphericModel initialized")

    def apply_aerial_perspective(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        params: AtmosphericParameters,
        max_distance: float = 1000.0
    ) -> np.ndarray:
        """Apply aerial perspective (atmospheric depth cues).

        Distant objects appear:
        - Lighter (scattered light adds to image)
        - Less saturated (color diluted by atmosphere)
        - Bluer/cooler (Rayleigh scattering favors blue)

        Args:
            image: RGB image (H, W, 3)
            depth_map: Normalized depth map (H, W) where 0=near, 1=far
            params: Atmospheric parameters
            max_distance: Maximum distance in meters for depth=1.0

        Returns:
            Image with aerial perspective applied
        """
        # Atmospheric color (sky color from scattering)
        atmo_color = self._get_atmospheric_color(params)

        # Calculate distance in meters
        distance = depth_map * max_distance

        # Calculate transmission (Beer-Lambert law)
        transmission = self._calculate_transmission(distance, params)

        # Apply aerial perspective
        # I_final = I_original * transmission + I_atmosphere * (1 - transmission)
        atmospheric = image.astype(np.float32) / 255.0

        # Add atmospheric scattering
        for i, color in enumerate(['red', 'green', 'blue']):
            channel_transmission = transmission ** self.RAYLEIGH_COEFFICIENTS[color]

            atmospheric[:, :, 2 - i] = (
                atmospheric[:, :, 2 - i] * channel_transmission +
                atmo_color[2 - i] * (1 - channel_transmission)
            )

        # Desaturate with distance
        desaturation = 1.0 - (1.0 - transmission) * 0.3
        hsv = self._rgb_to_hsv(atmospheric)
        hsv[:, :, 1] *= desaturation
        atmospheric = self._hsv_to_rgb(hsv)

        # Convert back to uint8
        atmospheric = (atmospheric * 255).clip(0, 255).astype(np.uint8)

        return atmospheric

    def _calculate_transmission(
        self,
        distance: np.ndarray,
        params: AtmosphericParameters
    ) -> np.ndarray:
        """Calculate atmospheric transmission using Beer-Lambert law.

        Args:
            distance: Distance in meters
            params: Atmospheric parameters

        Returns:
            Transmission coefficient (0-1)
        """
        # Extinction coefficient (higher = more haze)
        beta = params.turbidity / params.visibility * 1000  # Per km to per m

        # Add aerosol contribution
        beta += params.aerosol_density * 0.1

        # Beer-Lambert law: T = exp(-beta * distance)
        transmission = np.exp(-beta * distance / 1000.0)  # Convert to km

        return transmission

    def _get_atmospheric_color(
        self,
        params: AtmosphericParameters
    ) -> np.ndarray:
        """Get atmospheric scattering color.

        Args:
            params: Atmospheric parameters

        Returns:
            RGB color (3,) in range [0, 1]
        """
        # Base sky color (blue from Rayleigh scattering)
        base_color = np.array([0.4, 0.6, 0.9])

        # Add marine influence (slightly greenish-blue)
        if params.marine_influence > 0:
            marine_tint = np.array([0.3, 0.65, 0.85])
            base_color = (
                base_color * (1 - params.marine_influence * 0.3) +
                marine_tint * params.marine_influence * 0.3
            )

        # Haze makes atmosphere whiter
        haze_color = np.array([0.9, 0.9, 0.85])  # Slightly warm white
        haze_amount = (params.turbidity - 1.0) / 9.0  # Normalize turbidity to 0-1

        color = base_color * (1 - haze_amount * 0.5) + haze_color * haze_amount * 0.5

        return color

    def simulate_marine_layer(
        self,
        image: np.ndarray,
        height_map: np.ndarray,
        marine_params: MarineLayerParameters,
        camera_height: float = 2.0
    ) -> np.ndarray:
        """Simulate marine layer fog effects.

        Args:
            image: RGB image
            height_map: Height map (meters above ground)
            marine_params: Marine layer parameters
            camera_height: Camera height in meters

        Returns:
            Image with marine layer fog
        """
        if not marine_params.present:
            return image

        # Calculate which pixels are within marine layer
        pixel_height = height_map + camera_height

        # Marine layer density varies with height
        # Dense at bottom, fades toward top
        layer_bottom = 0
        layer_top = marine_params.height

        # Calculate fog density for each pixel
        fog_density = np.zeros_like(height_map)

        in_layer = (pixel_height >= layer_bottom) & (pixel_height <= layer_top)

        # Density decreases linearly from bottom to top
        fog_density[in_layer] = marine_params.density * (
            1.0 - (pixel_height[in_layer] - layer_bottom) / (layer_top - layer_bottom)
        )

        # Fog color (cool gray-blue)
        fog_color = np.array([0.85, 0.87, 0.90])

        # Apply fog
        fogged = image.astype(np.float32) / 255.0

        for i in range(3):
            fogged[:, :, i] = (
                fogged[:, :, i] * (1 - fog_density) +
                fog_color[i] * fog_density
            )

        fogged = (fogged * 255).clip(0, 255).astype(np.uint8)

        return fogged

    def calculate_sundowner_clarity(
        self,
        base_visibility: float,
        sundowner_active: bool = False
    ) -> float:
        """Calculate visibility during Sundowner wind conditions.

        Sundowner winds cause exceptional atmospheric clarity
        through warm, dry offshore flow.

        Args:
            base_visibility: Base visibility in km
            sundowner_active: Whether Sundowner conditions present

        Returns:
            Enhanced visibility in km
        """
        if sundowner_active:
            # Sundowner can increase visibility 30-50%
            return base_visibility * 1.4
        return base_visibility

    def get_seasonal_atmospheric_profile(
        self,
        season: str,
        location: str = "santa_barbara"
    ) -> AtmosphericParameters:
        """Get seasonal atmospheric parameters.

        Args:
            season: "spring", "summer", "fall", "winter"
            location: Location identifier

        Returns:
            Seasonal atmospheric parameters
        """
        # Santa Barbara/Montecito seasonal profiles
        profiles = {
            "spring": AtmosphericParameters(
                turbidity=2.5,
                humidity=0.68,
                aerosol_density=0.14,
                marine_influence=0.7,
                visibility=25.0
            ),
            "summer": AtmosphericParameters(
                turbidity=3.0,  # June gloom
                humidity=0.75,
                aerosol_density=0.15,
                marine_influence=0.8,
                visibility=20.0
            ),
            "fall": AtmosphericParameters(
                turbidity=1.5,  # Sundowner season - exceptional clarity
                humidity=0.50,
                aerosol_density=0.10,
                marine_influence=0.5,
                visibility=40.0
            ),
            "winter": AtmosphericParameters(
                turbidity=1.8,  # Rain-washed clarity
                humidity=0.62,
                aerosol_density=0.11,
                marine_influence=0.6,
                visibility=35.0
            )
        }

        return profiles.get(season, profiles["spring"])

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space.

        Args:
            rgb: RGB image (H, W, 3) in range [0, 1]

        Returns:
            HSV image (H, W, 3)
        """
        import cv2
        # OpenCV uses BGR and range [0, 180] for H
        bgr = rgb[:, :, ::-1]
        hsv = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        return hsv.astype(np.float32) / np.array([180, 255, 255])

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB color space.

        Args:
            hsv: HSV image (H, W, 3) in range [0, 1]

        Returns:
            RGB image (H, W, 3) in range [0, 1]
        """
        import cv2
        hsv_scaled = (hsv * np.array([180, 255, 255])).astype(np.uint8)
        bgr = cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2BGR)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        return rgb

    def __repr__(self) -> str:
        return "AtmosphericModel()"
