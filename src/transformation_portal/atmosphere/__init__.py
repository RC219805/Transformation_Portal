"""SkyGAN integration for location-specific atmospheric rendering.

SkyGAN combines physics-based clear-sky models with StyleGAN3-generated
atmospheric features (clouds, haze, horizons) learned from 39,000 HDR photographs.

Key Features:
- Physics-informed neural sky generation
- User control over sun position (azimuth, elevation)
- Near-real-time generation on modern GPUs
- Extended Dynamic Range (14EV) for accurate IBL
- Location-specific atmospheric parameters
- Marine layer and coastal atmosphere simulation

For Montecito/Santa Barbara (34.4°N):
- Marine layer characteristics
- Sundowner wind clarity effects
- Coastal aerosol modeling
- Seasonal sun path variations
"""

from transformation_portal.atmosphere.skygan_generator import SkyGANGenerator
from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel
from transformation_portal.atmosphere.location_presets import LocationPresets
from transformation_portal.atmosphere.sky_blending import SkyBlender

__all__ = [
    'SkyGANGenerator',
    'AtmosphericModel',
    'LocationPresets',
    'SkyBlender',
]
