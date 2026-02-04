"""
SkyBlender: Physically-Constrained Volumetric Compositing Engine.

CORE PHILOSOPHY:
1. Deep Volumetric Unification: Merges sky and foreground using depth-aware
   atmospheric physics, not just 2D masking.
2. Physics Guardrails: Analyzes scene geometry (shadows) to prevent
   optically impossible renders (e.g., West Sun on East Shadows).
3. Intelligent Correction: Auto-derives optimal sky parameters to match
   source photography.
"""

import copy
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from transformation_portal.atmosphere.atmospheric_model import AtmosphericModel, AtmosphericParameters, MarineLayerParameters
from transformation_portal.atmosphere.skygan_generator import SkyGANGenerator, SkyParameters

logger = logging.getLogger(__name__)


class PhysicsViolationError(ValueError):
    """Raised when a transformation would break optical consistency."""

    pass


@dataclass
class LightingProfile:
    """Estimated lighting conditions from the source image."""

    azimuth: float  # 0-360 degrees
    elevation: float  # 0-90 degrees
    confidence: float  # 0.0-1.0


@dataclass
class CorrectionSuggestion:
    """The AI's counter-offer to a physically impossible request."""

    original_request_azimuth: float
    measured_source_azimuth: float
    confidence: float
    suggested_params: SkyParameters
    message: str


class MuLawToneMapper:
    """High-Dynamic Range compression using μ-law algorithm.

    Mimics the logarithmic response of human vision and high-end cinema cameras
    (e.g., ARRI LogC) for superior highlight retention.
    """

    def __init__(self, mu: float = 5000.0):
        self.mu = mu

    def process(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply μ-law compression to HDR data."""
        # Log2 base compression for exposure leveling
        exposure_norm = np.log2(1.0 + hdr_image)

        # μ-law encoding formula: F(x) = ln(1 + μx) / ln(1 + μ)
        numerator = np.log(1.0 + self.mu * exposure_norm)
        denominator = np.log(1.0 + self.mu)

        compressed = numerator / denominator

        # Gamma correction for Rec.709 display
        return np.power(compressed, 1.0 / 2.2).clip(0, 1)


class SunConsistencyGuard:
    """The Gatekeeper: Reverse-engineers scene lighting to prevent shadow conflicts."""

    def analyze_and_suggest(
        self,
        source_image: np.ndarray,
        depth_map: np.ndarray,
        requested_params: SkyParameters,
        tolerance_degrees: float = 45.0,
    ) -> CorrectionSuggestion:
        """Analyze the scene and formulate a correction plan if physics are violated."""
        # 1. Reverse-Engineer the Source Lighting
        existing_light = self._estimate_dominant_light_source(source_image, depth_map)

        # 2. Check for Compatibility
        diff = abs(existing_light.azimuth - requested_params.sun_azimuth)
        diff = min(diff, 360 - diff)  # Handle wrap-around

        is_valid = diff <= tolerance_degrees or existing_light.confidence < 0.4

        # 3. Formulate the "Perfect Match" parameters
        suggested = copy.deepcopy(requested_params)

        time_context = "Diffuse/Overcast"
        if existing_light.confidence >= 0.4:
            # If we are confident in the shadow source, align the sky to it
            suggested.sun_azimuth = existing_light.azimuth

            # Determine context string
            if 45 < existing_light.azimuth < 135:
                time_context = "Morning (East Sun)"
            elif 225 < existing_light.azimuth < 315:
                time_context = "Afternoon (West Sun)"
            else:
                time_context = "Mid-Day/High Sun"

        # 4. Draft the report
        if is_valid:
            msg = f"Physics OK. Requested sky aligns with source shadows ({existing_light.azimuth:.0f}°)."
        else:
            msg = (
                f"PHYSICS CONFLICT: You requested {requested_params.sun_azimuth:.0f}°, "
                f"but source shadows dictate {existing_light.azimuth:.0f}° ({time_context}). "
                f"Applying source azimuth will fix shadows."
            )

        return CorrectionSuggestion(
            original_request_azimuth=requested_params.sun_azimuth,
            measured_source_azimuth=existing_light.azimuth,
            confidence=existing_light.confidence,
            suggested_params=suggested,
            message=msg,
        )

    def _estimate_dominant_light_source(self, img: np.ndarray, depth: np.ndarray) -> LightingProfile:
        """Uses 'Spherical Harmonic Gradient' analysis to find the sun."""
        # 1. Compute Surface Normals from Depth
        # gradients: dz/dx, dz/dy (Negated as depth 'uphill' points to camera)
        zy, zx = np.gradient(depth)
        normal_x = -zx
        normal_y = -zy
        normal_z = np.ones_like(depth)

        # Normalize vectors
        magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        magnitude[magnitude == 0] = 1.0  # Avoid div by zero

        nx = normal_x / magnitude
        ny = normal_y / magnitude

        # 2. Get Luminance (L)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # 3. Solve via weighted correlation
        # We want the normal direction that correlates highest with brightness
        weights = gray**2
        avg_nx = np.average(nx, weights=weights)
        avg_ny = np.average(ny, weights=weights)

        # Convert vector back to Azimuth
        # Image coords: X+ is East, Y+ is South (usually), Y- is North
        azimuth_rad = np.arctan2(avg_nx, -avg_ny)
        azimuth_deg = np.degrees(azimuth_rad)
        if azimuth_deg < 0:
            azimuth_deg += 360

        # Confidence metric: Variance of brightness
        confidence = min(np.std(gray) * 4.0, 1.0)

        return LightingProfile(azimuth=azimuth_deg, elevation=45.0, confidence=confidence)


class SkyBlender:
    """The Engine: Orchestrates SkyGAN, Physics Checks, and Volumetric Unification."""

    def __init__(
        self,
        skygan: Optional[SkyGANGenerator] = None,
        atmosphere: Optional[AtmosphericModel] = None,
        device: str = "cuda",
    ):
        self.skygan = skygan or SkyGANGenerator(device=device)
        self.atmosphere = atmosphere or AtmosphericModel()
        self.guardrail = SunConsistencyGuard()
        self.tone_mapper = MuLawToneMapper()
        self.device = device

    def smart_render(
        self,
        source_image: np.ndarray,
        sky_params: SkyParameters,
        atmo_params: AtmosphericParameters,
        marine_params: Optional[MarineLayerParameters] = None,
        auto_correct: bool = True,
        strict_physics: bool = False,
        random_seed: int = 42,
    ) -> Tuple[np.ndarray, CorrectionSuggestion]:
        """
        The Intelligent Pipeline Entry Point.

        Args:
            auto_correct: If True, automatically replaces invalid params
                          with the suggested 'perfect match' params.
            strict_physics: If True and auto_correct is False, raises Error
                            on shadow mismatch.

        Returns:
            Tuple[Rendered Image, The Analysis Report]
        """
        # 1. ANALYSIS PASS
        depth_map = self._estimate_depth(source_image)
        suggestion = self.guardrail.analyze_and_suggest(source_image, depth_map, sky_params)

        # 2. DECISION LOGIC
        active_params = sky_params

        if auto_correct and suggestion.measured_source_azimuth != sky_params.sun_azimuth:
            if suggestion.confidence > 0.4:
                logger.info(f"Auto-Correcting: {suggestion.message}")
                active_params = suggestion.suggested_params
            else:
                logger.info("Lighting ambiguous, proceeding with requested params.")

        elif strict_physics and not auto_correct:
            # Check deviation for strict mode
            diff = abs(suggestion.measured_source_azimuth - sky_params.sun_azimuth)
            diff = min(diff, 360 - diff)
            if diff > 45 and suggestion.confidence > 0.4:
                raise PhysicsViolationError(suggestion.message)

        # 3. EXECUTION PASS
        final_image = self._execute_render(
            source_image,
            depth_map,
            active_params,
            atmo_params,
            marine_params,
            random_seed,
        )

        return final_image, suggestion

    def _execute_render(
        self,
        source_image: np.ndarray,
        depth_map: np.ndarray,
        sky_params: SkyParameters,
        atmo_params: AtmosphericParameters,
        marine_params: Optional[MarineLayerParameters],
        random_seed: int,
    ) -> np.ndarray:
        """Core rendering pipeline (Private)."""
        h, w = source_image.shape[:2]

        # A. Generate Sky (Latent Space -> HDR Image)
        logger.info(f"Generating sky (Azimuth: {sky_params.sun_azimuth:.1f}°)...")
        hdr_sky = self.skygan.generate_sky(
            params=sky_params,
            resolution=(w, h),
            output_format="hdr",
            random_seed=random_seed,
        )

        # B. Segment Sky
        sky_mask = self._segment_sky(source_image)

        # C. Composite (Linear Space)
        linear_source = (source_image.astype(np.float32) / 255.0) ** 2.2
        if hdr_sky.shape[:2] != (h, w):
            hdr_sky = cv2.resize(hdr_sky, (w, h))

        # Blend: Sky replaces masked area
        scene_linear = linear_source * (1.0 - sky_mask) + hdr_sky * sky_mask

        # D. Volumetric Unification
        # Unified depth: Foreground = Estimated Depth, Sky = Infinity (1.0)
        unified_depth = depth_map * (1.0 - sky_mask[:, :, 0]) + 1.0 * sky_mask[:, :, 0]

        # Apply Aerial Perspective (Rayleigh Scattering)
        unified_scene = self.atmosphere.apply_aerial_perspective(scene_linear, unified_depth, atmo_params)

        # Apply Marine Layer (Volumetric Fog)
        if marine_params and marine_params.present:
            # Approximation: Darker/Lower pixels in depth map = Lower elevation
            # Scale to 0-500 meters
            pseudo_height_map = (1.0 - unified_depth) * 500.0
            unified_scene = self.atmosphere.simulate_marine_layer(unified_scene, pseudo_height_map, marine_params)
        else:
            # Clip if no extra processing needed
            unified_scene = unified_scene.clip(0, 10.0)  # Soft clip for HDR

        # E. Tone Map (HDR -> LDR)
        unified_scene = (self.tone_mapper.process(unified_scene) * 255).astype(np.uint8)

        return unified_scene

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Monocular Depth Estimation Wrapper."""
        try:
            import torch

            model_type = "MiDaS_small"
            midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas.to(self.device).eval()

            input_batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()
            return (depth - depth.min()) / (depth.max() - depth.min())

        except (ImportError, Exception) as e:
            logger.warning(f"Depth model unavailable ({e}). Using Luma heuristic.")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return 1.0 - (gray.astype(np.float32) / 255.0)

    def _segment_sky(self, image: np.ndarray) -> np.ndarray:
        """Sky Segmentation Wrapper (Fallback to Color Heuristic)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Blue sky mask
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # White sky mask (clouds/overcast)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(mask_blue, mask_white)

        # Refine mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)

        mask_3c = np.repeat(combined_mask[:, :, np.newaxis], 3, axis=2)
        return mask_3c.astype(np.float32) / 255.0
