"""Color harmony analysis using CIELAB color space.

Research shows:
- Harmonious color combinations activate medial orbitofrontal cortex
- Disharmonious palettes trigger automatic amygdala responses
- Asymmetric processing: disharmony detection is universal, harmony appreciation varies

CIELAB color space provides perceptually uniform distance metric:
- L*: Lightness (0-100)
- a*: Green-Red axis
- b*: Blue-Yellow axis

Harmony principles:
- Analogous: Adjacent on color wheel (30° apart)
- Complementary: Opposite on color wheel (180° apart)
- Triadic: Evenly spaced (120° apart)
- Warm palettes trigger nostalgia (golden hour, heritage)
- Cool palettes suggest luxury (coastal, modern)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


class HarmonyType(Enum):
    """Color harmony types."""
    ANALOGOUS = "analogous"
    COMPLEMENTARY = "complementary"
    TRIADIC = "triadic"
    TETRADIC = "tetradic"
    MONOCHROMATIC = "monochromatic"
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


@dataclass
class ColorPalette:
    """Extracted color palette.

    Attributes:
        colors_rgb: Dominant colors in RGB (N, 3)
        colors_lab: Colors in CIELAB (N, 3)
        proportions: Proportion of each color (sums to 1)
        hues: Hue angles in degrees (0-360)
        saturations: Saturation values (0-100)
        lightnesses: Lightness values (0-100)
    """
    colors_rgb: np.ndarray
    colors_lab: np.ndarray
    proportions: np.ndarray
    hues: np.ndarray
    saturations: np.ndarray
    lightnesses: np.ndarray


@dataclass
class HarmonyAnalysis:
    """Color harmony analysis results.

    Attributes:
        harmony_score: Overall harmony (0-1, higher = more harmonious)
        harmony_type: Detected harmony type
        palette: Extracted color palette
        temperature: Warm/cool balance (-1=cool, 0=neutral, 1=warm)
        emotional_profile: Emotional associations
        recommendations: Improvement suggestions
        disharmony_factors: Identified disharmony issues
    """
    harmony_score: float
    harmony_type: HarmonyType
    palette: ColorPalette
    temperature: float
    emotional_profile: Dict[str, float]
    recommendations: List[str]
    disharmony_factors: List[str]


class ColorHarmonyAnalyzer:
    """Analyze color harmony using perceptual color science.

    Uses CIELAB color space for perceptually uniform analysis.
    Identifies harmony patterns and emotional resonance.

    Example:
        >>> analyzer = ColorHarmonyAnalyzer()
        >>> analysis = analyzer.analyze("luxury_interior.jpg")
        >>> print(f"Harmony score: {analysis.harmony_score:.2f}")
        >>> print(f"Harmony type: {analysis.harmony_type.value}")
        >>> print(f"Temperature: {'warm' if analysis.temperature > 0 else 'cool'}")
        >>> print(f"Emotional profile: {analysis.emotional_profile}")
    """

    # Harmony angle thresholds (degrees)
    ANALOGOUS_THRESHOLD = 30
    COMPLEMENTARY_THRESHOLD = 30  # ±30° from 180°
    TRIADIC_THRESHOLD = 30  # ±30° from 120°

    # Temperature thresholds (hue degrees)
    WARM_HUE_RANGES = [(0, 60), (300, 360)]  # Reds, oranges, yellows
    COOL_HUE_RANGES = [(180, 270)]  # Blues, cyans

    def __init__(
        self,
        num_colors: int = 5,
        min_proportion: float = 0.05
    ):
        """Initialize color harmony analyzer.

        Args:
            num_colors: Number of dominant colors to extract
            min_proportion: Minimum color proportion to consider
        """
        self.num_colors = num_colors
        self.min_proportion = min_proportion

        logger.info(f"ColorHarmonyAnalyzer initialized (n_colors={num_colors})")

    def analyze(
        self,
        image: Union[str, np.ndarray, Image.Image],
        sample_fraction: float = 0.1
    ) -> HarmonyAnalysis:
        """Analyze color harmony of image.

        Args:
            image: Input image
            sample_fraction: Fraction of pixels to sample (for speed)

        Returns:
            Complete harmony analysis
        """
        # Load image
        image_rgb = self._load_image(image)

        # Extract color palette
        palette = self._extract_palette(image_rgb, sample_fraction)

        # Detect harmony type
        harmony_type = self._detect_harmony_type(palette)

        # Calculate harmony score
        harmony_score = self._calculate_harmony_score(palette, harmony_type)

        # Calculate temperature
        temperature = self._calculate_temperature(palette)

        # Generate emotional profile
        emotional_profile = self._generate_emotional_profile(
            palette, harmony_type, temperature
        )

        # Identify disharmony factors
        disharmony_factors = self._identify_disharmony(palette)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            harmony_score, harmony_type, temperature, disharmony_factors
        )

        return HarmonyAnalysis(
            harmony_score=harmony_score,
            harmony_type=harmony_type,
            palette=palette,
            temperature=temperature,
            emotional_profile=emotional_profile,
            recommendations=recommendations,
            disharmony_factors=disharmony_factors
        )

    def _extract_palette(
        self,
        image: np.ndarray,
        sample_fraction: float
    ) -> ColorPalette:
        """Extract dominant color palette using K-means.

        Args:
            image: RGB image
            sample_fraction: Fraction of pixels to sample

        Returns:
            ColorPalette with dominant colors
        """
        # Reshape image to pixel array
        pixels = image.reshape(-1, 3)

        # Sample pixels for efficiency
        if sample_fraction < 1.0:
            n_samples = int(len(pixels) * sample_fraction)
            indices = np.random.choice(len(pixels), n_samples, replace=False)
            pixels = pixels[indices]

        # K-means clustering in RGB
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors_rgb = kmeans.cluster_centers_.astype(np.uint8)

        # Calculate proportions
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        proportions = counts / counts.sum()

        # Filter by minimum proportion
        mask = proportions >= self.min_proportion
        colors_rgb = colors_rgb[mask]
        proportions = proportions[mask]
        proportions /= proportions.sum()  # Renormalize

        # Convert to CIELAB
        colors_lab = self._rgb_to_lab(colors_rgb)

        # Calculate HSL for hue/saturation
        hues, saturations, lightnesses = self._lab_to_hsl(colors_lab)

        return ColorPalette(
            colors_rgb=colors_rgb,
            colors_lab=colors_lab,
            proportions=proportions,
            hues=hues,
            saturations=saturations,
            lightnesses=lightnesses
        )

    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to CIELAB.

        Args:
            rgb: RGB colors (N, 3) in range [0, 255]

        Returns:
            CIELAB colors (N, 3)
        """
        # OpenCV expects BGR
        bgr = rgb[:, ::-1]

        # Convert to float [0, 1]
        bgr_float = bgr.astype(np.float32) / 255.0

        # Add dimension for OpenCV
        bgr_img = bgr_float.reshape(1, -1, 3)

        # Convert BGR to LAB
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

        # Remove extra dimension
        lab = lab_img.reshape(-1, 3)

        return lab

    def _lab_to_hsl(
        self,
        lab: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert CIELAB to HSL components.

        Args:
            lab: CIELAB colors (N, 3)

        Returns:
            Tuple of (hues, saturations, lightnesses)
        """
        L = lab[:, 0]
        a = lab[:, 1]
        b = lab[:, 2]

        # Hue from a*, b*
        hues = np.arctan2(b, a) * 180 / np.pi
        hues = (hues + 360) % 360  # Normalize to [0, 360)

        # Chroma (saturation proxy)
        chroma = np.sqrt(a**2 + b**2)
        saturations = chroma  # Simplified saturation

        # Lightness directly from L*
        lightnesses = L

        return hues, saturations, lightnesses

    def _detect_harmony_type(self, palette: ColorPalette) -> HarmonyType:
        """Detect color harmony type from palette.

        Args:
            palette: Color palette

        Returns:
            Detected harmony type
        """
        hues = palette.hues

        if len(hues) < 2:
            return HarmonyType.MONOCHROMATIC

        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                # Handle wraparound
                diff = min(diff, 360 - diff)
                hue_diffs.append(diff)

        hue_diffs = np.array(hue_diffs)

        # Check for analogous (small differences)
        if np.all(hue_diffs < self.ANALOGOUS_THRESHOLD):
            return HarmonyType.ANALOGOUS

        # Check for complementary (~180°)
        complementary_diffs = np.abs(hue_diffs - 180)
        if np.any(complementary_diffs < self.COMPLEMENTARY_THRESHOLD):
            return HarmonyType.COMPLEMENTARY

        # Check for triadic (~120°)
        triadic_diffs = np.abs(hue_diffs - 120)
        if np.any(triadic_diffs < self.TRIADIC_THRESHOLD):
            return HarmonyType.TRIADIC

        # Check for warm/cool
        warm_count = sum(
            any(start <= h <= end for start, end in self.WARM_HUE_RANGES)
            for h in hues
        )
        cool_count = sum(
            any(start <= h <= end for start, end in self.COOL_HUE_RANGES)
            for h in hues
        )

        if warm_count > cool_count * 2:
            return HarmonyType.WARM
        elif cool_count > warm_count * 2:
            return HarmonyType.COOL

        return HarmonyType.NEUTRAL

    def _calculate_harmony_score(
        self,
        palette: ColorPalette,
        harmony_type: HarmonyType
    ) -> float:
        """Calculate overall harmony score.

        Args:
            palette: Color palette
            harmony_type: Detected harmony type

        Returns:
            Harmony score (0-1)
        """
        scores = []

        # Lightness variation score (avoid too similar or too extreme)
        lightness_std = np.std(palette.lightnesses)
        lightness_score = 1.0 - abs(lightness_std - 20) / 20  # Optimal ~20
        lightness_score = max(0, lightness_score)
        scores.append(lightness_score * 0.3)

        # Saturation balance score
        saturation_std = np.std(palette.saturations)
        saturation_score = 1.0 - abs(saturation_std - 15) / 15
        saturation_score = max(0, saturation_score)
        scores.append(saturation_score * 0.2)

        # Harmony type score
        harmony_type_scores = {
            HarmonyType.ANALOGOUS: 0.9,
            HarmonyType.COMPLEMENTARY: 0.85,
            HarmonyType.TRIADIC: 0.8,
            HarmonyType.WARM: 0.75,
            HarmonyType.COOL: 0.75,
            HarmonyType.MONOCHROMATIC: 0.7,
            HarmonyType.NEUTRAL: 0.6,
            HarmonyType.TETRADIC: 0.75,
        }
        scores.append(harmony_type_scores.get(harmony_type, 0.5) * 0.5)

        return sum(scores)

    def _calculate_temperature(self, palette: ColorPalette) -> float:
        """Calculate color temperature (-1=cool, 0=neutral, 1=warm).

        Args:
            palette: Color palette

        Returns:
            Temperature score
        """
        warm_weight = 0.0
        cool_weight = 0.0

        for hue, proportion in zip(palette.hues, palette.proportions):
            # Check if warm
            is_warm = any(
                start <= hue <= end
                for start, end in self.WARM_HUE_RANGES
            )

            # Check if cool
            is_cool = any(
                start <= hue <= end
                for start, end in self.COOL_HUE_RANGES
            )

            if is_warm:
                warm_weight += proportion
            elif is_cool:
                cool_weight += proportion

        # Normalize to [-1, 1]
        temperature = warm_weight - cool_weight

        return temperature

    def _generate_emotional_profile(
        self,
        palette: ColorPalette,
        harmony_type: HarmonyType,
        temperature: float
    ) -> Dict[str, float]:
        """Generate emotional association profile.

        Args:
            palette: Color palette
            harmony_type: Harmony type
            temperature: Temperature score

        Returns:
            Dictionary mapping emotions to scores (0-1)
        """
        profile = {
            "nostalgia": 0.0,
            "aspiration": 0.0,
            "luxury": 0.0,
            "comfort": 0.0,
            "energy": 0.0,
            "serenity": 0.0
        }

        # Nostalgia triggered by warm colors
        if temperature > 0.3:
            profile["nostalgia"] = temperature * 0.8

        # Aspiration from high lightness and harmony
        avg_lightness = np.mean(palette.lightnesses)
        if avg_lightness > 60:
            profile["aspiration"] = 0.7

        # Luxury from low saturation + high lightness (sophisticated)
        avg_saturation = np.mean(palette.saturations)
        if avg_saturation < 30 and avg_lightness > 50:
            profile["luxury"] = 0.8

        # Comfort from analogous warm colors
        if harmony_type == HarmonyType.ANALOGOUS and temperature > 0:
            profile["comfort"] = 0.7

        # Energy from high saturation and complementary
        if avg_saturation > 40:
            profile["energy"] = 0.6
            if harmony_type == HarmonyType.COMPLEMENTARY:
                profile["energy"] = 0.8

        # Serenity from cool analogous
        if harmony_type == HarmonyType.ANALOGOUS and temperature < 0:
            profile["serenity"] = 0.7
        elif temperature < -0.3:
            profile["serenity"] = abs(temperature) * 0.6

        return profile

    def _identify_disharmony(self, palette: ColorPalette) -> List[str]:
        """Identify disharmony factors.

        Args:
            palette: Color palette

        Returns:
            List of disharmony issues
        """
        issues = []

        # Check for extreme saturation variation
        saturation_std = np.std(palette.saturations)
        if saturation_std > 30:
            issues.append("High saturation variation may cause visual tension")

        # Check for extreme lightness contrast
        lightness_range = palette.lightnesses.max() - palette.lightnesses.min()
        if lightness_range > 70:
            issues.append("Extreme lightness contrast may be jarring")

        # Check for muddy colors (low saturation + mid lightness)
        muddy_count = sum(
            (s < 20 and 30 < l < 70)
            for s, l in zip(palette.saturations, palette.lightnesses)
        )
        if muddy_count > len(palette.colors_rgb) // 2:
            issues.append("Many muddy/dull colors detected")

        # Check for clashing saturated colors
        high_sat_colors = palette.saturations > 50
        if np.sum(high_sat_colors) >= 2:
            high_sat_hues = palette.hues[high_sat_colors]
            # Check for clashing hues (not harmonious)
            for i in range(len(high_sat_hues)):
                for j in range(i + 1, len(high_sat_hues)):
                    diff = abs(high_sat_hues[i] - high_sat_hues[j])
                    diff = min(diff, 360 - diff)
                    # Clashing if not complementary or triadic
                    if 50 < diff < 150 or 210 < diff < 310:
                        issues.append("Clashing saturated colors detected")
                        break

        return issues

    def _generate_recommendations(
        self,
        harmony_score: float,
        harmony_type: HarmonyType,
        temperature: float,
        disharmony_factors: List[str]
    ) -> List[str]:
        """Generate color harmony recommendations.

        Args:
            harmony_score: Harmony score
            harmony_type: Detected harmony type
            temperature: Temperature score
            disharmony_factors: Identified issues

        Returns:
            List of recommendations
        """
        recommendations = []

        if harmony_score >= 0.8:
            recommendations.append(
                "Excellent color harmony! Palette activates positive neural responses."
            )
        elif harmony_score >= 0.6:
            recommendations.append(
                "Good color harmony with room for optimization."
            )
        else:
            recommendations.append(
                "Color palette could benefit from harmonization to improve emotional resonance."
            )

        # Temperature-specific recommendations
        if abs(temperature) < 0.2:
            recommendations.append(
                "Neutral temperature - consider shifting toward warm (nostalgia) "
                "or cool (luxury) depending on target emotion."
            )

        # Harmony type recommendations
        if harmony_type == HarmonyType.NEUTRAL:
            recommendations.append(
                "No clear harmony pattern detected. Consider unifying palette "
                "using analogous or complementary scheme."
            )

        # Address specific disharmony factors
        for factor in disharmony_factors:
            recommendations.append(f"Resolve: {factor}")

        return recommendations

    def _load_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image as RGB numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        else:
            pil_img = Image.open(image).convert("RGB")
            return np.array(pil_img)

    def __repr__(self) -> str:
        return f"ColorHarmonyAnalyzer(n_colors={self.num_colors})"
