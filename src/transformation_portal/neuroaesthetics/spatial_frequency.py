"""Spatial frequency analysis for visual comfort.

Research findings:
- Human visual system most sensitive to 2-6 cycles per degree
- Low spatial frequencies (LSF) convey coarse information via magnocellular pathways
- High spatial frequencies (HSF) provide detail via parvocellular pathways
- Coarse-to-fine sequence: LSF processed before HSF
- Excessive HSF content causes visual discomfort and fatigue
- Balanced spatial frequency distribution reduces visual stress

For architectural photography:
- Overall composition (LSF) forms first impressions
- Textural detail (HSF) provides richness
- Balance prevents visual exhaustion

This module analyzes and optimizes spatial frequency content.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import fftpack


logger = logging.getLogger(__name__)


@dataclass
class SpatialFrequencyAnalysis:
    """Spatial frequency analysis results.

    Attributes:
        lsf_energy: Low spatial frequency energy
        msf_energy: Mid spatial frequency energy
        hsf_energy: High spatial frequency energy
        balance_score: Balance score (0-1, higher = better balanced)
        dominant_frequencies: Dominant frequency bands
        visual_comfort_score: Predicted visual comfort (0-1)
        recommendations: Optimization suggestions
    """
    lsf_energy: float
    msf_energy: float
    hsf_energy: float
    balance_score: float
    dominant_frequencies: List[str]
    visual_comfort_score: float
    recommendations: List[str]


class SpatialFrequencyAnalyzer:
    """Analyze spatial frequency content for visual comfort.

    Uses Fourier transform to decompose image into frequency bands.
    Evaluates balance and predicts visual comfort.

    Example:
        >>> analyzer = SpatialFrequencyAnalyzer()
        >>> analysis = analyzer.analyze("architectural_photo.jpg")
        >>> print(f"Visual comfort: {analysis.visual_comfort_score:.2f}")
        >>> print(f"Balance: {analysis.balance_score:.2f}")
        >>> if analysis.visual_comfort_score < 0.7:
        ...     print("Recommendations:", analysis.recommendations)
    """

    # Frequency band definitions (cycles per degree approximations)
    # Assuming ~60 pixels per degree at typical viewing distance
    LSF_THRESHOLD = 0.1  # Low: 0-0.1 (large structures)
    MSF_THRESHOLD = 0.3  # Mid: 0.1-0.3 (medium details)
    # High: >0.3 (fine details)

    # Optimal energy distribution
    OPTIMAL_LSF = 0.50  # 50% low frequency
    OPTIMAL_MSF = 0.35  # 35% mid frequency
    OPTIMAL_HSF = 0.15  # 15% high frequency

    def __init__(self):
        """Initialize spatial frequency analyzer."""
        logger.info("SpatialFrequencyAnalyzer initialized")

    def analyze(
        self,
        image: Union[str, np.ndarray, Image.Image],
        viewing_distance_factor: float = 1.0
    ) -> SpatialFrequencyAnalysis:
        """Analyze spatial frequency distribution.

        Args:
            image: Input image
            viewing_distance_factor: Viewing distance adjustment (1.0 = standard)

        Returns:
            Spatial frequency analysis
        """
        # Load and convert to grayscale
        image_gray = self._load_image_gray(image)

        # Compute frequency spectrum
        spectrum = self._compute_spectrum(image_gray)

        # Calculate frequency band energies
        lsf_energy, msf_energy, hsf_energy = self._calculate_band_energies(
            spectrum,
            viewing_distance_factor
        )

        # Normalize energies
        total_energy = lsf_energy + msf_energy + hsf_energy
        lsf_norm = lsf_energy / total_energy if total_energy > 0 else 0
        msf_norm = msf_energy / total_energy if total_energy > 0 else 0
        hsf_norm = hsf_energy / total_energy if total_energy > 0 else 0

        # Calculate balance score
        balance_score = self._calculate_balance_score(
            lsf_norm, msf_norm, hsf_norm
        )

        # Identify dominant frequencies
        dominant_frequencies = self._identify_dominant_frequencies(
            lsf_norm, msf_norm, hsf_norm
        )

        # Calculate visual comfort score
        visual_comfort_score = self._calculate_comfort_score(
            lsf_norm, msf_norm, hsf_norm, balance_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            lsf_norm, msf_norm, hsf_norm, visual_comfort_score
        )

        return SpatialFrequencyAnalysis(
            lsf_energy=lsf_norm,
            msf_energy=msf_norm,
            hsf_energy=hsf_norm,
            balance_score=balance_score,
            dominant_frequencies=dominant_frequencies,
            visual_comfort_score=visual_comfort_score,
            recommendations=recommendations
        )

    def _compute_spectrum(self, image: np.ndarray) -> np.ndarray:
        """Compute 2D Fourier spectrum.

        Args:
            image: Grayscale image

        Returns:
            Power spectrum
        """
        # Apply 2D FFT
        fft = fftpack.fft2(image)
        fft_shifted = fftpack.fftshift(fft)

        # Compute power spectrum
        spectrum = np.abs(fft_shifted) ** 2

        return spectrum

    def _calculate_band_energies(
        self,
        spectrum: np.ndarray,
        viewing_distance_factor: float
    ) -> Tuple[float, float, float]:
        """Calculate energy in each frequency band.

        Args:
            spectrum: Power spectrum
            viewing_distance_factor: Viewing distance adjustment

        Returns:
            Tuple of (LSF, MSF, HSF) energies
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # Create frequency coordinate grids
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Normalize radius to [0, 1]
        max_radius = np.sqrt(center_x**2 + center_y**2)
        norm_radius = radius / max_radius

        # Adjust thresholds by viewing distance
        lsf_thresh = self.LSF_THRESHOLD * viewing_distance_factor
        msf_thresh = self.MSF_THRESHOLD * viewing_distance_factor

        # Create masks for each band
        lsf_mask = norm_radius <= lsf_thresh
        msf_mask = (norm_radius > lsf_thresh) & (norm_radius <= msf_thresh)
        hsf_mask = norm_radius > msf_thresh

        # Calculate energies
        lsf_energy = np.sum(spectrum[lsf_mask])
        msf_energy = np.sum(spectrum[msf_mask])
        hsf_energy = np.sum(spectrum[hsf_mask])

        return lsf_energy, msf_energy, hsf_energy

    def _calculate_balance_score(
        self,
        lsf: float,
        msf: float,
        hsf: float
    ) -> float:
        """Calculate balance score based on optimal distribution.

        Args:
            lsf: Normalized LSF energy
            msf: Normalized MSF energy
            hsf: Normalized HSF energy

        Returns:
            Balance score (0-1)
        """
        # Calculate deviations from optimal
        lsf_dev = abs(lsf - self.OPTIMAL_LSF)
        msf_dev = abs(msf - self.OPTIMAL_MSF)
        hsf_dev = abs(hsf - self.OPTIMAL_HSF)

        # Average deviation
        avg_dev = (lsf_dev + msf_dev + hsf_dev) / 3

        # Convert to score (lower deviation = higher score)
        balance_score = 1.0 - avg_dev

        return max(0.0, min(1.0, balance_score))

    def _identify_dominant_frequencies(
        self,
        lsf: float,
        msf: float,
        hsf: float
    ) -> List[str]:
        """Identify dominant frequency bands.

        Args:
            lsf: LSF energy
            msf: MSF energy
            hsf: HSF energy

        Returns:
            List of dominant bands
        """
        bands = [
            ("Low (structure)", lsf),
            ("Mid (detail)", msf),
            ("High (texture)", hsf)
        ]

        # Sort by energy
        bands.sort(key=lambda x: x[1], reverse=True)

        # Return bands above 25% threshold
        dominant = [name for name, energy in bands if energy > 0.25]

        return dominant if dominant else [bands[0][0]]

    def _calculate_comfort_score(
        self,
        lsf: float,
        msf: float,
        hsf: float,
        balance_score: float
    ) -> float:
        """Calculate visual comfort score.

        Args:
            lsf: LSF energy
            msf: MSF energy
            hsf: HSF energy
            balance_score: Balance score

        Returns:
            Visual comfort score (0-1)
        """
        # Penalize excessive HSF (causes fatigue)
        hsf_penalty = 0.0
        if hsf > 0.25:
            hsf_penalty = (hsf - 0.25) * 2  # Up to 0.5 penalty

        # Reward adequate LSF (provides structure)
        lsf_bonus = 0.0
        if lsf >= 0.4:
            lsf_bonus = 0.2

        # Combine factors
        comfort = balance_score + lsf_bonus - hsf_penalty

        return max(0.0, min(1.0, comfort))

    def _generate_recommendations(
        self,
        lsf: float,
        msf: float,
        hsf: float,
        comfort_score: float
    ) -> List[str]:
        """Generate optimization recommendations.

        Args:
            lsf: LSF energy
            msf: MSF energy
            hsf: HSF energy
            comfort_score: Visual comfort score

        Returns:
            List of recommendations
        """
        recommendations = []

        if comfort_score >= 0.8:
            recommendations.append(
                "Excellent spatial frequency balance - optimal visual comfort."
            )
            return recommendations

        # LSF issues
        if lsf < 0.35:
            recommendations.append(
                "Increase low-frequency content (overall structure). "
                "Consider slight blur or reduce excessive detail."
            )

        # HSF issues
        if hsf > 0.25:
            recommendations.append(
                "Reduce high-frequency content to prevent visual fatigue. "
                "Apply subtle smoothing or reduce sharpening."
            )

        # MSF issues
        if msf < 0.25:
            recommendations.append(
                "Enhance mid-frequency content (medium details). "
                "Increase local contrast or clarity."
            )

        # Balance issues
        if abs(lsf - hsf) < 0.1 and lsf < 0.4:
            recommendations.append(
                "Strengthen hierarchical structure: boost low frequencies "
                "relative to high frequencies for better visual processing."
            )

        return recommendations

    def create_frequency_visualization(
        self,
        image: Union[str, np.ndarray, Image.Image],
        analysis: Optional[SpatialFrequencyAnalysis] = None
    ) -> np.ndarray:
        """Create visualization of frequency content.

        Args:
            image: Input image
            analysis: Pre-computed analysis (computes if None)

        Returns:
            Visualization image
        """
        # Load image
        image_gray = self._load_image_gray(image)

        # Get or compute analysis
        if analysis is None:
            analysis = self.analyze(image)

        # Compute spectrum
        spectrum = self._compute_spectrum(image_gray)

        # Log scale for visualization
        spectrum_log = np.log1p(spectrum)

        # Normalize to [0, 255]
        spectrum_vis = ((spectrum_log - spectrum_log.min()) /
                       (spectrum_log.max() - spectrum_log.min()) * 255)
        spectrum_vis = spectrum_vis.astype(np.uint8)

        # Convert to RGB for colormap
        spectrum_colored = cv2.applyColorMap(spectrum_vis, cv2.COLORMAP_JET)

        # Add text with analysis results
        text_lines = [
            f"LSF: {analysis.lsf_energy:.2f} | MSF: {analysis.msf_energy:.2f} | HSF: {analysis.hsf_energy:.2f}",
            f"Balance: {analysis.balance_score:.2f} | Comfort: {analysis.visual_comfort_score:.2f}",
            f"Dominant: {', '.join(analysis.dominant_frequencies)}"
        ]

        y_offset = 30
        for line in text_lines:
            cv2.putText(
                spectrum_colored,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                spectrum_colored,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 25

        return spectrum_colored

    def _load_image_gray(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image as grayscale numpy array."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image
        elif isinstance(image, Image.Image):
            return np.array(image.convert("L"))
        else:
            pil_img = Image.open(image).convert("L")
            return np.array(pil_img)

    def __repr__(self) -> str:
        return "SpatialFrequencyAnalyzer()"
