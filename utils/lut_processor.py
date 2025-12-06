"""
LUT Processor - .cube LUT Loading and Application

Implements professional color grading via LUT (Look-Up Table) application with:
- .cube file format parsing (1D and 3D LUTs)
- Trilinear interpolation for smooth color mapping
- Configurable strength/opacity control
- Support for film emulation, location aesthetics, and material response LUTs
- 16-bit precision preservation

Part of Transformation Portal Phase 3 integration.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class LUTCategory(Enum):
    """LUT category types."""
    FILM_EMULATION = "film_emulation"
    LOCATION_AESTHETIC = "location_aesthetic"
    MATERIAL_RESPONSE = "material_response"
    CUSTOM = "custom"


@dataclass
class LUTConfig:
    """Configuration for LUT processing."""
    lut_path: Optional[Path] = None
    strength: float = 0.7  # LUT application strength [0, 1]
    category: LUTCategory = LUTCategory.FILM_EMULATION
    preserve_highlights: bool = True
    preserve_blacks: bool = True
    blend_mode: str = "normal"  # normal, multiply, screen
    
    def __post_init__(self):
        """Validate configuration."""
        if self.lut_path is not None:
            self.lut_path = Path(self.lut_path)
            if not self.lut_path.exists():
                raise FileNotFoundError(f"LUT file not found: {self.lut_path}")
        
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Strength must be in [0, 1], got {self.strength}")
        
        if self.blend_mode not in ["normal", "multiply", "screen"]:
            raise ValueError(f"Invalid blend mode: {self.blend_mode}")


class LUTProcessor:
    """
    Professional LUT processor for color grading.
    
    Supports:
    - 1D and 3D .cube LUT format
    - Trilinear interpolation
    - Configurable strength/opacity
    - Highlight/black preservation
    - Multiple blend modes
    """
    
    def __init__(self, config: LUTConfig):
        """
        Initialize LUT processor.
        
        Args:
            config: LUT processing configuration
        """
        self.config = config
        self.lut_data = None
        self.lut_size = None
        self.is_3d = False
        self.title = None
        
        if config.lut_path:
            self._load_lut(config.lut_path)
            logger.info(f"LUT processor initialized with {config.lut_path.name}")
    
    def _load_lut(self, lut_path: Path):
        """
        Load .cube LUT file.
        
        Args:
            lut_path: Path to .cube file
        """
        logger.info(f"Loading LUT: {lut_path}")
        
        with open(lut_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        lut_size = None
        domain_min = np.array([0.0, 0.0, 0.0])
        domain_max = np.array([1.0, 1.0, 1.0])
        data_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse title
            if line.startswith('TITLE'):
                self.title = line.split('"')[1] if '"' in line else line.split()[1]
            
            # Parse LUT size
            elif line.startswith('LUT_3D_SIZE'):
                lut_size = int(line.split()[1])
                self.is_3d = True
            elif line.startswith('LUT_1D_SIZE'):
                lut_size = int(line.split()[1])
                self.is_3d = False
            
            # Parse domain
            elif line.startswith('DOMAIN_MIN'):
                domain_min = np.array([float(x) for x in line.split()[1:4]])
            elif line.startswith('DOMAIN_MAX'):
                domain_max = np.array([float(x) for x in line.split()[1:4]])
            
            # Data line (three float values)
            elif re.match(r'^[\d\.\-\+eE\s]+$', line):
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    data_lines.append(values)
        
        if lut_size is None:
            raise ValueError(f"Could not parse LUT size from {lut_path}")
        
        # Convert to numpy array
        lut_array = np.array(data_lines, dtype=np.float32)
        
        # Validate dimensions
        if self.is_3d:
            expected_size = lut_size ** 3
            if len(lut_array) != expected_size:
                raise ValueError(
                    f"3D LUT size mismatch: expected {expected_size}, got {len(lut_array)}"
                )
            # Reshape to 3D grid
            self.lut_data = lut_array.reshape((lut_size, lut_size, lut_size, 3))
        else:
            if len(lut_array) != lut_size:
                raise ValueError(
                    f"1D LUT size mismatch: expected {lut_size}, got {len(lut_array)}"
                )
            self.lut_data = lut_array
        
        self.lut_size = lut_size
        self.domain_min = domain_min
        self.domain_max = domain_max
        
        logger.info(
            f"Loaded {'3D' if self.is_3d else '1D'} LUT: "
            f"size={lut_size}, title={self.title}"
        )
    
    def apply(
        self,
        image: np.ndarray,
        strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply LUT to image.
        
        Args:
            image: Input image (H, W, 3) in [0, 1] range
            strength: Override config strength (optional)
        
        Returns:
            LUT-processed image in [0, 1] range
        """
        if self.lut_data is None:
            raise ValueError("No LUT loaded")
        
        strength = strength if strength is not None else self.config.strength
        
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Clamp to [0, 1]
        image = np.clip(image, 0.0, 1.0)
        
        # Apply LUT
        if self.is_3d:
            result = self._apply_3d_lut(image)
        else:
            result = self._apply_1d_lut(image)
        
        # Blend with original based on strength
        result = image * (1.0 - strength) + result * strength
        
        # Preserve highlights and blacks if requested
        if self.config.preserve_highlights:
            result = self._preserve_highlights(image, result)
        if self.config.preserve_blacks:
            result = self._preserve_blacks(image, result)
        
        return np.clip(result, 0.0, 1.0)
    
    def _apply_3d_lut(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 3D LUT using trilinear interpolation.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            LUT-processed image
        """
        h, w, c = image.shape
        
        # Flatten image for interpolation
        pixels = image.reshape(-1, 3)
        
        # Create interpolator for each channel
        coords = np.linspace(0, self.lut_size - 1, self.lut_size)
        
        # Interpolate each channel
        result = np.zeros_like(pixels)
        for ch in range(3):
            interpolator = RegularGridInterpolator(
                (coords, coords, coords),
                self.lut_data[:, :, :, ch],
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            
            # Map pixel values to LUT grid coordinates
            sample_points = pixels * (self.lut_size - 1)
            result[:, ch] = interpolator(sample_points)
        
        return result.reshape(h, w, c)
    
    def _apply_1d_lut(self, image: np.ndarray) -> np.ndarray:
        """
        Apply 1D LUT using linear interpolation per channel.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            LUT-processed image
        """
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        for ch in range(3):
            # Linear interpolation
            indices = image[:, :, ch] * (self.lut_size - 1)
            indices_floor = np.floor(indices).astype(np.int32)
            indices_ceil = np.ceil(indices).astype(np.int32)
            
            # Clamp indices
            indices_floor = np.clip(indices_floor, 0, self.lut_size - 1)
            indices_ceil = np.clip(indices_ceil, 0, self.lut_size - 1)
            
            # Interpolation weight
            weight = indices - indices_floor
            
            # Interpolate
            result[:, :, ch] = (
                self.lut_data[indices_floor, ch] * (1 - weight) +
                self.lut_data[indices_ceil, ch] * weight
            )
        
        return result
    
    def _preserve_highlights(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        threshold: float = 0.9
    ) -> np.ndarray:
        """
        Preserve bright highlights by blending based on luminance.
        
        Args:
            original: Original image
            processed: LUT-processed image
            threshold: Luminance threshold for preservation
        
        Returns:
            Image with preserved highlights
        """
        # Calculate luminance (Rec. 709)
        luminance = (
            0.2126 * original[:, :, 0] +
            0.7152 * original[:, :, 1] +
            0.0722 * original[:, :, 2]
        )
        
        # Create blend mask (smooth transition above threshold)
        blend_mask = np.clip((luminance - threshold) / (1.0 - threshold), 0, 1)
        blend_mask = blend_mask[:, :, np.newaxis]  # Add channel dimension
        
        # Blend original highlights back in
        return processed * (1 - blend_mask) + original * blend_mask
    
    def _preserve_blacks(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Preserve deep blacks by blending based on luminance.
        
        Args:
            original: Original image
            processed: LUT-processed image
            threshold: Luminance threshold for preservation
        
        Returns:
            Image with preserved blacks
        """
        # Calculate luminance
        luminance = (
            0.2126 * original[:, :, 0] +
            0.7152 * original[:, :, 1] +
            0.0722 * original[:, :, 2]
        )
        
        # Create blend mask (smooth transition below threshold)
        blend_mask = np.clip((threshold - luminance) / threshold, 0, 1)
        blend_mask = blend_mask[:, :, np.newaxis]
        
        # Blend original blacks back in
        return processed * (1 - blend_mask) + original * blend_mask
    
    def load_lut(self, lut_path: Path):
        """
        Load a new LUT file.
        
        Args:
            lut_path: Path to .cube file
        """
        self.config.lut_path = Path(lut_path)
        self._load_lut(self.config.lut_path)


def create_lut_processor(
    lut_path: Optional[Path] = None,
    strength: float = 0.7,
    category: LUTCategory = LUTCategory.FILM_EMULATION,
    preserve_highlights: bool = True,
    preserve_blacks: bool = True
) -> LUTProcessor:
    """
    Convenience function to create LUT processor.
    
    Args:
        lut_path: Path to .cube LUT file
        strength: LUT application strength [0, 1]
        category: LUT category type
        preserve_highlights: Preserve bright highlights
        preserve_blacks: Preserve deep blacks
    
    Returns:
        Configured LUT processor
    """
    config = LUTConfig(
        lut_path=lut_path,
        strength=strength,
        category=category,
        preserve_highlights=preserve_highlights,
        preserve_blacks=preserve_blacks
    )
    return LUTProcessor(config)


def discover_luts(base_path: Path) -> dict:
    """
    Discover available LUT files in directory structure.
    
    Args:
        base_path: Base path to search (e.g., assets/luts/)
    
    Returns:
        Dictionary mapping category -> list of LUT paths
    """
    luts = {
        LUTCategory.FILM_EMULATION: [],
        LUTCategory.LOCATION_AESTHETIC: [],
        LUTCategory.MATERIAL_RESPONSE: [],
        LUTCategory.CUSTOM: []
    }
    
    base_path = Path(base_path)
    
    for category in luts.keys():
        category_path = base_path / category.value
        if category_path.exists():
            cube_files = sorted(category_path.rglob("*.cube"))
            luts[category] = cube_files
            logger.debug(f"Found {len(cube_files)} LUTs in {category.value}")
    
    return luts
