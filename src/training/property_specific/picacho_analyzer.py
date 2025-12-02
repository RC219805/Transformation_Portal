#!/usr/bin/env python3
"""
Property Analyzer for 750 Picacho Lane Luxury Estate.

This module provides comprehensive analysis of property images including:
- Material detection (stone, glass, water, wood, metal, fabric)
- Color palette extraction
- Architectural feature identification
- Scene composition analysis
- Quality metrics assessment

Designed for generating property-specific training data for luxury real estate
image enhancement.

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging

import numpy as np
from PIL import Image

# Optional scipy for advanced image processing
try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_gaussian_filter = None

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """Types of materials detected in architectural imagery."""
    STONE = "stone"
    GLASS = "glass"
    WATER = "water"
    WOOD = "wood"
    METAL = "metal"
    FABRIC = "fabric"
    STUCCO = "stucco"
    TILE = "tile"
    CONCRETE = "concrete"
    VEGETATION = "vegetation"
    SKY = "sky"
    UNKNOWN = "unknown"


class RoomType(Enum):
    """Types of rooms/spaces in the property."""
    EXTERIOR = "exterior"
    LIVING_ROOM = "living_room"
    KITCHEN = "kitchen"
    POOL = "pool"
    PRIMARY_BATHROOM = "primary_bathroom"
    PRIMARY_BEDROOM = "primary_bedroom"
    DINING_ROOM = "dining_room"
    OFFICE = "office"
    AERIAL = "aerial"
    COURTYARD = "courtyard"
    UNKNOWN = "unknown"


@dataclass
class ColorPalette:
    """Extracted color palette from an image."""
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    color_weights: List[float] = field(default_factory=list)
    average_color: Tuple[int, int, int] = (128, 128, 128)
    color_temperature: str = "neutral"
    saturation_level: str = "moderate"
    brightness_level: str = "moderate"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dominant_colors": [list(c) for c in self.dominant_colors],
            "color_weights": self.color_weights,
            "average_color": list(self.average_color),
            "color_temperature": self.color_temperature,
            "saturation_level": self.saturation_level,
            "brightness_level": self.brightness_level,
        }


@dataclass
class MaterialDetection:
    """Results of material detection in an image."""
    detected_materials: Dict[MaterialType, float] = field(default_factory=dict)
    primary_materials: List[MaterialType] = field(default_factory=list)
    material_regions: Dict[MaterialType, List[Tuple[int, int, int, int]]] = field(
        default_factory=dict
    )
    total_coverage: Dict[MaterialType, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detected_materials": {m.value: v for m, v in self.detected_materials.items()},
            "primary_materials": [m.value for m in self.primary_materials],
            "total_coverage": {m.value: v for m, v in self.total_coverage.items()},
        }


@dataclass
class ArchitecturalFeatures:
    """Detected architectural features in an image."""
    has_infinity_edge: bool = False
    has_floor_to_ceiling_windows: bool = False
    has_open_floor_plan: bool = False
    has_outdoor_living: bool = False
    ceiling_type: str = "standard"
    lighting_type: str = "natural"
    view_type: str = "interior"
    architectural_style: str = "contemporary"
    notable_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_infinity_edge": self.has_infinity_edge,
            "has_floor_to_ceiling_windows": self.has_floor_to_ceiling_windows,
            "has_open_floor_plan": self.has_open_floor_plan,
            "has_outdoor_living": self.has_outdoor_living,
            "ceiling_type": self.ceiling_type,
            "lighting_type": self.lighting_type,
            "view_type": self.view_type,
            "architectural_style": self.architectural_style,
            "notable_features": self.notable_features,
        }


@dataclass
class ImageAnalysis:
    """Complete analysis of a single image."""
    image_path: Path = field(default_factory=Path)
    room_type: RoomType = RoomType.UNKNOWN
    dimensions: Tuple[int, int] = (0, 0)
    bit_depth: int = 8
    color_palette: ColorPalette = field(default_factory=ColorPalette)
    materials: MaterialDetection = field(default_factory=MaterialDetection)
    architectural_features: ArchitecturalFeatures = field(default_factory=ArchitecturalFeatures)
    quality_score: float = 0.0
    histogram_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": str(self.image_path),
            "room_type": self.room_type.value,
            "dimensions": list(self.dimensions),
            "bit_depth": self.bit_depth,
            "color_palette": self.color_palette.to_dict(),
            "materials": self.materials.to_dict(),
            "architectural_features": self.architectural_features.to_dict(),
            "quality_score": self.quality_score,
            "histogram_stats": self.histogram_stats,
        }


@dataclass
class PropertyReport:
    """Complete property analysis report."""
    property_name: str = "750 Picacho Lane"
    property_address: str = "750 Picacho Lane, Montecito, CA"
    project_number: str = "24098.00"
    analysis_date: str = ""
    total_images: int = 0
    image_analyses: List[ImageAnalysis] = field(default_factory=list)
    property_materials: Dict[str, float] = field(default_factory=dict)
    property_color_palette: ColorPalette = field(default_factory=ColorPalette)
    room_distribution: Dict[str, int] = field(default_factory=dict)
    average_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "property_name": self.property_name,
            "property_address": self.property_address,
            "project_number": self.project_number,
            "analysis_date": self.analysis_date,
            "total_images": self.total_images,
            "image_analyses": [a.to_dict() for a in self.image_analyses],
            "property_materials": self.property_materials,
            "property_color_palette": self.property_color_palette.to_dict(),
            "room_distribution": self.room_distribution,
            "average_quality_score": self.average_quality_score,
            "recommendations": self.recommendations,
        }

    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Property report saved to {output_path}")


class PicachoAnalyzer:
    """
    Comprehensive analyzer for 750 Picacho Lane property images.

    Analyzes all 6 property images (Exterior, LivingRoom, Kitchen, Pool,
    PrimaryBathroom, PrimaryBedroom) for materials, colors, and architectural
    features to generate property-specific training data.

    Attributes:
        property_dir: Path to property images directory
        image_paths: List of discovered image paths
        analyses: Completed image analyses
    """

    # Expected property images for 750 Picacho Lane
    EXPECTED_ROOMS = [
        "Exterior",
        "LivingRoom",
        "Kitchen",
        "Pool",
        "PrimaryBathroom",
        "PrimaryBedroom",
    ]

    # Material color signatures (approximate HSV ranges)
    MATERIAL_SIGNATURES = {
        MaterialType.STONE: {"h_range": (15, 45), "s_range": (10, 50), "v_range": (30, 80)},
        MaterialType.GLASS: {"h_range": (180, 220), "s_range": (5, 30), "v_range": (60, 100)},
        MaterialType.WATER: {"h_range": (180, 220), "s_range": (30, 80), "v_range": (40, 90)},
        MaterialType.WOOD: {"h_range": (10, 40), "s_range": (30, 70), "v_range": (25, 70)},
        MaterialType.METAL: {"h_range": (0, 360), "s_range": (0, 20), "v_range": (50, 95)},
        MaterialType.FABRIC: {"h_range": (0, 360), "s_range": (10, 60), "v_range": (40, 90)},
        MaterialType.STUCCO: {"h_range": (30, 60), "s_range": (5, 25), "v_range": (70, 100)},
        MaterialType.VEGETATION: {"h_range": (60, 150), "s_range": (25, 80), "v_range": (20, 80)},
        MaterialType.SKY: {"h_range": (180, 240), "s_range": (20, 80), "v_range": (60, 100)},
    }

    def __init__(
        self,
        property_dir: Optional[Path] = None,
        project_root: Optional[Path] = None
    ):
        """
        Initialize property analyzer.

        Args:
            property_dir: Path to property images directory
            project_root: Root path of the project (for relative paths)
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.property_dir = property_dir or (
            self.project_root / "projects" / "750_picacho_lane" / "Final_Production_UltraQuality"
        )
        self.property_dir = Path(self.property_dir)

        self.image_paths: List[Path] = []
        self.analyses: List[ImageAnalysis] = []
        self._discover_images()

    def _discover_images(self) -> None:
        """Discover property images in the directory."""
        if not self.property_dir.exists():
            logger.warning(f"Property directory not found: {self.property_dir}")
            return

        # Look for TIFF and common image formats
        extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
        for ext in extensions:
            self.image_paths.extend(self.property_dir.glob(ext))

        # Filter out hidden files
        self.image_paths = [p for p in self.image_paths if not p.name.startswith(".")]
        self.image_paths = sorted(self.image_paths)

        logger.info(f"Discovered {len(self.image_paths)} property images")

    def analyze_property(self) -> PropertyReport:
        """
        Perform comprehensive analysis of all property images.

        Returns:
            PropertyReport with complete analysis results
        """
        from datetime import datetime

        logger.info(f"Analyzing {len(self.image_paths)} property images...")

        self.analyses = []
        for image_path in self.image_paths:
            try:
                analysis = self._analyze_image(image_path)
                self.analyses.append(analysis)
                logger.info(f"  ✓ Analyzed: {image_path.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed to analyze {image_path.name}: {e}")

        # Generate property report
        report = self._generate_report()
        report.analysis_date = datetime.now().isoformat()

        return report

    def _analyze_image(self, image_path: Path) -> ImageAnalysis:
        """Analyze a single image for materials, colors, and features."""
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)

        # Determine room type from filename
        room_type = self._detect_room_type(image_path)

        # Extract color palette
        color_palette = self._extract_color_palette(img_array)

        # Detect materials
        materials = self._detect_materials(img_array, room_type)

        # Detect architectural features
        arch_features = self._detect_architectural_features(img_array, room_type)

        # Calculate quality score
        quality_score = self._calculate_quality_score(img_array)

        # Calculate histogram statistics
        histogram_stats = self._calculate_histogram_stats(img_array)

        return ImageAnalysis(
            image_path=image_path,
            room_type=room_type,
            dimensions=(img.width, img.height),
            bit_depth=self._get_bit_depth(img),
            color_palette=color_palette,
            materials=materials,
            architectural_features=arch_features,
            quality_score=quality_score,
            histogram_stats=histogram_stats,
        )

    def _detect_room_type(self, image_path: Path) -> RoomType:
        """Detect room type from filename."""
        filename = image_path.stem.lower()

        room_mapping = {
            "exterior": RoomType.EXTERIOR,
            "livingroom": RoomType.LIVING_ROOM,
            "living_room": RoomType.LIVING_ROOM,
            "greatroom": RoomType.LIVING_ROOM,
            "great_room": RoomType.LIVING_ROOM,
            "kitchen": RoomType.KITCHEN,
            "pool": RoomType.POOL,
            "primarybathroom": RoomType.PRIMARY_BATHROOM,
            "primary_bathroom": RoomType.PRIMARY_BATHROOM,
            "bathroom": RoomType.PRIMARY_BATHROOM,
            "primarybedroom": RoomType.PRIMARY_BEDROOM,
            "primary_bedroom": RoomType.PRIMARY_BEDROOM,
            "bedroom": RoomType.PRIMARY_BEDROOM,
            "aerial": RoomType.AERIAL,
            "courtyard": RoomType.COURTYARD,
            "dining": RoomType.DINING_ROOM,
            "office": RoomType.OFFICE,
        }

        for key, room in room_mapping.items():
            if key in filename:
                return room

        return RoomType.UNKNOWN

    def _extract_color_palette(self, img_array: np.ndarray, n_colors: int = 5) -> ColorPalette:
        """Extract dominant colors from image."""
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Normalize to 8-bit for analysis
        if img_array.dtype != np.uint8:
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        # Downsample for speed
        h, w = img_array.shape[:2]
        scale = max(1, min(h, w) // 256)
        img_small = img_array[::scale, ::scale]

        # Reshape to pixel list
        pixels = img_small.reshape(-1, 3)

        # Simple k-means-like clustering using numpy
        dominant_colors, weights = self._simple_color_clustering(pixels, n_colors)

        # Calculate average color
        avg_color = tuple(int(c) for c in pixels.mean(axis=0))

        # Determine color temperature
        r_avg, g_avg, b_avg = avg_color
        if r_avg > b_avg + 20:
            temperature = "warm"
        elif b_avg > r_avg + 20:
            temperature = "cool"
        else:
            temperature = "neutral"

        # Calculate saturation and brightness levels
        hsv_pixels = self._rgb_to_hsv_batch(pixels)
        avg_saturation = hsv_pixels[:, 1].mean()
        avg_brightness = hsv_pixels[:, 2].mean()

        saturation_level = (
            "high" if avg_saturation > 0.5
            else "moderate" if avg_saturation > 0.25
            else "low"
        )
        brightness_level = (
            "high" if avg_brightness > 0.7
            else "moderate" if avg_brightness > 0.4
            else "low"
        )

        return ColorPalette(
            dominant_colors=dominant_colors,
            color_weights=weights,
            average_color=avg_color,
            color_temperature=temperature,
            saturation_level=saturation_level,
            brightness_level=brightness_level,
        )

    def _simple_color_clustering(
        self,
        pixels: np.ndarray,
        n_clusters: int
    ) -> Tuple[List[Tuple[int, int, int]], List[float]]:
        """Simple color clustering using random sampling and binning."""
        # Quantize colors to reduce complexity
        quantized = (pixels // 32) * 32 + 16

        # Find unique colors and their counts
        unique_colors, indices, counts = np.unique(
            quantized, axis=0, return_inverse=True, return_counts=True
        )

        # Get top n_clusters by count
        top_indices = np.argsort(-counts)[:n_clusters]

        dominant_colors = [tuple(int(c) for c in unique_colors[i]) for i in top_indices]
        weights = [float(counts[i]) / len(pixels) for i in top_indices]

        return dominant_colors, weights

    def _rgb_to_hsv_batch(self, rgb_array: np.ndarray) -> np.ndarray:
        """Convert RGB array to HSV (normalized 0-1)."""
        rgb_normalized = rgb_array.astype(np.float32) / 255.0

        r, g, b = rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]

        v = np.maximum(np.maximum(r, g), b)
        c = v - np.minimum(np.minimum(r, g), b)

        s = np.where(v != 0, c / v, 0)

        h = np.zeros_like(v)
        mask_c = c != 0

        mask_r = mask_c & (v == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / c[mask_r]) % 6

        mask_g = mask_c & (v == g)
        h[mask_g] = (b[mask_g] - r[mask_g]) / c[mask_g] + 2

        mask_b = mask_c & (v == b)
        h[mask_b] = (r[mask_b] - g[mask_b]) / c[mask_b] + 4

        h = h / 6.0  # Normalize to 0-1

        return np.stack([h, s, v], axis=1)

    def _detect_materials(
        self,
        img_array: np.ndarray,
        room_type: RoomType
    ) -> MaterialDetection:
        """Detect materials present in the image."""
        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        # Normalize if needed
        if img_array.dtype != np.uint8:
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        # Downsample for analysis
        h, w = img_array.shape[:2]
        scale = max(1, min(h, w) // 512)
        img_small = img_array[::scale, ::scale]

        # Convert to HSV for material detection
        pixels = img_small.reshape(-1, 3)
        hsv_pixels = self._rgb_to_hsv_batch(pixels)

        # Scale HSV to standard ranges (H: 0-360, S: 0-100, V: 0-100)
        hsv_scaled = hsv_pixels.copy()
        hsv_scaled[:, 0] *= 360
        hsv_scaled[:, 1] *= 100
        hsv_scaled[:, 2] *= 100

        # Detect each material type
        detected_materials: Dict[MaterialType, float] = {}

        for material, sig in self.MATERIAL_SIGNATURES.items():
            h_min, h_max = sig["h_range"]
            s_min, s_max = sig["s_range"]
            v_min, v_max = sig["v_range"]

            # Check if pixels fall within material signature
            h_match = (hsv_scaled[:, 0] >= h_min) & (hsv_scaled[:, 0] <= h_max)
            s_match = (hsv_scaled[:, 1] >= s_min) & (hsv_scaled[:, 1] <= s_max)
            v_match = (hsv_scaled[:, 2] >= v_min) & (hsv_scaled[:, 2] <= v_max)

            match_ratio = (h_match & s_match & v_match).sum() / len(pixels)
            if match_ratio > 0.01:  # At least 1% coverage
                detected_materials[material] = match_ratio

        # Apply room-type-specific adjustments
        detected_materials = self._adjust_materials_for_room(detected_materials, room_type)

        # Determine primary materials (top 3)
        sorted_materials = sorted(
            detected_materials.items(), key=lambda x: x[1], reverse=True
        )
        primary_materials = [m[0] for m in sorted_materials[:3]]

        return MaterialDetection(
            detected_materials=detected_materials,
            primary_materials=primary_materials,
            total_coverage=detected_materials,
        )

    def _adjust_materials_for_room(
        self,
        materials: Dict[MaterialType, float],
        room_type: RoomType
    ) -> Dict[MaterialType, float]:
        """Adjust material detection confidence based on room type."""
        adjusted = materials.copy()

        # Room-specific material expectations
        room_material_boost = {
            RoomType.POOL: {MaterialType.WATER: 1.5, MaterialType.STONE: 1.2},
            RoomType.KITCHEN: {MaterialType.METAL: 1.3, MaterialType.STONE: 1.2},
            RoomType.PRIMARY_BATHROOM: {MaterialType.STONE: 1.3, MaterialType.GLASS: 1.2},
            RoomType.PRIMARY_BEDROOM: {MaterialType.FABRIC: 1.3, MaterialType.WOOD: 1.2},
            RoomType.LIVING_ROOM: {MaterialType.FABRIC: 1.2, MaterialType.WOOD: 1.2},
            RoomType.EXTERIOR: {MaterialType.STUCCO: 1.3, MaterialType.VEGETATION: 1.2},
        }

        boosts = room_material_boost.get(room_type, {})
        for material, boost in boosts.items():
            if material in adjusted:
                adjusted[material] *= boost

        return adjusted

    def _detect_architectural_features(
        self,
        img_array: np.ndarray,
        room_type: RoomType
    ) -> ArchitecturalFeatures:
        """Detect architectural features based on image analysis and room type."""
        features = ArchitecturalFeatures()

        # Analyze image for feature detection
        h, w = img_array.shape[:2]
        aspect_ratio = w / h

        # Room-type-specific feature detection
        if room_type == RoomType.POOL:
            features.has_infinity_edge = True
            features.has_outdoor_living = True
            features.view_type = "ocean_view"
            features.notable_features = ["infinity_pool", "ocean_view", "stone_deck"]

        elif room_type == RoomType.LIVING_ROOM:
            features.has_floor_to_ceiling_windows = True
            features.has_open_floor_plan = True
            features.ceiling_type = "high_ceiling"
            features.notable_features = ["open_concept", "natural_light", "luxury_finishes"]

        elif room_type == RoomType.KITCHEN:
            features.has_open_floor_plan = True
            features.lighting_type = "mixed"
            features.notable_features = ["custom_cabinetry", "stone_counters", "high_end_appliances"]

        elif room_type == RoomType.PRIMARY_BATHROOM:
            features.lighting_type = "natural"
            features.notable_features = ["spa_bathroom", "stone_finishes", "soaking_tub"]

        elif room_type == RoomType.PRIMARY_BEDROOM:
            features.view_type = "scenic"
            features.ceiling_type = "high_ceiling"
            features.notable_features = ["panoramic_views", "luxury_finishes", "natural_light"]

        elif room_type == RoomType.EXTERIOR:
            features.architectural_style = "contemporary_mediterranean"
            features.view_type = "hillside"
            features.notable_features = ["montecito_coastal", "landscaped_gardens", "dramatic_entry"]

        # Wide aspect ratio suggests panoramic or open space
        if aspect_ratio > 1.5:
            features.notable_features.append("panoramic_composition")

        return features

    def _calculate_quality_score(self, img_array: np.ndarray) -> float:
        """Calculate overall quality score for the image."""
        scores = []

        # Normalize if needed
        if img_array.dtype != np.uint8:
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)

        # Resolution score (higher is better, max at 4K)
        h, w = img_array.shape[:2]
        resolution = h * w
        res_score = min(1.0, resolution / (3840 * 2160))
        scores.append(res_score)

        # Dynamic range score
        min_val, max_val = img_array.min(), img_array.max()
        range_score = (max_val - min_val) / 255.0
        scores.append(range_score)

        # Contrast score (standard deviation)
        std = img_array.std()
        contrast_score = min(1.0, std / 60.0)
        scores.append(contrast_score)

        # Sharpness estimate (Laplacian variance)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.float32)
        else:
            gray = img_array.astype(np.float32)

        # Simple Laplacian-like edge detection
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_var = self._convolve_variance(gray, kernel)
        sharpness_score = min(1.0, laplacian_var / 500.0)
        scores.append(sharpness_score)

        return np.mean(scores)

    def _convolve_variance(self, img: np.ndarray, kernel: np.ndarray) -> float:
        """Calculate variance of convolved image (simple edge detection)."""
        h, w = img.shape
        # kernel.shape is used implicitly via patch size (3x3) in the loop

        # Sample center region for speed
        center_h, center_w = h // 2, w // 2
        region_size = min(256, min(h, w) // 2)
        region = img[
            center_h - region_size:center_h + region_size,
            center_w - region_size:center_w + region_size
        ]

        # Simple convolution approximation
        edges = np.zeros_like(region)
        for i in range(1, region.shape[0] - 1):
            for j in range(1, region.shape[1] - 1):
                patch = region[i - 1:i + 2, j - 1:j + 2]
                edges[i, j] = np.abs(np.sum(patch * kernel))

        return edges.var()

    def _calculate_histogram_stats(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate histogram statistics for the image."""
        # Normalize if needed
        if img_array.dtype != np.uint8:
            if img_array.max() > 255:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)

        if len(img_array.shape) == 3:
            channels = ["red", "green", "blue"]
            stats = {}
            for i, channel in enumerate(channels):
                channel_data = img_array[:, :, i]
                stats[f"{channel}_mean"] = float(channel_data.mean())
                stats[f"{channel}_std"] = float(channel_data.std())
                stats[f"{channel}_min"] = float(channel_data.min())
                stats[f"{channel}_max"] = float(channel_data.max())
        else:
            stats = {
                "gray_mean": float(img_array.mean()),
                "gray_std": float(img_array.std()),
                "gray_min": float(img_array.min()),
                "gray_max": float(img_array.max()),
            }

        return stats

    def _get_bit_depth(self, img: Image.Image) -> int:
        """Determine bit depth of image."""
        mode_to_depth = {
            "1": 1,
            "L": 8,
            "P": 8,
            "RGB": 8,
            "RGBA": 8,
            "CMYK": 8,
            "YCbCr": 8,
            "LAB": 8,
            "HSV": 8,
            "I": 32,
            "F": 32,
            "I;16": 16,
            "I;16B": 16,
            "I;16L": 16,
        }
        return mode_to_depth.get(img.mode, 8)

    def _generate_report(self) -> PropertyReport:
        """Generate comprehensive property report from analyses."""
        if not self.analyses:
            return PropertyReport()

        # Aggregate materials across all images
        all_materials: Dict[str, float] = {}
        for analysis in self.analyses:
            for material, coverage in analysis.materials.detected_materials.items():
                key = material.value
                all_materials[key] = all_materials.get(key, 0) + coverage

        # Normalize material coverage
        total = sum(all_materials.values()) or 1
        property_materials = {k: v / total for k, v in all_materials.items()}

        # Aggregate color palettes
        all_colors = []
        for analysis in self.analyses:
            all_colors.extend(analysis.color_palette.dominant_colors)

        # Simple average for property palette
        if all_colors:
            avg_colors = np.mean(all_colors, axis=0).astype(int)
            property_palette = ColorPalette(
                dominant_colors=all_colors[:5],
                average_color=tuple(int(c) for c in avg_colors),
            )
        else:
            property_palette = ColorPalette()

        # Room distribution
        room_dist = {}
        for analysis in self.analyses:
            room = analysis.room_type.value
            room_dist[room] = room_dist.get(room, 0) + 1

        # Average quality
        avg_quality = np.mean([a.quality_score for a in self.analyses])

        # Generate recommendations
        recommendations = self._generate_recommendations()

        return PropertyReport(
            total_images=len(self.analyses),
            image_analyses=self.analyses,
            property_materials=property_materials,
            property_color_palette=property_palette,
            room_distribution=room_dist,
            average_quality_score=avg_quality,
            recommendations=recommendations,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations based on analysis."""
        recommendations = []

        if not self.analyses:
            recommendations.append("No images found for analysis.")
            return recommendations

        # Check material diversity
        all_materials = set()
        for analysis in self.analyses:
            all_materials.update(analysis.materials.primary_materials)

        if len(all_materials) >= 5:
            recommendations.append(
                "Good material diversity detected. Multi-material training recommended."
            )
        else:
            recommendations.append(
                f"Limited material diversity ({len(all_materials)} types). "
                "Consider augmentation with material-specific transforms."
            )

        # Check quality scores
        avg_quality = np.mean([a.quality_score for a in self.analyses])
        if avg_quality > 0.7:
            recommendations.append(
                "High-quality source images. Suitable for direct training."
            )
        elif avg_quality > 0.5:
            recommendations.append(
                "Moderate quality sources. Pre-processing recommended."
            )
        else:
            recommendations.append(
                "Lower quality sources detected. Significant pre-processing required."
            )

        # Training-specific recommendations
        recommendations.extend([
            "Use multi-scale crops (512, 1024, 2048) for comprehensive training.",
            "Apply depth-aware augmentation for architectural coherence.",
            "Implement material-specific loss weighting for balanced learning.",
            "Consider 3-stage training: Material → Architectural → Full-Resolution.",
        ])

        return recommendations

    def get_image_paths(self) -> List[Path]:
        """Return list of discovered image paths."""
        return self.image_paths

    def get_room_types(self) -> List[RoomType]:
        """Return list of detected room types."""
        return [a.room_type for a in self.analyses]

    def __repr__(self) -> str:
        return (
            f"PicachoAnalyzer(property_dir={self.property_dir}, "
            f"images={len(self.image_paths)})"
        )
