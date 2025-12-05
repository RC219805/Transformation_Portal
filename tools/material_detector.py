#!/usr/bin/env python3
"""
Material Detection with Confidence Scores
==========================================
Advanced material detection system that provides probability maps and confidence
scores for each material type (wood, metal, glass, stone, fabric, water).

Features:
- Per-pixel confidence heatmaps for each material type
- Adaptive enhancement strength based on confidence levels
- Material detection reports with statistics
- Visualization overlays for confidence maps
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


class MaterialType(Enum):
    """Supported material types for detection."""
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    STONE = "stone"
    FABRIC = "fabric"
    WATER = "water"
    CONCRETE = "concrete"
    CERAMIC = "ceramic"


@dataclass
class MaterialConfidence:
    """Confidence score for a material type."""
    material: MaterialType
    confidence: float  # 0.0 to 1.0
    pixel_count: int
    percentage: float
    mean_confidence: float
    std_confidence: float


@dataclass
class MaterialDetectionResult:
    """Complete material detection result."""
    image_path: Path
    materials: Dict[MaterialType, MaterialConfidence]
    confidence_maps: Dict[MaterialType, np.ndarray]  # Per-pixel confidence [0, 1]
    dominant_material: MaterialType
    processing_time: float
    timestamp: str


class MaterialDetector:
    """
    Advanced material detector with confidence scoring.
    
    Uses color histograms, texture analysis, and spatial patterns to identify
    materials and provide confidence scores.
    """
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize material detector.
        
        Args:
            min_confidence: Minimum confidence threshold for reporting materials
        """
        self.min_confidence = min_confidence
        self._load_material_profiles()
        
    def _load_material_profiles(self):
        """Load material characteristic profiles."""
        # Color profiles (HSV ranges) for each material
        self.material_profiles = {
            MaterialType.WOOD: {
                'hue_range': (10, 40),  # Browns, tans
                'saturation_range': (0.2, 0.8),
                'value_range': (0.2, 0.7),
                'texture_strength': 0.4,
            },
            MaterialType.METAL: {
                'hue_range': None,  # Achromatic
                'saturation_range': (0.0, 0.2),
                'value_range': (0.4, 1.0),
                'texture_strength': 0.1,  # Low texture, high specular
                'specular_threshold': 0.7,
            },
            MaterialType.GLASS: {
                'hue_range': None,
                'saturation_range': (0.0, 0.3),
                'value_range': (0.5, 1.0),
                'texture_strength': 0.05,  # Very smooth
                'transparency_indicator': True,
            },
            MaterialType.STONE: {
                'hue_range': (0, 60),  # Grays, browns
                'saturation_range': (0.0, 0.4),
                'value_range': (0.3, 0.8),
                'texture_strength': 0.6,  # High texture
            },
            MaterialType.FABRIC: {
                'hue_range': None,  # Any hue
                'saturation_range': (0.2, 1.0),
                'value_range': (0.2, 0.9),
                'texture_strength': 0.5,
            },
            MaterialType.WATER: {
                'hue_range': (180, 220),  # Blues, cyans
                'saturation_range': (0.3, 0.9),
                'value_range': (0.3, 0.9),
                'texture_strength': 0.2,
                'specular_threshold': 0.6,
            },
            MaterialType.CONCRETE: {
                'hue_range': None,
                'saturation_range': (0.0, 0.2),
                'value_range': (0.4, 0.7),
                'texture_strength': 0.3,
            },
            MaterialType.CERAMIC: {
                'hue_range': None,
                'saturation_range': (0.0, 0.8),
                'value_range': (0.5, 1.0),
                'texture_strength': 0.1,
                'specular_threshold': 0.5,
            },
        }
    
    def detect(self, image_path: Path) -> MaterialDetectionResult:
        """
        Detect materials in an image with confidence scores.
        
        Args:
            image_path: Path to input image
            
        Returns:
            MaterialDetectionResult with confidence maps and statistics
        """
        import time
        start_time = time.time()
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to HSV for color analysis
        hsv_img = self._rgb_to_hsv(img_array)
        
        # Compute texture strength map
        texture_map = self._compute_texture_strength(img_array)
        
        # Compute specular map (highlights)
        specular_map = self._compute_specular_map(img_array)
        
        # Detect each material type
        confidence_maps = {}
        materials = {}
        
        for material_type in MaterialType:
            confidence_map = self._detect_material(
                hsv_img, texture_map, specular_map, material_type
            )
            confidence_maps[material_type] = confidence_map
            
            # Compute statistics
            valid_pixels = confidence_map > self.min_confidence
            pixel_count = int(np.sum(valid_pixels))
            
            if pixel_count > 0:
                mean_conf = float(np.mean(confidence_map[valid_pixels]))
                std_conf = float(np.std(confidence_map[valid_pixels]))
                percentage = 100.0 * pixel_count / (img_array.shape[0] * img_array.shape[1])
                
                materials[material_type] = MaterialConfidence(
                    material=material_type,
                    confidence=mean_conf,
                    pixel_count=pixel_count,
                    percentage=percentage,
                    mean_confidence=mean_conf,
                    std_confidence=std_conf
                )
        
        # Find dominant material
        dominant_material = max(
            materials.items(),
            key=lambda x: x[1].percentage
        )[0] if materials else MaterialType.STONE
        
        processing_time = time.time() - start_time
        
        return MaterialDetectionResult(
            image_path=image_path,
            materials=materials,
            confidence_maps=confidence_maps,
            dominant_material=dominant_material,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        # Normalize RGB to [0, 1]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Hue calculation
        h = np.zeros_like(max_c)
        
        mask = diff != 0
        r_mask = (max_c == r) & mask
        g_mask = (max_c == g) & mask
        b_mask = (max_c == b) & mask
        
        h[r_mask] = 60 * (((g[r_mask] - b[r_mask]) / diff[r_mask]) % 6)
        h[g_mask] = 60 * (((b[g_mask] - r[g_mask]) / diff[g_mask]) + 2)
        h[b_mask] = 60 * (((r[b_mask] - g[b_mask]) / diff[b_mask]) + 4)
        
        # Saturation
        s = np.zeros_like(max_c)
        s[max_c != 0] = diff[max_c != 0] / max_c[max_c != 0]
        
        # Value
        v = max_c
        
        return np.stack([h, s, v], axis=-1)
    
    def _compute_texture_strength(self, image: np.ndarray) -> np.ndarray:
        """Compute texture strength using gradient magnitude."""
        gray = np.mean(image, axis=-1)
        
        # Sobel filters
        dy, dx = np.gradient(gray)
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Normalize
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        
        return magnitude
    
    def _compute_specular_map(self, image: np.ndarray) -> np.ndarray:
        """Compute specular highlights map."""
        # High value, low saturation = specular highlight
        v = np.max(image, axis=-1)
        s = (v - np.min(image, axis=-1)) / (v + 1e-6)
        
        specular = v * (1 - s)
        return specular
    
    def _detect_material(
        self,
        hsv_img: np.ndarray,
        texture_map: np.ndarray,
        specular_map: np.ndarray,
        material_type: MaterialType
    ) -> np.ndarray:
        """
        Detect specific material type and return confidence map.
        
        Args:
            hsv_img: HSV image (H in [0, 360], S/V in [0, 1])
            texture_map: Texture strength map [0, 1]
            specular_map: Specular highlight map [0, 1]
            material_type: Material to detect
            
        Returns:
            Confidence map [0, 1] for the material
        """
        profile = self.material_profiles[material_type]
        h, s, v = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
        
        confidence = np.ones_like(s)
        
        # Hue matching
        if profile['hue_range'] is not None:
            hue_min, hue_max = profile['hue_range']
            hue_match = ((h >= hue_min) & (h <= hue_max)).astype(np.float32)
            confidence *= hue_match
        
        # Saturation matching
        sat_min, sat_max = profile['saturation_range']
        sat_match = 1.0 - np.abs(s - (sat_min + sat_max) / 2) / ((sat_max - sat_min) / 2 + 1e-6)
        sat_match = np.clip(sat_match, 0, 1)
        confidence *= sat_match
        
        # Value matching
        val_min, val_max = profile['value_range']
        val_match = 1.0 - np.abs(v - (val_min + val_max) / 2) / ((val_max - val_min) / 2 + 1e-6)
        val_match = np.clip(val_match, 0, 1)
        confidence *= val_match
        
        # Texture matching
        tex_target = profile['texture_strength']
        tex_match = 1.0 - np.abs(texture_map - tex_target)
        tex_match = np.clip(tex_match, 0, 1)
        confidence *= (0.7 + 0.3 * tex_match)  # Texture is weighted less
        
        # Specular matching (for metals, glass, water, ceramic)
        if 'specular_threshold' in profile:
            spec_thresh = profile['specular_threshold']
            spec_match = (specular_map > spec_thresh).astype(np.float32)
            confidence *= (0.5 + 0.5 * spec_match)
        
        return confidence
    
    def generate_heatmap(
        self,
        result: MaterialDetectionResult,
        material_type: MaterialType,
        output_path: Path
    ):
        """
        Generate confidence heatmap overlay for a material type.
        
        Args:
            result: Material detection result
            material_type: Material to visualize
            output_path: Path to save heatmap image
        """
        # Load original image
        img = Image.open(result.image_path).convert('RGB')
        img_array = np.array(img)
        
        # Get confidence map
        confidence_map = result.confidence_maps[material_type]
        
        # Create heatmap (red = high confidence)
        heatmap = self._confidence_to_heatmap(confidence_map)
        
        # Blend with original
        alpha = 0.5
        blended = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
        
        # Add label
        blended_img = Image.fromarray(blended)
        draw = ImageDraw.Draw(blended_img)
        
        label = f"{material_type.value.upper()} Confidence"
        if material_type in result.materials:
            mat_conf = result.materials[material_type]
            label += f" | {mat_conf.percentage:.1f}% coverage | {mat_conf.mean_confidence:.2f} avg"
        
        # Draw label background
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle(
            [(10, 10), (20 + text_width, 20 + text_height)],
            fill=(0, 0, 0, 180)
        )
        draw.text((15, 15), label, fill=(255, 255, 255), font=font)
        
        blended_img.save(output_path, quality=95)
        print(f"✓ Saved heatmap: {output_path}")
    
    def _confidence_to_heatmap(self, confidence: np.ndarray) -> np.ndarray:
        """Convert confidence map to RGB heatmap."""
        # Red-yellow-white color scheme
        heatmap = np.zeros((*confidence.shape, 3), dtype=np.uint8)
        
        # Red channel: always high where confidence > 0
        heatmap[..., 0] = (255 * np.clip(confidence * 1.2, 0, 1)).astype(np.uint8)
        
        # Green channel: increases with confidence
        heatmap[..., 1] = (255 * np.clip(confidence * 0.8, 0, 1)).astype(np.uint8)
        
        # Blue channel: minimal
        heatmap[..., 2] = (50 * confidence).astype(np.uint8)
        
        return heatmap
    
    def generate_report(
        self,
        result: MaterialDetectionResult,
        output_path: Path
    ):
        """
        Generate JSON report with material detection statistics.
        
        Args:
            result: Material detection result
            output_path: Path to save JSON report
        """
        report = {
            'image_path': str(result.image_path),
            'timestamp': result.timestamp,
            'processing_time_seconds': result.processing_time,
            'dominant_material': result.dominant_material.value,
            'materials': {}
        }
        
        for material_type, confidence in result.materials.items():
            report['materials'][material_type.value] = {
                'pixel_count': confidence.pixel_count,
                'coverage_percentage': round(confidence.percentage, 2),
                'mean_confidence': round(confidence.mean_confidence, 3),
                'std_confidence': round(confidence.std_confidence, 3),
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved report: {output_path}")
    
    def enhance_with_confidence(
        self,
        image: np.ndarray,
        result: MaterialDetectionResult,
        enhancement_func: callable,
        base_strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply enhancement with strength modulated by material confidence.
        
        Args:
            image: Input image array [0, 1]
            result: Material detection result
            enhancement_func: Function(image, material_type, strength) -> enhanced_image
            base_strength: Base enhancement strength
            
        Returns:
            Enhanced image with confidence-weighted strength
        """
        enhanced = image.copy()
        
        for material_type, confidence_data in result.materials.items():
            if confidence_data.mean_confidence < self.min_confidence:
                continue
            
            # Get confidence map for this material
            confidence_map = result.confidence_maps[material_type]
            
            # Apply enhancement with adaptive strength
            material_enhanced = enhancement_func(
                image, material_type, base_strength * confidence_data.mean_confidence
            )
            
            # Blend based on confidence map
            confidence_3d = confidence_map[..., np.newaxis]
            enhanced = enhanced * (1 - confidence_3d) + material_enhanced * confidence_3d
        
        return enhanced


def main():
    """CLI for material detection."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Material Detection with Confidence Scores"
    )
    parser.add_argument('input', type=Path, help='Input image path')
    parser.add_argument('--output-dir', type=Path, default=Path('output_material_detection'),
                       help='Output directory for results')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                       help='Minimum confidence threshold (0-1)')
    parser.add_argument('--generate-heatmaps', action='store_true',
                       help='Generate heatmap overlays for all materials')
    parser.add_argument('--materials', nargs='+', choices=[m.value for m in MaterialType],
                       help='Specific materials to detect (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = MaterialDetector(min_confidence=args.min_confidence)
    
    print(f"\n{'='*60}")
    print(f"Material Detection Analysis")
    print(f"{'='*60}\n")
    print(f"Input: {args.input}")
    print(f"Min Confidence: {args.min_confidence}")
    
    # Detect materials
    print("\n🔍 Detecting materials...")
    result = detector.detect(args.input)
    
    print(f"✓ Detection complete in {result.processing_time:.2f}s\n")
    
    # Print results
    print(f"Dominant Material: {result.dominant_material.value.upper()}\n")
    print(f"{'Material':<12} {'Coverage':<10} {'Confidence':<12} {'Pixels':<10}")
    print(f"{'-'*50}")
    
    sorted_materials = sorted(
        result.materials.items(),
        key=lambda x: x[1].percentage,
        reverse=True
    )
    
    for material_type, confidence in sorted_materials:
        print(
            f"{material_type.value:<12} "
            f"{confidence.percentage:>6.2f}%   "
            f"{confidence.mean_confidence:>6.3f} ± {confidence.std_confidence:.3f}   "
            f"{confidence.pixel_count:>8}"
        )
    
    # Generate report
    report_path = args.output_dir / f"{args.input.stem}_material_report.json"
    detector.generate_report(result, report_path)
    
    # Generate heatmaps
    if args.generate_heatmaps:
        print(f"\n🎨 Generating heatmaps...")
        materials_to_viz = (
            [MaterialType(m) for m in args.materials]
            if args.materials
            else result.materials.keys()
        )
        
        for material_type in materials_to_viz:
            heatmap_path = args.output_dir / f"{args.input.stem}_{material_type.value}_heatmap.png"
            detector.generate_heatmap(result, material_type, heatmap_path)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
