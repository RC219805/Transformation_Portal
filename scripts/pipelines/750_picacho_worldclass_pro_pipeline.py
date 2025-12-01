#!/usr/bin/env python3
"""
750 Picacho - World-Class Professional Pipeline
===============================================

Custom pipeline for transforming 6 luxury real estate images with:
- Scene-specific depth processing
- Adaptive material enhancement
- Atmospheric refinement
- Professional color grading
- Ultra-high-quality output (16-bit TIFF)

Optimized for architectural photography showcasing luxury residential spaces.

Author: Transformation Portal
Date: November 11, 2025
"""

from src.transformation_portal.io.tiff_handler import TIFFHandler
from src.transformation_portal.depth.processors.zone_tone_mapping import ZoneToneMapper
from src.transformation_portal.depth.processors.atmospheric_effects import AtmosphericEffects
from src.transformation_portal.depth.models.depth_anything_v2 import DepthAnythingV2Model
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Import core pipeline components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SceneConfig:
    """Scene-specific configuration for optimal processing."""

    AERIAL = {
        'name': 'Aerial',
        'depth_strength': 0.7,
        'atmospheric_strength': 0.8,
        'clarity_boost': 1.3,
        'saturation': 1.15,
        'contrast': 1.1,
        'focus_zones': ['middle', 'far'],
        'sky_enhancement': True,
        'vegetation_boost': True
    }

    INTERIOR_LARGE = {  # Great Room, Primary Bedroom
        'name': 'Interior Large',
        'depth_strength': 0.85,
        'atmospheric_strength': 0.4,
        'clarity_boost': 1.4,
        'saturation': 1.1,
        'contrast': 1.15,
        'focus_zones': ['near', 'middle'],
        'warm_tone': True,
        'window_recovery': True
    }

    INTERIOR_DETAIL = {  # Kitchen, Bathroom
        'name': 'Interior Detail',
        'depth_strength': 0.9,
        'atmospheric_strength': 0.3,
        'clarity_boost': 1.5,
        'saturation': 1.12,
        'contrast': 1.2,
        'focus_zones': ['near'],
        'material_enhancement': True,
        'specular_refinement': True
    }

    OUTDOOR = {  # Pool
        'name': 'Outdoor',
        'depth_strength': 0.75,
        'atmospheric_strength': 0.6,
        'clarity_boost': 1.35,
        'saturation': 1.18,
        'contrast': 1.12,
        'focus_zones': ['near', 'middle'],
        'water_enhancement': True,
        'sky_enhancement': True
    }


class WorldClassProPipeline:
    """
    World-class professional pipeline for luxury real estate photography.

    Features:
    - Adaptive depth processing per scene type
    - Material-aware enhancement
    - Professional color grading
    - Atmospheric refinement
    - Ultra-high-quality 16-bit output
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model_size: str = 'large'
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize depth model
        logger.info("Initializing Depth Anything V2 model...")
        self.depth_model = DepthAnythingV2Model(encoder=model_size)

        # Initialize processors
        self.atmospheric = AtmosphericEffects()
        self.zone_mapper = ZoneToneMapper()
        self.tiff_handler = TIFFHandler()

        # Scene mapping
        self.scene_map = {
            'Aerial': SceneConfig.AERIAL,
            'GreatRoom': SceneConfig.INTERIOR_LARGE,
            'Kitchen': SceneConfig.INTERIOR_DETAIL,
            'Pool': SceneConfig.OUTDOOR,
            'PrimaryBathroom': SceneConfig.INTERIOR_DETAIL,
            'PrimaryBedroom': SceneConfig.INTERIOR_LARGE
        }

        # Processing metadata
        self.metadata = {
            'pipeline': 'WorldClassProPipeline',
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'model': f'DepthAnythingV2-{model_size}',
            'processed_images': []
        }

    def detect_scene_type(self, filename: str) -> Dict:
        """Detect scene type from filename and return config."""
        for scene_name, config in self.scene_map.items():
            if scene_name in filename:
                logger.info(f"Detected scene: {config['name']} for {filename}")
                return config

        # Default to interior large
        logger.warning(f"Unknown scene type for {filename}, using Interior Large config")
        return SceneConfig.INTERIOR_LARGE

    def load_image(self, path: Path) -> np.ndarray:
        """Load and prepare image for processing."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate depth map with caching."""
        logger.info("Generating depth map...")
        depth = self.depth_model.infer(image)
        return depth

    def apply_depth_aware_enhancement(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Apply depth-aware enhancement based on scene config."""
        enhanced = image.copy()

        # Depth-aware clarity enhancement
        if config.get('clarity_boost', 1.0) > 1.0:
            clarity_strength = config['clarity_boost']
            enhanced = self._apply_clarity_boost(enhanced, depth, clarity_strength)

        # Atmospheric effects for depth
        if config.get('atmospheric_strength', 0) > 0:
            atmo_strength = config['atmospheric_strength']
            enhanced = self.atmospheric.apply(
                enhanced,
                depth,
                strength=atmo_strength
            )

        return enhanced

    def _apply_clarity_boost(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply depth-guided clarity enhancement."""
        from scipy.ndimage import gaussian_filter

        # Multi-scale unsharp mask
        blur_sigma = 2.0
        blurred = gaussian_filter(image, sigma=(blur_sigma, blur_sigma, 0))
        detail = image - blurred

        # Depth-weighted enhancement (focus on near/middle zones)
        depth_weight = 1.0 - (depth ** 2)  # Boost near content
        depth_weight = np.clip(depth_weight, 0.3, 1.0)

        # Apply weighted clarity boost
        enhanced = image + detail * (strength - 1.0) * depth_weight[..., np.newaxis]
        return np.clip(enhanced, 0, 1)

    def apply_zone_tone_mapping(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Apply zone-based tone mapping."""
        return self.zone_mapper.apply(
            image,
            depth,
            zones=config.get('focus_zones', ['near', 'middle'])
        )

    def apply_color_grading(
        self,
        image: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Apply professional color grading."""
        graded = image.copy()

        # Saturation adjustment
        saturation = config.get('saturation', 1.1)
        if saturation != 1.0:
            # Convert to HSV for saturation adjustment
            from skimage import color
            hsv = color.rgb2hsv(graded)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
            graded = color.hsv2rgb(hsv)

        # Contrast adjustment
        contrast = config.get('contrast', 1.1)
        if contrast != 1.0:
            graded = self._apply_contrast(graded, contrast)

        # Warm tone for interiors
        if config.get('warm_tone', False):
            graded = self._apply_warm_tone(graded, strength=0.15)

        return graded

    def _apply_contrast(
        self,
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """Apply contrast adjustment."""
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 1)

    def _apply_warm_tone(
        self,
        image: np.ndarray,
        strength: float = 0.15
    ) -> np.ndarray:
        """Apply subtle warm tone for luxury interiors."""
        warm = image.copy()
        warm[..., 0] = np.clip(warm[..., 0] * (1 + strength * 0.5), 0, 1)  # Boost red
        warm[..., 1] = np.clip(warm[..., 1] * (1 + strength * 0.2), 0, 1)  # Slight green boost
        warm[..., 2] = np.clip(warm[..., 2] * (1 - strength * 0.1), 0, 1)  # Reduce blue
        return warm

    def apply_scene_specific_enhancements(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Dict
    ) -> np.ndarray:
        """Apply scene-specific enhancements."""
        enhanced = image.copy()

        # Sky enhancement for aerial/outdoor
        if config.get('sky_enhancement', False):
            enhanced = self._enhance_sky(enhanced, depth)

        # Water enhancement for pool
        if config.get('water_enhancement', False):
            enhanced = self._enhance_water(enhanced, depth)

        # Material enhancement for detail shots
        if config.get('material_enhancement', False):
            enhanced = self._enhance_materials(enhanced)

        return enhanced

    def _enhance_sky(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """Enhance sky regions (far depth, blue hue)."""
        # Detect sky: far depth + blue color
        sky_mask = depth > 0.7
        blue_channel = image[..., 2]
        sky_mask = sky_mask & (blue_channel > image[..., 0]) & (blue_channel > image[..., 1])

        enhanced = image.copy()
        if np.any(sky_mask):
            # Boost blue saturation and slight brightness
            enhanced[sky_mask, 2] = np.clip(enhanced[sky_mask, 2] * 1.1, 0, 1)

        return enhanced

    def _enhance_water(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """Enhance water regions (pool)."""
        # Detect water: blue-cyan hue
        water_mask = (image[..., 2] > image[..., 0]) & (image[..., 2] > image[..., 1] * 0.9)

        enhanced = image.copy()
        if np.any(water_mask):
            # Boost blue-cyan saturation
            enhanced[water_mask, 2] = np.clip(enhanced[water_mask, 2] * 1.15, 0, 1)
            enhanced[water_mask, 1] = np.clip(enhanced[water_mask, 1] * 1.05, 0, 1)

        return enhanced

    def _enhance_materials(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Enhance material details (surfaces, textures)."""
        from scipy.ndimage import gaussian_filter

        # Micro-contrast enhancement
        blur = gaussian_filter(image, sigma=(0.5, 0.5, 0))
        detail = image - blur
        enhanced = image + detail * 0.3

        return np.clip(enhanced, 0, 1)

    def save_output(
        self,
        image: np.ndarray,
        output_path: Path,
        metadata: Dict
    ):
        """Save output as 16-bit TIFF with metadata."""
        logger.info(f"Saving 16-bit TIFF: {output_path.name}")

        # Convert to 16-bit
        image_16bit = np.clip(image * 65535, 0, 65535).astype(np.uint16)

        # Save with PIL
        img_pil = Image.fromarray(image_16bit, mode='RGB')

        # Add metadata to EXIF
        from PIL import TiffImagePlugin
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[270] = json.dumps(metadata)  # ImageDescription tag

        img_pil.save(
            output_path,
            format='TIFF',
            compression='lzw',
            tiffinfo=info
        )

    def process_image(
        self,
        input_path: Path
    ) -> Dict:
        """Process a single image through the full pipeline."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"{'='*80}")

        # Detect scene configuration
        config = self.detect_scene_type(input_path.name)

        # Load image
        logger.info("Loading image...")
        image = self.load_image(input_path)
        original_size = image.shape[:2]

        # Generate depth map
        depth = self.estimate_depth(image)

        # Stage 1: Depth-aware enhancement
        logger.info("Stage 1: Depth-aware enhancement...")
        enhanced = self.apply_depth_aware_enhancement(image, depth, config)

        # Stage 2: Zone tone mapping
        logger.info("Stage 2: Zone-based tone mapping...")
        enhanced = self.apply_zone_tone_mapping(enhanced, depth, config)

        # Stage 3: Scene-specific enhancements
        logger.info("Stage 3: Scene-specific enhancements...")
        enhanced = self.apply_scene_specific_enhancements(enhanced, depth, config)

        # Stage 4: Professional color grading
        logger.info("Stage 4: Professional color grading...")
        enhanced = self.apply_color_grading(enhanced, config)

        # Save outputs
        stem = input_path.stem

        # Save enhanced image
        output_path = self.output_dir / f"{stem}_WorldClassPro.tif"
        image_metadata = {
            'source': input_path.name,
            'scene_type': config['name'],
            'size': f"{original_size[1]}x{original_size[0]}",
            'processing_stages': [
                'depth_aware_enhancement',
                'zone_tone_mapping',
                'scene_specific_enhancements',
                'professional_color_grading'
            ],
            'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        }
        self.save_output(enhanced, output_path, image_metadata)

        # Save depth map for reference
        depth_path = self.output_dir / f"{stem}_DepthMap.tif"
        depth_16bit = np.clip(depth * 65535, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_16bit, mode='I;16').save(depth_path, compression='lzw')
        logger.info(f"Saved depth map: {depth_path.name}")

        result = {
            'input': input_path.name,
            'output': output_path.name,
            'depth_map': depth_path.name,
            'scene_type': config['name'],
            'size': original_size
        }

        self.metadata['processed_images'].append(result)

        logger.info(f"✓ Complete: {output_path.name}\n")
        return result

    def process_all(self) -> List[Dict]:
        """Process all JPEG files in input directory."""
        # Find all JPEG files
        jpeg_files = sorted(self.input_dir.glob("*.jpg")) + sorted(self.input_dir.glob("*.jpeg"))

        if not jpeg_files:
            logger.error(f"No JPEG files found in {self.input_dir}")
            return []

        logger.info(f"\n{'='*80}")
        logger.info("750 Picacho - World-Class Professional Pipeline")
        logger.info(f"{'='*80}")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Files to process: {len(jpeg_files)}")
        logger.info(f"{'='*80}\n")

        results = []
        for img_path in tqdm(jpeg_files, desc="Processing images"):
            try:
                result = self.process_image(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                import traceback
                traceback.print_exc()

        # Save metadata
        metadata_path = self.output_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"\nMetadata saved: {metadata_path}")

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Processed: {len(results)}/{len(jpeg_files)} images")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*80}\n")

        return results


def main():
    """Main execution."""
    # Configuration
    input_dir = Path("input_images/750Picacho_Source_Files")
    output_dir = Path("outputs/750_picacho") / f"WorldClassPro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Verify input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Create pipeline
    pipeline = WorldClassProPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_size='large'  # Use large model for best quality
    )

    # Process all images
    results = pipeline.process_all()

    # Success message
    if results:
        logger.info("✓ All images processed successfully!")
        logger.info(f"\nView outputs in: {output_dir}")
    else:
        logger.error("✗ No images were processed")
        sys.exit(1)


if __name__ == "__main__":
    main()
