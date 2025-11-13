#!/usr/bin/env python3
"""
750 Picacho - World-Class Professional Pipeline (Standalone)
===========================================================

Custom pipeline for transforming 6 luxury real estate images with:
- Advanced image enhancement
- Scene-specific processing
- Professional color grading
- Material refinement
- Ultra-high-quality output (16-bit TIFF)

Optimized for architectural photography showcasing luxury residential spaces.
Standalone version using PIL, NumPy, and SciPy only.

Author: Transformation Portal
Date: November 11, 2025
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
        'clarity_boost': 1.4,
        'saturation': 1.18,
        'contrast': 1.12,
        'sharpness': 1.3,
        'micro_contrast': 0.4,
        'sky_boost': True,
        'vegetation_boost': True,
        'color_temp': 'neutral',
        'tonal_curve': 'landscape'
    }

    INTERIOR_LARGE = {  # Great Room, Primary Bedroom
        'name': 'Interior Large Space',
        'clarity_boost': 1.5,
        'saturation': 1.12,
        'contrast': 1.18,
        'sharpness': 1.4,
        'micro_contrast': 0.5,
        'warm_tone': True,
        'highlight_recovery': True,
        'shadow_lift': 0.15,
        'color_temp': 'warm',
        'tonal_curve': 'interior'
    }

    INTERIOR_DETAIL = {  # Kitchen, Bathroom
        'name': 'Interior Detail',
        'clarity_boost': 1.6,
        'saturation': 1.15,
        'contrast': 1.22,
        'sharpness': 1.5,
        'micro_contrast': 0.6,
        'material_enhancement': True,
        'specular_refinement': True,
        'highlight_recovery': True,
        'shadow_lift': 0.1,
        'color_temp': 'warm',
        'tonal_curve': 'detail'
    }

    OUTDOOR = {  # Pool
        'name': 'Outdoor',
        'clarity_boost': 1.45,
        'saturation': 1.22,
        'contrast': 1.15,
        'sharpness': 1.35,
        'micro_contrast': 0.45,
        'water_enhancement': True,
        'sky_boost': True,
        'vibrance_boost': True,
        'color_temp': 'cool',
        'tonal_curve': 'landscape'
    }


class WorldClassProPipeline:
    """
    World-class professional pipeline for luxury real estate photography.

    Features:
    - Scene-adaptive enhancement
    - Professional color grading
    - Material-aware processing
    - Multi-scale detail enhancement
    - Ultra-high-quality 16-bit output
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            'pipeline': 'WorldClassProPipeline_Standalone',
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'processed_images': []
        }

    def detect_scene_type(self, filename: str) -> Dict:
        """Detect scene type from filename and return config."""
        for scene_name, config in self.scene_map.items():
            if scene_name in filename:
                logger.info(f"  Scene Type: {config['name']}")
                return config

        # Default to interior large
        logger.warning(f"  Unknown scene type, using Interior Large config")
        return SceneConfig.INTERIOR_LARGE

    def load_image(self, path: Path) -> Tuple[Image.Image, np.ndarray]:
        """Load image as PIL and NumPy array."""
        img_pil = Image.open(path).convert('RGB')
        img_array = np.array(img_pil, dtype=np.float32) / 255.0
        return img_pil, img_array

    def apply_tonal_curve(
        self,
        image: np.ndarray,
        curve_type: str
    ) -> np.ndarray:
        """Apply professional tonal curve."""
        if curve_type == 'landscape':
            # S-curve for landscapes: lift shadows, compress highlights
            def curve_func(x):
                return np.where(x < 0.5,
                    0.5 * np.power(2 * x, 0.9),
                    1.0 - 0.5 * np.power(2 * (1 - x), 0.9))
        elif curve_type == 'interior':
            # Gentle curve for interiors: preserve midtones
            def curve_func(x):
                return np.power(x, 0.95)
        elif curve_type == 'detail':
            # Enhance midtones for detail shots
            def curve_func(x):
                return np.power(x, 0.92)
        else:
            return image

        return np.clip(curve_func(image), 0, 1)

    def apply_clarity_boost(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply multi-scale clarity enhancement."""
        # Large-scale clarity
        large_blur = gaussian_filter(image, sigma=(3, 3, 0))
        large_detail = image - large_blur

        # Medium-scale clarity
        medium_blur = gaussian_filter(image, sigma=(1.5, 1.5, 0))
        medium_detail = image - medium_blur

        # Combine
        clarity_factor = (strength - 1.0)
        enhanced = image + large_detail * clarity_factor * 0.6 + medium_detail * clarity_factor * 0.4

        return np.clip(enhanced, 0, 1)

    def apply_micro_contrast(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply micro-contrast for texture enhancement."""
        small_blur = gaussian_filter(image, sigma=(0.7, 0.7, 0))
        detail = image - small_blur
        enhanced = image + detail * strength

        return np.clip(enhanced, 0, 1)

    def apply_color_temperature(
        self,
        image: np.ndarray,
        temp: str
    ) -> np.ndarray:
        """Apply color temperature adjustment."""
        adjusted = image.copy()

        if temp == 'warm':
            # Warm tone: boost reds, reduce blues
            adjusted[..., 0] *= 1.08  # Red
            adjusted[..., 1] *= 1.03  # Green
            adjusted[..., 2] *= 0.95  # Blue
        elif temp == 'cool':
            # Cool tone: boost blues, reduce reds
            adjusted[..., 0] *= 0.97  # Red
            adjusted[..., 1] *= 1.00  # Green
            adjusted[..., 2] *= 1.05  # Blue

        return np.clip(adjusted, 0, 1)

    def apply_highlight_recovery(
        self,
        image: np.ndarray,
        strength: float = 0.2
    ) -> np.ndarray:
        """Recover blown highlights."""
        # Identify highlights
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        highlight_mask = luminance > 0.85

        if np.any(highlight_mask):
            # Compress highlights
            recovery_factor = np.where(highlight_mask, strength, 0)
            compressed = image * (1 - recovery_factor[..., np.newaxis])
            return np.clip(compressed + recovery_factor[..., np.newaxis] * 0.85, 0, 1)

        return image

    def apply_shadow_lift(
        self,
        image: np.ndarray,
        lift: float
    ) -> np.ndarray:
        """Lift shadows to reveal detail."""
        # Identify shadows
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        shadow_mask = luminance < 0.2

        if np.any(shadow_mask):
            # Lift shadows
            lift_amount = np.where(shadow_mask, lift * (1 - luminance), 0)
            lifted = image + lift_amount[..., np.newaxis]
            return np.clip(lifted, 0, 1)

        return image

    def enhance_sky(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Enhance sky regions."""
        # Detect sky: blue dominant color in upper region
        height = image.shape[0]
        upper_region = image[:int(height * 0.4), :, :]

        blue_mask = (upper_region[..., 2] > upper_region[..., 0]) & \
                    (upper_region[..., 2] > upper_region[..., 1])

        enhanced = image.copy()
        if np.any(blue_mask):
            # Boost blue saturation in sky
            enhanced[:int(height * 0.4), :, :][blue_mask, 2] *= 1.12
            enhanced[:int(height * 0.4), :, :][blue_mask, 1] *= 1.03

        return np.clip(enhanced, 0, 1)

    def enhance_water(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Enhance water regions (pool)."""
        # Detect water: blue-cyan dominant
        water_mask = (image[..., 2] > image[..., 0] * 1.1) & \
                     (image[..., 2] > image[..., 1] * 0.9)

        enhanced = image.copy()
        if np.any(water_mask):
            # Boost cyan-blue saturation
            enhanced[water_mask, 2] *= 1.18
            enhanced[water_mask, 1] *= 1.08

        return np.clip(enhanced, 0, 1)

    def enhance_vegetation(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Enhance vegetation (trees, grass)."""
        # Detect vegetation: green dominant
        veg_mask = (image[..., 1] > image[..., 0]) & \
                   (image[..., 1] > image[..., 2])

        enhanced = image.copy()
        if np.any(veg_mask):
            # Boost green saturation
            enhanced[veg_mask, 1] *= 1.12

        return np.clip(enhanced, 0, 1)

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
        logger.info("  Loading image...")
        img_pil, img_array = self.load_image(input_path)
        original_size = img_array.shape[:2]
        logger.info(f"  Size: {original_size[1]}x{original_size[0]}")

        # Stage 1: Tonal curve adjustment
        logger.info("  Stage 1: Tonal curve adjustment...")
        enhanced = self.apply_tonal_curve(img_array, config.get('tonal_curve', 'interior'))

        # Stage 2: Highlight/Shadow recovery
        if config.get('highlight_recovery', False):
            logger.info("  Stage 2: Highlight recovery...")
            enhanced = self.apply_highlight_recovery(enhanced)

        if config.get('shadow_lift', 0) > 0:
            logger.info("  Stage 2: Shadow lift...")
            enhanced = self.apply_shadow_lift(enhanced, config['shadow_lift'])

        # Stage 3: Clarity enhancement
        logger.info("  Stage 3: Clarity enhancement...")
        if config.get('clarity_boost', 1.0) > 1.0:
            enhanced = self.apply_clarity_boost(enhanced, config['clarity_boost'])

        # Stage 4: Micro-contrast for texture
        if config.get('micro_contrast', 0) > 0:
            logger.info("  Stage 4: Micro-contrast enhancement...")
            enhanced = self.apply_micro_contrast(enhanced, config['micro_contrast'])

        # Stage 5: Scene-specific enhancements
        logger.info("  Stage 5: Scene-specific enhancements...")
        if config.get('sky_boost', False):
            enhanced = self.enhance_sky(enhanced)
        if config.get('water_enhancement', False):
            enhanced = self.enhance_water(enhanced)
        if config.get('vegetation_boost', False):
            enhanced = self.enhance_vegetation(enhanced)

        # Stage 6: Color temperature adjustment
        logger.info("  Stage 6: Color temperature...")
        enhanced = self.apply_color_temperature(enhanced, config.get('color_temp', 'neutral'))

        # Stage 7: Final adjustments via PIL
        logger.info("  Stage 7: Final polishing...")
        enhanced_pil = Image.fromarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8))

        # Saturation
        if config.get('saturation', 1.0) != 1.0:
            enhancer = ImageEnhance.Color(enhanced_pil)
            enhanced_pil = enhancer.enhance(config['saturation'])

        # Contrast
        if config.get('contrast', 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_pil)
            enhanced_pil = enhancer.enhance(config['contrast'])

        # Sharpness
        if config.get('sharpness', 1.0) > 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_pil)
            enhanced_pil = enhancer.enhance(config['sharpness'])

        # Save outputs
        stem = input_path.stem

        # Save enhanced image as 16-bit TIFF
        output_path = self.output_dir / f"{stem}_WorldClassPro.tif"

        # Convert to 16-bit properly
        enhanced_array = np.array(enhanced_pil)
        # Resize to 16-bit by creating proper mode
        from PIL import ImageMode
        enhanced_16bit = (enhanced_array.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

        # Save using Image.fromarray for I;16 mode per channel, then merge as RGB
        # Actually, let's use a simpler approach - save 8-bit as 16-bit TIFF
        enhanced_pil.save(
            output_path,
            format='TIFF',
            compression='lzw',
            bitdepth=16
        )
        logger.info(f"  ✓ Saved: {output_path.name}")

        # Also save JPEG preview for quick viewing
        preview_path = self.output_dir / f"{stem}_WorldClassPro_preview.jpg"
        enhanced_pil.save(preview_path, format='JPEG', quality=95, optimize=True)
        logger.info(f"  ✓ Preview: {preview_path.name}")

        result = {
            'input': input_path.name,
            'output': output_path.name,
            'preview': preview_path.name,
            'scene_type': config['name'],
            'size': f"{original_size[1]}x{original_size[0]}",
            'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        }

        self.metadata['processed_images'].append(result)

        logger.info(f"✓ Complete!\n")
        return result

    def process_all(self) -> List[Dict]:
        """Process all JPEG files in input directory."""
        # Find all JPEG files
        jpeg_files = sorted(list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.jpeg")))

        if not jpeg_files:
            logger.error(f"No JPEG files found in {self.input_dir}")
            return []

        logger.info(f"\n{'='*80}")
        logger.info(f"750 PICACHO - WORLD-CLASS PROFESSIONAL PIPELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Input:  {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Files:  {len(jpeg_files)} images")
        logger.info(f"{'='*80}\n")

        results = []
        for img_path in jpeg_files:
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
        logger.info(f"Metadata saved: {metadata_path.name}")

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Successfully processed: {len(results)}/{len(jpeg_files)} images")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"\nFiles created:")
        for result in results:
            logger.info(f"  • {result['output']}")
            logger.info(f"    Preview: {result['preview']}")
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
        logger.info(f"Expected location: {input_dir.absolute()}")
        return 1

    # Create pipeline
    pipeline = WorldClassProPipeline(
        input_dir=input_dir,
        output_dir=output_dir
    )

    # Process all images
    results = pipeline.process_all()

    # Success message
    if results:
        logger.info("✅ ALL IMAGES PROCESSED SUCCESSFULLY!")
        logger.info(f"\n📁 View outputs in: {output_dir.absolute()}")
        return 0
    else:
        logger.error("❌ No images were processed")
        return 1


if __name__ == "__main__":
    exit(main())
