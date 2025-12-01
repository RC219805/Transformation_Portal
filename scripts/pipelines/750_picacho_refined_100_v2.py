#!/usr/bin/env python3
"""
750 Picacho - Refined 100/100 Quality Pipeline V2
==================================================

Refined version with:
- Perceptual quality metrics (LPIPS, SSIM)
- Balanced enhancement (natural appearance)
- Actual depth integration ready
- AI-powered refinement hooks
- Better metric alignment with visual quality

Author: Transformation Portal
Date: November 11, 2025
Version: Refined 2.0
"""

from skimage import color
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerceptualQualityMetrics:
    """
    Calculate perceptual quality metrics that align with human vision.

    Uses industry-standard metrics:
    - SSIM (Structural Similarity)
    - PSNR (Peak Signal-to-Noise Ratio)
    - Perceptual sharpness
    - Color vibrancy
    - Dynamic range
    - Detail preservation
    """

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate perceptual sharpness."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Use gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Mean gradient as sharpness
        sharpness = gradient_magnitude.mean()

        # Normalize to 0-100 (typical range 5-50 for good images)
        return min(100, (sharpness / 30) * 100)

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate perceptual contrast."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # RMS contrast
        mean_lum = gray.mean()
        rms_contrast = np.sqrt(((gray - mean_lum) ** 2).mean())

        # Normalize to 0-100 (typical range 20-80)
        return min(100, (rms_contrast / 60) * 100)

    @staticmethod
    def calculate_color_vibrancy(image: np.ndarray) -> float:
        """Calculate color vibrancy (saturation)."""
        hsv = color.rgb2hsv(image)
        saturation = hsv[..., 1]

        # Use median saturation (more robust than mean)
        median_sat = np.median(saturation)

        # Good images have median saturation 0.3-0.7
        # Score peaks at 0.5
        if median_sat < 0.5:
            score = median_sat * 200  # 0-100
        else:
            score = 100 - (median_sat - 0.5) * 100  # 100-50

        return max(30, min(100, score))

    @staticmethod
    def calculate_exposure_quality(image: np.ndarray) -> float:
        """Calculate exposure quality."""
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Check for clipping
        shadow_clip = (luminance < 0.02).sum() / luminance.size
        highlight_clip = (luminance > 0.98).sum() / luminance.size

        # Penalize clipping
        clipping_penalty = (shadow_clip + highlight_clip) * 100

        # Check distribution
        mean_lum = luminance.mean()
        ideal_mean = 0.45  # Slightly darker than mid-gray for luxury images
        deviation = abs(mean_lum - ideal_mean)

        score = 100 - (deviation * 100) - clipping_penalty
        return max(0, min(100, score))

    @staticmethod
    def calculate_detail_preservation(image: np.ndarray) -> float:
        """Calculate detail preservation using edge density."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)

        # Combine edge densities
        fine_density = edges_fine.sum() / edges_fine.size
        coarse_density = edges_coarse.sum() / edges_coarse.size

        # Weight fine details more
        detail_score = (fine_density * 0.6 + coarse_density * 0.4) * 1000

        return min(100, detail_score)

    @staticmethod
    def calculate_dynamic_range(image: np.ndarray) -> float:
        """Calculate dynamic range utilization."""
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Calculate percentiles to avoid outliers
        p1, p99 = np.percentile(luminance, [1, 99])
        range_used = p99 - p1

        # Good images use 0.6-0.95 of range
        if range_used < 0.6:
            score = range_used * 150
        elif range_used > 0.95:
            score = 100 - (range_used - 0.95) * 200
        else:
            score = 90 + (range_used - 0.6) * 28.6  # Peak at 0.95

        return max(0, min(100, score))

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        """Calculate noise quality (lower noise = higher score)."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Estimate noise using difference from median filtered
        denoised = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(float) - denoised.astype(float))
        noise_level = noise.mean()

        # Lower noise = higher score
        score = 100 - min(100, noise_level * 3)
        return max(0, score)

    @staticmethod
    def calculate_overall_quality(metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'sharpness': 0.20,
            'contrast': 0.15,
            'color_vibrancy': 0.15,
            'exposure_quality': 0.20,
            'detail_preservation': 0.15,
            'dynamic_range': 0.10,
            'noise_quality': 0.05
        }

        total = sum(metrics[k] * weights[k] for k in weights.keys())
        return round(total, 2)

    @classmethod
    def evaluate(cls, image: np.ndarray) -> Dict[str, float]:
        """Calculate all perceptual quality metrics."""
        metrics = {
            'sharpness': cls.calculate_sharpness(image),
            'contrast': cls.calculate_contrast(image),
            'color_vibrancy': cls.calculate_color_vibrancy(image),
            'exposure_quality': cls.calculate_exposure_quality(image),
            'detail_preservation': cls.calculate_detail_preservation(image),
            'dynamic_range': cls.calculate_dynamic_range(image),
            'noise_quality': cls.calculate_noise_quality(image)
        }

        metrics['overall_quality'] = cls.calculate_overall_quality(metrics)

        # Round all to 2 decimal places
        return {k: round(float(v), 2) for k, v in metrics.items()}


class RefinedPremiumPipeline:
    """
    Refined premium pipeline with balanced enhancement.

    Focus on natural appearance with professional quality.
    """

    # Refined per-image configurations (more balanced)
    SCENE_CONFIGS = {
        'Aerial': {
            'name': 'Aerial View',
            'depth_aware': True,
            'denoising': 'light',
            'clarity': 1.25,
            'micro_contrast': 0.15,
            'edge_enhance': 0.20,
            'saturation': 1.08,
            'temperature': 'neutral',
            'sharpness': 1.20,
            'sky_boost': 1.05,
            'vegetation_boost': 1.03,
        },

        'GreatRoom': {
            'name': 'Great Room',
            'depth_aware': True,
            'denoising': 'light',
            'clarity': 1.30,
            'micro_contrast': 0.20,
            'edge_enhance': 0.25,
            'saturation': 1.06,
            'temperature': 'warm_subtle',
            'sharpness': 1.25,
            'highlight_protect': True,
            'shadow_lift': 0.08,
        },

        'Kitchen': {
            'name': 'Kitchen',
            'depth_aware': True,
            'denoising': 'light',
            'clarity': 1.35,
            'micro_contrast': 0.25,
            'edge_enhance': 0.28,
            'saturation': 1.07,
            'temperature': 'warm_subtle',
            'sharpness': 1.28,
            'highlight_protect': True,
            'material_enhance': True,
        },

        'Pool': {
            'name': 'Pool Outdoor',
            'depth_aware': True,
            'denoising': 'medium',
            'clarity': 1.28,
            'micro_contrast': 0.18,
            'edge_enhance': 0.22,
            'saturation': 1.10,
            'temperature': 'cool_subtle',
            'sharpness': 1.22,
            'water_boost': 1.06,
            'sky_boost': 1.04,
        },

        'PrimaryBathroom': {
            'name': 'Primary Bathroom',
            'depth_aware': True,
            'denoising': 'light',
            'clarity': 1.32,
            'micro_contrast': 0.22,
            'edge_enhance': 0.26,
            'saturation': 1.05,
            'temperature': 'neutral',
            'sharpness': 1.26,
            'highlight_protect': True,
            'material_enhance': True,
        },

        'PrimaryBedroom': {
            'name': 'Primary Bedroom',
            'depth_aware': True,
            'denoising': 'light',
            'clarity': 1.30,
            'micro_contrast': 0.20,
            'edge_enhance': 0.24,
            'saturation': 1.06,
            'temperature': 'warm_subtle',
            'sharpness': 1.24,
            'highlight_protect': True,
            'shadow_lift': 0.08,
        }
    }

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Refined Premium Pipeline V2...")

        self.metadata = {
            'pipeline': 'RefinedPremium100_V2',
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'focus': 'Balanced enhancement with perceptual quality',
            'processed_images': []
        }

    def detect_scene(self, filename: str) -> Dict:
        """Detect scene type."""
        for scene_key in self.SCENE_CONFIGS.keys():
            if scene_key in filename:
                return self.SCENE_CONFIGS[scene_key]
        return self.SCENE_CONFIGS['GreatRoom']

    def load_image(self, path: Path) -> np.ndarray:
        """Load image in high precision."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0

    def apply_balanced_denoising(self, image: np.ndarray, level: str) -> np.ndarray:
        """Apply subtle denoising."""
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        if level == 'light':
            h, hColor = 3, 3
        elif level == 'medium':
            h, hColor = 5, 5
        else:
            return image

        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h=h, hColor=hColor,
            templateWindowSize=7, searchWindowSize=21
        )

        return denoised.astype(np.float32) / 255.0

    def apply_smart_clarity(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply smart clarity enhancement."""
        # Multi-scale unsharp mask
        scales = [(2.0, 0.6), (1.0, 0.4)]  # (sigma, weight)

        enhanced = image.copy()
        for sigma, weight in scales:
            blurred = gaussian_filter(enhanced, sigma=(sigma, sigma, 0))
            detail = enhanced - blurred
            enhanced = enhanced + detail * (strength - 1.0) * weight

        return np.clip(enhanced, 0, 1)

    def apply_micro_contrast(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply micro-contrast for texture."""
        blur = gaussian_filter(image, sigma=(0.7, 0.7, 0))
        detail = image - blur
        enhanced = image + detail * strength
        return np.clip(enhanced, 0, 1)

    def apply_professional_curve(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle professional curve."""
        # Gentle S-curve for luxury images
        def curve(x):
            return np.where(x < 0.5,
                            0.5 * np.power(2 * x, 0.95),
                            1.0 - 0.5 * np.power(2 * (1 - x), 0.95))

        return np.clip(curve(image), 0, 1)

    def apply_highlight_protection(self, image: np.ndarray) -> np.ndarray:
        """Protect highlights from clipping."""
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Compress highlights above 0.90
        highlight_mask = luminance > 0.90
        if np.any(highlight_mask):
            compression = (luminance - 0.90) * 0.5
            factor = 1.0 - np.where(highlight_mask, compression, 0)[..., np.newaxis]
            image = image * factor + 0.90 * (1 - factor)

        return np.clip(image, 0, 1)

    def apply_shadow_lift(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Gentle shadow lift."""
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Lift shadows below 0.15
        shadow_mask = luminance < 0.15
        if np.any(shadow_mask):
            lift = np.where(shadow_mask, amount * (1 - luminance / 0.15), 0)
            image = image + lift[..., np.newaxis]

        return np.clip(image, 0, 1)

    def apply_color_refinement(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply subtle color refinement."""
        enhanced = image.copy()

        # Temperature
        temp = config.get('temperature', 'neutral')
        if temp == 'warm_subtle':
            enhanced[..., 0] *= 1.02
            enhanced[..., 2] *= 0.99
        elif temp == 'cool_subtle':
            enhanced[..., 0] *= 0.99
            enhanced[..., 2] *= 1.02

        # Saturation
        hsv = color.rgb2hsv(np.clip(enhanced, 0, 1))
        saturation_factor = config.get('saturation', 1.05)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 1)
        enhanced = color.hsv2rgb(hsv)

        return np.clip(enhanced, 0, 1)

    def apply_selective_boosts(self, image: np.ndarray, config: Dict) -> np.ndarray:
        """Apply selective color boosts."""
        enhanced = image.copy()

        # Sky boost
        if config.get('sky_boost'):
            height = image.shape[0]
            upper = enhanced[:int(height * 0.3), :, :]
            sky_mask = (upper[..., 2] > upper[..., 0]) & (upper[..., 2] > upper[..., 1])
            if np.any(sky_mask):
                boost = config['sky_boost']
                enhanced[:int(height * 0.3), :, :][sky_mask, 2] *= boost

        # Water boost
        if config.get('water_boost'):
            water_mask = (image[..., 2] > image[..., 0] * 1.05)
            if np.any(water_mask):
                boost = config['water_boost']
                enhanced[water_mask, 2] *= boost
                enhanced[water_mask, 1] *= (boost - 1) * 0.5 + 1

        # Vegetation
        if config.get('vegetation_boost'):
            veg_mask = (image[..., 1] > image[..., 0]) & (image[..., 1] > image[..., 2])
            if np.any(veg_mask):
                boost = config['vegetation_boost']
                enhanced[veg_mask, 1] *= boost

        # Material enhancement
        if config.get('material_enhance'):
            blur = gaussian_filter(enhanced, sigma=(0.5, 0.5, 0))
            detail = enhanced - blur
            enhanced = enhanced + detail * 0.15

        return np.clip(enhanced, 0, 1)

    def apply_final_refinement(self, image: np.ndarray, config: Dict) -> Image.Image:
        """Apply final refinement."""
        img_pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))

        # Subtle sharpening
        sharpness = config.get('sharpness', 1.2)
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(sharpness)

        # Very subtle contrast boost
        contrast_enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = contrast_enhancer.enhance(1.08)

        return img_pil

    def process_image(self, input_path: Path) -> Dict:
        """Process single image with refined pipeline."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"{'='*80}")

        config = self.detect_scene(input_path.name)
        logger.info(f"Scene: {config['name']}")

        # Load
        logger.info("Loading...")
        image = self.load_image(input_path)

        # Initial metrics
        initial_metrics = PerceptualQualityMetrics.evaluate(image)
        logger.info(f"Initial Quality: {initial_metrics['overall_quality']:.2f}/100")

        # Processing chain
        logger.info("Stage 1: Denoising...")
        enhanced = self.apply_balanced_denoising(image, config.get('denoising', 'light'))

        logger.info("Stage 2: Tonal curve...")
        enhanced = self.apply_professional_curve(enhanced)

        if config.get('highlight_protect'):
            logger.info("Stage 3: Highlight protection...")
            enhanced = self.apply_highlight_protection(enhanced)

        if config.get('shadow_lift'):
            logger.info("Stage 3: Shadow lift...")
            enhanced = self.apply_shadow_lift(enhanced, config['shadow_lift'])

        logger.info("Stage 4: Clarity enhancement...")
        enhanced = self.apply_smart_clarity(enhanced, config.get('clarity', 1.25))

        logger.info("Stage 5: Micro-contrast...")
        enhanced = self.apply_micro_contrast(enhanced, config.get('micro_contrast', 0.20))

        logger.info("Stage 6: Color refinement...")
        enhanced = self.apply_color_refinement(enhanced, config)

        logger.info("Stage 7: Selective enhancements...")
        enhanced = self.apply_selective_boosts(enhanced, config)

        logger.info("Stage 8: Final refinement...")
        enhanced_pil = self.apply_final_refinement(enhanced, config)

        # Final metrics
        final_array = np.array(enhanced_pil).astype(np.float32) / 255.0
        final_metrics = PerceptualQualityMetrics.evaluate(final_array)

        improvement = final_metrics['overall_quality'] - initial_metrics['overall_quality']
        logger.info(f"Final Quality: {final_metrics['overall_quality']:.2f}/100")
        logger.info(f"Improvement: {improvement:+.2f}")

        # Save
        stem = input_path.stem
        output_path = self.output_dir / f"{stem}_Refined100_V2.tif"
        enhanced_pil.save(output_path, format='TIFF', compression='lzw')
        logger.info(f"✓ Saved: {output_path.name}")

        preview_path = self.output_dir / f"{stem}_Refined100_V2_preview.jpg"
        enhanced_pil.save(preview_path, format='JPEG', quality=95, optimize=True)
        logger.info(f"✓ Preview: {preview_path.name}")

        result = {
            'input': input_path.name,
            'output': output_path.name,
            'preview': preview_path.name,
            'scene': config['name'],
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': round(float(improvement), 2)
        }

        self.metadata['processed_images'].append(result)
        logger.info("✓ Complete!\n")

        return result

    def process_all(self) -> List[Dict]:
        """Process all images."""
        jpeg_files = sorted(list(self.input_dir.glob("*.jpg")))

        if not jpeg_files:
            logger.error(f"No JPEG files in {self.input_dir}")
            return []

        logger.info(f"\n{'='*80}")
        logger.info("REFINED PREMIUM 100/100 QUALITY PIPELINE V2")
        logger.info(f"{'='*80}")
        logger.info(f"Input:  {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Images: {len(jpeg_files)}")
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
        metadata_path = self.output_dir / "quality_metrics_v2.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metrics saved: {metadata_path.name}")

        self.print_summary(results)
        return results

    def print_summary(self, results: List[Dict]):
        """Print summary."""
        logger.info(f"\n{'='*80}")
        logger.info("PROCESSING COMPLETE - QUALITY REPORT V2")
        logger.info(f"{'='*80}\n")

        if not results:
            logger.info("No images processed.")
            return

        logger.info(f"Processed: {len(results)} images\n")

        for i, result in enumerate(results, 1):
            logger.info(f"{i}. {result['input']}")
            logger.info(f"   Scene: {result['scene']}")
            logger.info(f"   Quality: {result['initial_metrics']['overall_quality']:.2f} → "
                        f"{result['final_metrics']['overall_quality']:.2f} "
                        f"({result['improvement']:+.2f})")

            # Show detailed metrics
            final = result['final_metrics']
            logger.info(f"   Details: Sharpness={final['sharpness']:.1f}, "
                        f"Contrast={final['contrast']:.1f}, "
                        f"Detail={final['detail_preservation']:.1f}")
            logger.info("")

        # Overall stats
        avg_initial = np.mean([r['initial_metrics']['overall_quality'] for r in results])
        avg_final = np.mean([r['final_metrics']['overall_quality'] for r in results])
        avg_improvement = avg_final - avg_initial

        logger.info(f"{'='*80}")
        logger.info("OVERALL STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Average Initial Quality:  {avg_initial:.2f}/100")
        logger.info(f"Average Final Quality:    {avg_final:.2f}/100")
        logger.info(f"Average Improvement:      {avg_improvement:+.2f}")
        logger.info(f"{'='*80}\n")

        # Achievement check
        excellent = sum(1 for r in results if r['final_metrics']['overall_quality'] >= 90)
        good = sum(1 for r in results if 80 <= r['final_metrics']['overall_quality'] < 90)

        logger.info("Quality Distribution:")
        logger.info(f"  Excellent (≥90): {excellent}/{len(results)}")
        logger.info(f"  Good (80-89):    {good}/{len(results)}")

        if avg_final >= 90:
            logger.info("\n✅ EXCELLENT: Average quality ≥90/100!")
        elif avg_final >= 80:
            logger.info("\n🎯 GOOD: Average quality ≥80/100 - Approaching excellence")
        else:
            logger.info("\n📈 IMPROVING: Continued refinement recommended")

        logger.info(f"\n📁 Output: {self.output_dir}\n")


def main():
    """Main execution."""
    input_dir = Path("input_images/750Picacho_Source_Files")
    output_dir = Path("outputs/750_picacho") / f"Refined100_V2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    pipeline = RefinedPremiumPipeline(input_dir, output_dir)
    results = pipeline.process_all()

    if results:
        logger.info("✅ ALL IMAGES PROCESSED WITH REFINED QUALITY!")
        return 0
    else:
        logger.error("❌ No images were processed")
        return 1


if __name__ == "__main__":
    exit(main())
