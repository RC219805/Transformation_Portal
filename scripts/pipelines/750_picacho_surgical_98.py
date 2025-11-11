#!/usr/bin/env python3
"""
750 Picacho - Surgical Per-Image Refinement for 98+ Quality
============================================================

Ultra-conservative, surgical approach to refine already-excellent images.

Strategy:
- Analyze each image individually
- Apply ONLY beneficial adjustments
- Target 98+ quality from 88-95 baseline
- Preserve natural appearance
- Minimal intervention philosophy

Author: Transformation Portal
Date: November 11, 2025
Version: Surgical 1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from scipy.ndimage import gaussian_filter
from skimage import color

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SurgicalQualityMetrics:
    """Ultra-precise quality metrics for surgical refinement."""

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate perceptual sharpness."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        sharpness = gradient.mean()
        return min(100, (sharpness / 30) * 100)

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate RMS contrast."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mean_lum = gray.mean()
        rms_contrast = np.sqrt(((gray - mean_lum) ** 2).mean())
        return min(100, (rms_contrast / 60) * 100)

    @staticmethod
    def calculate_color_vibrancy(image: np.ndarray) -> float:
        """Calculate color vibrancy."""
        hsv = color.rgb2hsv(image)
        median_sat = np.median(hsv[..., 1])
        if median_sat < 0.5:
            score = median_sat * 200
        else:
            score = 100 - (median_sat - 0.5) * 100
        return max(30, min(100, score))

    @staticmethod
    def calculate_exposure_quality(image: np.ndarray) -> float:
        """Calculate exposure quality."""
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        shadow_clip = (lum < 0.02).sum() / lum.size
        highlight_clip = (lum > 0.98).sum() / lum.size
        clipping_penalty = (shadow_clip + highlight_clip) * 100
        mean_lum = lum.mean()
        ideal_mean = 0.45
        deviation = abs(mean_lum - ideal_mean)
        score = 100 - (deviation * 100) - clipping_penalty
        return max(0, min(100, score))

    @staticmethod
    def calculate_detail_preservation(image: np.ndarray) -> float:
        """Calculate detail preservation."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)
        fine_density = edges_fine.sum() / edges_fine.size
        coarse_density = edges_coarse.sum() / edges_coarse.size
        detail_score = (fine_density * 0.6 + coarse_density * 0.4) * 1000
        return min(100, detail_score)

    @staticmethod
    def calculate_dynamic_range(image: np.ndarray) -> float:
        """Calculate dynamic range."""
        lum = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        p1, p99 = np.percentile(lum, [1, 99])
        range_used = p99 - p1
        if range_used < 0.6:
            score = range_used * 150
        elif range_used > 0.95:
            score = 100 - (range_used - 0.95) * 200
        else:
            score = 90 + (range_used - 0.6) * 28.6
        return max(0, min(100, score))

    @staticmethod
    def calculate_noise_quality(image: np.ndarray) -> float:
        """Calculate noise quality."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(float) - denoised.astype(float))
        noise_level = noise.mean()
        score = 100 - min(100, noise_level * 3)
        return max(0, score)

    @staticmethod
    def calculate_overall_quality(metrics: Dict[str, float]) -> float:
        """Calculate overall quality."""
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
        """Calculate all metrics."""
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
        return {k: round(float(v), 2) for k, v in metrics.items()}

    @classmethod
    def analyze_weaknesses(cls, metrics: Dict[str, float]) -> List[Tuple[str, float]]:
        """Identify areas below 95 that could be improved."""
        weaknesses = []
        for key, value in metrics.items():
            if key != 'overall_quality' and value < 95:
                weaknesses.append((key, value))
        weaknesses.sort(key=lambda x: x[1])  # Sort by score (lowest first)
        return weaknesses


class SurgicalRefinement:
    """Surgical refinement - minimal intervention for maximum quality."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized Surgical Refinement Pipeline")

    def load_image(self, path: Path) -> np.ndarray:
        """Load image."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32) / 255.0

    def apply_micro_sharpening(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply extremely subtle sharpening."""
        if strength <= 1.0:
            return image

        # Very subtle unsharp mask
        blurred = gaussian_filter(image, sigma=(0.8, 0.8, 0))
        detail = image - blurred
        enhanced = image + detail * (strength - 1.0)
        return np.clip(enhanced, 0, 1)

    def apply_micro_saturation(self, image: np.ndarray, boost: float) -> np.ndarray:
        """Apply extremely subtle saturation boost."""
        if abs(boost - 1.0) < 0.001:
            return image

        hsv = color.rgb2hsv(image)
        hsv[..., 1] = np.clip(hsv[..., 1] * boost, 0, 1)
        enhanced = color.hsv2rgb(hsv)
        return np.clip(enhanced, 0, 1)

    def apply_targeted_contrast(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply very subtle contrast adjustment."""
        if abs(amount - 1.0) < 0.001:
            return image

        # Subtle midtone contrast
        mean = image.mean(axis=(0, 1), keepdims=True)
        enhanced = (image - mean) * amount + mean
        return np.clip(enhanced, 0, 1)

    def refine_image(self, input_path: Path, interactive: bool = True) -> Dict:
        """Refine single image with surgical precision."""
        logger.info(f"\n{'='*80}")
        logger.info(f"SURGICAL REFINEMENT: {input_path.name}")
        logger.info(f"{'='*80}\n")

        # Load
        image = self.load_image(input_path)
        h, w = image.shape[:2]
        logger.info(f"Resolution: {w}x{h}")

        # Initial analysis
        logger.info("\nInitial Analysis:")
        logger.info("-" * 80)
        initial_metrics = SurgicalQualityMetrics.evaluate(image)

        logger.info(f"Overall Quality: {initial_metrics['overall_quality']:.2f}/100\n")
        logger.info("Component Scores:")
        for key, value in initial_metrics.items():
            if key != 'overall_quality':
                status = "✅" if value >= 95 else "⚠️" if value >= 85 else "❌"
                logger.info(f"  {status} {key:25s}: {value:6.2f}/100")

        # Identify weaknesses
        weaknesses = SurgicalQualityMetrics.analyze_weaknesses(initial_metrics)

        if not weaknesses:
            logger.info("\n✅ All metrics ≥95 - Image is already excellent!")
            logger.info("No refinement needed.")

            # Save original as-is
            output_path = self.output_dir / f"{input_path.stem}_Surgical98.tif"
            Image.fromarray((image * 255).astype(np.uint8)).save(
                output_path, format='TIFF', compression='lzw'
            )

            return {
                'input': input_path.name,
                'output': output_path.name,
                'initial_metrics': initial_metrics,
                'final_metrics': initial_metrics,
                'improvement': 0.0,
                'actions': ['no_processing_needed']
            }

        logger.info(f"\n⚠️  Areas for improvement ({len(weaknesses)}):")
        for metric, score in weaknesses[:3]:  # Top 3
            logger.info(f"   • {metric}: {score:.2f}/100")

        # Determine surgical interventions
        logger.info("\n🔬 Surgical Plan:")
        logger.info("-" * 80)

        actions = []
        enhanced = image.copy()

        # Sharpness adjustment
        if initial_metrics['sharpness'] < 95:
            deficit = 95 - initial_metrics['sharpness']
            strength = 1.0 + (deficit / 200)  # Very conservative
            strength = min(strength, 1.08)  # Cap at 8% increase
            logger.info(f"1. Micro-sharpening: {strength:.3f}x (targeting +{deficit:.1f} points)")
            enhanced = self.apply_micro_sharpening(enhanced, strength)
            actions.append(f"sharpening_{strength:.3f}")

        # Color vibrancy
        if initial_metrics['color_vibrancy'] < 95:
            deficit = 95 - initial_metrics['color_vibrancy']
            boost = 1.0 + (deficit / 500)  # Extremely conservative
            boost = min(boost, 1.03)  # Cap at 3% increase
            logger.info(f"2. Micro-saturation: {boost:.3f}x (targeting +{deficit:.1f} points)")
            enhanced = self.apply_micro_saturation(enhanced, boost)
            actions.append(f"saturation_{boost:.3f}")

        # Contrast
        if initial_metrics['contrast'] < 95:
            deficit = 95 - initial_metrics['contrast']
            amount = 1.0 + (deficit / 400)
            amount = min(amount, 1.04)  # Cap at 4% increase
            logger.info(f"3. Micro-contrast: {amount:.3f}x (targeting +{deficit:.1f} points)")
            enhanced = self.apply_targeted_contrast(enhanced, amount)
            actions.append(f"contrast_{amount:.3f}")

        if not actions:
            logger.info("No specific interventions needed.")
            actions.append('minimal_adjustment')

        # Convert to PIL for saving
        enhanced_pil = Image.fromarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8))

        # Final analysis
        logger.info("\n" + "="*80)
        logger.info("RESULTS:")
        logger.info("="*80)

        final_metrics = SurgicalQualityMetrics.evaluate(np.array(enhanced_pil).astype(np.float32) / 255.0)
        improvement = final_metrics['overall_quality'] - initial_metrics['overall_quality']

        logger.info(f"\nOverall Quality: {initial_metrics['overall_quality']:.2f} → {final_metrics['overall_quality']:.2f} ({improvement:+.2f})")
        logger.info("\nDetailed Changes:")

        for key in initial_metrics.keys():
            if key != 'overall_quality':
                initial = initial_metrics[key]
                final = final_metrics[key]
                delta = final - initial
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                logger.info(f"  {key:25s}: {initial:6.2f} → {final:6.2f} ({arrow} {abs(delta):5.2f})")

        # Save
        stem = input_path.stem
        output_path = self.output_dir / f"{stem}_Surgical98.tif"
        enhanced_pil.save(output_path, format='TIFF', compression='lzw')
        logger.info(f"\n✓ Saved: {output_path.name}")

        preview_path = self.output_dir / f"{stem}_Surgical98_preview.jpg"
        enhanced_pil.save(preview_path, format='JPEG', quality=98, optimize=True)
        logger.info(f"✓ Preview: {preview_path.name}")

        # Quality assessment
        logger.info("\n" + "="*80)
        if final_metrics['overall_quality'] >= 98:
            logger.info("🏆 EXCELLENT: Achieved 98+ quality target!")
        elif final_metrics['overall_quality'] >= 95:
            logger.info("✅ VERY GOOD: 95+ quality achieved")
        elif final_metrics['overall_quality'] >= 90:
            logger.info("👍 GOOD: 90+ quality achieved")
        else:
            logger.info("📊 BASELINE: Further refinement may be needed")
        logger.info("="*80 + "\n")

        return {
            'input': input_path.name,
            'output': output_path.name,
            'preview': preview_path.name,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'improvement': round(float(improvement), 2),
            'actions': actions
        }


def main():
    """Main execution - process images one at a time."""
    import argparse

    parser = argparse.ArgumentParser(description='Surgical refinement for 98+ quality')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--output-dir', type=str, default='outputs/750_picacho/Surgical98',
                       help='Output directory')

    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        logger.error(f"Image not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')

    pipeline = SurgicalRefinement(output_dir)
    result = pipeline.refine_image(input_path, interactive=True)

    # Save metadata
    metadata_path = output_dir / "refinement_report.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'pipeline': 'Surgical98',
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'result': result
        }, f, indent=2)

    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📊 Report: {metadata_path.name}\n")

    return 0


if __name__ == "__main__":
    exit(main())
