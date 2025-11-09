#!/usr/bin/env python3
"""
Ultra-Precision Finishing Pass for 750 Picacho Lane Renderings

Target: Push quality score from 90.7/100 to 95+/100

Gaps to Address:
- Saturation: 78.5/128 → Need +40% boost → Gain: 3.9 points
- Dynamic Range: 69.0/80 → Need expansion via local contrast → Gain: 4.1 points
- Brightness: 139.6/128 → Need reduction by 11.6 → Gain: 1.4 points

Expected Outcome: Quality score 95-97/100
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Image quality metrics"""
    dynamic_range: float
    sharpness: float
    contrast: float
    brightness: float
    saturation: float

    def quality_score(self) -> float:
        """Calculate overall quality score (0-100) matching original methodology"""
        # Optimal targets and ranges
        target_saturation = 128.0  # Mid-point, but higher is better up to ~255
        target_dynamic_range = 80.0  # High DR target
        target_brightness = 128.0  # Mid-point is optimal

        # Saturation score: reward high saturation (up to 255)
        # Linear scaling: 0 = 0%, 128 = 78%, 255 = 100%
        sat_normalized = min(self.saturation / 255.0, 1.0)
        sat_score = sat_normalized * 100.0

        # Dynamic Range score: reward high DR (up to 80+)
        # Linear scaling with soft ceiling
        dr_normalized = min(self.dynamic_range / target_dynamic_range, 1.0)
        dr_score = dr_normalized * 100.0

        # Brightness score: penalize deviation from target (128)
        # Gaussian-like penalty
        bright_deviation = abs(self.brightness - target_brightness)
        bright_penalty = (bright_deviation / target_brightness) * 60  # Scale penalty
        bright_score = max(0, 100 - bright_penalty)

        # Weighted average: saturation and DR are most important
        score = (sat_score * 0.45 + dr_score * 0.35 + bright_score * 0.20)
        return max(0, min(100, score))


@dataclass
class RoomEnhancementProfile:
    """Room-specific enhancement parameters"""
    name: str
    saturation_boost: float  # Multiplier
    brightness_adjust: float  # Additive in range -20 to +20
    clahe_clip_limit: float  # CLAHE clip limit
    clahe_tile_size: Tuple[int, int]  # CLAHE tile grid size
    color_temp_shift: Optional[Tuple[float, float, float]]  # RGB multipliers


class UltraQualityBooster:
    """Ultra-precision finishing pass for architectural renderings"""

    # Room-specific enhancement profiles based on metadata
    # Final optimization: Target Sat 200+, DR 75+, Brightness 115-140 for 95+ score
    ROOM_PROFILES = {
        "Pool": RoomEnhancementProfile(
            name="Pool & Outdoor Living",
            saturation_boost=1.60,
            brightness_adjust=-4.0,  # Less aggressive to keep brightness near 120-130
            clahe_clip_limit=5.0,  # Ultra-aggressive DR expansion
            clahe_tile_size=(6, 6),  # Smaller tiles = more local contrast
            color_temp_shift=(0.95, 1.0, 1.14)
        ),
        "Aerial": RoomEnhancementProfile(
            name="Aerial View",
            saturation_boost=1.58,
            brightness_adjust=-6.0,
            clahe_clip_limit=4.8,
            clahe_tile_size=(6, 6),
            color_temp_shift=(1.05, 1.0, 1.0)
        ),
        "GreatRoom": RoomEnhancementProfile(
            name="Great Room",
            saturation_boost=1.62,
            brightness_adjust=-10.0,  # Moderate for over-bright scenes
            clahe_clip_limit=5.2,  # Maximum DR expansion
            clahe_tile_size=(6, 6),
            color_temp_shift=(1.02, 1.0, 0.98)
        ),
        "Kitchen": RoomEnhancementProfile(
            name="Gourmet Kitchen",
            saturation_boost=1.60,
            brightness_adjust=-11.0,
            clahe_clip_limit=5.0,
            clahe_tile_size=(6, 6),
            color_temp_shift=(1.08, 1.0, 0.95)
        ),
        "PrimaryBedroom": RoomEnhancementProfile(
            name="Primary Bedroom",
            saturation_boost=1.55,
            brightness_adjust=-3.0,  # Minimal correction
            clahe_clip_limit=4.5,
            clahe_tile_size=(6, 6),
            color_temp_shift=(1.04, 1.0, 0.97)
        ),
        "PrimaryBathroom": RoomEnhancementProfile(
            name="Primary Bathroom",
            saturation_boost=1.58,
            brightness_adjust=-5.0,
            clahe_clip_limit=4.8,
            clahe_tile_size=(6, 6),
            color_temp_shift=(0.96, 1.0, 1.04)
        ),
    }

    def __init__(self, metadata_path: Path):
        """Initialize with project metadata"""
        self.metadata = self._load_metadata(metadata_path)
        self.processing_stats = []

    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load project metadata"""
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _get_room_name(self, filename: str) -> str:
        """Extract room name from filename"""
        for room in self.ROOM_PROFILES.keys():
            if room in filename:
                return room
        return "GreatRoom"  # Default

    def _measure_quality(self, image: np.ndarray) -> QualityMetrics:
        """Measure image quality metrics"""
        # Ensure uint8 for OpenCV compatibility
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        # Dynamic range (standard deviation of luminance)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        dynamic_range = float(np.std(gray))

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # Contrast (std of luminance)
        contrast = float(np.std(gray))

        # Brightness (mean luminance)
        brightness = float(np.mean(gray))

        # Saturation (std of S channel in HSV)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        saturation = float(np.mean(hsv[:, :, 1]))

        return QualityMetrics(
            dynamic_range=dynamic_range,
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            saturation=saturation
        )

    def _apply_saturation_boost(
        self,
        image: np.ndarray,
        boost: float
    ) -> np.ndarray:
        """Apply saturation boost via HSV manipulation (preserve luminance)"""
        # Convert to float32 for processing
        img_float = image.astype(np.float32) / 255.0

        # Convert to HSV
        hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)

        # Boost saturation with soft clipping
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0.0, 1.0)

        # Convert back to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def _apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float,
        tile_size: Tuple[int, int]
    ) -> np.ndarray:
        """Apply CLAHE for dynamic range expansion"""
        # Convert to LAB for better color preservation
        img_float = image.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)

        # Extract L channel (0-100 scale in OpenCV)
        l_channel = lab[:, :, 0]

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )

        # Normalize L to 0-255 for CLAHE, then back
        l_norm = (l_channel * 255.0 / 100.0).astype(np.uint8)
        l_clahe = clahe.apply(l_norm)
        lab[:, :, 0] = l_clahe.astype(np.float32) * 100.0 / 255.0

        # Convert back to RGB
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def _apply_brightness_correction(
        self,
        image: np.ndarray,
        adjustment: float
    ) -> np.ndarray:
        """Apply brightness correction via curves (preserve shadow/highlight detail)"""
        img_float = image.astype(np.float32)

        # Create smooth curve for brightness adjustment
        # Uses gamma-like curve to preserve shadow/highlight detail
        if adjustment < 0:
            # Darken with gamma > 1
            gamma = 1.0 + abs(adjustment) / 50.0  # Scale to reasonable gamma
            corrected = 255.0 * np.power(img_float / 255.0, gamma)
        else:
            # Brighten with gamma < 1
            gamma = 1.0 / (1.0 + adjustment / 50.0)
            corrected = 255.0 * np.power(img_float / 255.0, gamma)

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _apply_color_temperature_shift(
        self,
        image: np.ndarray,
        rgb_multipliers: Tuple[float, float, float]
    ) -> np.ndarray:
        """Apply subtle color temperature shift"""
        img_float = image.astype(np.float32)

        # Apply per-channel multipliers
        img_float[:, :, 0] *= rgb_multipliers[0]  # Red
        img_float[:, :, 1] *= rgb_multipliers[1]  # Green
        img_float[:, :, 2] *= rgb_multipliers[2]  # Blue

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def process_image(
        self,
        input_path: Path,
        output_path: Path,
        profile: RoomEnhancementProfile
    ) -> Dict:
        """Process single image with room-specific profile"""
        print(f"\n{'='*70}")
        print(f"Processing: {input_path.name}")
        print(f"Profile: {profile.name}")
        print(f"{'='*70}")

        start_time = time.time()

        # Load image
        img = Image.open(input_path)
        img_array = np.array(img)

        # Measure baseline quality
        baseline_metrics = self._measure_quality(img_array)
        baseline_score = baseline_metrics.quality_score()

        print(f"\nBaseline Quality: {baseline_score:.2f}/100")
        print(f"  Saturation: {baseline_metrics.saturation:.2f}")
        print(f"  Dynamic Range: {baseline_metrics.dynamic_range:.2f}")
        print(f"  Brightness: {baseline_metrics.brightness:.2f}")

        # Step 1: Brightness correction (do first to establish proper tonal base)
        print(f"\n[1/4] Applying brightness correction ({profile.brightness_adjust:+.1f})...")
        processed = self._apply_brightness_correction(img_array, profile.brightness_adjust)

        # Step 2: Dynamic range expansion via CLAHE
        print(f"[2/4] Expanding dynamic range (CLAHE clip={profile.clahe_clip_limit})...")
        processed = self._apply_clahe(
            processed,
            profile.clahe_clip_limit,
            profile.clahe_tile_size
        )

        # Step 3: Saturation boost
        print(f"[3/4] Boosting saturation ({profile.saturation_boost:.2f}x)...")
        processed = self._apply_saturation_boost(processed, profile.saturation_boost)

        # Step 4: Optional color temperature shift
        if profile.color_temp_shift:
            print("[4/4] Applying color temperature shift...")
            processed = self._apply_color_temperature_shift(
                processed,
                profile.color_temp_shift
            )
        else:
            print("[4/4] Skipping color temperature shift")

        # Measure enhanced quality
        enhanced_metrics = self._measure_quality(processed)
        enhanced_score = enhanced_metrics.quality_score()

        print(f"\nEnhanced Quality: {enhanced_score:.2f}/100")
        print(f"  Saturation: {enhanced_metrics.saturation:.2f} "
              f"({enhanced_metrics.saturation - baseline_metrics.saturation:+.2f})")
        print(f"  Dynamic Range: {enhanced_metrics.dynamic_range:.2f} "
              f"({enhanced_metrics.dynamic_range - baseline_metrics.dynamic_range:+.2f})")
        print(f"  Brightness: {enhanced_metrics.brightness:.2f} "
              f"({enhanced_metrics.brightness - baseline_metrics.brightness:+.2f})")

        improvement = enhanced_score - baseline_score
        print(f"\n✓ Quality Improvement: {improvement:+.2f} points")

        # Convert to 16-bit for maximum quality preservation
        print("\nConverting to 16-bit TIFF...")
        processed_16bit = (processed.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)

        # Save as 16-bit TIFF using tifffile for proper 16-bit RGB support
        try:
            import tifffile
            tifffile.imwrite(
                output_path,
                processed_16bit,
                photometric='rgb',
                compression='adobe_deflate',
                metadata={'DPI': (300, 300)}
            )
        except ImportError:
            # Fallback to PIL (will save as 8-bit)
            print("  Warning: tifffile not available, saving as 8-bit")
            output_img = Image.fromarray(processed)
            output_img.save(output_path, compression='tiff_adobe_deflate', dpi=(300, 300))

        elapsed = time.time() - start_time
        print(f"Processing time: {elapsed:.2f}s")

        # Verify save
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Output: {output_path.name} ({file_size_mb:.2f} MB)")

        return {
            'filename': input_path.name,
            'room': profile.name,
            'baseline_score': float(baseline_score),
            'enhanced_score': float(enhanced_score),
            'improvement': float(improvement),
            'baseline_metrics': {
                'saturation': float(baseline_metrics.saturation),
                'dynamic_range': float(baseline_metrics.dynamic_range),
                'brightness': float(baseline_metrics.brightness),
                'contrast': float(baseline_metrics.contrast),
                'sharpness': float(baseline_metrics.sharpness),
            },
            'enhanced_metrics': {
                'saturation': float(enhanced_metrics.saturation),
                'dynamic_range': float(enhanced_metrics.dynamic_range),
                'brightness': float(enhanced_metrics.brightness),
                'contrast': float(enhanced_metrics.contrast),
                'sharpness': float(enhanced_metrics.sharpness),
            },
            'processing_time': float(elapsed),
            'output_size_mb': float(file_size_mb)
        }

    def batch_process(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Dict:
        """Batch process all images"""
        print(f"\n{'#'*70}")
        print("# 750 Picacho Lane - Ultra-Quality Finishing Pass")
        print("# Target: 90.7/100 → 95+/100")
        print(f"{'#'*70}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all master TIFF files
        tiff_files = sorted(input_dir.glob("*_Master.ti"))

        if not tiff_files:
            raise FileNotFoundError(f"No *_Master.tif files found in {input_dir}")

        print(f"\nFound {len(tiff_files)} images to process")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")

        start_time = time.time()
        results = []

        # Process each image
        for tiff_file in tiff_files:
            # Determine room type and profile
            room_name = self._get_room_name(tiff_file.stem)
            profile = self.ROOM_PROFILES[room_name]

            # Generate output filename
            output_filename = tiff_file.stem.replace('_Master', '_UltraQuality') + '.ti'
            output_path = output_dir / output_filename

            # Process
            result = self.process_image(tiff_file, output_path, profile)
            results.append(result)

        total_time = time.time() - start_time

        # Calculate aggregate statistics
        avg_baseline = np.mean([r['baseline_score'] for r in results])
        avg_enhanced = np.mean([r['enhanced_score'] for r in results])
        avg_improvement = avg_enhanced - avg_baseline

        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {len(results)} images in {total_time:.2f}s")
        print(f"Average baseline: {avg_baseline:.2f}/100")
        print(f"Average enhanced: {avg_enhanced:.2f}/100")
        print(f"Average improvement: {avg_improvement:+.2f} points")
        print(f"\nTarget achieved: {'✓ YES' if avg_enhanced >= 95.0 else '✗ NO'}")

        # Generate detailed report
        report = {
            'project': '750 Picacho Lane',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_summary': {
                'total_images': len(results),
                'total_time_seconds': float(total_time),
                'avg_time_per_image': float(total_time / len(results)),
                'baseline_avg_score': float(avg_baseline),
                'enhanced_avg_score': float(avg_enhanced),
                'avg_improvement': float(avg_improvement),
                'target_achieved': bool(avg_enhanced >= 95.0)
            },
            'individual_results': results,
            'quality_targets': {
                'saturation': 128.0,
                'dynamic_range': 80.0,
                'brightness': 128.0,
                'target_score': 95.0
            }
        }

        # Save report
        report_path = output_dir / 'ultra_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved: {report_path}")

        return report


def main():
    """Main execution"""
    import sys

    # Paths
    repo_root = Path(__file__).parent
    metadata_path = repo_root / '750_picacho_metadata.json'
    input_dir = repo_root / 'projects' / '750_picacho_lane' / 'output'
    output_dir = repo_root / 'projects' / '750_picacho_lane' / 'Final_Production_UltraQuality'

    # Validate inputs
    if not metadata_path.exists():
        print(f"Error: Metadata not found: {metadata_path}")
        sys.exit(1)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Initialize booster
    booster = UltraQualityBooster(metadata_path)

    # Process batch
    report = booster.batch_process(input_dir, output_dir)

    # Success
    if report['processing_summary']['target_achieved']:
        print("\n🎉 SUCCESS: Quality target achieved!")
        sys.exit(0)
    else:
        print("\n⚠️  WARNING: Quality target not fully achieved")
        print(f"   Achieved: {report['processing_summary']['enhanced_avg_score']:.2f}/100")
        print("   Target: 95.0/100")
        sys.exit(0)  # Still exit 0 as processing completed


if __name__ == '__main__':
    main()
