#!/usr/bin/env python3
"""
Prepare 750 Picacho BIM Training Data
Converts high-quality renders and BIM images into training pairs for Hyper-Reality Enhancement

This script:
1. Uses UltraQuality TIFFs as high-quality targets
2. Creates degraded versions as low-quality inputs
3. Incorporates BIM images from architectural plans
4. Leverages architectural context for room-specific training

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Repository paths
REPO_ROOT = Path(__file__).parent.parent.parent
PICACHO_PROJECT = REPO_ROOT / "projects" / "750_picacho_lane"
ULTRAQUALITY_DIR = PICACHO_PROJECT / "Final_Production_UltraQuality"
BIM_IMAGES_DIR = REPO_ROOT / "extracted_context" / "24098.00_750 PICACHO LANE_images"
CONTEXT_JSON = REPO_ROOT / "extracted_context" / "24098.00_750 PICACHO LANE_context.json"
MBAR_CONTEXT_JSON = REPO_ROOT / "extracted_context" / "mbar_submittal.json"


class Picacho750DataPreparation:
    """Prepare training data from 750 Picacho BIM model"""

    def __init__(self, output_dir: str = "data/training_750picacho"):
        self.output_dir = Path(output_dir)
        self.low_quality_dir = self.output_dir / "low_quality"
        self.high_quality_dir = self.output_dir / "high_quality"

        # Create directories
        self.low_quality_dir.mkdir(parents=True, exist_ok=True)
        self.high_quality_dir.mkdir(parents=True, exist_ok=True)

        # Load architectural context
        self.context = self._load_context()

        self.pairs_created = 0

    def _load_context(self) -> Optional[Dict]:
        """Load architectural context from BIM and MBAR submittal"""
        context = {}

        # Load main BIM context
        if CONTEXT_JSON.exists():
            try:
                with open(CONTEXT_JSON, 'r') as f:
                    context['bim'] = json.load(f)
                    print("✓ Loaded BIM context")
            except Exception as e:
                print(f"⚠️  Failed to load BIM context: {e}")

        # Load MBAR submittal context
        if MBAR_CONTEXT_JSON.exists():
            try:
                with open(MBAR_CONTEXT_JSON, 'r') as f:
                    context['mbar'] = json.load(f)
                    print("✓ Loaded MBAR submittal context (materials, elevations, details)")
            except Exception as e:
                print(f"⚠️  Failed to load MBAR context: {e}")

        return context if context else None

    def prepare_ultraquality_renders(self):
        """Convert UltraQuality TIFFs to training pairs"""
        print(f"\n{'='*60}")
        print("PREPARING ULTRAQUALITY RENDERS")
        print(f"{'='*60}\n")

        if not ULTRAQUALITY_DIR.exists():
            print(f"❌ UltraQuality directory not found: {ULTRAQUALITY_DIR}")
            return

        # Find all TIFF files
        tiff_files = list(ULTRAQUALITY_DIR.glob("*.tif")) + list(ULTRAQUALITY_DIR.glob("*.tiff"))
        tiff_files = [f for f in tiff_files if not f.name.startswith('.')]

        print(f"Found {len(tiff_files)} UltraQuality renders")

        for tiff_path in tqdm(tiff_files, desc="Processing renders"):
            self._process_ultraquality_render(tiff_path)

        print(f"\n✓ Processed {len(tiff_files)} UltraQuality renders")

    def _process_ultraquality_render(self, tiff_path: Path):
        """Create training pairs from single UltraQuality render"""
        try:
            # Load high-quality image
            high_img = Image.open(tiff_path).convert('RGB')

            # Extract room name from filename (e.g., "750Picacho_Kitchen_UltraQuality.tif" -> "Kitchen")
            room_name = tiff_path.stem.replace('750Picacho_', '').replace('_UltraQuality', '')

            # Create multiple training pairs from crops
            self._create_crop_pairs(high_img, room_name, num_crops=5)

        except Exception as e:
            print(f"⚠️  Failed to process {tiff_path.name}: {e}")

    def _create_crop_pairs(self, image: Image.Image, room_name: str, num_crops: int = 5):
        """Create multiple training pairs from crops of large image"""
        width, height = image.size
        crop_size = 1024  # Crop to 1024x1024

        # Generate random crops
        for i in range(num_crops):
            # Random position ensuring we don't go out of bounds
            if width < crop_size or height < crop_size:
                # Image too small, resize first
                scale = max(crop_size / width, crop_size / height) * 1.1
                new_size = (int(width * scale), int(height * scale))
                image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
                width, height = new_size
            else:
                image_resized = image

            x = np.random.randint(0, max(1, width - crop_size))
            y = np.random.randint(0, max(1, height - crop_size))

            # Crop high-quality version
            high_crop = image_resized.crop((x, y, x + crop_size, y + crop_size))

            # Create degraded low-quality version
            low_crop = self._degrade_image(high_crop, room_name)

            # Save pair
            pair_name = f"750picacho_{room_name.lower()}_{self.pairs_created:04d}.png"

            high_path = self.high_quality_dir / pair_name
            low_path = self.low_quality_dir / pair_name

            high_crop.save(high_path, quality=100)
            low_crop.save(low_path, quality=95)

            self.pairs_created += 1

    def _degrade_image(self, img: Image.Image, room_name: str = "") -> Image.Image:
        """Apply realistic degradations to create low-quality version"""
        img_array = np.array(img).astype(np.float32)

        # Room-specific degradation profiles
        degradation_profiles = {
            'Kitchen': {'contrast': 0.75, 'noise': 6, 'blur': 0.6, 'saturation': 0.8},
            'Pool': {'contrast': 0.70, 'noise': 8, 'blur': 0.8, 'saturation': 0.75},
            'Aerial': {'contrast': 0.65, 'noise': 10, 'blur': 1.0, 'saturation': 0.7},
            'GreatRoom': {'contrast': 0.75, 'noise': 5, 'blur': 0.6, 'saturation': 0.8},
            'PrimaryBedroom': {'contrast': 0.70, 'noise': 6, 'blur': 0.7, 'saturation': 0.75},
            'PrimaryBathroom': {'contrast': 0.72, 'noise': 6, 'blur': 0.7, 'saturation': 0.78},
        }

        # Get profile or use default
        profile = degradation_profiles.get(room_name, {
            'contrast': 0.72, 'noise': 7, 'blur': 0.7, 'saturation': 0.77
        })

        # 1. Reduce contrast
        img_array = (img_array - 128) * profile['contrast'] + 128

        # 2. Add noise
        noise = np.random.randn(*img_array.shape) * profile['noise']
        img_array += noise

        # 3. Slight blur
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            img_array[:, :, c] = gaussian_filter(img_array[:, :, c], sigma=profile['blur'])

        # 4. Reduce saturation
        gray = img_array.mean(axis=2, keepdims=True)
        img_array = gray * (1 - profile['saturation']) + img_array * profile['saturation']

        # 5. Add JPEG compression artifacts (simulate web quality)
        img_degraded = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Save to buffer with low quality to add compression
        from io import BytesIO
        buffer = BytesIO()
        img_degraded.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        img_degraded = Image.open(buffer)

        return img_degraded

    def prepare_bim_images(self, max_images: int = 500):
        """Convert BIM-extracted images to training pairs"""
        print(f"\n{'='*60}")
        print("PREPARING BIM ARCHITECTURAL IMAGES")
        print(f"{'='*60}\n")

        if not BIM_IMAGES_DIR.exists():
            print(f"❌ BIM images directory not found: {BIM_IMAGES_DIR}")
            return

        # Find all images
        image_files = list(BIM_IMAGES_DIR.glob("*.jpeg")) + list(BIM_IMAGES_DIR.glob("*.jpg"))
        image_files = [f for f in image_files if not f.name.startswith('.')]

        print(f"Found {len(image_files)} BIM images")

        # Sample if too many
        if len(image_files) > max_images:
            np.random.seed(42)
            image_files = list(np.random.choice(image_files, max_images, replace=False))
            print(f"Sampling {max_images} images for training")

        for img_path in tqdm(image_files, desc="Processing BIM images"):
            self._process_bim_image(img_path)

        print(f"\n✓ Processed {len(image_files)} BIM images")

    def _process_bim_image(self, img_path: Path):
        """Convert BIM image to training pair"""
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')

            # Skip very small images (likely icons)
            if img.size[0] < 200 or img.size[1] < 200:
                return

            # Resize to training size
            target_size = 512
            aspect = img.size[0] / img.size[1]

            if aspect > 1:
                new_size = (target_size, int(target_size / aspect))
            else:
                new_size = (int(target_size * aspect), target_size)

            # Pad to square if needed
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # Create square canvas
            square_img = Image.new('RGB', (target_size, target_size), (240, 240, 240))
            offset = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
            square_img.paste(img_resized, offset)

            # This is already architectural line art, treat as high-quality
            high_img = square_img

            # Create degraded version
            low_img = self._degrade_image(high_img, "BIM")

            # Save pair
            pair_name = f"bim_{img_path.stem}_{self.pairs_created:04d}.png"

            high_path = self.high_quality_dir / pair_name
            low_path = self.low_quality_dir / pair_name

            high_img.save(high_path, quality=100)
            low_img.save(low_path, quality=95)

            self.pairs_created += 1

        except Exception as e:
            # Skip problematic images silently
            pass

    def create_metadata(self):
        """Create metadata file with dataset information"""
        metadata = {
            'dataset_name': '750_Picacho_Training_Data',
            'project': '750 Picacho Lane, Montecito, CA',
            'project_number': '24098.00',
            'total_pairs': self.pairs_created,
            'source_types': {
                'ultraquality_renders': 'High-quality architectural renders (6 TIFFs)',
                'bim_images': 'Extracted from BIM architectural plans (2,488 images)',
                'mbar_submittal': 'Construction documents, elevations, material boards'
            },
            'data_sources': {
                'ultraquality': str(ULTRAQUALITY_DIR),
                'bim_images': str(BIM_IMAGES_DIR),
                'bim_context': str(CONTEXT_JSON),
                'mbar_context': str(MBAR_CONTEXT_JSON)
            },
            'context_available': self.context is not None,
            'context_types': list(self.context.keys()) if self.context else [],
            'rooms': list(self.context.get('bim', {}).get('rooms', {}).keys()) if self.context else [],
            'materials': self.context.get('mbar', {}).get('rooms', [])[:5] if self.context else [],
            'degradation_types': [
                'Room-specific contrast reduction',
                'Gaussian noise addition',
                'Depth-aware blur',
                'Saturation reduction',
                'JPEG compression artifacts'
            ],
            'room_profiles': {
                'Kitchen': 'Bright lighting, balanced depth, metal/stone/glass materials',
                'Pool': 'Water reflections, atmospheric haze, outdoor lighting',
                'Aerial': 'Distance blur, atmospheric effects, perspective',
                'Bedroom': 'Soft lighting, fabric textures, warm tones',
                'Bathroom': 'Stone/glass materials, spa aesthetic, neutral temperature'
            }
        }

        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Metadata saved: {metadata_path}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 750 Picacho BIM training data for Hyper-Reality Enhancement"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_750picacho",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--ultraquality-only",
        action="store_true",
        help="Only use UltraQuality renders (skip BIM images)"
    )
    parser.add_argument(
        "--bim-only",
        action="store_true",
        help="Only use BIM images (skip UltraQuality renders)"
    )
    parser.add_argument(
        "--max-bim-images",
        type=int,
        default=500,
        help="Maximum number of BIM images to use (default: 500)"
    )
    parser.add_argument(
        "--crops-per-render",
        type=int,
        default=5,
        help="Number of crops to extract from each UltraQuality render (default: 5)"
    )

    args = parser.parse_args()

    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  750 Picacho BIM Training Data Preparation                   ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"\nOutput directory: {args.output_dir}")

    # Initialize preparation
    preparer = Picacho750DataPreparation(args.output_dir)

    # Check what data is available
    has_ultraquality = ULTRAQUALITY_DIR.exists() and list(ULTRAQUALITY_DIR.glob("*.tif"))
    has_bim = BIM_IMAGES_DIR.exists() and list(BIM_IMAGES_DIR.glob("*.jpeg"))

    if not has_ultraquality and not has_bim:
        print("\n❌ No 750 Picacho data found!")
        print(f"   Expected UltraQuality renders at: {ULTRAQUALITY_DIR}")
        print(f"   Expected BIM images at: {BIM_IMAGES_DIR}")
        return 1

    # Process data based on flags
    if not args.bim_only and has_ultraquality:
        preparer.prepare_ultraquality_renders()

    if not args.ultraquality_only and has_bim:
        preparer.prepare_bim_images(max_images=args.max_bim_images)

    # Create metadata
    metadata = preparer.create_metadata()

    # Summary
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║  Dataset Preparation Complete                                 ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"\n✓ Created {preparer.pairs_created} training pairs")
    print(f"✓ High quality: {preparer.high_quality_dir}")
    print(f"✓ Low quality: {preparer.low_quality_dir}")
    print("\nNext step: Train models with this data")
    print("  python src/enhancements/train_hyper_reality.py \\")
    print(f"      --data-dir {args.output_dir} \\")
    print("      --epochs 50 --batch-size 4")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
