#!/usr/bin/env python3
"""
Process 750 Picacho Lane source JPEGs through the luxury pipeline
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transformation_portal.pipelines.lux_render_pipeline import LuxRenderPipeline

# Scene definitions with their source files
SCENES = ["Aerial", "GreatRoom", "Kitchen", "Pool", "PrimaryBathroom", "PrimaryBedroom"]


def process_750_picacho(source_dir: Path, output_base_dir: Path):
    """Process all 6 source images for 750 Picacho Lane"""

    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"750_Picacho_Processed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("750 PICACHO LANE - LUXURY PIPELINE PROCESSING")
    print("=" * 80)
    print(f"\nSource Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    # Initialize pipeline
    pipeline = LuxRenderPipeline(use_depth_estimation=True, use_advanced_grading=True, output_format="jpg", quality=98)

    results = {}

    for i, scene in enumerate(SCENES, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(SCENES)}: {scene}")
        print(f"{'='*80}")

        # Build paths
        source_file = source_dir / f"750Picacho_{scene}.jpg"
        output_file = output_dir / f"750Picacho_{scene}_luxury.jpg"

        if not source_file.exists():
            print(f"❌ Source file not found: {source_file}")
            results[scene] = "MISSING"
            continue

        try:
            # Process the image
            print(f"📥 Input: {source_file.name}")
            print(f"📤 Output: {output_file.name}")
            print(f"⚙️  Processing...")

            result = pipeline.process_image(input_path=str(source_file), output_path=str(output_file))

            if result and output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Success! Output size: {size_mb:.2f} MB")
                results[scene] = "SUCCESS"
            else:
                print(f"❌ Processing failed")
                results[scene] = "FAILED"

        except Exception as e:
            print(f"❌ Error processing {scene}: {e}")
            results[scene] = f"ERROR: {str(e)}"

    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}\n")

    success_count = sum(1 for v in results.values() if v == "SUCCESS")
    total_count = len(SCENES)

    for scene, status in results.items():
        icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{icon} {scene:20s} {status}")

    print(f"\n{'='*80}")
    print(f"Completed: {success_count}/{total_count} scenes processed successfully")
    print(f"Output location: {output_dir}")
    print(f"{'='*80}\n")

    return success_count == total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 750 Picacho Lane source JPEGs through Lux Render.")
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/tp-750-picacho-processed"))
    args = parser.parse_args()
    success = process_750_picacho(args.source_dir, args.output_dir)
    sys.exit(0 if success else 1)
