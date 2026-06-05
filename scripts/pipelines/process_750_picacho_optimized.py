#!/usr/bin/env python3
"""
Optimized batch processor for 750 Picacho Lane using canonical sources only.

This script ensures:
1. Only canonical (deduplicated) sources are processed
2. Consistent output naming
3. Robust error handling with continuation
4. Progress tracking and reporting
5. Quality validation at each stage

Author: Transformation Portal
Date: 2025-11-08
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PIPELINES_DIR = Path(__file__).resolve().parent
DEFAULT_750_BASE_DIR = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))


def load_canonical_manifest(manifest_path: Path) -> Dict:
    """Load the canonical sources manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def process_single_image(source_path: Path, output_dir: Path, scene_name: str, pipeline_config: Optional[Dict] = None) -> Dict:
    """
    Process a single image through the optimized pipeline.

    Args:
        source_path: Path to canonical source file
        output_dir: Output directory for results
        scene_name: Clean scene name (without version suffixes)
        pipeline_config: Optional configuration overrides

    Returns:
        Processing results dictionary
    """
    from unified_luxury_pipeline import LuxuryConfig, UnifiedLuxuryPipeline

    print(f"\n{'='*80}")
    print(f"Processing: {scene_name}")
    print(f"Source: {source_path.name}")
    print(f"{'='*80}")

    start_time = datetime.now()
    results = {
        "scene": scene_name,
        "source": str(source_path),
        "start_time": start_time.isoformat(),
        "status": "started",
        "outputs": [],
        "errors": [],
    }

    try:
        # Configure pipeline
        config = LuxuryConfig(
            # Use scene name (not source filename) for outputs
            output_name_override=scene_name,
            # Enable all output formats
            output_formats=["jpg", "ti", "png"],
            # Maximum quality settings
            depth_estimation=True,
            material_response=True,
            color_grading=True,
            # Processing parameters
            exposure_adjustment=0.0,  # Neutral - already graded in LightRoom
            contrast=1.05,
            saturation=1.02,
            clarity=0.15,
            # Device optimization
            device="mps",  # Use Apple Silicon
        )

        # Apply any custom overrides
        if pipeline_config:
            for key, value in pipeline_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Initialize pipeline
        pipeline = UnifiedLuxuryPipeline(config)

        # Process image
        print("\n🚀 Starting pipeline processing...")
        output_paths = pipeline.process_image(source_path, output_dir)

        results["status"] = "success"
        results["outputs"] = [str(p) for p in output_paths.values()]
        results["output_details"] = {format_name: str(path) for format_name, path in output_paths.items()}

        # Validation
        print("\n✅ Processing complete!")
        print(f"   Outputs generated: {len(output_paths)}")
        for format_name, path in output_paths.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✓ {format_name.upper()}: {path.name} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {format_name.upper()}: MISSING")
                results["errors"].append(f"Output missing: {format_name}")

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"\n❌ Processing failed: {e}")
        print(f"   Traceback:\n{results['traceback']}")

    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = duration
        print(f"\n⏱  Duration: {duration:.1f} seconds")

    return results


def batch_process_canonical_sources(
    manifest_path: Path, output_base_dir: Path, scene_filter: Optional[List[str]] = None, continue_on_error: bool = True
) -> Dict:
    """
    Batch process all canonical sources.

    Args:
        manifest_path: Path to canonical sources manifest
        output_base_dir: Base output directory
        scene_filter: Optional list of scene names to process (None = all)
        continue_on_error: Continue processing if individual scenes fail

    Returns:
        Batch processing summary
    """
    print("\n" + "=" * 80)
    print("750 Picacho Lane - Optimized Batch Processing")
    print("=" * 80)

    # Load manifest
    print("\n📋 Loading canonical manifest...")
    manifest = load_canonical_manifest(manifest_path)
    print(f"   Found {len(manifest['canonical_sources'])} canonical sources")

    if manifest.get("duplicates_found"):
        print(f"   Note: {len(manifest['duplicates_found'])} scenes had duplicates (now resolved)")

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"Optimized_Processing_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Filter scenes if requested
    sources_to_process = manifest["canonical_sources"]
    if scene_filter:
        sources_to_process = {k: v for k, v in sources_to_process.items() if k in scene_filter}
        print(f"\n🎯 Filtering to {len(sources_to_process)} scenes: {', '.join(scene_filter)}")

    # Process each source
    batch_start = datetime.now()
    results_summary = {
        "manifest_used": str(manifest_path),
        "output_directory": str(output_dir),
        "start_time": batch_start.isoformat(),
        "scenes_processed": [],
        "successes": [],
        "failures": [],
        "skipped": [],
    }

    total_scenes = len(sources_to_process)

    for idx, (scene_name, source_info) in enumerate(sources_to_process.items(), 1):
        print(f"\n\n{'#'*80}")
        print(f"# Scene {idx}/{total_scenes}: {scene_name}")
        print(f"{'#'*80}")

        source_path = Path(source_info["path"])

        # Validate source exists
        if not source_path.exists():
            print(f"⚠️  Source file not found, skipping: {source_path}")
            results_summary["skipped"].append({"scene": scene_name, "reason": "source_not_found", "path": str(source_path)})
            continue

        # Process the scene
        try:
            scene_results = process_single_image(source_path, output_dir, scene_name)

            results_summary["scenes_processed"].append(scene_results)

            if scene_results["status"] == "success":
                results_summary["successes"].append(scene_name)
            else:
                results_summary["failures"].append({"scene": scene_name, "error": scene_results.get("error", "Unknown error")})

                if not continue_on_error:
                    print("\n❌ Stopping batch processing due to error (continue_on_error=False)")
                    break

        except Exception as e:
            error_info = {"scene": scene_name, "error": str(e), "traceback": traceback.format_exc()}
            results_summary["failures"].append(error_info)
            print(f"\n❌ Unexpected error processing {scene_name}: {e}")
            print(f"   Traceback:\n{error_info['traceback']}")

            if not continue_on_error:
                print("\n❌ Stopping batch processing")
                break

    # Finalize summary
    batch_end = datetime.now()
    batch_duration = (batch_end - batch_start).total_seconds()
    results_summary["end_time"] = batch_end.isoformat()
    results_summary["total_duration_seconds"] = batch_duration
    results_summary["summary"] = {
        "total_scenes": total_scenes,
        "processed": len(results_summary["scenes_processed"]),
        "successful": len(results_summary["successes"]),
        "failed": len(results_summary["failures"]),
        "skipped": len(results_summary["skipped"]),
        "success_rate": len(results_summary["successes"]) / max(len(results_summary["scenes_processed"]), 1) * 100,
    }

    # Save results
    results_file = output_dir / "batch_processing_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    # Print summary
    print("\n\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print("\n📊 Summary:")
    print(f"   Total scenes: {results_summary['summary']['total_scenes']}")
    print(f"   Processed: {results_summary['summary']['processed']}")
    print(f"   ✅ Successful: {results_summary['summary']['successful']}")
    print(f"   ❌ Failed: {results_summary['summary']['failed']}")
    print(f"   ⏭  Skipped: {results_summary['summary']['skipped']}")
    print(f"   Success rate: {results_summary['summary']['success_rate']:.1f}%")
    print(f"\n⏱  Total duration: {batch_duration / 60:.1f} minutes")
    print(f"\n💾 Results saved to: {results_file}")

    if results_summary["failures"]:
        print("\n⚠️  Failed scenes:")
        for failure in results_summary["failures"]:
            print(f"   - {failure['scene']}: {failure['error']}")

    print(f"\n✨ All outputs saved to: {output_dir}")
    print()

    return results_summary


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimized batch processor for 750 Picacho Lane")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_750_BASE_DIR / "canonical_sources_manifest.json",
        help="Path to canonical sources manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_750_BASE_DIR,
        help="Base output directory",
    )
    parser.add_argument("--scenes", nargs="+", default=None, help="Specific scenes to process (space-separated)")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop processing if any scene fails (default: continue)")

    args = parser.parse_args()

    # Validate manifest
    if not args.manifest.exists():
        print(f"❌ Manifest not found: {args.manifest}")
        print("\nRun resolve_750_picacho_duplicates.py first to generate manifest")
        sys.exit(1)

    # Run batch processing
    results = batch_process_canonical_sources(
        args.manifest, args.output_dir, scene_filter=args.scenes, continue_on_error=not args.stop_on_error
    )

    # Exit with appropriate code
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
