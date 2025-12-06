#!/usr/bin/env python3
"""
Batch Processing Example
========================

Process an entire directory of images with progress tracking.

Requirements:
    - Input images in 'input' directory
"""
from pathlib import Path
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


def main():
    # Configure for batch processing
    config = PipelineConfig(
        preset=Preset.INTERIOR_LUXURY,
        
        input_dir=Path("input"),
        output_dir=Path("output_batch"),
        depth_dir=Path("depth_maps"),
        
        device="auto",
        upscale=4,
        upscaler_backend="none",  # Use Real-ESRGAN for production
        
        # Batch-friendly options
        skip_existing=True,  # Resume interrupted batches
        enable_material=True,
        save_preview_jpg=True,
        preview_scale=0.25,
    )
    
    # Initialize pipeline
    print("Initializing batch pipeline...")
    pipeline = LuxPipelineV2(config)
    
    # Process entire directory
    print(f"\nProcessing directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print("-" * 60)
    
    results = pipeline.process_directory()
    
    # Summarize results
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    
    total = len(results)
    success = sum(1 for r in results if r['status'] == 'ok')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Total images: {total}")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
    if success > 0:
        timings = [r['timing_s'] for r in results if r['status'] == 'ok']
        avg_time = sum(timings) / len(timings)
        total_time = sum(timings)
        print(f"\nTiming:")
        print(f"  Average: {avg_time:.2f}s per image")
        print(f"  Total: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"  Throughput: {3600/avg_time:.1f} images/hour")
    
    # Print errors if any
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r['status'] == 'error':
                print(f"  {Path(r['image']).name}: {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
