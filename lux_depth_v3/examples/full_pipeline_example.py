#!/usr/bin/env python3
"""
Full pipeline example for Lux Depth V3.

Demonstrates complete workflow from input to export with validation.
"""

from pathlib import Path
import numpy as np
from PIL import Image

from lux_depth_v3 import (
    DA3Config,
    Preset,
    InputManager,
    DA3InferenceEngine,
)
from lux_depth_v3.postprocessing import Postprocessor
from lux_depth_v3.validation import DepthValidator, ValidationReport
from lux_depth_v3.export import Exporter
from lux_depth_v3.config import ExportFormat


def main():
    """Run full DA3 pipeline example."""
    
    print("=" * 60)
    print("Lux Depth V3 - Full Pipeline Example")
    print("=" * 60)
    
    # Configuration
    print("\n1. Configuration")
    config = DA3Config.from_preset(Preset.INTERIOR_LUXURY)
    config.export.output_dir = Path("example_output")
    config.export.formats = [ExportFormat.PNG, ExportFormat.NPZ, ExportFormat.PLY]
    
    print(f"   Model: {config.model_variant.value}")
    print(f"   Mode: {config.inference_mode.value}")
    print(f"   Device: {config.device.device}")
    print(f"   Output: {config.export.output_dir}")
    
    # Input
    print("\n2. Input Setup")
    manager = InputManager()
    
    # Create sample images for demonstration
    sample_dir = Path("example_input")
    sample_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        # Generate sample image (gradient pattern)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 512).reshape(1, -1)  # Red gradient
        img[:, :, 1] = np.linspace(0, 255, 512).reshape(-1, 1)  # Green gradient
        img[:, :, 2] = 128  # Constant blue
        
        img_path = sample_dir / f"sample_{i:02d}.jpg"
        Image.fromarray(img).save(img_path)
        manager.add_image(path=img_path)
    
    print(f"   Loaded {len(manager.inputs)} images")
    stats = manager.get_statistics()
    print(f"   Average size: {stats['avg_size']}")
    
    # Inference
    print("\n3. Model Loading & Inference")
    engine = DA3InferenceEngine(config)
    engine.load_model()
    
    print("   Running inference...")
    results = []
    for i, img_input in enumerate(manager.get_images()):
        result = engine.inference(img_input)
        results.append(result)
        print(f"   ✓ Image {i+1}: depth range {result.get_depth_range()}")
    
    # Postprocessing
    print("\n4. Postprocessing")
    postprocessor = Postprocessor(config.postprocessing)
    
    for i, result in enumerate(results):
        result = postprocessor.process(result)
        results[i] = result
        print(f"   ✓ Processed image {i+1}")
    
    # Validation (optional, requires ground truth)
    print("\n5. Validation")
    validator = DepthValidator()
    report = ValidationReport()
    
    for i, result in enumerate(results):
        # Create synthetic ground truth for demo
        synthetic_gt = result.depth_map + np.random.normal(0, 0.01, result.depth_map.shape)
        
        metrics = validator.validate(result, ground_truth=synthetic_gt)
        report.add_result(metrics)
        
        if metrics.passes_quality_gate():
            print(f"   ✓ Image {i+1}: Quality gate passed")
            print(f"     RMSE: {metrics.rmse:.4f}, δ1: {metrics.delta_1:.3f}")
        else:
            print(f"   ✗ Image {i+1}: Quality gate failed")
    
    # Save validation report
    config.export.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.export.output_dir / "validation_report.json"
    report.save(report_path)
    
    summary = report.compute_summary()
    print(f"\n   Validation Summary:")
    print(f"   - Mean RMSE: {summary['mean_rmse']:.4f}")
    print(f"   - Mean δ1: {summary['mean_delta_1']:.3f}")
    
    # Export
    print("\n6. Export Results")
    exporter = Exporter(config.export)
    
    for i, result in enumerate(results):
        filename_base = f"sample_{i:02d}"
        exported = exporter.export(result, filename_base)
        
        print(f"   ✓ Exported image {i+1}:")
        for fmt, path in exported.items():
            print(f"     - {fmt}: {path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.export.output_dir}")
    print(f"Validation report: {report_path}")
    print("\nNext steps:")
    print("  - Review exported depth maps")
    print("  - Check validation report")
    print("  - Visualize point clouds (PLY files)")
    print("  - Integrate into production pipeline")


if __name__ == "__main__":
    main()
