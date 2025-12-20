#!/usr/bin/env python3
"""
Tiled Inference Validation: Prove "No Internal Resize" Claim
=============================================================

Critical validation to confirm the core architectural claim:
"Tiles are processed at native model resolution, not quietly resized."

This addresses the most important implementation fact that must be proven
before claiming the tiled approach delivers genuine high-resolution inference.

Reference: User feedback 2025-12-17 follow-up
"This is a common failure point: you can tile at 1024, but the processor 
quietly resizes to 518/384/256 internally—making tiling largely pointless."
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_instrumented_estimator():
    """Create depth estimator with preprocessing inspection."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.error("PyTorch and transformers required")
        return None
    
    # Create pipeline
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    logger.info(f"Loading model: {model_name}")
    
    pipe = hf_pipeline(
        "depth-estimation",
        model=model_name,
        device=-1  # CPU for validation (MPS has different behavior)
    )
    
    return pipe


def validate_tile_inference(
    tile_sizes: list = [512, 1024, 1536],
    test_image_size: Tuple[int, int] = (2048, 2048)
) -> Dict[str, any]:
    """
    Validate that tiles are processed at intended resolution.
    
    Returns:
        Validation report with actual tensor sizes
    """
    logger.info("=" * 60)
    logger.info("VALIDATION: Tile Inference Resolution")
    logger.info("=" * 60)
    
    # Create synthetic test image
    logger.info(f"Creating test image: {test_image_size}")
    test_img = np.random.randint(0, 255, (*test_image_size, 3), dtype=np.uint8)
    test_img_pil = Image.fromarray(test_img)
    
    # Results
    results = {
        'test_image_size': test_image_size,
        'tile_tests': []
    }
    
    # Create instrumented estimator
    pipe = create_instrumented_estimator()
    if pipe is None:
        return {'error': 'Failed to create estimator'}
    
    # Inspect image processor config
    if hasattr(pipe, 'image_processor'):
        processor = pipe.image_processor
        logger.info(f"\nImage processor config:")
        logger.info(f"  size: {getattr(processor, 'size', 'N/A')}")
        logger.info(f"  do_resize: {getattr(processor, 'do_resize', 'N/A')}")
        logger.info(f"  resample: {getattr(processor, 'resample', 'N/A')}")
    
    # Test each tile size
    for tile_size in tile_sizes:
        logger.info(f"\n--- Testing tile_size={tile_size} ---")
        
        # Create tile
        tile = test_img_pil.crop((0, 0, tile_size, tile_size))
        logger.info(f"Input tile PIL size: {tile.size}")
        
        # Run inference with hooks to capture preprocessing
        try:
            # Preprocess to see actual model input
            if hasattr(pipe, 'image_processor'):
                inputs = pipe.image_processor(images=tile, return_tensors="pt")
                if 'pixel_values' in inputs:
                    preprocessed_shape = tuple(inputs['pixel_values'].shape)
                    logger.info(f"Preprocessed tensor shape: {preprocessed_shape}")
                    
                    # Check for resize
                    batch, channels, input_h, input_w = preprocessed_shape
                    resize_detected = (input_h != tile_size or input_w != tile_size)
                    
                    if resize_detected:
                        logger.warning(f"⚠️  RESIZE DETECTED: {tile_size}×{tile_size} → {input_h}×{input_w}")
                        resize_factor = min(input_h / tile_size, input_w / tile_size)
                        verdict = 'FAIL: Internal resize detected'
                    else:
                        logger.info(f"✓ No resize: tile={tile_size}×{tile_size}, model input={input_h}×{input_w}")
                        resize_factor = 1.0
                        verdict = 'PASS: High-res preserved'
            
            # Run full pipeline
            result = pipe(tile)
            
            # Extract depth map
            if hasattr(result, 'depth'):
                depth_map = np.array(result.depth)
            elif isinstance(result, dict) and 'depth' in result:
                depth_map = np.array(result['depth'])
            else:
                depth_map = np.array(result)
            
            logger.info(f"Output depth shape: {depth_map.shape}")
            
            test_result = {
                'tile_size': tile_size,
                'preprocessed_shape': preprocessed_shape,
                'output_depth_shape': depth_map.shape,
                'resize_detected': resize_detected,
                'resize_factor': resize_factor,
                'verdict': verdict
            }
            
            results['tile_tests'].append(test_result)
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            results['tile_tests'].append({
                'tile_size': tile_size,
                'error': str(e)
            })
    
    # Overall verdict
    all_passed = all(
        t.get('verdict', '').startswith('PASS') 
        for t in results['tile_tests']
        if 'verdict' in t
    )
    
    results['overall_verdict'] = 'PASS: Tiled inference is genuinely high-res' if all_passed else 'FAIL: Internal resizing detected'
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    for test in results['tile_tests']:
        logger.info(f"Tile {test.get('tile_size')}px: {test.get('verdict', test.get('error', 'N/A'))}")
    logger.info(f"\nOverall: {results['overall_verdict']}")
    
    return results


def validate_global_consistency():
    """
    Validate tile blending for global consistency.
    
    Tests:
    1. Tile seam visibility
    2. Low-frequency banding
    3. Plane continuity across tiles
    """
    logger.info("=" * 60)
    logger.info("VALIDATION: Global Consistency")
    logger.info("=" * 60)
    
    # Create test pattern with known structure
    # Horizontal gradient + vertical stripes
    h, w = 2048, 2048
    test_pattern = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Horizontal gradient
    for i in range(h):
        test_pattern[i, :] = int(255 * i / h)
    
    # Vertical stripes (every 256px)
    for j in range(0, w, 256):
        test_pattern[:, j:j+32] = 255
    
    logger.info(f"Created test pattern: {test_pattern.shape}")
    logger.info("Pattern: Horizontal gradient + vertical stripes (256px period)")
    
    # TODO: Run tiled inference and analyze
    # 1. Extract tiles
    # 2. Process each
    # 3. Measure consistency in overlap regions
    # 4. Check for seam artifacts
    
    results = {
        'test_pattern_size': (h, w),
        'validation_status': 'TODO: Implement tile blending consistency checks',
        'recommended_fixes': [
            'Global anchor pass (low-res full frame)',
            'High-frequency residual from tiles',
            'Prevents global drift and plane warping'
        ]
    }
    
    return results


def generate_validation_report(output_path: Path):
    """Generate comprehensive validation report."""
    report = {
        'validation_date': '2025-12-18',
        'purpose': 'Validate core claims before calling Phase 1 complete',
        'critical_validations': []
    }
    
    # Validation 1: No internal resize
    logger.info("\n\n")
    tile_results = validate_tile_inference()
    report['critical_validations'].append({
        'name': 'No Internal Resize',
        'status': tile_results.get('overall_verdict', 'N/A'),
        'details': tile_results
    })
    
    # Validation 2: Global consistency
    logger.info("\n\n")
    consistency_results = validate_global_consistency()
    report['critical_validations'].append({
        'name': 'Global Consistency',
        'status': consistency_results.get('validation_status', 'N/A'),
        'details': consistency_results
    })
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✓ Validation report saved: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate tiled depth inference implementation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_report_tiled_inference.json"),
        help="Output path for validation report"
    )
    parser.add_argument(
        "--tile-sizes",
        nargs='+',
        type=int,
        default=[512, 1024, 1536],
        help="Tile sizes to test"
    )
    
    args = parser.parse_args()
    
    report = generate_validation_report(args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for validation in report['critical_validations']:
        print(f"{validation['name']}: {validation['status']}")
