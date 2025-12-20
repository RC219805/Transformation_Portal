#!/usr/bin/env python3
"""Test DA3 normalization fix on sample images."""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import json
import pytest

# Skip if torch not available (ML dependency)
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent / "lux_depth_v3"))
sys.path.insert(0, str(Path(__file__).parent / "high_fidelity_depth"))

from lux_depth_v3.config import DA3Config, DA3APIConfig, ModelVariant, InferenceMode
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput
from high_fidelity_depth.quality_metrics import validate_depth_quality

def normalize_depth_old(depth):
    """Old normalization (min-max)."""
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def normalize_depth_new(depth):
    """New normalization (inverse depth aware)."""
    depth_range = depth.max() - depth.min()
    if depth_range < 1.0 and depth.mean() > 0.5:
        # Narrow range near 1.0 suggests inverse depth
        depth_disparity = 1.0 / (depth + 1e-6)
        return (depth_disparity - depth_disparity.min()) / (depth_disparity.max() - depth_disparity.min() + 1e-8)
    else:
        # Standard min-max normalization
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# Test images (structure and texture)
test_images = [
    "data/validation_full/800-picacho-12.jpg",  # Structure
    "data/validation_full/800-picacho-11.jpg",  # Texture
]

# Find actual available images
available = []
for img in test_images:
    if Path(img).exists():
        available.append(Path(img))

if not available:
    available = list(Path("data/validation_full").glob("*.jpg"))[:3]


@pytest.mark.skipif(not available, reason="No test images available")
def test_da3_normalization_fix():
    """Validate DA3 normalization methods (skip if DA3 not available)."""
    print("="*80)
    print("DA3 NORMALIZATION FIX VALIDATION")
    print("="*80)

    # Initialize DA3 with higher resolution for architectural detail
    api_config = DA3APIConfig(
        process_res=1022,  # 2x increase vs default 504px
        process_res_method="upper_bound_resize",
    )

    config = DA3Config(
        model_variant=ModelVariant.DA3_LARGE_V1_1,
        inference_mode=InferenceMode.MONOCULAR,
        api=api_config,
    )

    print("\nInitializing DA3 engine (1022px processing resolution)...")
    try:
        engine = DA3InferenceEngine(config, commercial_use=False)
    except RuntimeError as e:
        pytest.skip(f"DA3 Python API not available: {e}")

    # Load baseline metrics for comparison
    baseline_dir = Path("validation_v1_baseline_pack/46img_validation_results")

    results = []

    for img_path in available[:3]:  # Test first 3 images
        print(f"\n{'='*80}")
        print(f"Image: {img_path.name}")
        print(f"{'='*80}")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Run DA3
        image_input = ImageInput(path=img_path)
        da3_result = engine.infer([image_input])
        depth_raw = da3_result.depth[0]
        
        print(f"\nRaw depth: min={depth_raw.min():.6f}, max={depth_raw.max():.6f}, range={depth_raw.ptp():.6f}")
        
        # Apply both normalizations
        depth_old = normalize_depth_old(depth_raw)
        depth_new = normalize_depth_new(depth_raw)
        
        # Compute quality metrics
        metrics_old = validate_depth_quality(image_np, depth_old, image_filename=img_path.stem)
        metrics_new = validate_depth_quality(image_np, depth_new, image_filename=img_path.stem)
        
        # Convert to dicts
        if hasattr(metrics_old, '__dict__'):
            metrics_old = {k: v for k, v in metrics_old.__dict__.items()}
        if hasattr(metrics_new, '__dict__'):
            metrics_new = {k: v for k, v in metrics_new.__dict__.items()}
        
        # Find baseline
        baseline_file = baseline_dir / f"{img_path.stem}_metrics.json"
        da2_metrics = None
        if baseline_file.exists():
            with open(baseline_file) as f:
                da2_metrics = json.load(f)
        
        print(f"\n📊 OLD Normalization (min-max):")
        print(f"   Edge F1: {metrics_old['edge_f1']:.4f}")
        print(f"   Chamfer: {metrics_old['chamfer_distance']:.2f}")
        print(f"   Scene: {metrics_old['scene_type']}")
        print(f"   Lenient Pass: {metrics_old.get('lenient_pass', False)}")
        
        print(f"\n📊 NEW Normalization (inverse-aware):")
        print(f"   Edge F1: {metrics_new['edge_f1']:.4f}")
        print(f"   Chamfer: {metrics_new['chamfer_distance']:.2f}")
        print(f"   Scene: {metrics_new['scene_type']}")
        print(f"   Lenient Pass: {metrics_new.get('lenient_pass', False)}")
        
        if da2_metrics:
            print(f"\n📊 DA2 Baseline:")
            print(f"   Edge F1: {da2_metrics['edge_f1']:.4f}")
            print(f"   Chamfer: {da2_metrics.get('chamfer_px', da2_metrics.get('chamfer_distance', 0)):.2f}")
            print(f"   Scene: {da2_metrics['scene_type']}")
            print(f"   Lenient Pass: {da2_metrics.get('lenient_pass', False)}")
        
        improvement = metrics_new['edge_f1'] - metrics_old['edge_f1']
        print(f"\n✨ Improvement: {improvement:+.4f} Edge F1")
        
        if improvement > 0.05:
            print("   ✅ SIGNIFICANT IMPROVEMENT")
        elif improvement > 0:
            print("   ✓ Minor improvement")
        else:
            print("   ⚠️ No improvement or regression")
        
        results.append({
            'image': img_path.stem,
            'old': metrics_old,
            'new': metrics_new,
            'da2': da2_metrics,
            'improvement': improvement
        })

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    avg_improvement = np.mean([r['improvement'] for r in results])
    old_pass_count = sum(1 for r in results if r['old'].get('lenient_pass', False))
    new_pass_count = sum(1 for r in results if r['new'].get('lenient_pass', False))

    print(f"\nTested {len(results)} images:")
    print(f"  Average F1 improvement: {avg_improvement:+.4f}")
    print(f"  Old pass rate: {old_pass_count}/{len(results)} ({100*old_pass_count/len(results):.1f}%)")
    print(f"  New pass rate: {new_pass_count}/{len(results)} ({100*new_pass_count/len(results):.1f}%)")

    if new_pass_count > old_pass_count:
        print(f"\n✅ FIX SUCCESSFUL: +{new_pass_count - old_pass_count} passes")
    elif avg_improvement > 0.05:
        print(f"\n✓ Metrics improved but may need threshold adjustment")
    else:
        print(f"\n⚠️ Fix did not improve results - try alternative approach")
    
    # Assertions for test validation
    assert len(results) > 0, "No images processed"
    assert all('edge_f1' in r['old'] for r in results), "Missing old metrics"
    assert all('edge_f1' in r['new'] for r in results), "Missing new metrics"
