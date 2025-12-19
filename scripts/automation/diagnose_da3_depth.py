#!/usr/bin/env python3
"""
Diagnose DA3 vs DA2 depth output differences.
Compare raw depth values, distributions, and edge characteristics.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def load_depth_tiff(path):
    """Load depth map from TIFF."""
    img = Image.open(path)
    depth = np.array(img, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]  # Take first channel if RGB
    return depth

def analyze_depth_stats(depth, name):
    """Compute comprehensive depth statistics."""
    stats = {
        'name': name,
        'shape': depth.shape,
        'min': float(np.min(depth)),
        'max': float(np.max(depth)),
        'mean': float(np.mean(depth)),
        'median': float(np.median(depth)),
        'std': float(np.std(depth)),
        'p2': float(np.percentile(depth, 2)),
        'p98': float(np.percentile(depth, 98)),
        'range': float(np.ptp(depth)),
        'non_zero_min': float(np.min(depth[depth > 0])) if np.any(depth > 0) else 0.0,
    }
    
    # Check for inverse depth characteristics
    depth_flat = depth.flatten()
    depth_nonzero = depth_flat[depth_flat > 0]
    if len(depth_nonzero) > 0:
        # Inverse depth would have small values for far objects
        inv_depth = 1.0 / (depth_nonzero + 1e-6)
        stats['inv_depth_range'] = float(np.ptp(inv_depth))
        stats['inv_depth_mean'] = float(np.mean(inv_depth))
    
    return stats

def visualize_comparison(da3_depth, da2_depth, output_path, image_name):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'DA3 vs DA2 Depth Comparison: {image_name}', fontsize=16)
    
    # Normalize for visualization
    da3_norm = (da3_depth - da3_depth.min()) / (da3_depth.max() - da3_depth.min() + 1e-8)
    da2_norm = (da2_depth - da2_depth.min()) / (da2_depth.max() - da2_depth.min() + 1e-8)
    
    # Row 1: Depth maps
    axes[0, 0].imshow(da3_norm, cmap='viridis')
    axes[0, 0].set_title('DA3 Depth')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(da2_norm, cmap='viridis')
    axes[0, 1].set_title('DA2 Depth (Baseline)')
    axes[0, 1].axis('off')
    
    diff = np.abs(da3_norm - da2_norm)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title(f'Absolute Difference (mean={diff.mean():.3f})')
    axes[0, 2].axis('off')
    
    # Row 2: Histograms
    axes[1, 0].hist(da3_depth.flatten(), bins=100, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('DA3 Raw Distribution')
    axes[1, 0].set_xlabel('Depth Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(da2_depth.flatten(), bins=100, alpha=0.7, color='green', density=True)
    axes[1, 1].set_title('DA2 Raw Distribution')
    axes[1, 1].set_xlabel('Depth Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overlay normalized histograms
    axes[1, 2].hist(da3_norm.flatten(), bins=50, alpha=0.5, color='blue', label='DA3', density=True)
    axes[1, 2].hist(da2_norm.flatten(), bins=50, alpha=0.5, color='green', label='DA2', density=True)
    axes[1, 2].set_title('Normalized Distributions')
    axes[1, 2].set_xlabel('Normalized Depth [0-1]')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {output_path}")

def main():
    """Run diagnostic comparison."""
    # Find sample images across scene types
    da3_dir = Path("outputs/da3_ab_validation_FIXED_20251219_123409")
    da2_dir = Path("validation_v1_baseline_pack/46img_validation_results")
    
    # Load DA3 metrics to find samples
    with open(da3_dir / "da3_metrics.json") as f:
        da3_metrics = json.load(f)
    
    # Select 3 representative images
    sample_images = []
    for key, metrics in da3_metrics.items():
        scene_type = metrics.get('scene_type', 'unknown')
        if 'structure' in scene_type and len([s for s in sample_images if s[1] == 'structure']) == 0:
            sample_images.append((key, 'structure', metrics))
        elif 'texture' in scene_type and len([s for s in sample_images if s[1] == 'texture']) == 0:
            sample_images.append((key, 'texture', metrics))
        if len(sample_images) >= 3:
            break
    
    if len(sample_images) < 2:
        # Just take first 3
        sample_images = list(da3_metrics.items())[:3]
        sample_images = [(k, v.get('scene_type', 'unknown'), v) for k, v in sample_images]
    
    print("=" * 80)
    print("DA3 DEPTH QUALITY DIAGNOSTIC")
    print("=" * 80)
    print(f"\nAnalyzing {len(sample_images)} sample images:\n")
    
    results = []
    
    for img_key, scene_type, da3_met in sample_images:
        # Find corresponding DA2 baseline
        # DA3 keys might have prefixes, need to match to DA2 filenames
        print(f"\n{'='*80}")
        print(f"Image: {img_key} (Scene: {scene_type})")
        print(f"{'='*80}")
        
        # Find DA2 baseline file
        da2_metrics_files = list(da2_dir.glob("*_metrics.json"))
        da2_match = None
        for da2_file in da2_metrics_files:
            # Try to find matching image
            basename = da2_file.stem.replace('_metrics', '')
            if basename in img_key or img_key.replace('V2_', '') in basename:
                da2_match = da2_file
                break
        
        if not da2_match:
            print(f"⚠ Could not find DA2 baseline for {img_key}")
            continue
        
        # Load DA2 metrics
        with open(da2_match) as f:
            da2_met = json.load(f)
        
        print(f"\nDA3 Metrics:")
        print(f"  Edge F1: {da3_met['edge_f1']:.4f}")
        print(f"  Chamfer: {da3_met['chamfer_distance']:.2f}")
        print(f"  Scene: {da3_met['scene_type']}")
        print(f"  Edge Count Ratio: {da3_met['edge_count_ratio']:.2f}")
        
        print(f"\nDA2 Metrics:")
        print(f"  Edge F1: {da2_met['edge_f1']:.4f}")
        print(f"  Chamfer: {da2_met['chamfer_px']:.2f}")
        print(f"  Scene: {da2_met['scene_type']}")
        print(f"  Edge Count Ratio: {da2_met['edge_count_ratio']:.2f}")
        print(f"  Pass: {da2_met.get('lenient_pass', False)}")
        
        # Try to load depth TIFFs
        da3_depth_file = None
        da2_depth_file = da2_match.parent / da2_match.name.replace('_metrics.json', '_depth.tiff')
        
        # Find DA3 depth file
        validation_dir = Path("data/validation_full")
        possible_names = [
            img_key + ".jpg",
            img_key.replace('V2_', '') + ".jpg",
            img_key.replace('V2_', '').replace('_', '') + ".jpg",
        ]
        
        for poss_name in possible_names:
            depth_name = poss_name.replace('.jpg', '_depth.tiff')
            candidate = da3_dir / depth_name
            if candidate.exists():
                da3_depth_file = candidate
                break
        
        if not da3_depth_file or not da3_depth_file.exists():
            print(f"⚠ DA3 depth not found, checking alternate locations...")
            continue
        
        if not da2_depth_file.exists():
            print(f"⚠ DA2 depth not found: {da2_depth_file}")
            continue
        
        # Load and analyze depths
        print(f"\n📊 Loading depth maps...")
        da3_depth = load_depth_tiff(da3_depth_file)
        da2_depth = load_depth_tiff(da2_depth_file)
        
        da3_stats = analyze_depth_stats(da3_depth, "DA3")
        da2_stats = analyze_depth_stats(da2_depth, "DA2")
        
        print(f"\nDA3 Depth Stats:")
        print(f"  Shape: {da3_stats['shape']}")
        print(f"  Range: [{da3_stats['min']:.6f}, {da3_stats['max']:.6f}]")
        print(f"  Mean ± Std: {da3_stats['mean']:.6f} ± {da3_stats['std']:.6f}")
        print(f"  Median: {da3_stats['median']:.6f}")
        print(f"  P2-P98: [{da3_stats['p2']:.6f}, {da3_stats['p98']:.6f}]")
        
        print(f"\nDA2 Depth Stats:")
        print(f"  Shape: {da2_stats['shape']}")
        print(f"  Range: [{da2_stats['min']:.6f}, {da2_stats['max']:.6f}]")
        print(f"  Mean ± Std: {da2_stats['mean']:.6f} ± {da2_stats['std']:.6f}")
        print(f"  Median: {da2_stats['median']:.6f}")
        print(f"  P2-P98: [{da2_stats['p2']:.6f}, {da2_stats['p98']:.6f}]")
        
        # Key diagnostic: check if ranges differ significantly
        da3_range = da3_stats['p98'] - da3_stats['p2']
        da2_range = da2_stats['p98'] - da2_stats['p2']
        range_ratio = da3_range / (da2_range + 1e-8)
        
        print(f"\n🔍 Key Diagnostics:")
        print(f"  DA3 effective range (P2-P98): {da3_range:.6f}")
        print(f"  DA2 effective range (P2-P98): {da2_range:.6f}")
        print(f"  Range ratio (DA3/DA2): {range_ratio:.3f}")
        
        if abs(range_ratio - 1.0) > 0.5:
            print(f"  ⚠️ SIGNIFICANT RANGE DIFFERENCE DETECTED")
        
        # Create visualization
        vis_path = Path("outputs") / f"depth_comparison_{img_key[:30]}.png"
        visualize_comparison(da3_depth, da2_depth, vis_path, img_key)
        
        results.append({
            'image': img_key,
            'scene_type': scene_type,
            'da3_stats': da3_stats,
            'da2_stats': da2_stats,
            'da3_metrics': da3_met,
            'da2_metrics': da2_met,
            'range_ratio': range_ratio,
        })
    
    # Save diagnostic report
    report_path = Path("outputs/da3_depth_diagnostic_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved diagnostic report: {report_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nAnalyzed {len(results)} images")
    
    if results:
        avg_range_ratio = np.mean([r['range_ratio'] for r in results])
        print(f"Average DA3/DA2 range ratio: {avg_range_ratio:.3f}")
        
        if avg_range_ratio < 0.5 or avg_range_ratio > 2.0:
            print("\n⚠️ DIAGNOSIS: DEPTH SCALE MISMATCH DETECTED")
            print("   DA3 outputs depth in different scale than DA2")
            print("   Recommendation: Adjust normalization in da3_wrapper.py")

if __name__ == '__main__':
    main()
