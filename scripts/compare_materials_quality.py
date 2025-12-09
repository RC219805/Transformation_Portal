#!/usr/bin/env python3
"""Compare Materials v2 enhanced outputs vs baseline.

Analyzes quality improvements in material rendering:
- Color/luma differences (should be minimal)
- Material fidelity (enhancement quality)
- Edge quality (soft vs hard transitions)
- Overall realism improvements
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class MaterialQualityMetrics:
    """Quality metrics for material rendering comparison."""
    
    image_name: str
    
    # Color accuracy
    mean_color_diff: float = 0.0
    max_color_diff: float = 0.0
    
    # Luminance
    mean_luma_diff: float = 0.0
    max_luma_diff: float = 0.0
    
    # Structural similarity
    structural_similarity: float = 1.0
    
    # Material-specific metrics
    material_regions: Dict[str, float] = None  # Region coverage
    enhancement_strength: float = 0.0
    
    # Edge quality
    edge_softness_score: float = 0.0
    edge_preservation_score: float = 1.0
    
    # Overall assessment
    realism_score: Optional[float] = None  # 1-5 scale
    quality_improvement: Optional[float] = None  # -1 to 1 scale
    
    def __post_init__(self):
        if self.material_regions is None:
            self.material_regions = {}


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img, dtype=np.float32) / 255.0


def calculate_color_diff(baseline: np.ndarray, enhanced: np.ndarray) -> Tuple[float, float]:
    """Calculate mean and max color difference."""
    diff = np.abs(baseline - enhanced)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    return mean_diff, max_diff


def calculate_luma_diff(baseline: np.ndarray, enhanced: np.ndarray) -> Tuple[float, float]:
    """Calculate luminance difference using Rec. 709 coefficients."""
    # Rec. 709 luma coefficients
    luma_coef = np.array([0.2126, 0.7152, 0.0722])
    
    baseline_luma = np.dot(baseline, luma_coef)
    enhanced_luma = np.dot(enhanced, luma_coef)
    
    diff = np.abs(baseline_luma - enhanced_luma)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    return mean_diff, max_diff


def calculate_ssim_simple(baseline: np.ndarray, enhanced: np.ndarray, 
                         window_size: int = 11) -> float:
    """Calculate simplified SSIM (structural similarity)."""
    # Convert to grayscale
    luma_coef = np.array([0.2126, 0.7152, 0.0722])
    baseline_gray = np.dot(baseline, luma_coef)
    enhanced_gray = np.dot(enhanced, luma_coef)
    
    # Calculate means
    mu1 = baseline_gray.mean()
    mu2 = enhanced_gray.mean()
    
    # Calculate variances and covariance
    var1 = np.var(baseline_gray)
    var2 = np.var(enhanced_gray)
    cov = np.cov(baseline_gray.flatten(), enhanced_gray.flatten())[0, 1]
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
    
    return float(ssim)


def detect_edges(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Simple edge detection using gradient magnitude."""
    # Convert to grayscale
    luma_coef = np.array([0.2126, 0.7152, 0.0722])
    gray = np.dot(image, luma_coef)
    
    # Calculate gradients
    gy, gx = np.gradient(gray)
    
    # Gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Threshold
    edges = magnitude > threshold
    
    return edges


def calculate_edge_quality(baseline: np.ndarray, enhanced: np.ndarray) -> Tuple[float, float]:
    """Calculate edge softness and preservation scores."""
    
    # Detect edges in both images
    baseline_edges = detect_edges(baseline)
    enhanced_edges = detect_edges(enhanced)
    
    # Edge preservation: how many baseline edges are preserved
    if baseline_edges.sum() > 0:
        preserved = np.logical_and(baseline_edges, enhanced_edges).sum()
        preservation_score = preserved / baseline_edges.sum()
    else:
        preservation_score = 1.0
    
    # Edge softness: measure gradient transition smoothness
    # (Lower magnitude = softer edges)
    luma_coef = np.array([0.2126, 0.7152, 0.0722])
    enhanced_gray = np.dot(enhanced, luma_coef)
    gy, gx = np.gradient(enhanced_gray)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Normalize softness score (inverse of mean gradient magnitude)
    softness_score = 1.0 - min(magnitude.mean() * 5, 1.0)
    
    return softness_score, preservation_score


def analyze_material_regions(enhanced_path: Path, cache_dir: Optional[Path] = None) -> Dict[str, float]:
    """Analyze material region coverage from cached masks."""
    
    if not cache_dir:
        cache_dir = Path(".materials_v2_cache")
    
    if not cache_dir.exists():
        return {}
    
    # Look for mask files
    image_stem = enhanced_path.stem
    mask_files = list(cache_dir.glob(f"{image_stem}_*.npy"))
    
    regions = {}
    for mask_file in mask_files:
        # Extract material type from filename
        # Format: {image_stem}_{material_type}_mask.npy
        parts = mask_file.stem.split('_')
        if len(parts) >= 2:
            material_type = parts[-2]  # Second to last part
            
            try:
                mask = np.load(mask_file)
                coverage = mask.sum() / mask.size * 100  # Percentage
                regions[material_type] = coverage
            except:
                pass
    
    return regions


def compare_images(
    baseline_path: Path,
    enhanced_path: Path,
    cache_dir: Optional[Path] = None,
) -> MaterialQualityMetrics:
    """Compare baseline and enhanced images."""
    
    print(f"Comparing: {baseline_path.name}")
    
    # Load images
    baseline = load_image(baseline_path)
    enhanced = load_image(enhanced_path)
    
    # Ensure same dimensions
    if baseline.shape != enhanced.shape:
        print(f"  Warning: Dimension mismatch - {baseline.shape} vs {enhanced.shape}")
        # Resize enhanced to match baseline
        enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
        enhanced_img = enhanced_img.resize((baseline.shape[1], baseline.shape[0]), Image.LANCZOS)
        enhanced = np.array(enhanced_img, dtype=np.float32) / 255.0
    
    # Calculate color differences
    mean_color_diff, max_color_diff = calculate_color_diff(baseline, enhanced)
    
    # Calculate luminance differences
    mean_luma_diff, max_luma_diff = calculate_luma_diff(baseline, enhanced)
    
    # Calculate structural similarity
    ssim = calculate_ssim_simple(baseline, enhanced)
    
    # Calculate edge quality
    edge_softness, edge_preservation = calculate_edge_quality(baseline, enhanced)
    
    # Analyze material regions
    material_regions = analyze_material_regions(enhanced_path, cache_dir)
    
    # Calculate enhancement strength (based on color difference)
    enhancement_strength = mean_color_diff
    
    metrics = MaterialQualityMetrics(
        image_name=baseline_path.stem,
        mean_color_diff=float(mean_color_diff),
        max_color_diff=float(max_color_diff),
        mean_luma_diff=float(mean_luma_diff),
        max_luma_diff=float(max_luma_diff),
        structural_similarity=float(ssim),
        material_regions=material_regions,
        enhancement_strength=float(enhancement_strength),
        edge_softness_score=float(edge_softness),
        edge_preservation_score=float(edge_preservation),
    )
    
    print(f"  Mean color diff: {mean_color_diff:.4f}")
    print(f"  Mean luma diff: {mean_luma_diff:.4f}")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  Edge preservation: {edge_preservation:.4f}")
    
    if material_regions:
        print(f"  Material coverage:")
        for material, coverage in material_regions.items():
            print(f"    {material}: {coverage:.1f}%")
    
    return metrics


def compare_directories(
    baseline_dir: Path,
    enhanced_dir: Path,
    cache_dir: Optional[Path] = None,
    output_report: str = "quality_comparison_report.json",
):
    """Compare all images in baseline vs enhanced directories."""
    
    print(f"\n{'=' * 60}")
    print("Materials v2 Quality Comparison")
    print(f"{'=' * 60}\n")
    print(f"Baseline: {baseline_dir}")
    print(f"Enhanced: {enhanced_dir}")
    print()
    
    # Find matching images
    baseline_images = sorted(baseline_dir.glob("*.tif*"))
    if not baseline_images:
        baseline_images = sorted(baseline_dir.glob("*.png"))
    
    results = []
    
    for baseline_path in baseline_images:
        # Find corresponding enhanced image
        enhanced_path = enhanced_dir / baseline_path.name
        
        # Try different extensions
        if not enhanced_path.exists():
            enhanced_path = enhanced_dir / baseline_path.with_suffix('.png').name
        if not enhanced_path.exists():
            enhanced_path = enhanced_dir / baseline_path.with_suffix('.tiff').name
        
        if not enhanced_path.exists():
            print(f"Warning: Enhanced image not found for {baseline_path.name}")
            continue
        
        # Compare images
        metrics = compare_images(baseline_path, enhanced_path, cache_dir)
        results.append(metrics)
        print()
    
    # Generate report
    report = {
        "comparison": "Materials v2 Enhanced vs Baseline",
        "baseline_dir": str(baseline_dir),
        "enhanced_dir": str(enhanced_dir),
        "image_count": len(results),
        "results": [asdict(m) for m in results],
    }
    
    # Calculate aggregate statistics
    if results:
        report["summary"] = {
            "avg_color_diff": sum(r.mean_color_diff for r in results) / len(results),
            "avg_luma_diff": sum(r.mean_luma_diff for r in results) / len(results),
            "avg_ssim": sum(r.structural_similarity for r in results) / len(results),
            "avg_edge_preservation": sum(r.edge_preservation_score for r in results) / len(results),
        }
    
    # Save report
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Images compared: {len(results)}")
    
    if "summary" in report:
        print(f"\nAverage color difference: {report['summary']['avg_color_diff']:.4f}")
        print(f"Average luma difference: {report['summary']['avg_luma_diff']:.4f}")
        print(f"Average SSIM: {report['summary']['avg_ssim']:.4f}")
        print(f"Average edge preservation: {report['summary']['avg_edge_preservation']:.4f}")
    
    print(f"\nReport saved to: {output_report}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Materials v2 quality")
    parser.add_argument("--baseline-dir", required=True,
                       help="Directory with baseline (no Materials v2) outputs")
    parser.add_argument("--enhanced-dir", required=True,
                       help="Directory with Materials v2 enhanced outputs")
    parser.add_argument("--cache-dir", default=".materials_v2_cache",
                       help="Materials v2 cache directory")
    parser.add_argument("--output", default="materials_v2_quality_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    baseline_dir = Path(args.baseline_dir)
    enhanced_dir = Path(args.enhanced_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        exit(1)
    
    if not enhanced_dir.exists():
        print(f"Error: Enhanced directory not found: {enhanced_dir}")
        exit(1)
    
    # Run comparison
    report = compare_directories(baseline_dir, enhanced_dir, cache_dir, args.output)
