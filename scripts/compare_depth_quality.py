#!/usr/bin/env python3
"""
Depth Map Quality Comparison Tool

Compares depth maps generated from different source inputs (e.g., master vs upscaled TIFF)
to establish canonical depth source strategy for each quality tier.

Usage:
    python scripts/compare_depth_quality.py \
        --master input_images/750_Picacho/Source_TIFFs/750Picacho_Kitchen_16bit.tiff \
        --upscaled output_kitchen_apex/750Picacho_Kitchen_16bit_upscaled16.tif \
        --output-dir depth_comparison_results/

Metrics:
    - L1/L2 error between normalized depth maps
    - SSIM (structural similarity)
    - Edge consistency (gradient correlation along object boundaries)
    - Noise profile in flat surfaces (pool water, sky, walls)

Outputs:
    - Side-by-side depth visualizations
    - Edge maps (Sobel over depth)
    - Quantitative metrics JSON
    - Recommendation summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# Optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.filters import sobel
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_generate_depth(image_path: Path, model: str = "depth-anything/Depth-Anything-V2-Large") -> np.ndarray:
    """
    Load image and generate depth map using Depth Anything V2.
    
    Args:
        image_path: Path to input image
        model: HuggingFace model ID
    
    Returns:
        Normalized depth map (0-1 range, float32)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and torch required. Install with: pip install transformers torch")
    
    logger.info(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    logger.info(f"Generating depth map with {model}")
    depth_pipe = pipeline(task="depth-estimation", model=model, device="mps" if torch.backends.mps.is_available() else "cpu")
    depth_result = depth_pipe(img)
    
    # Convert to numpy and normalize
    depth_map = np.array(depth_result["depth"])
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    logger.info(f"Depth map generated: shape={depth_map.shape}, dtype={depth_map.dtype}")
    return depth_map.astype(np.float32)


def compute_metrics(depth1: np.ndarray, depth2: np.ndarray) -> Dict[str, float]:
    """
    Compute quantitative comparison metrics between two depth maps.
    
    Metrics:
        - L1 error (MAE)
        - L2 error (RMSE)
        - SSIM (structural similarity, requires scikit-image)
        - Edge correlation (Sobel gradient similarity)
        - Flat region noise (std dev in low-gradient areas)
    
    Args:
        depth1: First depth map (0-1 range)
        depth2: Second depth map (0-1 range, must match depth1 shape)
    
    Returns:
        Dictionary of metric name -> value
    """
    if depth1.shape != depth2.shape:
        raise ValueError(f"Depth maps must have same shape: {depth1.shape} vs {depth2.shape}")
    
    metrics = {}
    
    # L1 and L2 error
    l1_error = np.mean(np.abs(depth1 - depth2))
    l2_error = np.sqrt(np.mean((depth1 - depth2) ** 2))
    metrics["l1_mae"] = float(l1_error)
    metrics["l2_rmse"] = float(l2_error)
    
    # SSIM (if available)
    if SKIMAGE_AVAILABLE:
        ssim_val = ssim(depth1, depth2, data_range=1.0)
        metrics["ssim"] = float(ssim_val)
        
        # Edge consistency using Sobel
        edges1 = sobel(depth1)
        edges2 = sobel(depth2)
        edge_corr = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
        metrics["edge_correlation"] = float(edge_corr)
        
        # Noise in flat regions (low gradient areas)
        gradient_mag1 = sobel(depth1)
        flat_mask = gradient_mag1 < np.percentile(gradient_mag1, 20)  # Bottom 20% gradients
        if np.any(flat_mask):
            flat_noise1 = np.std(depth1[flat_mask])
            flat_noise2 = np.std(depth2[flat_mask])
            metrics["flat_noise_depth1"] = float(flat_noise1)
            metrics["flat_noise_depth2"] = float(flat_noise2)
            metrics["flat_noise_ratio"] = float(flat_noise2 / (flat_noise1 + 1e-8))
    else:
        logger.warning("scikit-image not available; skipping SSIM and edge metrics")
    
    return metrics


def visualize_comparison(
    depth1: np.ndarray,
    depth2: np.ndarray,
    output_dir: Path,
    prefix: str = "comparison"
) -> None:
    """
    Generate side-by-side visualizations and edge maps.
    
    Outputs:
        - {prefix}_depth_master.png
        - {prefix}_depth_upscaled.png
        - {prefix}_edges_master.png (if scikit-image available)
        - {prefix}_edges_upscaled.png
        - {prefix}_difference.png
    
    Args:
        depth1: Master depth map
        depth2: Upscaled depth map
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Depth visualizations (grayscale)
    Image.fromarray((depth1 * 255).astype(np.uint8)).save(output_dir / f"{prefix}_depth_master.png")
    Image.fromarray((depth2 * 255).astype(np.uint8)).save(output_dir / f"{prefix}_depth_upscaled.png")
    
    # Difference map (highlight discrepancies)
    diff = np.abs(depth1 - depth2)
    diff_vis = (diff / diff.max() * 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(output_dir / f"{prefix}_difference.png")
    logger.info(f"Saved difference map: max_diff={diff.max():.4f}")
    
    # Edge maps (if available)
    if SKIMAGE_AVAILABLE:
        edges1 = sobel(depth1)
        edges2 = sobel(depth2)
        Image.fromarray((edges1 / edges1.max() * 255).astype(np.uint8)).save(output_dir / f"{prefix}_edges_master.png")
        Image.fromarray((edges2 / edges2.max() * 255).astype(np.uint8)).save(output_dir / f"{prefix}_edges_upscaled.png")
        logger.info("Saved edge maps")


def generate_recommendation(metrics: Dict[str, float], output_path: Path) -> str:
    """
    Generate human-readable recommendation based on metrics.
    
    Decision thresholds:
        - SSIM > 0.95: "Nearly identical, use either"
        - SSIM 0.90-0.95: "Very similar, prefer master for APEX, upscaled OK for Max"
        - SSIM 0.80-0.90: "Moderate difference, use master for quality-critical work"
        - SSIM < 0.80: "Significant difference, always use master for depth"
    
    Args:
        metrics: Computed metrics dictionary
        output_path: Where to write recommendation
    
    Returns:
        Recommendation text
    """
    ssim_val = metrics.get("ssim", None)
    l1_mae = metrics["l1_mae"]
    l2_rmse = metrics["l2_rmse"]
    
    recommendation = f"""
# Depth Map Quality Comparison - Recommendation

## Metrics Summary

- **L1 MAE:** {l1_mae:.6f}
- **L2 RMSE:** {l2_rmse:.6f}
"""
    
    if ssim_val is not None:
        recommendation += f"- **SSIM:** {ssim_val:.4f}\n"
        recommendation += f"- **Edge Correlation:** {metrics.get('edge_correlation', 0.0):.4f}\n"
        recommendation += f"- **Flat Noise Ratio:** {metrics.get('flat_noise_ratio', 1.0):.4f}\n\n"
        
        if ssim_val > 0.95:
            tier = "NEARLY IDENTICAL"
            advice = "Use either source for depth. Prefer master for archival consistency."
        elif ssim_val > 0.90:
            tier = "VERY SIMILAR"
            advice = "Prefer master for APEX tier. Upscaled acceptable for Standard/Max."
        elif ssim_val > 0.80:
            tier = "MODERATE DIFFERENCE"
            advice = "Use master for APEX and Max Quality. Upscaled OK for Standard/preview."
        else:
            tier = "SIGNIFICANT DIFFERENCE"
            advice = "Always use master TIFF for depth generation across all tiers."
        
        recommendation += f"## Verdict: {tier}\n\n"
        recommendation += f"**Recommendation:** {advice}\n\n"
    else:
        recommendation += "\n## Verdict: METRICS INCOMPLETE\n\n"
        recommendation += "Install scikit-image for full analysis: `pip install scikit-image`\n\n"
    
    recommendation += f"""
## Quality Tier Guidelines

| Tier | Depth Source | Rationale |
|------|--------------|-----------|
| **Standard** | Master or Upscaled | Faster processing, minimal visual difference in previews |
| **Max Quality** | Master | Better edge preservation for client stills |
| **APEX** | Master | Maximum accuracy for archival/hero frames |

## Technical Notes

- Edge correlation measures how well depth gradients align along object boundaries.
- Flat noise ratio >1.0 indicates upscaled source adds noise in smooth areas (sky, water).
- L1/L2 errors are normalized (0-1 range); typical acceptable threshold: L1 <0.02, L2 <0.03.

---
Generated by: scripts/compare_depth_quality.py
"""
    
    output_path.write_text(recommendation)
    logger.info(f"Recommendation written to: {output_path}")
    return recommendation


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare depth map quality from different source inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--master", type=Path, required=True, help="Master 16-bit TIFF input")
    parser.add_argument("--upscaled", type=Path, required=True, help="Upscaled 16-bit TIFF input")
    parser.add_argument("--output-dir", type=Path, default=Path("depth_comparison_results"), help="Output directory")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large", help="Depth model ID")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    if not args.master.exists():
        logger.error(f"Master image not found: {args.master}")
        return 1
    
    if not args.upscaled.exists():
        logger.error(f"Upscaled image not found: {args.upscaled}")
        return 1
    
    # Generate depth maps
    logger.info("=" * 80)
    logger.info("DEPTH MAP GENERATION - MASTER SOURCE")
    logger.info("=" * 80)
    depth_master = load_and_generate_depth(args.master, args.model)
    
    logger.info("=" * 80)
    logger.info("DEPTH MAP GENERATION - UPSCALED SOURCE")
    logger.info("=" * 80)
    depth_upscaled = load_and_generate_depth(args.upscaled, args.model)
    
    # Ensure same size (resize if needed)
    if depth_master.shape != depth_upscaled.shape:
        logger.warning(f"Shape mismatch: master={depth_master.shape}, upscaled={depth_upscaled.shape}")
        logger.info("Resizing upscaled to match master")
        from PIL import Image as PILImage
        depth_upscaled_pil = PILImage.fromarray((depth_upscaled * 255).astype(np.uint8))
        depth_upscaled_pil = depth_upscaled_pil.resize(
            (depth_master.shape[1], depth_master.shape[0]),
            PILImage.Resampling.BICUBIC
        )
        depth_upscaled = np.array(depth_upscaled_pil).astype(np.float32) / 255.0
    
    # Compute metrics
    logger.info("=" * 80)
    logger.info("COMPUTING METRICS")
    logger.info("=" * 80)
    metrics = compute_metrics(depth_master, depth_upscaled)
    
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.6f}")
    
    # Save metrics JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics saved: {metrics_path}")
    
    # Generate visualizations
    if not args.skip_viz:
        logger.info("=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        visualize_comparison(depth_master, depth_upscaled, args.output_dir, prefix="comparison")
    
    # Generate recommendation
    logger.info("=" * 80)
    logger.info("GENERATING RECOMMENDATION")
    logger.info("=" * 80)
    recommendation_path = args.output_dir / "RECOMMENDATION.md"
    recommendation = generate_recommendation(metrics, recommendation_path)
    print("\n" + recommendation)
    
    logger.info("=" * 80)
    logger.info("✅ COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
