#!/usr/bin/env python3
"""
Script 05: Validate Model for 750 Picacho Lane.

This script validates the trained model against test data:
- Computes quality metrics (PSNR, SSIM, LPIPS)
- Generates visual comparisons
- Produces validation report

Usage:
    python scripts/training/750_picacho/05_validate_model.py [options]

Author: Transformation_Portal Enhancement Team
Version: 1.0.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from training.property_specific.picacho_inference import PicachoInference, InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Optional: Import torch and lpips for LPIPS metric
LPIPS_AVAILABLE = False
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    logger.warning(
        "torch or lpips not installed. LPIPS metric will be skipped. "
        "Install with: pip install torch lpips"
    )


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0 if img1.dtype == np.uint8 else 65535.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (simplified)."""
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants for numerical stability
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Compute variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # SSIM formula
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return float(ssim)


def init_lpips_model(device: str = "auto") -> Optional[Any]:
    """Initialize LPIPS model for perceptual similarity.

    Args:
        device: Compute device ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        LPIPS model or None if not available.
    """
    if not LPIPS_AVAILABLE:
        return None

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Initializing LPIPS model with net='alex' on device '{device}'")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    return lpips_model


def compute_lpips(
    img1: np.ndarray,
    img2: np.ndarray,
    lpips_model: Any
) -> Optional[float]:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity) distance.

    Args:
        img1: First image as numpy array (RGB, 0-255, HWC format).
        img2: Second image as numpy array (RGB, 0-255, HWC format).
        lpips_model: Initialized LPIPS model.

    Returns:
        LPIPS distance (lower is better) or None if LPIPS is not available.
    """
    if lpips_model is None or not LPIPS_AVAILABLE:
        return None

    # Convert numpy images (RGB, 0-255) to PyTorch tensors (NCHW, normalized to [-1, 1])
    # Step 1: Normalize from [0, 255] to [0, 1]
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0

    # Step 2: Normalize from [0, 1] to [-1, 1]
    img1_normalized = img1_float * 2.0 - 1.0
    img2_normalized = img2_float * 2.0 - 1.0

    # Step 3: Convert HWC to NCHW format
    img1_tensor = torch.from_numpy(img1_normalized).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2_normalized).permute(2, 0, 1).unsqueeze(0)

    # Move to same device as model
    device = next(lpips_model.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # Compute LPIPS distance
    with torch.no_grad():
        lpips_distance = lpips_model(img1_tensor, img2_tensor)

    return float(lpips_distance.item())


def validate_model(
    model_path: Path,
    test_dir: Path,
    output_dir: Path,
    device: str = "auto"
) -> Dict[str, Any]:
    """Validate model against test data."""
    # Initialize inference
    config = InferenceConfig(
        model_path=model_path,
        device=device,
        output_dir=output_dir,
    )
    inference = PicachoInference(config=config)

    # Find test images
    test_images_dir = test_dir / "images"
    if not test_images_dir.exists():
        raise ValueError(f"Test images directory not found: {test_images_dir}")

    test_images = list(test_images_dir.glob("*.png"))
    if not test_images:
        raise ValueError(f"No test images found in: {test_images_dir}")

    print(f"Found {len(test_images)} test images")

    # Initialize LPIPS model outside the loop for efficiency
    lpips_model = init_lpips_model(device)

    # Process and compute metrics
    metrics: Dict[str, List[float]] = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "processing_time": [],
    }

    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for i, test_image_path in enumerate(test_images):
        print(f"  Processing {i + 1}/{len(test_images)}: {test_image_path.name}")

        try:
            # Load original
            original = np.array(Image.open(test_image_path))

            # Process
            result = inference.process(test_image_path)

            # Convert to comparable format
            enhanced = result.image
            if enhanced.dtype == np.uint16:
                enhanced = (enhanced / 256).astype(np.uint8)
            if original.dtype == np.uint16:
                original = (original / 256).astype(np.uint8)

            # Compute metrics
            psnr = compute_psnr(original, enhanced)
            ssim = compute_ssim(original, enhanced)
            lpips_value = compute_lpips(original, enhanced, lpips_model)

            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            if lpips_value is not None:
                metrics["lpips"].append(lpips_value)
            metrics["processing_time"].append(result.processing_time)

            # Save comparison
            comparison_path = comparison_dir / f"{test_image_path.stem}_comparison.png"
            save_comparison(original, enhanced, comparison_path)

        except Exception as e:
            logger.error(f"Failed to process {test_image_path.name}: {e}")

    # Compute summary statistics
    summary: Dict[str, Any] = {
        "num_samples": len(metrics["psnr"]),
        "psnr": {
            "mean": float(np.mean(metrics["psnr"])),
            "std": float(np.std(metrics["psnr"])),
            "min": float(np.min(metrics["psnr"])),
            "max": float(np.max(metrics["psnr"])),
        },
        "ssim": {
            "mean": float(np.mean(metrics["ssim"])),
            "std": float(np.std(metrics["ssim"])),
            "min": float(np.min(metrics["ssim"])),
            "max": float(np.max(metrics["ssim"])),
        },
        "processing_time": {
            "mean": float(np.mean(metrics["processing_time"])),
            "total": float(np.sum(metrics["processing_time"])),
        },
        "individual_metrics": metrics,
    }

    # Add LPIPS statistics if available
    if metrics["lpips"]:
        summary["lpips"] = {
            "mean": float(np.mean(metrics["lpips"])),
            "std": float(np.std(metrics["lpips"])),
            "min": float(np.min(metrics["lpips"])),
            "max": float(np.max(metrics["lpips"])),
        }

    return summary


def save_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    output_path: Path
) -> None:
    """Save side-by-side comparison image."""
    # Resize if needed to match
    if original.shape != enhanced.shape:
        h = min(original.shape[0], enhanced.shape[0])
        w = min(original.shape[1], enhanced.shape[1])
        original = np.array(Image.fromarray(original).resize((w, h)))
        enhanced = np.array(Image.fromarray(enhanced).resize((w, h)))

    # Create side-by-side comparison
    comparison = np.concatenate([original, enhanced], axis=1)

    # Add labels
    # (Simple version without text overlay)

    Image.fromarray(comparison).save(output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate 750 Picacho Lane enhancement model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/750_picacho/best_model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/training_750picacho/test"),
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/750_picacho/validation"),
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("750 PICACHO LANE MODEL VALIDATION")
    print("=" * 60 + "\n")

    # Verify paths
    if not args.model.exists():
        print(f"❌ Model not found: {args.model}")
        print("   Please run 04_train_model.py first.")
        return 1

    if not args.test_dir.exists():
        print(f"❌ Test directory not found: {args.test_dir}")
        print("   Please run 03_generate_dataset.py first.")
        return 1

    print(f"Model: {args.model}")
    print(f"Test data: {args.test_dir}")
    print(f"Output: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    print("\nRunning validation...")
    try:
        results = validate_model(
            model_path=args.model,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=args.device
        )

        # Save results
        results_path = args.output_dir / "validation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"\n✓ Samples validated: {results['num_samples']}")
        print("\nQuality Metrics:")
        print(f"  PSNR: {results['psnr']['mean']:.2f} dB (±{results['psnr']['std']:.2f})")
        print(f"  SSIM: {results['ssim']['mean']:.4f} (±{results['ssim']['std']:.4f})")
        if "lpips" in results:
            print(f"  LPIPS: {results['lpips']['mean']:.4f} (±{results['lpips']['std']:.4f})")
        else:
            print("  LPIPS: N/A (torch/lpips not installed)")
        print("\nProcessing Performance:")
        print(f"  Avg time: {results['processing_time']['mean']:.2f}s per image")
        print(f"  Total time: {results['processing_time']['total']:.2f}s")

        # Quality assessment
        psnr_mean = results["psnr"]["mean"]
        ssim_mean = results["ssim"]["mean"]
        lpips_mean = results.get("lpips", {}).get("mean")

        print("\nQuality Assessment:")

        # Determine quality level based on all available metrics
        # Target thresholds from TRAINING_PROTOCOL.md:
        # PSNR: Excellent ≥35 dB, Good ≥30 dB
        # SSIM: Excellent ≥0.92, Good ≥0.85
        # LPIPS: Excellent ≤0.15, Good ≤0.20
        psnr_excellent = psnr_mean >= 35
        psnr_good = psnr_mean >= 30
        ssim_excellent = ssim_mean >= 0.92
        ssim_good = ssim_mean >= 0.85

        # LPIPS assessment (lower is better)
        if lpips_mean is not None:
            lpips_excellent = lpips_mean <= 0.15
            lpips_good = lpips_mean <= 0.20
        else:
            # If LPIPS not available, don't penalize
            lpips_excellent = True
            lpips_good = True

        if psnr_excellent and ssim_excellent and lpips_excellent:
            print("  ✓ EXCELLENT: Model meets all quality thresholds")
            if lpips_mean is not None:
                print("    PSNR ≥35 dB ✓, SSIM ≥0.92 ✓, LPIPS ≤0.15 ✓")
            else:
                print("    PSNR ≥35 dB ✓, SSIM ≥0.92 ✓")
        elif psnr_good and ssim_good and lpips_good:
            print("  ✓ GOOD: Model meets baseline quality thresholds")
            if lpips_mean is not None:
                print("    PSNR ≥30 dB ✓, SSIM ≥0.85 ✓, LPIPS ≤0.20 ✓")
            else:
                print("    PSNR ≥30 dB ✓, SSIM ≥0.85 ✓")
        else:
            print("  ⚠ NEEDS IMPROVEMENT: Consider additional training")
            issues = []
            if not psnr_good:
                issues.append(f"PSNR ({psnr_mean:.2f} dB) < 30 dB")
            if not ssim_good:
                issues.append(f"SSIM ({ssim_mean:.4f}) < 0.85")
            if lpips_mean is not None and not lpips_good:
                issues.append(f"LPIPS ({lpips_mean:.4f}) > 0.20")
            print(f"    Issues: {', '.join(issues)}")

        print(f"\n✓ Results saved to: {results_path}")
        print(f"✓ Comparisons saved to: {args.output_dir / 'comparisons'}")

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nNext step: Run 06_process_final_output.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
