#!/usr/bin/env python3
"""Visual comparison: Bicubic vs Real-ESRGAN upscaling.

This script demonstrates the quality difference between bicubic and Real-ESRGAN
upscaling backends. Requires ML dependencies for Real-ESRGAN.

Usage:
    # Test with bicubic (always available)
    python examples/upscaling_comparison.py --backend bicubic

    # Test with Real-ESRGAN (requires: pip install basicsr)
    python examples/upscaling_comparison.py --backend realesrgan --device cuda

    # Compare both backends
    python examples/upscaling_comparison.py --backend both --device cuda
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from transformation_portal.upscaling import UpscalerRegistry


def create_test_image(size: int = 512) -> np.ndarray:
    """Create a test image with high-frequency details.

    Args:
        size: Image size (size x size)

    Returns:
        Test image as numpy array (size, size, 3)
    """
    # Create gradient with text and patterns
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # Gradient background
    for i in range(size):
        image[i, :] = [int(255 * i / size), 128, int(255 * (1 - i / size))]

    # Add checkerboard pattern (high-frequency)
    checker_size = 16
    for i in range(0, size, checker_size):
        for j in range(0, size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                image[i : i + checker_size, j : j + checker_size] = [255, 255, 255]

    return image


def benchmark_upscaler(backend: str, image: np.ndarray, scale_factor: float, device: str) -> tuple[np.ndarray, float]:
    """Benchmark upscaling backend.

    Args:
        backend: Backend name (bicubic, realesrgan)
        image: Input image
        scale_factor: Upscaling factor
        device: Device to use

    Returns:
        Tuple of (upscaled_image, time_seconds)
    """
    registry = UpscalerRegistry()

    print(f"\n{'=' * 60}")
    print(f"Testing {backend.upper()} backend")
    print(f"{'=' * 60}")
    print(f"Input shape: {image.shape}")
    print(f"Scale factor: {scale_factor}x")
    print(f"Device: {device}")

    # Get backend
    start_load = time.time()
    if backend == "bicubic":
        upscaler = registry.get("bicubic")
    else:
        upscaler = registry.get("realesrgan", device=device, model="RealESRGAN_x2plus", fallback_to_bicubic=False)
    load_time = time.time() - start_load
    print(f"Backend load time: {load_time:.3f}s")

    # Upscale
    start_upscale = time.time()
    upscaled = upscaler.upscale(image, scale_factor=scale_factor)
    upscale_time = time.time() - start_upscale

    print(f"Output shape: {upscaled.shape}")
    print(f"Upscale time: {upscale_time:.3f}s")
    print(f"Total time: {load_time + upscale_time:.3f}s")

    return upscaled, upscale_time


def save_comparison(
    original: np.ndarray,
    bicubic: np.ndarray | None,
    realesrgan: np.ndarray | None,
    output_dir: Path,
):
    """Save comparison images.

    Args:
        original: Original image
        bicubic: Bicubic upscaled image (or None)
        realesrgan: Real-ESRGAN upscaled image (or None)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    Image.fromarray(original).save(output_dir / "original.png")
    print(f"\nSaved: {output_dir / 'original.png'}")

    if bicubic is not None:
        Image.fromarray(bicubic).save(output_dir / "bicubic_upscaled.png")
        print(f"Saved: {output_dir / 'bicubic_upscaled.png'}")

    if realesrgan is not None:
        Image.fromarray(realesrgan).save(output_dir / "realesrgan_upscaled.png")
        print(f"Saved: {output_dir / 'realesrgan_upscaled.png'}")

    # Create side-by-side comparison if both available
    if bicubic is not None and realesrgan is not None:
        # Crop central region for detail comparison
        h, w = bicubic.shape[:2]
        crop_size = 512
        y = (h - crop_size) // 2
        x = (w - crop_size) // 2

        bicubic_crop = bicubic[y : y + crop_size, x : x + crop_size]
        realesrgan_crop = realesrgan[y : y + crop_size, x : x + crop_size]

        comparison = np.hstack([bicubic_crop, realesrgan_crop])
        Image.fromarray(comparison).save(output_dir / "comparison_side_by_side.png")
        print(f"Saved: {output_dir / 'comparison_side_by_side.png'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare upscaling backends")
    parser.add_argument(
        "--backend",
        choices=["bicubic", "realesrgan", "both"],
        default="both",
        help="Backend to test",
    )
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscaling factor (default: 2.0)")
    parser.add_argument("--size", type=int, default=512, help="Test image size (default: 512)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_upscaling_comparison"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create test image
    print(f"Creating test image ({args.size}x{args.size})...")
    image = create_test_image(args.size)

    bicubic_result = None
    realesrgan_result = None

    # Test backends
    if args.backend in ["bicubic", "both"]:
        bicubic_result, _ = benchmark_upscaler("bicubic", image, args.scale, args.device)

    if args.backend in ["realesrgan", "both"]:
        try:
            realesrgan_result, _ = benchmark_upscaler("realesrgan", image, args.scale, args.device)
        except ImportError as e:
            print(f"\n❌ Real-ESRGAN not available: {e}")
            print("Install ML dependencies: pip install basicsr")

    # Save results
    save_comparison(image, bicubic_result, realesrgan_result, args.output_dir)

    print(f"\n✅ Comparison complete! Check {args.output_dir}/")


if __name__ == "__main__":
    main()
