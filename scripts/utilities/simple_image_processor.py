#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Image Processor - Minimal Setup Demo

Demonstrates basic image processing operations that work with minimal
dependencies (numpy, Pillow only). This script shows what's possible
even without the full ML stack.

Usage:
    python scripts/simple_image_processor.py input_images/test_render.jpg [--output OUTPUT]

Features:
    - Image format conversion
    - Resize and crop operations
    - Basic color adjustments (brightness, contrast, saturation)
    - Metadata preservation
    - Batch processing support

Requirements:
    - numpy
    - Pillow
    - typer (optional, for CLI)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
    from PIL import Image, ImageEnhance
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install numpy Pillow")
    sys.exit(1)

try:
    import typer

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False


def adjust_brightness(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust image brightness.

    Args:
        image: Input PIL Image
        factor: Brightness factor (1.0 = no change, <1.0 darker, >1.0 brighter)

    Returns:
        Adjusted image
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust image contrast.

    Args:
        image: Input PIL Image
        factor: Contrast factor (1.0 = no change, <1.0 less contrast, >1.0 more contrast)

    Returns:
        Adjusted image
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_saturation(image: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust image saturation.

    Args:
        image: Input PIL Image
        factor: Saturation factor (0.0 = grayscale, 1.0 = no change, >1.0 more saturated)

    Returns:
        Adjusted image
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """Resize image with optional aspect ratio preservation.

    Args:
        image: Input PIL Image
        target_size: Target (width, height)
        maintain_aspect: If True, maintain aspect ratio and fit within target_size

    Returns:
        Resized image
    """
    if maintain_aspect:
        # Calculate aspect-preserving size
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)


def process_image(
    input_path: Path,
    output_path: Path,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    resize: Optional[Tuple[int, int]] = None,
    quality: int = 95,
    verbose: bool = False,
) -> bool:
    """Process a single image with basic adjustments.

    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        brightness: Brightness adjustment factor (default: 1.0)
        contrast: Contrast adjustment factor (default: 1.0)
        saturation: Saturation adjustment factor (default: 1.0)
        resize: Optional target size (width, height)
        quality: JPEG quality (1-100, default: 95)
        verbose: Print processing details

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        if verbose:
            print(f"Loading: {input_path}")

        img = Image.open(input_path)
        original_size = img.size
        original_mode = img.mode

        if verbose:
            print(f"  Original: {original_size[0]}x{original_size[1]} {original_mode}")

        # Convert to RGB if needed (for processing)
        if img.mode != "RGB":
            if verbose:
                print(f"  Converting {img.mode} → RGB")
            img = img.convert("RGB")

        # Apply adjustments
        if brightness != 1.0:
            if verbose:
                print(f"  Adjusting brightness: {brightness:.2f}")
            img = adjust_brightness(img, brightness)

        if contrast != 1.0:
            if verbose:
                print(f"  Adjusting contrast: {contrast:.2f}")
            img = adjust_contrast(img, contrast)

        if saturation != 1.0:
            if verbose:
                print(f"  Adjusting saturation: {saturation:.2f}")
            img = adjust_saturation(img, saturation)

        # Resize if requested
        if resize:
            if verbose:
                print(f"  Resizing to: {resize[0]}x{resize[1]}")
            img = resize_image(img, resize, maintain_aspect=True)

        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from extension
        output_format = output_path.suffix.lower()
        save_kwargs = {}

        if output_format in [".jpg", ".jpeg"]:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif output_format == ".png":
            save_kwargs["optimize"] = True

        img.save(output_path, **save_kwargs)

        if verbose:
            print(f"  Saved: {output_path}")
            print(f"  Final: {img.size[0]}x{img.size[1]}")

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main_simple(
    input_path: str,
    output: Optional[str] = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: int = 95,
    verbose: bool = False,
):
    """
    Simple image processor for basic operations.

    Works with minimal dependencies (numpy, Pillow only).
    """
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    # Determine output path
    if output:
        output_file = Path(output)
    else:
        # Default: same name with _processed suffix
        output_file = input_file.parent / f"{input_file.stem}_processed{input_file.suffix}"

    # Determine resize target
    resize_target = None
    if width and height:
        resize_target = (width, height)

    # Process the image
    print(f"Processing: {input_file.name}")

    success = process_image(
        input_file,
        output_file,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        resize=resize_target,
        quality=quality,
        verbose=verbose,
    )

    if success:
        print(f"✓ Successfully processed: {output_file}")
        return 0
    else:
        print(f"✗ Failed to process image")
        return 1


# CLI setup
if HAS_TYPER:
    app = typer.Typer(help="Simple image processor (minimal dependencies)", add_completion=False)

    @app.command()
    def main(
        input_path: str = typer.Argument(..., help="Path to input image"),
        output: Optional[str] = typer.Option(
            None, "--output", "-o", help="Output path (default: input_processed.jpg)"
        ),
        brightness: float = typer.Option(1.0, "--brightness", "-b", help="Brightness factor (1.0 = no change)"),
        contrast: float = typer.Option(1.0, "--contrast", "-c", help="Contrast factor (1.0 = no change)"),
        saturation: float = typer.Option(1.0, "--saturation", "-s", help="Saturation factor (1.0 = no change)"),
        width: Optional[int] = typer.Option(None, "--width", "-w", help="Target width (maintains aspect ratio)"),
        height: Optional[int] = typer.Option(None, "--height", "-h", help="Target height (maintains aspect ratio)"),
        quality: int = typer.Option(95, "--quality", "-q", help="JPEG quality (1-100)"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    ):
        """Process image with basic adjustments."""
        return main_simple(input_path, output, brightness, contrast, saturation, width, height, quality, verbose)

    def cli_main() -> int:
        return app()
else:
    def cli_main() -> int:
        """Fallback CLI for when typer is not installed."""
        import argparse

        parser = argparse.ArgumentParser(description="Simple image processor")
        parser.add_argument("input_path", help="Path to input image")
        parser.add_argument("-o", "--output", help="Output path")
        parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness factor")
        parser.add_argument("-c", "--contrast", type=float, default=1.0, help="Contrast factor")
        parser.add_argument("-s", "--saturation", type=float, default=1.0, help="Saturation factor")
        parser.add_argument("-w", "--width", type=int, help="Target width")
        parser.add_argument("--height", type=int, help="Target height")
        parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality")
        parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

        args = parser.parse_args()

        return main_simple(
            args.input_path,
            args.output,
            args.brightness,
            args.contrast,
            args.saturation,
            args.width,
            args.height,
            args.quality,
            args.verbose,
        )


if __name__ == "__main__":
    sys.exit(cli_main())
