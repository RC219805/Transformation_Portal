#!/usr/bin/env python3
"""
Phase 3: Scene-Specific Refinement
Applies targeted exposure and enhancement per scene type
"""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageEnhance

# Scene-specific configurations from action plan
SCENE_CONFIGS = {
    "Pool": {
        "exposure": +0.25,  # CRITICAL FIX
        "water_clarity": 0.85,
        "saturation": 1.12,
        "warmth": 0.10,
        "contrast": 1.05,
    },
    "Aerial": {
        "exposure": +0.10,
        "clarity": 0.30,
        "saturation": 1.10,
        "contrast": 1.08,
    },
    "GreatRoom": {
        "exposure": +0.05,
        "contrast": 1.12,
        "warmth": 0.20,
        "saturation": 1.05,
    },
    "Kitchen": {
        "exposure": 0.0,
        "contrast": 1.10,
        "saturation": 1.08,
    },
    "PrimaryBathroom": {
        "exposure": +0.05,
        "contrast": 1.05,
        "warmth": 0.12,
        "saturation": 1.05,
    },
    "PrimaryBedroom": {
        "exposure": 0.0,
        "contrast": 1.06,
        "warmth": 0.18,
        "saturation": 1.05,
    },
}


def detect_scene_type(filename: str) -> str:
    """Detect scene type from filename"""
    filename_lower = filename.lower()
    for scene in SCENE_CONFIGS.keys():
        if scene.lower() in filename_lower:
            return scene
    return "default"


def apply_exposure(img: np.ndarray, exposure_ev: float) -> np.ndarray:
    """
    Apply exposure adjustment in EV stops
    exposure_ev: +0.25 = +1/4 stop brighter
    """
    if exposure_ev == 0:
        return img

    # Convert EV to linear multiplier: 2^EV
    multiplier = 2**exposure_ev

    # Apply and clip
    img_adjusted = np.clip(img * multiplier, 0, 1)
    return img_adjusted.astype(np.float32)


def apply_warmth(img: np.ndarray, warmth: float) -> np.ndarray:
    """Add warmth (shift toward orange/red)"""
    if warmth == 0:
        return img

    # Boost red channel, slightly boost green, leave blue
    img_warm = img.copy()
    img_warm[..., 0] = np.clip(img_warm[..., 0] + warmth * 0.1, 0, 1)  # Red
    img_warm[..., 1] = np.clip(img_warm[..., 1] + warmth * 0.05, 0, 1)  # Green
    # Blue stays same or slightly reduced
    return img_warm


def enhance_water_clarity(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Enhance water regions (blue/cyan areas)
    Increases saturation and slight contrast in blue channels
    """
    if strength == 0:
        return img

    # Detect water regions (high blue, lower red)
    blue = img[..., 2]
    red = img[..., 0]
    water_mask = (blue > red + 0.1) & (blue > 0.3)

    # Enhance blue saturation in water regions
    img_enhanced = img.copy()
    if water_mask.any():
        img_enhanced[..., 2] = np.where(water_mask, np.clip(img[..., 2] * (1 + strength * 0.15), 0, 1), img[..., 2])

    return img_enhanced


def process_scene(input_path: Path, output_path: Path, scene_type: str) -> None:
    """Process a single image with scene-specific enhancements"""

    print(f"\n{'=' * 80}")
    print(f"Processing: {input_path.name}")
    print(f"Scene type: {scene_type}")
    print(f"{'=' * 80}\n")

    # Get scene config
    config = SCENE_CONFIGS.get(scene_type, {})
    if not config:
        print(f"⚠️  No config for scene type '{scene_type}', using defaults")
        config = {"exposure": 0.0, "contrast": 1.0, "saturation": 1.0}

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load image (16-bit TIFF)
    print("\nLoading 16-bit TIFF...")
    img = tifffile.imread(str(input_path))
    print(f"  Shape: {img.shape}, dtype: {img.dtype}")
    print(f"  Range: [{img.min()}, {img.max()}]")

    # Convert to float [0-1]
    if img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)

    # Apply scene-specific processing
    print("\nApplying enhancements...")

    # 1. Exposure correction
    if "exposure" in config and config["exposure"] != 0:
        print(f"  • Exposure: {config['exposure']:+.2f} EV")
        img_float = apply_exposure(img_float, config["exposure"])

    # 2. Warmth
    if "warmth" in config and config["warmth"] > 0:
        print(f"  • Warmth: {config['warmth']:.2f}")
        img_float = apply_warmth(img_float, config["warmth"])

    # 3. Water clarity (for pool scenes)
    if "water_clarity" in config and config["water_clarity"] > 0:
        print(f"  • Water clarity: {config['water_clarity']:.2f}")
        img_float = enhance_water_clarity(img_float, config["water_clarity"])

    # Convert back to uint16
    img_16bit = (img_float * 65535).astype(np.uint16)

    # Apply PIL enhancements (contrast, saturation) on 8-bit for compatibility
    img_8bit = (img_float * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_8bit, mode="RGB")

    # 4. Contrast
    if "contrast" in config and config["contrast"] != 1.0:
        print(f"  • Contrast: {config['contrast']:.2f}")
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(config["contrast"])

    # 5. Saturation
    if "saturation" in config and config["saturation"] != 1.0:
        print(f"  • Saturation: {config['saturation']:.2f}")
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(config["saturation"])

    # Convert PIL result back to 16-bit
    img_final_8bit = np.array(img_pil)
    img_final_float = img_final_8bit.astype(np.float32) / 255.0

    # Blend 16-bit exposure/warmth with 8-bit contrast/sat (preserve highlights)
    # Use exposure-corrected 16-bit for highlights, PIL for midtones/shadows
    highlight_mask = img_float > 0.7
    img_blended = img_final_float.copy()

    # Apply mask per channel (avoid broadcasting issues)
    for c in range(3):
        img_blended[..., c] = np.where(
            highlight_mask[..., c], img_float[..., c], img_final_float[..., c]  # Use 16-bit for highlights  # Use PIL for rest
        )

    # Final conversion to 16-bit
    img_output = (np.clip(img_blended, 0, 1) * 65535).astype(np.uint16)

    # Save 16-bit TIFF
    print("\nSaving refined 16-bit TIFF...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), img_output, compression="lzw", photometric="rgb")

    print(f"✓ Saved: {output_path.name}")
    print(f"  Shape: {img_output.shape}, dtype: {img_output.dtype}")
    print(f"  Range: [{img_output.min()}, {img_output.max()}]")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Also save high-quality JPEG
    jpeg_path = output_path.with_suffix(".jpg")
    img_jpeg = Image.fromarray((img_blended * 255).astype(np.uint8))
    img_jpeg.save(jpeg_path, "JPEG", quality=98, optimize=True, subsampling=0)
    print(f"✓ Saved JPEG: {jpeg_path.name}")


def main():
    """Process all luxury pipeline outputs with scene-specific refinements"""

    input_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Ultimate_Quality"
    output_dir = Path.home() / "Desktop" / "Cache" / "750_LightFiction_Final_Views" / "Phase3_Refined"

    print(f"\n{'#' * 80}")
    print("  PHASE 3: SCENE-SPECIFIC REFINEMENT")
    print("  750 Picacho Lane")
    print(f"{'#' * 80}\n")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")

    # Process luxury TIFFs
    tiff_files = sorted(input_dir.glob("*_luxury.ti"))

    if not tiff_files:
        print("⚠️  No *_luxury.tif files found!")
        print("Looking for alternative TIFF files...")
        tiff_files = sorted(input_dir.glob("*.ti"))

    print(f"Found {len(tiff_files)} TIFF files to refine\n")

    for i, tiff_path in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] " + "=" * 70)

        # Detect scene type
        scene_type = detect_scene_type(tiff_path.name)

        # Create output filename
        base_name = tiff_path.stem.replace("_luxury", "").replace("_ultimate", "")
        output_path = output_dir / f"{base_name}_refined.ti"

        # Process
        try:
            process_scene(tiff_path, output_path, scene_type)
            print(f"✅ Success!")
        except Exception as e:
            print(f"❌ Error processing {tiff_path.name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'#' * 80}")
    print("  PHASE 3 COMPLETE")
    print(f"{'#' * 80}\n")
    print(f"Refined outputs: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*')))}")


if __name__ == "__main__":
    main()
