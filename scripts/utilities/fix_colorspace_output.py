"""Add linear colorspace output support to pro_pipeline.py"""
import numpy as np
from PIL import Image


def srgb_to_linear(img_array):
    """Convert sRGB gamma-encoded image to linear colorspace."""
    # Normalize to 0-1 range
    img_float = img_array.astype(np.float32) / 255.0

    # Apply inverse sRGB gamma curve
    linear = np.where(
        img_float <= 0.04045,
        img_float / 12.92,
        np.power((img_float + 0.055) / 1.055, 2.4)
    )

    return linear

def save_linear_tiff(image: Image.Image, output_path, bit_depth=16):
    """Save image in linear colorspace as 16-bit TIFF."""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Convert to linear
    linear_array = srgb_to_linear(img_array)

    # Scale to bit depth
    if bit_depth == 16:
        linear_scaled = (linear_array * 65535).astype(np.uint16)
    elif bit_depth == 32:
        linear_scaled = linear_array.astype(np.float32)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Create PIL image from linear array
    if bit_depth == 16:
        linear_img = Image.fromarray(linear_scaled, mode='RGB')
    else:
        linear_img = Image.fromarray(linear_scaled, mode='F')

    # Save with compression
    linear_img.save(
        output_path,
        compression="tiff_adobe_deflate",
        tiffinfo={
            # Tag 259: Compression
            # Tag 262: PhotometricInterpretation (2 = RGB)
            # Could add custom tag to mark as linear
        }
    )

    return output_path

# Test with the current output
if __name__ == "__main__":
    input_path = "processed_images/pool_pro_full/750Picacho_Pool_compatible_pool-luxury.tif"
    output_path = "processed_images/pool_pro_full/750Picacho_Pool_compatible_pool-luxury_LINEAR.tif"

    print("Loading sRGB image...")
    img = Image.open(input_path)

    print("Converting to linear colorspace and saving as 16-bit TIFF...")
    save_linear_tiff(img, output_path, bit_depth=16)

    print(f"\n✓ Saved linear output to: {output_path}")

    # Verify
    print("\nVerifying output...")
    linear_img = Image.open(output_path)
    linear_array = np.array(linear_img)

    print(f"  Mode: {linear_img.mode}")
    print(f"  Data type: {linear_array.dtype}")
    print(f"  Shape: {linear_array.shape}")
    print(f"  Min: {linear_array.min()}, Max: {linear_array.max()}")
    print(f"  Mean: {linear_array.mean():.2f}")

    if linear_array.dtype == np.uint16 and linear_array.max() <= 65535:
        print("\n✓ Output is 16-bit linear TIFF")
