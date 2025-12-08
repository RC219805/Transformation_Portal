#!/usr/bin/env python3
"""Create comparison image for lux depth v2 test."""
from PIL import Image
from pathlib import Path

def create_comparison():
    # Load images
    original = Image.open("input_images/750_Picacho/Ultimate_TIFFs_Base/750Picacho_Pool_Ultimate.tif")
    master = Image.open("output_750_Picacho_Pool_LuxDepthV2_Test/750Picacho_Pool_Ultimate_master16.tif")
    
    # Convert to RGB for display
    original_rgb = original.convert("RGB")
    master_rgb = master.convert("RGB")
    
    # Resize for web viewing (25% of original)
    web_size = (int(original.width * 0.25), int(original.height * 0.25))
    original_web = original_rgb.resize(web_size, Image.Resampling.LANCZOS)
    master_web = master_rgb.resize(web_size, Image.Resampling.LANCZOS)
    
    # Create side-by-side comparison
    comparison = Image.new("RGB", (web_size[0] * 2 + 20, web_size[1] + 40), (255, 255, 255))
    comparison.paste(original_web, (0, 40))
    comparison.paste(master_web, (web_size[0] + 20, 40))
    
    # Add labels using ImageDraw
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((web_size[0]//2, 10), "Original", fill=(0, 0, 0), anchor="mm", font=font)
    draw.text((web_size[0] + 10 + web_size[0]//2, 10), "Lux Depth V2 (exterior_showcase)", fill=(0, 0, 0), anchor="mm", font=font)
    
    # Save comparison
    output_dir = Path("output_750_Picacho_Pool_LuxDepthV2_Test")
    comparison_path = output_dir / "comparison_original_vs_luxdepth.jpg"
    comparison.save(comparison_path, "JPEG", quality=95)
    print(f"✓ Saved comparison: {comparison_path}")
    print(f"  Size: {comparison.width}x{comparison.height}")

if __name__ == "__main__":
    create_comparison()
