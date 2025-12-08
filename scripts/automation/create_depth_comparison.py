#!/usr/bin/env python3
"""Create comparison showing depth-aware vs uniform processing."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_comparison():
    # Load images
    original = Image.open("input_images/750_Picacho/Ultimate_TIFFs_Base/750Picacho_Pool_Ultimate.tif")
    no_depth = Image.open("output_750_Picacho_Pool_LuxDepthV2_Upscale_Test/750Picacho_Pool_Ultimate_master16.tif")
    with_depth = Image.open("output_750_Picacho_Pool_LuxDepthV2_WithDepth/750Picacho_Pool_Ultimate_master16.tif")
    depth_viz = Image.open("output_750_Picacho_Depth_Maps_MaxQuality_20251206/V2_750Picacho_Pool_depth_viz.png")
    
    # Convert to RGB
    original_rgb = original.convert("RGB")
    no_depth_rgb = no_depth.convert("RGB")
    with_depth_rgb = with_depth.convert("RGB")
    
    # Resize for web viewing (25% of original)
    web_size = (int(original.width * 0.25), int(original.height * 0.25))
    original_web = original_rgb.resize(web_size, Image.Resampling.LANCZOS)
    no_depth_web = no_depth_rgb.resize(web_size, Image.Resampling.LANCZOS)
    with_depth_web = with_depth_rgb.resize(web_size, Image.Resampling.LANCZOS)
    depth_viz_web = depth_viz.resize(web_size, Image.Resampling.LANCZOS)
    
    # Create 2x2 grid comparison
    gap = 15
    label_height = 35
    grid_width = web_size[0] * 2 + gap * 3
    grid_height = web_size[1] * 2 + gap * 3 + label_height
    
    comparison = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    
    # Paste images
    x1, x2 = gap, gap + web_size[0] + gap
    y1, y2 = label_height, label_height + web_size[1] + gap
    
    comparison.paste(original_web, (x1, y1))
    comparison.paste(depth_viz_web, (x2, y1))
    comparison.paste(no_depth_web, (x1, y2))
    comparison.paste(with_depth_web, (x2, y2))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Title
    draw.text((grid_width // 2, 10), "Lux Depth V2: Depth-Aware Processing Comparison", 
              fill=(0, 0, 0), anchor="mm", font=font_title)
    
    # Image labels
    draw.text((x1 + web_size[0] // 2, y1 - 10), "Original Source", 
              fill=(0, 0, 0), anchor="mm", font=font_label)
    draw.text((x2 + web_size[0] // 2, y1 - 10), "Depth Map (Depth Anything V2)", 
              fill=(0, 0, 0), anchor="mm", font=font_label)
    draw.text((x1 + web_size[0] // 2, y2 - 10), "Lux V2: Uniform Weights", 
              fill=(100, 100, 100), anchor="mm", font=font_label)
    draw.text((x2 + web_size[0] // 2, y2 - 10), "Lux V2: Depth-Aware Zones ✓", 
              fill=(0, 120, 0), anchor="mm", font=font_label)
    
    # Save comparison
    output_dir = Path("output_750_Picacho_Pool_LuxDepthV2_WithDepth")
    comparison_path = output_dir / "comparison_depth_aware_processing.jpg"
    comparison.save(comparison_path, "JPEG", quality=95)
    print(f"✓ Saved comparison: {comparison_path}")
    print(f"  Size: {comparison.width}x{comparison.height}")

if __name__ == "__main__":
    create_comparison()
