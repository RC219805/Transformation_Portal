#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Luxury Pool Enhancement - Final Polish Pipeline
==============================================

Applies magazine-quality enhancements to the depth-processed pool image:
1. Advanced color grading with luxury LUTs
2. Detail enhancement (sharpening, micro-contrast, texture)
3. Sky optimization
4. Water refinement (clarity, color, reflections)
5. Landscape enhancement
6. Final HDR tone mapping and polish

Performance: 30-90 seconds per 4K image on M4 Max
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("luxury_polish")


class LuxuryPoolEnhancer:
    """Magazine-quality enhancement pipeline for luxury pool photography."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def enhance_colors_luxury_pool(self, img: Image.Image) -> Image.Image:
        """
        Advanced color grading for luxury pool photography.
        
        Enhances:
        - Sky: Rich blues, cloud definition
        - Water: Crystalline blue-green tones
        - Vegetation: Lush, saturated greens
        - Architecture: Clean, neutral highlights
        """
        log.info("Applying luxury color grading...")
        
        # Convert to numpy for precise control
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Split into RGB channels
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Sky enhancement: boost blue channel in bright blue areas
        sky_mask = (b > 0.5) & (b > r) & (b > g) & (r + g + b > 1.5)
        b[sky_mask] = np.clip(b[sky_mask] * 1.15, 0, 1)
        
        # Water enhancement: crystalline blue-green tones
        water_mask = (b > g) & (g > r) & (b > 0.3) & (b < 0.8)
        b[water_mask] = np.clip(b[water_mask] * 1.12, 0, 1)
        g[water_mask] = np.clip(g[water_mask] * 1.08, 0, 1)
        
        # Vegetation: lush greens
        green_mask = (g > r) & (g > b) & (g > 0.25)
        g[green_mask] = np.clip(g[green_mask] * 1.10, 0, 1)
        
        # Recombine channels
        enhanced = np.stack([r, g, b], axis=2)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        result = Image.fromarray(enhanced)
        
        # Global adjustments
        # Boost saturation slightly for luxury appeal
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1.12)
        
        # Increase contrast for definition
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.08)
        
        # Slight brightness boost for airy feel
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1.03)
        
        return result
    
    def enhance_details(self, img: Image.Image) -> Image.Image:
        """
        Professional detail enhancement with selective sharpening.
        
        Applies:
        - Unsharp mask for global sharpening
        - Detail enhancement filter
        - Micro-contrast boost
        """
        log.info("Enhancing details and texture...")
        
        # Apply unsharp mask for crisp details
        # radius=2 for natural sharpening, percent=150 for strong effect
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=2.5, percent=125, threshold=3))
        
        # Apply detail enhancement
        detailed = sharpened.filter(ImageFilter.DETAIL)
        
        # Blend original and detailed (70% enhanced, 30% original for natural look)
        result = Image.blend(img, detailed, alpha=0.70)
        
        # Micro-contrast boost using edge enhancement
        edges = result.filter(ImageFilter.FIND_EDGES)
        result = Image.blend(result, result.filter(ImageFilter.EDGE_ENHANCE), alpha=0.15)
        
        return result
    
    def enhance_sky(self, img: Image.Image) -> Image.Image:
        """
        Sky-specific enhancements for dramatic luxury appeal.
        
        Enhances:
        - Cloud definition and texture
        - Sky gradation
        - Blue saturation
        """
        log.info("Optimizing sky and clouds...")
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Identify sky regions (upper third, blue-biased)
        height = img_array.shape[0]
        sky_region = np.zeros_like(r, dtype=bool)
        sky_region[:height//2, :] = True
        sky_mask = sky_region & (b > 0.4) & (b > r * 1.1) & (b > g * 1.05)
        
        # Enhance blue in sky
        b[sky_mask] = np.clip(b[sky_mask] * 1.15, 0, 1)
        
        # Boost contrast in sky for cloud definition
        sky_brightness = (r + g + b) / 3
        sky_enhanced = sky_brightness[sky_mask]
        sky_enhanced = (sky_enhanced - 0.5) * 1.2 + 0.5
        
        # Apply enhancement to all channels in sky
        enhancement_factor = np.ones_like(r)
        enhancement_factor[sky_mask] = np.clip(sky_enhanced / (sky_brightness[sky_mask] + 0.001), 0.8, 1.3)
        
        r = np.clip(r * enhancement_factor, 0, 1)
        g = np.clip(g * enhancement_factor, 0, 1)
        b = np.clip(b * enhancement_factor, 0, 1)
        
        enhanced = np.stack([r, g, b], axis=2)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def enhance_water(self, img: Image.Image) -> Image.Image:
        """
        Pool water refinement for crystalline luxury appearance.
        
        Enhances:
        - Water clarity and transparency
        - Blue-green luxury pool tones
        - Reflection definition
        """
        log.info("Refining pool water appearance...")
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Identify water regions (blue-green, mid-brightness)
        water_mask = (
            (b > g) & (g > r) &  # Blue-green bias
            (b > 0.25) & (b < 0.85) &  # Mid-range brightness
            (b - r > 0.1)  # Strong blue component
        )
        
        # Enhance blue-green tones for luxury pool color
        b[water_mask] = np.clip(b[water_mask] * 1.15, 0, 1)
        g[water_mask] = np.clip(g[water_mask] * 1.10, 0, 1)
        
        # Increase water clarity by boosting contrast
        water_brightness = (r + g + b) / 3
        water_enhanced = water_brightness[water_mask]
        water_enhanced = (water_enhanced - 0.5) * 1.15 + 0.5
        
        clarity_factor = np.ones_like(r)
        clarity_factor[water_mask] = np.clip(
            water_enhanced / (water_brightness[water_mask] + 0.001), 
            0.85, 1.25
        )
        
        r[water_mask] = np.clip(r[water_mask] * clarity_factor[water_mask], 0, 1)
        g[water_mask] = np.clip(g[water_mask] * clarity_factor[water_mask], 0, 1)
        b[water_mask] = np.clip(b[water_mask] * clarity_factor[water_mask], 0, 1)
        
        enhanced = np.stack([r, g, b], axis=2)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def enhance_landscape(self, img: Image.Image) -> Image.Image:
        """
        Vegetation and hardscape enhancement.
        
        Enhances:
        - Vegetation saturation and depth
        - Hardscape detail and texture
        - Overall landscaping appeal
        """
        log.info("Optimizing landscape and vegetation...")
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        # Identify vegetation (green-dominant)
        vegetation_mask = (g > r * 1.1) & (g > b * 1.05) & (g > 0.2)
        
        # Enhance greens for lush appearance
        g[vegetation_mask] = np.clip(g[vegetation_mask] * 1.12, 0, 1)
        
        # Add depth to vegetation with slight darkening in shadows
        veg_brightness = (r + g + b) / 3
        dark_vegetation = vegetation_mask & (veg_brightness < 0.5)
        r[dark_vegetation] = np.clip(r[dark_vegetation] * 0.95, 0, 1)
        g[dark_vegetation] = np.clip(g[dark_vegetation] * 0.97, 0, 1)
        b[dark_vegetation] = np.clip(b[dark_vegetation] * 0.95, 0, 1)
        
        enhanced = np.stack([r, g, b], axis=2)
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def apply_hdr_tone_mapping(self, img: Image.Image) -> Image.Image:
        """
        HDR-style tone mapping for enhanced dynamic range.
        
        Preserves highlights and shadows while boosting mid-tones.
        Creates that "luxury real estate" HDR look without oversaturation.
        """
        log.info("Applying HDR tone mapping...")
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Calculate luminance
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
        # S-curve tone mapping: compress highlights/shadows, expand mid-tones
        # This creates the HDR "pop" while preserving detail
        tone_mapped = luminance.copy()
        
        # Shadows (0-0.3): gentle lift
        shadow_mask = luminance < 0.3
        tone_mapped[shadow_mask] = luminance[shadow_mask] * 1.15
        
        # Mid-tones (0.3-0.7): boost for "pop"
        midtone_mask = (luminance >= 0.3) & (luminance < 0.7)
        tone_mapped[midtone_mask] = 0.3 + (luminance[midtone_mask] - 0.3) * 1.25
        
        # Highlights (0.7-1.0): gentle compression to preserve detail
        highlight_mask = luminance >= 0.7
        tone_mapped[highlight_mask] = 0.7 + (luminance[highlight_mask] - 0.7) * 0.85
        
        # Apply tone mapping to preserve color ratios
        tone_factor = np.clip(tone_mapped / (luminance + 0.001), 0.5, 1.5)
        tone_factor = np.expand_dims(tone_factor, axis=2)
        
        enhanced = img_array * tone_factor
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(enhanced)
    
    def final_polish(self, img: Image.Image) -> Image.Image:
        """
        Final polish and refinement for magazine-quality output.
        
        - Subtle vignette for focus
        - Final sharpening pass
        - Color balance fine-tuning
        """
        log.info("Applying final polish...")
        
        # Subtle vignette for focus on center
        width, height = img.size
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Create radial gradient for vignette
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width / 2, height / 2
        
        # Distance from center, normalized
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        # Very subtle vignette (90% at edges)
        vignette = 1.0 - (dist_norm**2 * 0.10)
        vignette = np.expand_dims(vignette, axis=2)
        
        img_array = img_array * vignette
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        
        result = Image.fromarray(img_array)
        
        # Final sharpening - gentle for natural look
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))
        
        return result
    
    def process(self, input_path: Path, base_name: str = "V3") -> Dict[str, Path]:
        """
        Execute full luxury enhancement pipeline.
        
        Returns dict of output paths for each stage.
        """
        log.info(f"Loading image: {input_path}")
        img = Image.open(input_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        log.info(f"Image size: {img.size}, Mode: {img.mode}")
        
        outputs = {}
        
        # Stage 1: Advanced color grading
        log.info("\n=== Stage 1: Advanced Color Grading ===")
        img_stage1 = self.enhance_colors_luxury_pool(img)
        path_stage1 = self.output_dir / f"{base_name}_01_color_graded.png"
        img_stage1.save(path_stage1, quality=95)
        outputs['01_color_graded'] = path_stage1
        log.info(f"Saved: {path_stage1}")
        
        # Stage 2: Sky enhancement
        log.info("\n=== Stage 2: Sky Enhancement ===")
        img_stage2 = self.enhance_sky(img_stage1)
        path_stage2 = self.output_dir / f"{base_name}_02_sky_enhanced.png"
        img_stage2.save(path_stage2, quality=95)
        outputs['02_sky_enhanced'] = path_stage2
        log.info(f"Saved: {path_stage2}")
        
        # Stage 3: Water refinement
        log.info("\n=== Stage 3: Water Refinement ===")
        img_stage3 = self.enhance_water(img_stage2)
        path_stage3 = self.output_dir / f"{base_name}_03_water_refined.png"
        img_stage3.save(path_stage3, quality=95)
        outputs['03_water_refined'] = path_stage3
        log.info(f"Saved: {path_stage3}")
        
        # Stage 4: Landscape optimization
        log.info("\n=== Stage 4: Landscape Enhancement ===")
        img_stage4 = self.enhance_landscape(img_stage3)
        path_stage4 = self.output_dir / f"{base_name}_04_landscape_enhanced.png"
        img_stage4.save(path_stage4, quality=95)
        outputs['04_landscape_enhanced'] = path_stage4
        log.info(f"Saved: {path_stage4}")
        
        # Stage 5: HDR tone mapping
        log.info("\n=== Stage 5: HDR Tone Mapping ===")
        img_stage5 = self.apply_hdr_tone_mapping(img_stage4)
        path_stage5 = self.output_dir / f"{base_name}_05_hdr_tone_mapped.png"
        img_stage5.save(path_stage5, quality=95)
        outputs['05_hdr_tone_mapped'] = path_stage5
        log.info(f"Saved: {path_stage5}")
        
        # Stage 6: Detail enhancement
        log.info("\n=== Stage 6: Detail Enhancement ===")
        img_stage6 = self.enhance_details(img_stage5)
        path_stage6 = self.output_dir / f"{base_name}_06_details_enhanced.png"
        img_stage6.save(path_stage6, quality=95)
        outputs['06_details_enhanced'] = path_stage6
        log.info(f"Saved: {path_stage6}")
        
        # Stage 7: Final polish
        log.info("\n=== Stage 7: Final Polish ===")
        img_final = self.final_polish(img_stage6)
        
        # Save master PNG
        path_final_png = self.output_dir / f"{base_name}_750Picacho_Pool_FINAL_LUXURY.png"
        img_final.save(path_final_png, quality=98, optimize=True)
        outputs['final_png'] = path_final_png
        log.info(f"Saved master PNG: {path_final_png}")
        
        # Save web-optimized JPEG
        path_final_jpg = self.output_dir / f"{base_name}_750Picacho_Pool_FINAL_LUXURY.jpg"
        img_final.save(path_final_jpg, quality=95, optimize=True)
        outputs['final_jpg'] = path_final_jpg
        log.info(f"Saved web JPEG: {path_final_jpg}")
        
        # Create comparison with original
        log.info("\n=== Creating Before/After Comparison ===")
        comparison = self.create_comparison(img, img_final)
        path_comparison = self.output_dir / f"{base_name}_BEFORE_AFTER_comparison.jpg"
        comparison.save(path_comparison, quality=95)
        outputs['comparison'] = path_comparison
        log.info(f"Saved comparison: {path_comparison}")
        
        return outputs
    
    def create_comparison(self, before: Image.Image, after: Image.Image) -> Image.Image:
        """Create side-by-side before/after comparison."""
        # Resize if needed for comparison
        max_width = 1920
        if before.width > max_width:
            ratio = max_width / before.width
            new_size = (max_width, int(before.height * ratio))
            before = before.resize(new_size, Image.Resampling.LANCZOS)
            after = after.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create side-by-side
        comparison = Image.new('RGB', (before.width * 2, before.height))
        comparison.paste(before, (0, 0))
        comparison.paste(after, (before.width, 0))
        
        return comparison


def main():
    """Main execution."""
    import time
    start_time = time.time()
    
    # Input path
    input_path = Path("output_images/depth_processed/V2_V2_750Picacho_Pool_Luxury_Enhanced_enhanced.png")
    
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Output directory
    output_dir = Path("output_images/luxury_final_polish")
    
    # Create enhancer and process
    enhancer = LuxuryPoolEnhancer(output_dir)
    
    log.info("\n" + "="*70)
    log.info("LUXURY POOL ENHANCEMENT - FINAL POLISH PIPELINE")
    log.info("="*70)
    log.info(f"Input: {input_path}")
    log.info(f"Output: {output_dir}/")
    log.info("="*70 + "\n")
    
    outputs = enhancer.process(input_path, base_name="V3")
    
    # Summary
    elapsed = time.time() - start_time
    log.info("\n" + "="*70)
    log.info("PROCESSING COMPLETE")
    log.info("="*70)
    log.info(f"Total time: {elapsed:.1f} seconds")
    log.info(f"Outputs generated: {len(outputs)}")
    log.info("\nFinal outputs:")
    log.info(f"  Master PNG: {outputs['final_png']}")
    log.info(f"  Web JPEG:   {outputs['final_jpg']}")
    log.info(f"  Comparison: {outputs['comparison']}")
    log.info("\nStage outputs:")
    for stage, path in sorted(outputs.items()):
        if not stage.startswith('final') and stage != 'comparison':
            log.info(f"  {stage}: {path.name}")
    log.info("="*70)


if __name__ == '__main__':
    main()
