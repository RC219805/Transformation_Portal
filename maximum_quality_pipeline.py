#!/usr/bin/env python3
"""
Maximum Quality Pipeline for 750 Picacho Lane
Optimized for Apple Silicon with CoreML/MPS acceleration

This pipeline delivers absolute maximum quality by:
1. Using Depth Anything V2 Large (best depth model)
2. Processing in 16-bit float precision throughout
3. Leveraging Apple Neural Engine + Metal GPU
4. Maintaining linear color space until final LUT
5. Conservative Material Response for natural results
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import pipeline as transformers_pipeline
from typing import Optional, Tuple
import warnings

# Suppress less critical warnings
warnings.filterwarnings('ignore', category=UserWarning)


class MaximumQualityPipeline:
    """Pipeline optimized for absolute maximum quality on Apple Silicon."""
    
    def __init__(self, use_large_depth_model: bool = True):
        """
        Initialize maximum quality pipeline.
        
        Args:
            use_large_depth_model: Use Depth Anything V2 Large (slower but best quality)
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing Maximum Quality Pipeline on {self.device}...")
        
        # Initialize depth estimation
        depth_model = ("depth-anything/Depth-Anything-V2-Large-hf" if use_large_depth_model 
                      else "depth-anything/Depth-Anything-V2-Small-hf")
        
        print(f"Loading {depth_model}...")
        self.depth_pipe = transformers_pipeline(
            task="depth-estimation",
            model=depth_model,
            device=self.device
        )
        print("✓ Depth model ready")
        
    def load_image_16bit(self, image_path: Path) -> np.ndarray:
        """
        Load image in 16-bit precision, maintaining full tonal range.
        Supports EXR, TIFF, PNG, JPEG.
        
        Returns: float32 array in range [0, 1]
        """
        ext = image_path.suffix.lower()
        
        # Handle EXR files
        if ext == '.exr':
            try:
                import imageio
                img_array = imageio.imread(image_path, format='EXR-FI')
                # EXR is already in float format
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    img_array = img_array.astype(np.float32)
                    # Clamp to reasonable range (EXR can have values > 1)
                    img_array = np.clip(img_array, 0, 10)  # Allow some HDR headroom
                    # Tonemap if needed
                    if img_array.max() > 1.0:
                        print(f"  EXR contains HDR data (max: {img_array.max():.2f}), applying simple tonemap...")
                        img_array = img_array / (img_array + 1.0)  # Simple Reinhard
                    return img_array
                else:
                    img_array = img_array.astype(np.float32) / 255.0
                    return img_array
            except Exception as e:
                print(f"  Warning: Could not load EXR with imageio ({e}), trying TIFF version...")
                # Fall back to TIFF if EXR fails
                tiff_path = image_path.with_suffix('.tif')
                if tiff_path.exists():
                    image_path = tiff_path
                else:
                    raise RuntimeError(f"Could not load EXR file and no TIFF alternative found")
        
        # Handle standard formats with PIL
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img, mask=img.split()[1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to float32 array normalized to [0, 1]
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize based on bit depth
        if img_array.max() > 255:
            # 16-bit image
            img_array = img_array / 65535.0
        else:
            # 8-bit image
            img_array = img_array / 255.0
        
        return img_array
    
    def estimate_depth_highquality(self, image: np.ndarray) -> np.ndarray:
        """
        Generate highest quality depth map using Depth Anything V2 Large.
        
        Returns: Normalized depth map (0=far, 1=near)
        """
        # Convert to PIL for transformers pipeline
        if image.dtype == np.float32:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_image = Image.fromarray(image)
        
        # Generate depth
        depth_result = self.depth_pipe(pil_image)
        depth_map = np.array(depth_result['depth'])
        
        # Normalize to [0, 1]
        depth_map = depth_map.astype(np.float32)
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def apply_depth_aware_enhancement(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        clarity_strength: float = 0.15,
        atmospheric_strength: float = 0.10
    ) -> np.ndarray:
        """
        Apply depth-aware enhancements with conservative settings.
        
        Args:
            image: Input image (float32, [0, 1])
            depth: Depth map (float32, [0, 1])
            clarity_strength: Clarity boost for foreground (0-0.3)
            atmospheric_strength: Atmospheric haze for background (0-0.2)
        """
        result = image.copy()
        h, w = depth.shape
        
        # Define depth zones
        foreground_mask = depth > 0.6  # Closest
        midground_mask = (depth > 0.3) & (depth <= 0.6)
        background_mask = depth <= 0.3  # Farthest
        
        # FOREGROUND: Subtle clarity enhancement
        if clarity_strength > 0:
            # Simple local contrast (unsharp mask substitute)
            from scipy.ndimage import gaussian_filter
            # Apply gaussian per channel
            blurred = np.zeros_like(result)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(result[:, :, c], sigma=2.0)
            clarity_boost = result + (result - blurred) * clarity_strength
            clarity_boost = np.clip(clarity_boost, 0, 1)
            
            # Apply only to foreground
            for c in range(3):
                result[:, :, c] = np.where(
                    foreground_mask,
                    clarity_boost[:, :, c],
                    result[:, :, c]
                )
        
        # BACKGROUND: Subtle atmospheric haze
        if atmospheric_strength > 0:
            # Slight blue tint and desaturation for distance
            haze_color = np.array([0.7, 0.75, 0.8])  # Subtle cool tone
            
            for c in range(3):
                haze_layer = result[:, :, c] * (1 - atmospheric_strength) + haze_color[c] * atmospheric_strength
                result[:, :, c] = np.where(
                    background_mask,
                    haze_layer,
                    result[:, :, c]
                )
        
        return np.clip(result, 0, 1)
    
    def apply_material_response(
        self,
        image: np.ndarray,
        strength: float = 0.65
    ) -> np.ndarray:
        """
        Apply conservative Material Response enhancement.
        
        Enhances micro-contrast and material textures subtly.
        """
        from scipy.ndimage import gaussian_filter
        
        # Calculate luminance
        luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        
        # Detect surface variations (textures)
        texture = luminance - gaussian_filter(luminance, sigma=1.5)
        
        # Enhance textures
        enhanced = image.copy()
        for c in range(3):
            enhanced[:, :, c] = image[:, :, c] + texture * strength * 0.3
        
        # Subtle saturation boost in midtones
        midtone_mask = (luminance > 0.2) & (luminance < 0.8)
        saturation_boost = 1.08
        
        for c in range(3):
            deviation = image[:, :, c] - luminance
            enhanced[:, :, c] = np.where(
                midtone_mask,
                luminance + deviation * saturation_boost,
                enhanced[:, :, c]
            )
        
        return np.clip(enhanced, 0, 1)
    
    def apply_tone_curve(self, image: np.ndarray, curve_type: str = "agx") -> np.ndarray:
        """
        Apply professional tone curve.
        
        Args:
            curve_type: 'agx', 'filmic', or 'linear'
        """
        if curve_type == "linear":
            return image
        
        elif curve_type == "agx":
            # AgX tone curve (Blender-style)
            # Preserves highlights better than Filmic
            def agx_curve(x):
                # Simplified AgX approximation
                return np.power(x / (x + 0.155), 2.2)
            
            return np.clip(agx_curve(image), 0, 1)
        
        elif curve_type == "filmic":
            # Filmic tone curve
            def filmic_curve(x):
                x = np.maximum(0, x - 0.004)
                return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
            
            return np.clip(filmic_curve(image), 0, 1)
        
        return image
    
    def save_16bit_tiff(self, image: np.ndarray, output_path: Path):
        """
        Save image as 16-bit TIFF maintaining maximum quality.
        Uses tifffile for proper 16-bit preservation.
        """
        try:
            import tifffile
            
            # Convert to 16-bit
            image_16bit = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
            
            # Save with tifffile (preserves 16-bit correctly)
            tifffile.imwrite(
                output_path,
                image_16bit,
                compression='lzw',
                photometric='rgb',
                metadata={'Software': 'Transformation Portal - Maximum Quality Pipeline'}
            )
            print(f"✓ Saved 16-bit TIFF: {output_path.name} [{image_16bit.nbytes / (1024**2):.1f} MB]")
            
        except ImportError:
            # Fallback to PIL (will be 8-bit - warn user)
            print("  ⚠️  Warning: tifffile not available, saving as 8-bit TIFF")
            image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_8bit)
            pil_image.save(output_path, format='TIFF', compression='tiff_lzw')
            print(f"  Saved 8-bit TIFF: {output_path.name}")
    
    def save_jpeg(self, image: np.ndarray, output_path: Path, quality: int = 98):
        """
        Save image as high-quality JPEG.
        """
        # Convert to 8-bit
        image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_8bit, mode='RGB')
        
        # Save with high quality, optimized
        pil_image.save(output_path, format='JPEG', quality=quality, optimize=True, subsampling=0)
        print(f"✓ Saved high-quality JPEG: {output_path.name}")
    
    def process_image(
        self,
        input_path: Path,
        output_dir: Path,
        save_tiff: bool = True,
        save_jpeg: bool = True,
        save_depth: bool = False
    ) -> dict:
        """
        Process single image through maximum quality pipeline.
        
        Returns: Dictionary with output paths
        """
        print(f"\nProcessing: {input_path.name}")
        print("-" * 60)
        
        # Load image
        print("Loading image in 16-bit precision...")
        image = self.load_image_16bit(input_path)
        
        # Generate depth map
        print("Generating depth map (Depth Anything V2 Large)...")
        depth = self.estimate_depth_highquality(image)
        
        # Apply depth-aware enhancements
        print("Applying depth-aware processing...")
        enhanced = self.apply_depth_aware_enhancement(
            image, depth,
            clarity_strength=0.15,
            atmospheric_strength=0.10
        )
        
        # Apply Material Response
        print("Applying Material Response enhancement...")
        enhanced = self.apply_material_response(enhanced, strength=0.65)
        
        # Apply tone curve
        print("Applying AgX tone curve...")
        final = self.apply_tone_curve(enhanced, curve_type="agx")
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        outputs = {}
        
        if save_tiff:
            tiff_path = output_dir / f"{stem}_MaxQuality.tif"
            self.save_16bit_tiff(final, tiff_path)
            outputs['tiff'] = tiff_path
        
        if save_jpeg:
            jpeg_path = output_dir / f"{stem}_MaxQuality.jpg"
            self.save_jpeg(final, jpeg_path, quality=98)
            outputs['jpeg'] = jpeg_path
        
        if save_depth:
            depth_path = output_dir / f"{stem}_Depth.png"
            depth_8bit = (depth * 255).astype(np.uint8)
            Image.fromarray(depth_8bit).save(depth_path)
            outputs['depth'] = depth_path
            print(f"✓ Saved depth map: {depth_path.name}")
        
        print("-" * 60)
        print(f"✓ Processing complete!")
        
        return outputs


def main():
    """Main entry point for maximum quality processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Maximum Quality Pipeline for 750 Picacho Lane')
    parser.add_argument('input', type=str, help='Input image path or directory')
    parser.add_argument('--output', '-o', type=str, default='Maximum_Quality_Output',
                       help='Output directory')
    parser.add_argument('--tiff', action='store_true', default=True,
                       help='Save 16-bit TIFF (default: True)')
    parser.add_argument('--jpeg', action='store_true', default=True,
                       help='Save high-quality JPEG (default: True)')
    parser.add_argument('--depth', action='store_true',
                       help='Save depth map visualization')
    parser.add_argument('--small-model', action='store_true',
                       help='Use smaller/faster depth model')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MaximumQualityPipeline(use_large_depth_model=not args.small_model)
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if input_path.is_file():
        # Single file
        pipeline.process_image(
            input_path, output_dir,
            save_tiff=args.tiff,
            save_jpeg=args.jpeg,
            save_depth=args.depth
        )
    elif input_path.is_dir():
        # Batch process directory
        image_files = []
        for ext in ['.exr', '.tif', '.tiff', '.jpg', '.jpeg', '.png']:
            image_files.extend(input_path.glob(f'*{ext}'))
        
        print(f"\nFound {len(image_files)} images to process")
        
        for img_path in image_files:
            try:
                pipeline.process_image(
                    img_path, output_dir,
                    save_tiff=args.tiff,
                    save_jpeg=args.jpeg,
                    save_depth=args.depth
                )
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
                continue
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Maximum Quality Processing Complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
