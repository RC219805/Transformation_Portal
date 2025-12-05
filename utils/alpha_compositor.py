#!/usr/bin/env python3
"""
Alpha Channel Compositor
=========================
Advanced alpha channel handling with multiple compositing modes.
"""

from pathlib import Path
from typing import Tuple, Optional, Literal

import numpy as np
from PIL import Image


AlphaMode = Literal['preserve', 'flatten-white', 'flatten-black', 'flatten-gray', 'composite-gradient', 'composite-branded']


class AlphaCompositor:
    """Handle alpha channel compositing with various background options."""
    
    def __init__(self):
        self.supported_modes = [
            'preserve',
            'flatten-white',
            'flatten-black',
            'flatten-gray',
            'composite-gradient',
            'composite-branded'
        ]
    
    def has_alpha(self, image: np.ndarray) -> bool:
        """Check if image has alpha channel."""
        return image.shape[2] == 4 if len(image.shape) > 2 else False
    
    def separate_alpha(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Separate RGB and alpha channels."""
        if self.has_alpha(image):
            return image[:, :, :3], image[:, :, 3]
        return image, None
    
    def composite(
        self,
        image: np.ndarray,
        mode: AlphaMode = 'flatten-white',
        background_color: Optional[Tuple[float, float, float]] = None,
        gradient_colors: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        brand_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Composite image with alpha channel using specified mode.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (RGB or RGBA), float [0, 1] or uint8/uint16
        mode : AlphaMode
            Compositing mode
        background_color : Tuple[float, float, float], optional
            Custom background color for flatten modes (R, G, B in [0, 1])
        gradient_colors : Tuple[Tuple, Tuple], optional
            Top and bottom colors for gradient mode
        brand_image : np.ndarray, optional
            Background image for branded mode
        
        Returns:
        --------
        result : np.ndarray
            Composited image (same dtype as input)
        """
        # Normalize input to float [0, 1]
        original_dtype = image.dtype
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image_float = image.astype(np.float32) / 65535.0
        else:
            image_float = image.astype(np.float32)
        
        # Separate RGB and alpha
        rgb, alpha = self.separate_alpha(image_float)
        
        if alpha is None:
            # No alpha channel, return as-is
            return image
        
        # Ensure alpha is in [0, 1]
        alpha = np.clip(alpha, 0, 1)
        
        # Apply compositing based on mode
        if mode == 'preserve':
            result = image_float
        
        elif mode == 'flatten-white':
            bg_color = background_color if background_color else (1.0, 1.0, 1.0)
            result = self._flatten_to_color(rgb, alpha, bg_color)
        
        elif mode == 'flatten-black':
            bg_color = background_color if background_color else (0.0, 0.0, 0.0)
            result = self._flatten_to_color(rgb, alpha, bg_color)
        
        elif mode == 'flatten-gray':
            bg_color = background_color if background_color else (0.5, 0.5, 0.5)
            result = self._flatten_to_color(rgb, alpha, bg_color)
        
        elif mode == 'composite-gradient':
            if gradient_colors is None:
                # Default: white to light gray gradient (top to bottom)
                gradient_colors = ((1.0, 1.0, 1.0), (0.95, 0.95, 0.95))
            result = self._composite_gradient(rgb, alpha, gradient_colors)
        
        elif mode == 'composite-branded':
            if brand_image is None:
                raise ValueError("brand_image required for composite-branded mode")
            result = self._composite_branded(rgb, alpha, brand_image)
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        # Convert back to original dtype
        if original_dtype == np.uint8:
            result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            result = (np.clip(result, 0, 1) * 65535).astype(np.uint16)
        else:
            result = result.astype(original_dtype)
        
        return result
    
    def _flatten_to_color(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        bg_color: Tuple[float, float, float]
    ) -> np.ndarray:
        """Flatten alpha channel to solid background color."""
        alpha_3d = alpha[:, :, np.newaxis]
        bg = np.array(bg_color).reshape(1, 1, 3)
        
        # Alpha compositing: result = foreground * alpha + background * (1 - alpha)
        result = rgb * alpha_3d + bg * (1 - alpha_3d)
        
        return np.clip(result, 0, 1)
    
    def _composite_gradient(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        gradient_colors: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> np.ndarray:
        """Composite onto gradient background."""
        height, width = rgb.shape[:2]
        
        # Create gradient
        top_color = np.array(gradient_colors[0]).reshape(1, 1, 3)
        bottom_color = np.array(gradient_colors[1]).reshape(1, 1, 3)
        
        # Linear gradient (top to bottom)
        gradient_factor = np.linspace(0, 1, height).reshape(-1, 1, 1)
        gradient_bg = top_color * (1 - gradient_factor) + bottom_color * gradient_factor
        gradient_bg = np.broadcast_to(gradient_bg, (height, width, 3))
        
        # Alpha composite
        alpha_3d = alpha[:, :, np.newaxis]
        result = rgb * alpha_3d + gradient_bg * (1 - alpha_3d)
        
        return np.clip(result, 0, 1)
    
    def _composite_branded(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        brand_image: np.ndarray
    ) -> np.ndarray:
        """Composite onto branded background image."""
        height, width = rgb.shape[:2]
        
        # Normalize brand image
        if brand_image.dtype == np.uint8:
            brand_float = brand_image.astype(np.float32) / 255.0
        elif brand_image.dtype == np.uint16:
            brand_float = brand_image.astype(np.float32) / 65535.0
        else:
            brand_float = brand_image.astype(np.float32)
        
        # Handle alpha in brand image
        if brand_float.shape[2] == 4:
            brand_float = brand_float[:, :, :3]
        
        # Resize brand image to match
        if brand_float.shape[:2] != (height, width):
            from PIL import Image as PILImage
            brand_pil = PILImage.fromarray((np.clip(brand_float, 0, 1) * 255).astype(np.uint8))
            brand_pil = brand_pil.resize((width, height), PILImage.Resampling.LANCZOS)
            brand_float = np.array(brand_pil).astype(np.float32) / 255.0
        
        # Alpha composite
        alpha_3d = alpha[:, :, np.newaxis]
        result = rgb * alpha_3d + brand_float * (1 - alpha_3d)
        
        return np.clip(result, 0, 1)
    
    def generate_variants(
        self,
        image: np.ndarray,
        modes: Optional[list] = None
    ) -> dict:
        """
        Generate multiple variants with different alpha handling.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image with alpha channel
        modes : list, optional
            List of modes to generate (default: all flatten modes + preserve)
        
        Returns:
        --------
        variants : dict
            Dictionary mapping mode name to composited image
        """
        if modes is None:
            modes = ['preserve', 'flatten-white', 'flatten-black', 'flatten-gray']
        
        variants = {}
        
        for mode in modes:
            try:
                variants[mode] = self.composite(image, mode)
            except Exception as e:
                print(f"⚠ Warning: Could not generate variant '{mode}': {e}")
        
        return variants
    
    def save_variants(
        self,
        image: np.ndarray,
        output_dir: Path,
        base_name: str,
        modes: Optional[list] = None
    ) -> dict:
        """
        Generate and save multiple alpha variants.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image with alpha channel
        output_dir : Path
            Output directory
        base_name : str
            Base filename (without extension)
        modes : list, optional
            List of modes to generate
        
        Returns:
        --------
        paths : dict
            Dictionary mapping mode name to saved file path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        variants = self.generate_variants(image, modes)
        paths = {}
        
        for mode, variant in variants.items():
            # Determine file extension based on mode
            if mode == 'preserve':
                # Save as PNG to preserve alpha
                output_path = output_dir / f"{base_name}_{mode}.png"
                
                # Convert to 8-bit for PNG
                if variant.dtype in [np.float32, np.float64]:
                    variant_8bit = (np.clip(variant, 0, 1) * 255).astype(np.uint8)
                else:
                    variant_8bit = variant
                
                pil_img = Image.fromarray(variant_8bit)
                pil_img.save(output_path, format='PNG')
            else:
                # Save as JPEG (no alpha)
                output_path = output_dir / f"{base_name}_{mode}.jpg"
                
                # Ensure RGB only
                if variant.shape[2] == 4:
                    variant = variant[:, :, :3]
                
                # Convert to 8-bit
                if variant.dtype in [np.float32, np.float64]:
                    variant_8bit = (np.clip(variant, 0, 1) * 255).astype(np.uint8)
                elif variant.dtype == np.uint16:
                    variant_8bit = (variant / 257).astype(np.uint8)
                else:
                    variant_8bit = variant
                
                pil_img = Image.fromarray(variant_8bit)
                pil_img.save(output_path, format='JPEG', quality=98, subsampling=0, optimize=True)
            
            paths[mode] = output_path
            print(f"  ✓ Saved {mode}: {output_path.name}")
        
        return paths


def main():
    """CLI for alpha compositing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha channel compositor")
    parser.add_argument("input", type=Path, help="Input image with alpha channel")
    parser.add_argument("--output-dir", type=Path, default=Path("output_alpha_variants"), help="Output directory")
    parser.add_argument("--modes", nargs='+', help="Modes to generate (default: all flatten modes + preserve)")
    parser.add_argument("--single-mode", help="Generate only one mode and save to specific path")
    parser.add_argument("--single-output", type=Path, help="Output path for single mode")
    parser.add_argument("--bg-color", nargs=3, type=float, help="Background color for flatten (R G B in [0, 1])")
    parser.add_argument("--gradient", nargs=6, type=float, help="Gradient colors (top_R top_G top_B bottom_R bottom_G bottom_B)")
    
    args = parser.parse_args()
    
    print(f"📷 Loading image: {args.input}")
    
    # Load image
    img = Image.open(args.input)
    img_array = np.array(img)
    
    # Check for alpha
    compositor = AlphaCompositor()
    if not compositor.has_alpha(img_array):
        print("⚠ Warning: Image has no alpha channel")
        return
    
    print(f"✓ Alpha channel detected")
    
    if args.single_mode:
        # Single mode output
        if not args.single_output:
            print("❌ Error: --single-output required with --single-mode")
            return
        
        # Prepare kwargs
        kwargs = {'mode': args.single_mode}
        
        if args.bg_color:
            kwargs['background_color'] = tuple(args.bg_color)
        
        if args.gradient:
            top = tuple(args.gradient[:3])
            bottom = tuple(args.gradient[3:])
            kwargs['gradient_colors'] = (top, bottom)
        
        # Composite
        result = compositor.composite(img_array, **kwargs)
        
        # Save
        if result.shape[2] == 4 and args.single_mode == 'preserve':
            # Save as PNG with alpha
            if result.dtype in [np.float32, np.float64]:
                result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(result).save(args.single_output, format='PNG')
        else:
            # Save as JPEG
            if result.shape[2] == 4:
                result = result[:, :, :3]
            if result.dtype in [np.float32, np.float64]:
                result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(result).save(args.single_output, format='JPEG', quality=98)
        
        print(f"✅ Saved: {args.single_output}")
    else:
        # Generate variants
        print(f"\n🎨 Generating alpha variants...")
        
        base_name = args.input.stem
        paths = compositor.save_variants(img_array, args.output_dir, base_name, args.modes)
        
        print(f"\n✅ Generated {len(paths)} variants in: {args.output_dir}")


if __name__ == "__main__":
    main()
