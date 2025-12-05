#!/usr/bin/env python3
"""
Multi-Exposure Fusion
======================
Extract multiple exposure brackets from HDR data and generate optimized
variants for different output media (web, print, social).

Features:
- Automatic exposure bracketing from HDR/32-bit sources
- Exposure-optimized variants (web vs print)
- Laplacian pyramid fusion for maximum dynamic range
- Bracketed sequence export for client review
"""

from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image


class ExposureTarget(Enum):
    """Output target for exposure optimization."""
    WEB = "web"           # sRGB, moderate contrast
    PRINT = "print"       # Wide gamut, high dynamic range
    SOCIAL = "social"     # High contrast, saturated
    REFERENCE = "reference"  # Neutral, maximum fidelity


@dataclass
class ExposureVariant:
    """An exposure-optimized image variant."""
    target: ExposureTarget
    exposure_ev: float  # Exposure compensation in EV stops
    image: np.ndarray   # [0, 1] float array
    description: str


class ExposureFusion:
    """
    Multi-exposure fusion and bracketing system.
    """
    
    def __init__(self):
        """Initialize exposure fusion processor."""
        pass
    
    def extract_brackets(
        self,
        hdr_image: np.ndarray,
        num_brackets: int = 3,
        ev_range: float = 2.0
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Extract exposure brackets from HDR image.
        
        Args:
            hdr_image: HDR image in linear space [0, inf]
            num_brackets: Number of brackets to extract
            ev_range: Total EV range (e.g., 2.0 = ±1 EV)
            
        Returns:
            List of (ev_offset, image) tuples
        """
        brackets = []
        
        # Calculate EV stops
        if num_brackets == 1:
            ev_stops = [0.0]
        else:
            ev_stops = np.linspace(-ev_range/2, ev_range/2, num_brackets)
        
        for ev in ev_stops:
            # Apply exposure compensation
            exposed = hdr_image * (2 ** ev)
            
            # Tone map to [0, 1]
            tone_mapped = self._tone_map(exposed)
            
            brackets.append((ev, tone_mapped))
        
        return brackets
    
    def _tone_map(self, hdr: np.ndarray, method: str = 'reinhard') -> np.ndarray:
        """
        Tone map HDR to LDR.
        
        Args:
            hdr: HDR image in linear space
            method: Tone mapping operator
            
        Returns:
            Tone-mapped image [0, 1]
        """
        if method == 'reinhard':
            # Reinhard global operator
            lum = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
            lum_mapped = lum / (1.0 + lum)
            
            # Scale RGB channels
            scale = np.divide(
                lum_mapped,
                lum + 1e-8,
                where=lum > 0
            )[..., np.newaxis]
            
            result = hdr * scale
            
        elif method == 'clamp':
            result = np.clip(hdr, 0, 1)
        
        else:
            raise ValueError(f"Unknown tone mapping method: {method}")
        
        return result.clip(0, 1)
    
    def fuse_exposures(
        self,
        brackets: List[np.ndarray],
        method: str = 'laplacian'
    ) -> np.ndarray:
        """
        Fuse multiple exposures into single HDR-like image.
        
        Args:
            brackets: List of exposure-bracketed images [0, 1]
            method: Fusion method ('laplacian' or 'weighted_average')
            
        Returns:
            Fused image [0, 1]
        """
        if len(brackets) == 1:
            return brackets[0]
        
        if method == 'weighted_average':
            return self._weighted_average_fusion(brackets)
        elif method == 'laplacian':
            return self._laplacian_pyramid_fusion(brackets)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    def _weighted_average_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """Simple weighted average based on well-exposedness."""
        weights = []
        
        for bracket in brackets:
            # Weight based on distance from mid-gray
            weight = 1.0 - np.abs(bracket - 0.5) * 2.0
            weight = weight.mean(axis=-1, keepdims=True)
            weight = np.power(weight, 2)  # Emphasize well-exposed regions
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights) + 1e-8
        weights = [w / total_weight for w in weights]
        
        # Weighted sum
        fused = sum(b * w for b, w in zip(brackets, weights))
        
        return fused.clip(0, 1)
    
    def _laplacian_pyramid_fusion(self, brackets: List[np.ndarray]) -> np.ndarray:
        """Laplacian pyramid fusion for detail preservation."""
        # Build Laplacian pyramids
        pyramids = [self._build_laplacian_pyramid(b) for b in brackets]
        
        # Build weight pyramids
        weight_pyramids = []
        for bracket in brackets:
            weight = self._compute_quality_weight(bracket)
            weight_pyr = self._build_gaussian_pyramid(weight)
            weight_pyramids.append(weight_pyr)
        
        # Fuse pyramids level by level
        fused_pyramid = []
        num_levels = len(pyramids[0])
        
        for level in range(num_levels):
            level_images = [p[level] for p in pyramids]
            level_weights = [w[level] if level < len(w) else w[-1] for w in weight_pyramids]
            
            # Normalize weights
            total_weight = sum(level_weights) + 1e-8
            level_weights = [w / total_weight for w in level_weights]
            
            # Weighted fusion
            if level_images[0].ndim == 3:
                # Expand weight dimensions for RGB
                level_weights = [w[..., np.newaxis] for w in level_weights]
            
            fused_level = sum(img * w for img, w in zip(level_images, level_weights))
            fused_pyramid.append(fused_level)
        
        # Reconstruct from pyramid
        result = self._reconstruct_from_laplacian(fused_pyramid)
        
        return result.clip(0, 1)
    
    def _build_gaussian_pyramid(self, image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
        """Build Gaussian pyramid."""
        pyramid = [image]
        
        for _ in range(levels - 1):
            # Downsample
            downsampled = self._downsample(pyramid[-1])
            pyramid.append(downsampled)
        
        return pyramid
    
    def _build_laplacian_pyramid(self, image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
        """Build Laplacian pyramid."""
        gaussian_pyr = self._build_gaussian_pyramid(image, levels)
        laplacian_pyr = []
        
        for i in range(len(gaussian_pyr) - 1):
            # Upsample next level
            upsampled = self._upsample(gaussian_pyr[i + 1], gaussian_pyr[i].shape[:2])
            
            # Laplacian = current - upsampled_next
            laplacian = gaussian_pyr[i] - upsampled
            laplacian_pyr.append(laplacian)
        
        # Last level is the residual
        laplacian_pyr.append(gaussian_pyr[-1])
        
        return laplacian_pyr
    
    def _reconstruct_from_laplacian(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        result = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            # Upsample and add
            result = self._upsample(result, pyramid[i].shape[:2])
            result = result + pyramid[i]
        
        return result
    
    def _downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample by factor of 2."""
        from PIL import Image as PILImage
        
        h, w = image.shape[:2]
        new_h, new_w = h // 2, w // 2
        
        if image.ndim == 3:
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            downsampled = img_pil.resize((new_w, new_h), PILImage.BILINEAR)
            return np.array(downsampled).astype(np.float32) / 255.0
        else:
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            downsampled = img_pil.resize((new_w, new_h), PILImage.BILINEAR)
            return np.array(downsampled).astype(np.float32) / 255.0
    
    def _upsample(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Upsample to target shape."""
        from PIL import Image as PILImage
        
        target_h, target_w = target_shape
        
        if image.ndim == 3:
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            upsampled = img_pil.resize((target_w, target_h), PILImage.BILINEAR)
            return np.array(upsampled).astype(np.float32) / 255.0
        else:
            img_pil = PILImage.fromarray((image * 255).astype(np.uint8))
            upsampled = img_pil.resize((target_w, target_h), PILImage.BILINEAR)
            return np.array(upsampled).astype(np.float32) / 255.0
    
    def _compute_quality_weight(self, image: np.ndarray) -> np.ndarray:
        """Compute quality weight based on contrast, saturation, and well-exposedness."""
        # Convert to grayscale for some metrics
        if image.ndim == 3:
            gray = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        else:
            gray = image
        
        # Contrast weight (Laplacian)
        dy, dx = np.gradient(gray)
        contrast = np.sqrt(dx**2 + dy**2)
        contrast_weight = contrast / (contrast.max() + 1e-8)
        
        # Well-exposedness weight (Gaussian around 0.5)
        exposedness = np.exp(-((gray - 0.5) ** 2) / (2 * 0.2 ** 2))
        
        # Saturation weight
        if image.ndim == 3:
            saturation = image.std(axis=-1)
            saturation_weight = saturation / (saturation.max() + 1e-8)
        else:
            saturation_weight = np.ones_like(gray)
        
        # Combine weights
        weight = contrast_weight * exposedness * saturation_weight
        
        return weight
    
    def generate_variants(
        self,
        hdr_image: np.ndarray
    ) -> List[ExposureVariant]:
        """
        Generate exposure-optimized variants for different output targets.
        
        Args:
            hdr_image: HDR image in linear space
            
        Returns:
            List of ExposureVariant objects
        """
        variants = []
        
        # Web variant: slightly underexposed, moderate contrast
        web_exposed = hdr_image * (2 ** -0.3)
        web_tone = self._tone_map(web_exposed, method='reinhard')
        web_tone = self._adjust_for_web(web_tone)
        variants.append(ExposureVariant(
            target=ExposureTarget.WEB,
            exposure_ev=-0.3,
            image=web_tone,
            description="sRGB optimized, web-ready (slightly darker for detail preservation)"
        ))
        
        # Print variant: preserve highlights, wider dynamic range
        print_exposed = hdr_image * (2 ** 0.0)
        print_tone = self._tone_map(print_exposed, method='reinhard')
        print_tone = self._adjust_for_print(print_tone)
        variants.append(ExposureVariant(
            target=ExposureTarget.PRINT,
            exposure_ev=0.0,
            image=print_tone,
            description="Wide gamut, high dynamic range for professional printing"
        ))
        
        # Social media variant: punchy, high contrast
        social_exposed = hdr_image * (2 ** 0.2)
        social_tone = self._tone_map(social_exposed, method='reinhard')
        social_tone = self._adjust_for_social(social_tone)
        variants.append(ExposureVariant(
            target=ExposureTarget.SOCIAL,
            exposure_ev=0.2,
            image=social_tone,
            description="High contrast, saturated for social media impact"
        ))
        
        return variants
    
    def _adjust_for_web(self, image: np.ndarray) -> np.ndarray:
        """Apply web-specific adjustments."""
        # Slight contrast boost
        adjusted = (image - 0.5) * 1.1 + 0.5
        
        # Subtle saturation increase
        adjusted = self._adjust_saturation(adjusted, 1.05)
        
        return adjusted.clip(0, 1)
    
    def _adjust_for_print(self, image: np.ndarray) -> np.ndarray:
        """Apply print-specific adjustments."""
        # Preserve highlights and shadows
        # No additional contrast
        
        # Slight saturation boost for print vibrancy
        adjusted = self._adjust_saturation(image, 1.08)
        
        return adjusted.clip(0, 1)
    
    def _adjust_for_social(self, image: np.ndarray) -> np.ndarray:
        """Apply social media-specific adjustments."""
        # Strong contrast boost
        adjusted = (image - 0.5) * 1.25 + 0.5
        
        # Increased saturation
        adjusted = self._adjust_saturation(adjusted, 1.15)
        
        # Slight warmth
        adjusted[..., 0] = np.clip(adjusted[..., 0] * 1.05, 0, 1)  # More red
        adjusted[..., 2] = np.clip(adjusted[..., 2] * 0.98, 0, 1)  # Less blue
        
        return adjusted.clip(0, 1)
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust saturation by factor."""
        if image.ndim != 3:
            return image
        
        # Convert to HSV
        from colorsys import rgb_to_hsv, hsv_to_rgb
        
        result = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j]
                h, s, v = rgb_to_hsv(r, g, b)
                s = min(s * factor, 1.0)
                result[i, j] = hsv_to_rgb(h, s, v)
        
        return result


def main():
    """CLI for exposure fusion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Exposure Fusion")
    parser.add_argument('input', type=Path, help='Input HDR image (TIFF/EXR)')
    parser.add_argument('--output-dir', type=Path, default=Path('output_exposure_fusion'),
                       help='Output directory')
    parser.add_argument('--brackets', type=int, default=3, help='Number of brackets')
    parser.add_argument('--ev-range', type=float, default=2.0, help='EV range for brackets')
    parser.add_argument('--generate-variants', action='store_true',
                       help='Generate web/print/social variants')
    parser.add_argument('--fuse', action='store_true',
                       help='Fuse brackets into single image')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Multi-Exposure Fusion")
    print(f"{'='*60}\n")
    print(f"Input: {args.input}")
    
    # Load image
    print(f"📷 Loading HDR image...")
    
    try:
        import tifffile
        img = tifffile.imread(args.input)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
    except ImportError:
        img = np.array(Image.open(args.input)).astype(np.float32) / 255.0
    
    print(f"✓ Loaded: {img.shape}, dtype={img.dtype}")
    
    fusion = ExposureFusion()
    
    # Generate variants
    if args.generate_variants:
        print(f"\n🎨 Generating exposure variants...")
        variants = fusion.generate_variants(img)
        
        for variant in variants:
            output_path = args.output_dir / f"{args.input.stem}_{variant.target.value}.png"
            variant_img = Image.fromarray((variant.image * 255).astype(np.uint8))
            variant_img.save(output_path, quality=95)
            print(f"✓ {variant.target.value:8s} (EV {variant.exposure_ev:+.1f}): {output_path}")
            print(f"  └─ {variant.description}")
    
    # Extract brackets
    print(f"\n📊 Extracting {args.brackets} exposure brackets (±{args.ev_range/2:.1f} EV)...")
    brackets = fusion.extract_brackets(img, args.brackets, args.ev_range)
    
    for i, (ev, bracket) in enumerate(brackets):
        output_path = args.output_dir / f"{args.input.stem}_bracket_{i:02d}_ev{ev:+.1f}.png"
        bracket_img = Image.fromarray((bracket * 255).astype(np.uint8))
        bracket_img.save(output_path, quality=95)
        print(f"✓ Bracket {i+1}/{args.brackets} (EV {ev:+.1f}): {output_path}")
    
    # Fuse brackets
    if args.fuse and len(brackets) > 1:
        print(f"\n🔗 Fusing brackets...")
        bracket_images = [b[1] for b in brackets]
        fused = fusion.fuse_exposures(bracket_images, method='laplacian')
        
        output_path = args.output_dir / f"{args.input.stem}_fused.png"
        fused_img = Image.fromarray((fused * 255).astype(np.uint8))
        fused_img.save(output_path, quality=95)
        print(f"✓ Fused: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
