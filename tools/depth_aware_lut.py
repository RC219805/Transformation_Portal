#!/usr/bin/env python3
"""
Depth-Aware LUT Application
============================
Apply LUTs with depth-dependent strength for atmospheric perspective and
realistic color grading that respects spatial depth.

Features:
- Zone-based LUT strength (foreground/midground/background)
- Atmospheric perspective simulation
- Per-zone color temperature adjustments
- Multiple LUT stacking with depth weighting
"""

from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image


class DepthZone(Enum):
    """Depth zones for LUT application."""
    FOREGROUND = "foreground"  # Closest to camera
    MIDGROUND = "midground"    # Middle distance
    BACKGROUND = "background"  # Farthest from camera


@dataclass
class ZoneLUTConfig:
    """LUT configuration for a specific depth zone."""
    zone: DepthZone
    lut_path: Path
    strength: float = 1.0  # 0.0 to 1.0
    color_temp_shift: int = 0  # Kelvin shift (negative = cooler, positive = warmer)
    saturation_mult: float = 1.0  # Saturation multiplier for zone


@dataclass
class DepthAwareLUTConfig:
    """Complete depth-aware LUT configuration."""
    zone_configs: Dict[DepthZone, ZoneLUTConfig]
    atmospheric_strength: float = 0.0  # 0.0 to 1.0
    depth_falloff: float = 2.0  # Exponential falloff for zone blending
    preserve_edges: bool = True  # Preserve depth edges during blending


class LUTReader:
    """Read and apply 3D LUT files (.cube format)."""

    @staticmethod
    def read_cube_lut(lut_path: Path) -> Tuple[np.ndarray, int]:
        """
        Read .cube LUT file.

        Args:
            lut_path: Path to .cube file

        Returns:
            (lut_array, size) - 3D LUT array and grid size
        """
        with open(lut_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        size = None
        lut_data = []

        for line in lines:
            line = line.strip()

            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
            elif line and not line.startswith('#') and not line.startswith('TITLE') and not line.startswith('DOMAIN'):
                # Data line
                try:
                    values = [float(v) for v in line.split()]
                    if len(values) == 3:
                        lut_data.append(values)
                except ValueError:
                    continue

        if size is None:
            raise ValueError(f"Could not find LUT_3D_SIZE in {lut_path}")

        # Reshape to 3D grid
        lut_array = np.array(lut_data, dtype=np.float32)
        lut_array = lut_array.reshape((size, size, size, 3))

        return lut_array, size

    @staticmethod
    def apply_lut(image: np.ndarray, lut: np.ndarray, lut_size: int) -> np.ndarray:
        """
        Apply 3D LUT to image using trilinear interpolation.

        Args:
            image: Input image [0, 1] (H, W, 3)
            lut: 3D LUT array (size, size, size, 3)
            lut_size: LUT grid size

        Returns:
            LUT-transformed image [0, 1]
        """
        # Scale input to LUT coordinates
        scaled = image * (lut_size - 1)

        # Get integer and fractional parts
        r_idx = scaled[..., 0]
        g_idx = scaled[..., 1]
        b_idx = scaled[..., 2]

        # Floor indices
        r0 = np.floor(r_idx).astype(np.int32).clip(0, lut_size - 2)
        g0 = np.floor(g_idx).astype(np.int32).clip(0, lut_size - 2)
        b0 = np.floor(b_idx).astype(np.int32).clip(0, lut_size - 2)

        r1 = (r0 + 1).clip(0, lut_size - 1)
        g1 = (g0 + 1).clip(0, lut_size - 1)
        b1 = (b0 + 1).clip(0, lut_size - 1)

        # Fractional parts
        r_frac = (r_idx - r0).clip(0, 1)
        g_frac = (g_idx - g0).clip(0, 1)
        b_frac = (b_idx - b0).clip(0, 1)

        # Expand dimensions for broadcasting
        r_frac = r_frac[..., np.newaxis]
        g_frac = g_frac[..., np.newaxis]
        b_frac = b_frac[..., np.newaxis]

        # Trilinear interpolation (8 corner samples)
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        # Interpolate along r
        c00 = c000 * (1 - r_frac) + c100 * r_frac
        c01 = c001 * (1 - r_frac) + c101 * r_frac
        c10 = c010 * (1 - r_frac) + c110 * r_frac
        c11 = c011 * (1 - r_frac) + c111 * r_frac

        # Interpolate along g
        c0 = c00 * (1 - g_frac) + c10 * g_frac
        c1 = c01 * (1 - g_frac) + c11 * g_frac

        # Interpolate along b
        result = c0 * (1 - b_frac) + c1 * b_frac

        return result.clip(0, 1)


class DepthAwareLUT:
    """
    Apply LUTs with depth-dependent strength for realistic atmospheric effects.
    """

    def __init__(self, config: DepthAwareLUTConfig):
        """
        Initialize depth-aware LUT processor.

        Args:
            config: Depth-aware LUT configuration
        """
        self.config = config
        self.lut_reader = LUTReader()
        self._load_luts()

    def _load_luts(self):
        """Load all LUT files."""
        self.luts = {}
        self.lut_sizes = {}

        for zone, zone_config in self.config.zone_configs.items():
            if zone_config.lut_path.exists():
                lut, size = self.lut_reader.read_cube_lut(zone_config.lut_path)
                self.luts[zone] = lut
                self.lut_sizes[zone] = size
                print(f"✓ Loaded {zone.value} LUT: {zone_config.lut_path.name} (size: {size})")
            else:
                print(f"⚠ Warning: LUT not found: {zone_config.lut_path}")

    def apply(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply depth-aware LUT to image.

        Args:
            image: Input image [0, 1] (H, W, 3)
            depth_map: Normalized depth map [0, 1] (H, W) - 0=near, 1=far

        Returns:
            LUT-transformed image with depth-dependent strength
        """
        # Ensure depth map is 2D
        if depth_map.ndim == 3:
            depth_map = depth_map[..., 0]

        # Create zone masks with smooth transitions
        zone_masks = self._create_zone_masks(depth_map)

        # Apply LUT to entire image for each zone
        zone_outputs = {}
        for zone in DepthZone:
            if zone not in self.luts:
                continue

            zone_config = self.config.zone_configs[zone]

            # Apply LUT
            lut_applied = self.lut_reader.apply_lut(
                image, self.luts[zone], self.lut_sizes[zone]
            )

            # Blend with original based on strength
            lut_applied = image * (1 - zone_config.strength) + lut_applied * zone_config.strength

            # Apply color temperature shift
            if zone_config.color_temp_shift != 0:
                lut_applied = self._apply_color_temp(lut_applied, zone_config.color_temp_shift)

            # Apply saturation adjustment
            if zone_config.saturation_mult != 1.0:
                lut_applied = self._adjust_saturation(lut_applied, zone_config.saturation_mult)

            zone_outputs[zone] = lut_applied

        # Blend zones based on masks
        result = np.zeros_like(image)
        total_weight = np.zeros((*image.shape[:2], 1))

        for zone, mask in zone_masks.items():
            if zone in zone_outputs:
                mask_3d = mask[..., np.newaxis]
                result += zone_outputs[zone] * mask_3d
                total_weight += mask_3d

        # Normalize by total weight
        result = np.divide(result, total_weight, where=total_weight > 0)

        # Apply atmospheric effects if enabled
        if self.config.atmospheric_strength > 0:
            result = self._apply_atmospheric_perspective(result, depth_map)

        return result.clip(0, 1)

    def _create_zone_masks(self, depth_map: np.ndarray) -> Dict[DepthZone, np.ndarray]:
        """
        Create smooth zone masks from depth map.

        Args:
            depth_map: Normalized depth [0, 1] - 0=near, 1=far

        Returns:
            Dictionary of zone masks [0, 1]
        """
        masks = {}

        # Define zone boundaries
        fg_max = 0.33
        mg_max = 0.67

        # Foreground mask: strong near camera, fades out
        fg_mask = 1.0 - np.clip(depth_map / fg_max, 0, 1)
        fg_mask = np.power(fg_mask, self.config.depth_falloff)
        masks[DepthZone.FOREGROUND] = fg_mask

        # Background mask: strong far from camera, fades in
        bg_mask = np.clip((depth_map - mg_max) / (1.0 - mg_max), 0, 1)
        bg_mask = np.power(bg_mask, self.config.depth_falloff)
        masks[DepthZone.BACKGROUND] = bg_mask

        # Midground mask: peak in middle, fades both directions
        mg_center = (fg_max + mg_max) / 2
        mg_width = (mg_max - fg_max) / 2
        mg_mask = 1.0 - np.abs(depth_map - mg_center) / mg_width
        mg_mask = np.clip(mg_mask, 0, 1)
        mg_mask = np.power(mg_mask, self.config.depth_falloff / 2)
        masks[DepthZone.MIDGROUND] = mg_mask

        # Normalize so masks sum to 1
        total = fg_mask + mg_mask + bg_mask
        for zone in masks:
            masks[zone] = np.divide(masks[zone], total, where=total > 0)

        return masks

    def _apply_color_temp(self, image: np.ndarray, kelvin_shift: int) -> np.ndarray:
        """
        Apply color temperature shift.

        Args:
            image: Input image [0, 1]
            kelvin_shift: Temperature shift in Kelvin (negative=cooler, positive=warmer)

        Returns:
            Temperature-adjusted image
        """
        # Simple temperature adjustment via RGB multipliers
        shift_factor = kelvin_shift / 10000.0  # Normalize

        adjusted = image.copy()

        if shift_factor > 0:  # Warmer
            adjusted[..., 0] = np.clip(adjusted[..., 0] * (1 + shift_factor), 0, 1)  # More red
            adjusted[..., 2] = np.clip(adjusted[..., 2] * (1 - shift_factor * 0.5), 0, 1)  # Less blue
        else:  # Cooler
            adjusted[..., 2] = np.clip(adjusted[..., 2] * (1 - shift_factor), 0, 1)  # More blue
            adjusted[..., 0] = np.clip(adjusted[..., 0] * (1 + shift_factor * 0.5), 0, 1)  # Less red

        return adjusted

    def _adjust_saturation(self, image: np.ndarray, mult: float) -> np.ndarray:
        """
        Adjust saturation.

        Args:
            image: Input image [0, 1]
            mult: Saturation multiplier

        Returns:
            Saturation-adjusted image
        """
        # Convert to HSV
        hsv = self._rgb_to_hsv(image)

        # Adjust saturation
        hsv[..., 1] = np.clip(hsv[..., 1] * mult, 0, 1)

        # Convert back to RGB
        return self._hsv_to_rgb(hsv)

    def _apply_atmospheric_perspective(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Apply atmospheric haze effect based on depth.

        Args:
            image: Input image [0, 1]
            depth_map: Normalized depth [0, 1]

        Returns:
            Image with atmospheric perspective
        """
        # Atmospheric color (light blue-gray)
        atmo_color = np.array([0.7, 0.75, 0.85])

        # Depth-dependent haze strength
        haze_strength = np.power(depth_map, 1.5) * self.config.atmospheric_strength
        haze_strength = haze_strength[..., np.newaxis]

        # Blend with atmospheric color
        result = image * (1 - haze_strength) + atmo_color * haze_strength

        return result

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c

        # Hue
        h = np.zeros_like(max_c)
        mask = diff != 0
        r_mask = (max_c == r) & mask
        g_mask = (max_c == g) & mask
        b_mask = (max_c == b) & mask

        h[r_mask] = 60 * (((g[r_mask] - b[r_mask]) / diff[r_mask]) % 6)
        h[g_mask] = 60 * (((b[g_mask] - r[g_mask]) / diff[g_mask]) + 2)
        h[b_mask] = 60 * (((r[b_mask] - g[b_mask]) / diff[b_mask]) + 4)

        # Saturation
        s = np.zeros_like(max_c)
        s[max_c != 0] = diff[max_c != 0] / max_c[max_c != 0]

        # Value
        v = max_c

        return np.stack([h, s, v], axis=-1)

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        c = v * s
        h_prime = h / 60.0
        x = c * (1 - np.abs(h_prime % 2 - 1))

        rgb = np.zeros_like(hsv)

        mask = (0 <= h_prime) & (h_prime < 1)
        rgb[..., 0][mask] = c[mask]
        rgb[..., 1][mask] = x[mask]
        rgb[..., 2][mask] = 0

        mask = (1 <= h_prime) & (h_prime < 2)
        rgb[..., 0][mask] = x[mask]
        rgb[..., 1][mask] = c[mask]
        rgb[..., 2][mask] = 0

        mask = (2 <= h_prime) & (h_prime < 3)
        rgb[..., 0][mask] = 0
        rgb[..., 1][mask] = c[mask]
        rgb[..., 2][mask] = x[mask]

        mask = (3 <= h_prime) & (h_prime < 4)
        rgb[..., 0][mask] = 0
        rgb[..., 1][mask] = x[mask]
        rgb[..., 2][mask] = c[mask]

        mask = (4 <= h_prime) & (h_prime < 5)
        rgb[..., 0][mask] = x[mask]
        rgb[..., 1][mask] = 0
        rgb[..., 2][mask] = c[mask]

        mask = (5 <= h_prime) & (h_prime < 6)
        rgb[..., 0][mask] = c[mask]
        rgb[..., 1][mask] = 0
        rgb[..., 2][mask] = x[mask]

        m = v - c
        rgb += m[..., np.newaxis]

        return rgb.clip(0, 1)


def create_depth_map(image_path: Path) -> np.ndarray:
    """
    Create depth map using Depth Anything V2 (if available).

    Args:
        image_path: Path to input image

    Returns:
        Normalized depth map [0, 1] - 0=near, 1=far
    """
    try:
        from src.transformation_portal.depth.models.depth_anything_v2 import DepthAnythingV2Estimator

        estimator = DepthAnythingV2Estimator()
        depth = estimator.estimate_depth(image_path)

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    except ImportError:
        print("⚠ Warning: Depth Anything V2 not available, using fallback depth estimation")
        # Fallback: simple gradient-based depth (top=far, bottom=near)
        img = Image.open(image_path)
        h, w = img.height, img.width

        # Vertical gradient
        depth = np.linspace(0.3, 1.0, h)[:, np.newaxis]
        depth = np.repeat(depth, w, axis=1)

        return depth


def main():
    """CLI for depth-aware LUT application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Depth-Aware LUT Application"
    )
    parser.add_argument('input', type=Path, help='Input image path')
    parser.add_argument('--output', type=Path, required=True, help='Output image path')
    parser.add_argument('--depth-map', type=Path, help='Pre-computed depth map (optional)')

    # Zone LUT configurations
    parser.add_argument('--fg-lut', type=Path, help='Foreground LUT path')
    parser.add_argument('--fg-strength', type=float, default=0.8, help='Foreground LUT strength')
    parser.add_argument('--fg-temp', type=int, default=0, help='Foreground color temp shift (K)')

    parser.add_argument('--mg-lut', type=Path, help='Midground LUT path')
    parser.add_argument('--mg-strength', type=float, default=0.7, help='Midground LUT strength')
    parser.add_argument('--mg-temp', type=int, default=0, help='Midground color temp shift (K)')

    parser.add_argument('--bg-lut', type=Path, help='Background LUT path')
    parser.add_argument('--bg-strength', type=float, default=0.6, help='Background LUT strength')
    parser.add_argument('--bg-temp', type=int, default=200,
                        help='Background color temp shift (K, default: +200 for atmospheric warmth)')

    parser.add_argument('--atmospheric', type=float, default=0.3,
                        help='Atmospheric perspective strength (0-1)')
    parser.add_argument('--depth-falloff', type=float, default=2.0,
                        help='Depth zone falloff exponent')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Depth-Aware LUT Application")
    print(f"{'='*60}\n")

    # Load image
    print(f"📷 Loading image: {args.input}")
    img = Image.open(args.input).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0

    # Load or create depth map
    if args.depth_map:
        print(f"📊 Loading depth map: {args.depth_map}")
        depth = np.array(Image.open(args.depth_map).convert('L')).astype(np.float32) / 255.0
    else:
        print(f"📊 Creating depth map...")
        depth = create_depth_map(args.input)

    # Resize depth to match image if needed
    if depth.shape[:2] != img_array.shape[:2]:
        from PIL import Image as PILImage
        depth_img = PILImage.fromarray((depth * 255).astype(np.uint8))
        depth_img = depth_img.resize((img_array.shape[1], img_array.shape[0]), PILImage.BILINEAR)
        depth = np.array(depth_img).astype(np.float32) / 255.0

    # Build zone configurations
    zone_configs = {}

    if args.fg_lut:
        zone_configs[DepthZone.FOREGROUND] = ZoneLUTConfig(
            zone=DepthZone.FOREGROUND,
            lut_path=args.fg_lut,
            strength=args.fg_strength,
            color_temp_shift=args.fg_temp
        )

    if args.mg_lut:
        zone_configs[DepthZone.MIDGROUND] = ZoneLUTConfig(
            zone=DepthZone.MIDGROUND,
            lut_path=args.mg_lut,
            strength=args.mg_strength,
            color_temp_shift=args.mg_temp
        )

    if args.bg_lut:
        zone_configs[DepthZone.BACKGROUND] = ZoneLUTConfig(
            zone=DepthZone.BACKGROUND,
            lut_path=args.bg_lut,
            strength=args.bg_strength,
            color_temp_shift=args.bg_temp
        )

    if not zone_configs:
        print("❌ Error: At least one zone LUT must be specified")
        return 1

    # Create configuration
    config = DepthAwareLUTConfig(
        zone_configs=zone_configs,
        atmospheric_strength=args.atmospheric,
        depth_falloff=args.depth_falloff
    )

    # Apply depth-aware LUT
    print(f"\n🎨 Applying depth-aware LUT...")
    processor = DepthAwareLUT(config)
    result = processor.apply(img_array, depth)

    # Save result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_img = Image.fromarray((result * 255).astype(np.uint8))
    result_img.save(args.output, quality=95)

    print(f"✓ Saved result: {args.output}")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
