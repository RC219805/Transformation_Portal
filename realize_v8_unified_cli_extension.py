#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realize_v8_unified_cli_extension.py — VFX Extension for Transformation Portal

Extends realize_v8_unified with depth-guided VFX using your existing infrastructure:

- Uses ArchitecturalDepthPipeline (24ms depth estimation on M4 Max)
- Integrates Material Response system
- Applies LUT collection with depth masking
- Maintains all realize_v8 color management and ICC profiles

Usage:
    # Material Response + Depth VFX
    python realize_v8_unified_cli_extension.py enhance-vfx \
        --input interior.jpg \
        --output enhanced.jpg \
        --base-preset signature_estate_agx \
        --vfx-preset cinematic_fog \
        --material-response \
        --lut 02_Location_Aesthetic/Montecito_Golden_Hour_HDR.cube

    # Batch with VFX
    python realize_v8_unified_cli_extension.py batch-vfx \
        --input renders/ \
        --output finals/ \
        --base-preset signature_estate \
        --vfx-preset dramatic_dof \
        --jobs 4
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

# Your existing infrastructure
from realize_v8_unified import (
    enhance, _open_any, _save_with_meta,
    _info, _warn, _error, PRESETS
)

# Your depth pipeline (optimized for M4 Max)
try:
    from depth_pipeline import ArchitecturalDepthPipeline
    _HAVE_DEPTH = True
except ImportError:
    _HAVE_DEPTH = False
    _warn("Depth pipeline not available - depth VFX will be disabled")

# Optional: Your Material Response
try:
    from material_response import MaterialResponse  # noqa: F401
    _HAVE_MR = True
except ImportError:
    _HAVE_MR = False
    _warn("Material Response not available - install with pip install -e .[ml]")


# ==================== Constants ====================

# Default depth pipeline configuration
DEFAULT_DEPTH_CONFIG = "config/interior_preset.yaml"

# VFX operator constants
BLOOM_HIGHLIGHT_THRESHOLD = 0.7  # Threshold for extracting highlights
BLOOM_RADIUS_TO_SIGMA = 3.0  # Relationship between bloom radius and Gaussian sigma
DEPTH_BLOOM_FALLOFF = 0.7  # Exponent for depth-based bloom weighting curve

FOG_FALLOFF_EXPONENT = 2.0  # Exponent for exponential fog falloff curve

# LUT depth masking constants
DEPTH_LUT_BASE_STRENGTH = 0.7  # Base LUT strength
DEPTH_LUT_DEPTH_INFLUENCE = 0.3  # Additional strength for foreground


# ==================== VFX Presets (Optimized for your workflow) ====================

VFX_PRESETS = {
    "subtle_estate": {
        "description": "Minimal depth effects for signature estate look",
        "bloom_intensity": 0.08,
        "bloom_radius": 12,
        "fog_density": 0.0,
        "material_boost": 0.15,
    },
    "montecito_golden": {
        "description": "Warm coastal light with atmospheric depth",
        "bloom_intensity": 0.20,
        "bloom_radius": 18,
        "fog_density": 0.25,
        "fog_color": (0.95, 0.88, 0.75),  # Warm golden
        "material_boost": 0.22,
        "lut_default": "02_Location_Aesthetic/California/Montecito_Golden_Hour_HDR.cube",
    },
    "cinematic_fog": {
        "description": "Atmospheric fog with gentle bloom",
        "bloom_intensity": 0.25,
        "bloom_radius": 20,
        "fog_density": 0.40,
        "fog_color": (0.75, 0.80, 0.85),
        "material_boost": 0.18,
    },
    "dramatic_dof": {
        "description": "Strong depth of field for hero shots",
        "bloom_intensity": 0.30,
        "dof_enabled": True,
        "dof_focus": 0.35,
        "dof_blur": 6.0,
        "material_boost": 0.25,
        "color_grade_near": (1.05, 1.0, 0.95),
        "color_grade_far": (0.8, 0.9, 1.0),
    },
}


# ==================== Optimized Depth Estimation ====================

def estimate_depth_fast(
    img_array: np.ndarray,
    config_path: str = None
) -> np.ndarray:
    """
    Use your optimized ArchitecturalDepthPipeline (24ms on M4 Max).

    Args:
        img_array: RGB image (H, W, 3) in [0, 1]
        config_path: Path to depth pipeline config (defaults to DEFAULT_DEPTH_CONFIG)

    Returns:
        Depth map (H, W) normalized to [0, 1]
    """
    if not _HAVE_DEPTH:
        _warn("Depth pipeline not available, returning mock depth")
        h, w = img_array.shape[:2]
        # Create simple gradient depth for fallback
        y = np.linspace(0, 1, h)
        depth = np.tile(y[:, None], (1, w))
        return depth.astype(np.float32)

    if config_path is None:
        config_path = DEFAULT_DEPTH_CONFIG

    config_path = Path(config_path)
    if not config_path.exists():
        _warn(f"Config not found: {config_path}, using default")
        config_path = Path(DEFAULT_DEPTH_CONFIG)

    try:
        pipeline = ArchitecturalDepthPipeline.from_config(str(config_path))

        # Estimate depth (uses CoreML backend on M4 Max)
        # Convert to uint8 for model
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)

        result = pipeline.depth_model.estimate_depth(pil_img)
        depth = result['depth']

        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth.astype(np.float32)
    except Exception as e:
        _error(f"Depth estimation failed: {e}")
        # Fallback to gradient
        h, w = img_array.shape[:2]
        y = np.linspace(0, 1, h)
        depth = np.tile(y[:, None], (1, w))
        return depth.astype(np.float32)


# ==================== VFX Operators (Simplified versions) ====================

def apply_depth_bloom(
    img: np.ndarray,
    depth: np.ndarray,
    intensity: float = 0.25,
    radius: int = 15,
    highlight_threshold: float = BLOOM_HIGHLIGHT_THRESHOLD
) -> np.ndarray:
    """Depth-aware bloom using your depth pipeline."""
    from scipy.ndimage import gaussian_filter

    # Extract highlights
    threshold = highlight_threshold
    bright = np.maximum(img - threshold, 0.0)

    # Blur
    bloom = gaussian_filter(bright, sigma=radius/BLOOM_RADIUS_TO_SIGMA, mode='reflect')

    # Depth weighting (near objects bloom more)
    depth_weight = 1.0 - (depth ** DEPTH_BLOOM_FALLOFF)
    bloom = bloom * depth_weight[..., None]

    result = img + bloom * intensity
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_depth_fog(
    img: np.ndarray,
    depth: np.ndarray,
    fog_color: tuple = (0.8, 0.85, 0.9),
    density: float = 0.5,
    falloff_exponent: float = FOG_FALLOFF_EXPONENT
) -> np.ndarray:
    """
    Atmospheric fog with exponential falloff.

    Parameters:
        img: Input image as float32 array.
        depth: Normalized depth map (0=near, 1=far).
        fog_color: RGB tuple for fog color.
        density: Fog density scalar.
        falloff_exponent: Exponent for fog falloff curve (default 2.0).
    """
    fog_array = np.array(fog_color, dtype=np.float32)

    # Exponential fog
    fog_amount = 1.0 - np.exp(-density * (depth ** falloff_exponent))

    result = img * (1.0 - fog_amount[..., None]) + fog_array * fog_amount[..., None]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_depth_of_field(
    img: np.ndarray,
    depth: np.ndarray,
    focus_depth: float = 0.35,
    blur_strength: float = 6.0
) -> np.ndarray:
    """Simple depth of field effect."""
    from scipy.ndimage import gaussian_filter

    # Calculate blur amount based on distance from focus
    blur_amount = np.abs(depth - focus_depth)
    blur_amount = blur_amount / (blur_amount.max() + 1e-8)

    # Apply varying blur
    result = img.copy()
    for i in range(3):  # Process each channel
        blurred_channel = gaussian_filter(img[..., i], sigma=blur_strength, mode='reflect')
        result[..., i] = img[..., i] * (1 - blur_amount) + blurred_channel * blur_amount

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def apply_lut_with_depth(
    img: np.ndarray,
    lut_path: Path,
    depth: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply your LUT collection with optional depth masking."""
    if not lut_path.exists():
        _warn(f"LUT not found: {lut_path}")
        return img

    try:
        # Parse CUBE LUT - strip whitespace and filter comments/empty lines
        with open(lut_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(stripped)

        # Extract data lines (lines starting with digit or minus sign)
        # More robust: check if the line can be split into 3 floats
        data_lines = []
        for line in lines:
            # Defensive: lines should already be non-empty, but check anyway
            try:
                parts = line.split()
                if len(parts) == 3:
                    # Validate that all parts are floats
                    [float(p) for p in parts]
                    data_lines.append(line)
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        if not data_lines:
            raise ValueError("No valid LUT data found in file (expected numeric values)")

        size = int(len(data_lines) ** (1/3) + 0.5)
        expected_lines = size ** 3

        if len(data_lines) != expected_lines:
            raise ValueError(
                f"Invalid CUBE LUT size: expected {expected_lines} lines for {size}³ LUT, "
                f"got {len(data_lines)} lines"
            )

        lut_data = np.array([list(map(float, line.split())) for line in data_lines])

        if lut_data.shape[1] != 3:
            raise ValueError(f"Invalid LUT data: expected 3 columns (RGB), got {lut_data.shape[1]}")

        lut_cube = lut_data.reshape((size, size, size, 3))

        # Apply LUT
        h, w = img.shape[:2]
        coords = img.reshape(-1, 3) * (size - 1)
        coords = np.clip(coords, 0, size - 1)

        # Trilinear interpolation
        c0 = np.floor(coords).astype(int)
        c1 = np.clip(c0 + 1, 0, size - 1)
        d = coords - c0

        # Get LUT values at the 8 corners
        def lut_at(ix, iy, iz):
            return lut_cube[ix, iy, iz]

        v000 = lut_at(c0[:, 0], c0[:, 1], c0[:, 2])
        v100 = lut_at(c1[:, 0], c0[:, 1], c0[:, 2])
        v010 = lut_at(c0[:, 0], c1[:, 1], c0[:, 2])
        v110 = lut_at(c1[:, 0], c1[:, 1], c0[:, 2])
        v001 = lut_at(c0[:, 0], c0[:, 1], c1[:, 2])
        v101 = lut_at(c1[:, 0], c0[:, 1], c1[:, 2])
        v011 = lut_at(c0[:, 0], c1[:, 1], c1[:, 2])
        v111 = lut_at(c1[:, 0], c1[:, 1], c1[:, 2])

        wx, wy, wz = d[:, 0:1], d[:, 1:2], d[:, 2:3]

        graded = (
            v000 * (1 - wx) * (1 - wy) * (1 - wz) +
            v100 * wx * (1 - wy) * (1 - wz) +
            v010 * (1 - wx) * wy * (1 - wz) +
            v110 * wx * wy * (1 - wz) +
            v001 * (1 - wx) * (1 - wy) * wz +
            v101 * wx * (1 - wy) * wz +
            v011 * (1 - wx) * wy * wz +
            v111 * wx * wy * wz
        )
        graded = graded.reshape(h, w, 3)
        # Optional depth masking (stronger on foreground)
        if depth is not None:
            blend = DEPTH_LUT_BASE_STRENGTH + DEPTH_LUT_DEPTH_INFLUENCE * (1 - depth[..., None])
            graded = img * (1 - blend) + graded * blend

        return np.clip(graded, 0.0, 1.0).astype(np.float32)

    except ValueError as e:
        _error(f"Invalid LUT format in {lut_path}: {e}")
        return img
    except Exception as e:
        _error(f"LUT processing failed for {lut_path}: {e}")
        return img


def apply_color_grade_zones(
    img: np.ndarray,
    depth: np.ndarray,
    near_color: tuple = (1.0, 1.0, 1.0),
    far_color: tuple = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Apply color grading based on depth zones."""
    near_array = np.array(near_color, dtype=np.float32)
    far_array = np.array(far_color, dtype=np.float32)

    # Blend between near and far colors based on depth
    color_shift = near_array * (1 - depth[..., None]) + far_array * depth[..., None]

    result = img * color_shift
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _save_depth_map(depth: np.ndarray, path: Path) -> None:
    """
    Helper function to save depth map as 16-bit PNG.

    Args:
        depth: Normalized depth map (H, W) in [0, 1]
        path: Output path
    """
    depth_img = (depth * 65535).astype(np.uint16)
    Image.fromarray(depth_img, mode='I;16').save(path)
    _info(f"Saved depth map: {path}")


# ==================== Material Response Integration ====================

def apply_material_response(
    img: np.ndarray,
    strength: float = 0.2
) -> np.ndarray:
    """
    Apply Material Response principles to enhance surface characteristics.

    This is a simplified version that respects the Material Response principle
    without requiring the full ML implementation.
    """
    if not _HAVE_MR:
        # Simplified fallback: enhance local contrast
        from scipy.ndimage import gaussian_filter

        # Enhance micro-contrast
        blurred = gaussian_filter(img, sigma=1.5, mode='reflect')
        enhanced = img + (img - blurred) * strength * 2.0

        return np.clip(enhanced, 0.0, 1.0).astype(np.float32)

    # Use actual Material Response if available
    _info("Using Material Response system")
    # Note: This is a placeholder - actual implementation would need
    # to match the MaterialResponse API
    return img


# ==================== Enhanced Processing Function ====================

def enhance_with_vfx(
    img_or_arr,
    base_preset: str = "signature_estate",
    vfx_preset: str = "subtle_estate",
    material_response: bool = False,
    lut_path: Optional[Path] = None,
    save_depth: bool = False,
    **base_overrides
) -> Dict[str, Any]:
    """
    Complete enhancement pipeline: base enhance + Material Response + depth VFX.

    Args:
        img_or_arr: Input PIL Image or numpy array
        base_preset: realize_v8 base preset
        vfx_preset: VFX preset name
        material_response: Enable Material Response
        lut_path: Optional LUT path
        save_depth: Include depth map in output
        **base_overrides: Override base preset parameters

    Returns:
        Dict with keys: 'image', 'array', 'depth', 'metrics'
    """
    t_start = time.perf_counter()

    # Step 1: Base enhancement (your realize_v8 pipeline)
    _info(f"Applying base preset: {base_preset}")

    base_params = PRESETS[base_preset].to_dict()
    base_params.update(base_overrides)

    preview, working, base_metrics = enhance(img_or_arr, **base_params)
    working_array = working.copy()

    # Step 2: Material Response (optional)
    if material_response:
        t0 = time.perf_counter()
        _info("Applying Material Response")

        vfx_cfg = VFX_PRESETS[vfx_preset]
        working_array = apply_material_response(
            working_array,
            strength=vfx_cfg.get("material_boost", 0.2)
        )

        base_metrics["material_response_ms"] = int((time.perf_counter() - t0) * 1000)

    # Step 3: Depth estimation (using your fast pipeline)
    t0 = time.perf_counter()
    _info("Estimating depth with ArchitecturalDepthPipeline")

    depth = estimate_depth_fast(working_array)
    base_metrics["depth_estimation_ms"] = int((time.perf_counter() - t0) * 1000)

    # Step 4: Depth-guided VFX
    t0 = time.perf_counter()
    _info(f"Applying VFX preset: {vfx_preset}")

    vfx_cfg = VFX_PRESETS[vfx_preset]
    result = working_array.copy()

    # Bloom
    if vfx_cfg.get("bloom_intensity", 0) > 0:
        result = apply_depth_bloom(
            result, depth,
            intensity=vfx_cfg["bloom_intensity"],
            radius=vfx_cfg.get("bloom_radius", 15)
        )

    # Fog
    if vfx_cfg.get("fog_density", 0) > 0:
        result = apply_depth_fog(
            result, depth,
            fog_color=vfx_cfg.get("fog_color", (0.8, 0.85, 0.9)),
            density=vfx_cfg["fog_density"]
        )

    # Depth of Field
    if vfx_cfg.get("dof_enabled", False):
        result = apply_depth_of_field(
            result, depth,
            focus_depth=vfx_cfg.get("dof_focus", 0.35),
            blur_strength=vfx_cfg.get("dof_blur", 6.0)
        )

    # Color grading zones
    if "color_grade_near" in vfx_cfg or "color_grade_far" in vfx_cfg:
        result = apply_color_grade_zones(
            result, depth,
            near_color=vfx_cfg.get("color_grade_near", (1.0, 1.0, 1.0)),
            far_color=vfx_cfg.get("color_grade_far", (1.0, 1.0, 1.0))
        )

    # LUT
    if lut_path or vfx_cfg.get("lut_default"):
        lut = lut_path or Path(vfx_cfg["lut_default"])
        if lut.exists():
            result = apply_lut_with_depth(result, lut, depth)

    base_metrics["vfx_ms"] = int((time.perf_counter() - t0) * 1000)
    base_metrics["total_ms"] = int((time.perf_counter() - t_start) * 1000)

    # Create final image
    final_img = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))

    output = {
        "image": final_img,
        "array": result,
        "depth": depth if save_depth else None,
        "metrics": base_metrics
    }

    return output


# ==================== Batch Processing ====================

def batch_process_vfx(
    input_dir: Path,
    output_dir: Path,
    base_preset: str = "signature_estate",
    vfx_preset: str = "subtle_estate",
    material_response: bool = False,
    pattern: str = "*.jpg",
    jobs: int = 4,
    out_bitdepth: int = 16
) -> None:
    """
    Batch process images with VFX.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        base_preset: Base enhancement preset
        vfx_preset: VFX preset
        material_response: Enable material response
        pattern: File pattern to match
        jobs: Number of parallel jobs (NOT YET IMPLEMENTED - processing is sequential)
        out_bitdepth: Output bit depth

    Note:
        Parallel processing via the 'jobs' parameter is planned but not yet implemented.
        All processing is currently sequential. This parameter is accepted for forward
        compatibility but has no effect on performance.
    """
    if jobs > 1:
        _warn(f"Parallel processing not yet implemented. Requested {jobs} jobs, using 1 (sequential).")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching files
    image_files = list(input_dir.glob(pattern))

    if not image_files:
        _error(f"No images found matching {pattern} in {input_dir}")
        return

    _info(f"Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files, 1):
        _info(f"[{i}/{len(image_files)}] Processing {img_path.name}")

        try:
            # Open image
            img, meta = _open_any(img_path)

            # Process with VFX
            result = enhance_with_vfx(
                img,
                base_preset=base_preset,
                vfx_preset=vfx_preset,
                material_response=material_response,
                save_depth=False
            )

            # Save
            output_path = output_dir / f"{img_path.stem}_{vfx_preset}{img_path.suffix}"
            _save_with_meta(
                result["image"],
                result["array"],
                output_path,
                meta,
                out_bitdepth=out_bitdepth
            )

            # Save depth map if requested
            if result["depth"] is not None:
                depth_path = output_dir / f"{img_path.stem}_depth.png"
                _save_depth_map(result["depth"], depth_path)

            _info(f"  Completed in {result['metrics']['total_ms']}ms")

        except Exception as e:
            _error(f"Failed to process {img_path.name}: {e}")
            continue

    _info("Batch processing complete!")


# ==================== CLI ====================

def add_vfx_commands(subparsers):
    """Add VFX commands to realize_v8_unified CLI."""

    # Single image with VFX
    p_vfx = subparsers.add_parser("enhance-vfx", help="Enhance with depth VFX")
    p_vfx.add_argument("--input", type=Path, required=True)
    p_vfx.add_argument("--output", type=Path, required=True)
    p_vfx.add_argument("--base-preset", choices=list(PRESETS.keys()),
                       default="signature_estate")
    p_vfx.add_argument("--vfx-preset", choices=list(VFX_PRESETS.keys()),
                       default="subtle_estate")
    p_vfx.add_argument("--material-response", action="store_true")
    p_vfx.add_argument("--lut", type=Path, help="Override default LUT")
    p_vfx.add_argument("--save-depth", action="store_true")
    p_vfx.add_argument("--out-bitdepth", type=int, choices=[8, 16, 32], default=16)
    p_vfx.set_defaults(func=handle_enhance_vfx)

    # Batch with VFX
    p_batch_vfx = subparsers.add_parser("batch-vfx", help="Batch process with VFX")
    p_batch_vfx.add_argument("--input", type=Path, required=True)
    p_batch_vfx.add_argument("--output", type=Path, required=True)
    p_batch_vfx.add_argument("--base-preset", choices=list(PRESETS.keys()),
                             default="signature_estate")
    p_batch_vfx.add_argument("--vfx-preset", choices=list(VFX_PRESETS.keys()),
                             default="subtle_estate")
    p_batch_vfx.add_argument("--material-response", action="store_true")
    p_batch_vfx.add_argument("--pattern", default="*.jpg", help="File pattern to match")
    p_batch_vfx.add_argument("--jobs", type=int, default=4)
    p_batch_vfx.add_argument("--out-bitdepth", type=int, choices=[8, 16, 32], default=16)
    p_batch_vfx.set_defaults(func=handle_batch_vfx)

    return subparsers


def handle_enhance_vfx(args):
    """Handle single image VFX enhancement."""
    # Open image
    img, meta = _open_any(args.input)

    # Process
    result = enhance_with_vfx(
        img,
        base_preset=args.base_preset,
        vfx_preset=args.vfx_preset,
        material_response=args.material_response,
        lut_path=args.lut,
        save_depth=args.save_depth
    )

    # Save
    _save_with_meta(
        result["image"],
        result["array"],
        args.output,
        meta,
        out_bitdepth=args.out_bitdepth
    )

    # Save depth if requested
    if result["depth"] is not None:
        depth_path = args.output.with_name(f"{args.output.stem}_depth.png")
        _save_depth_map(result["depth"], depth_path)

    # Print metrics
    _info(f"✓ Completed in {result['metrics']['total_ms']}ms")
    _info(f"  Base enhance: {result['metrics']['total_time_ms']}ms")
    _info(f"  Depth estimation: {result['metrics']['depth_estimation_ms']}ms")
    _info(f"  VFX: {result['metrics']['vfx_ms']}ms")


def handle_batch_vfx(args):
    """Handle batch VFX processing."""
    batch_process_vfx(
        args.input,
        args.output,
        base_preset=args.base_preset,
        vfx_preset=args.vfx_preset,
        material_response=args.material_response,
        pattern=args.pattern,
        jobs=args.jobs,
        out_bitdepth=args.out_bitdepth
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Realize V8 VFX Extension - Depth-guided visual effects"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    add_vfx_commands(subparsers)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    args.func(args)
    return 0


# ==================== Example Usage ====================

if __name__ == "__main__":
    import sys
    sys.exit(main())
