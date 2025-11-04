#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the VFX extension for Transformation Portal.

This demonstrates how to use the realize_v8_unified_cli_extension to apply
depth-guided visual effects to architectural renderings.
"""

from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from realize_v8_unified import _info
from realize_v8_unified_cli_extension import (
    VFX_PRESETS,
    enhance_with_vfx,
    _open_any,
    _save_with_meta,
)


def example_single_image():
    """Example: Process a single image with VFX."""
    _info("=== Example: Single Image VFX Processing ===")
    
    # This is a demonstration - would need an actual image file
    img_path = Path("renders/interior.jpg")
    output_path = Path("output/interior_enhanced.jpg")
    
    if not img_path.exists():
        _info(f"Note: Example file {img_path} doesn't exist")
        _info("Usage: provide your own image path")
        return
    
    # Open image
    img, meta = _open_any(img_path)
    
    # Process with VFX
    result = enhance_with_vfx(
        img,
        base_preset="signature_estate_agx",
        vfx_preset="montecito_golden",
        material_response=True,
        save_depth=True
    )
    
    # Save result
    _save_with_meta(
        result["image"],
        result["array"],
        output_path,
        meta,
        out_bitdepth=16
    )
    
    # Save depth map
    if result["depth"] is not None:
        from PIL import Image
        import numpy as np
        depth_path = output_path.with_name(f"{output_path.stem}_depth.png")
        depth_img = (result["depth"] * 65535).astype(np.uint16)
        Image.fromarray(depth_img, mode='I;16').save(depth_path)
        _info(f"Saved depth map: {depth_path}")
    
    # Print metrics
    _info(f"✓ Completed in {result['metrics']['total_ms']}ms")
    _info(f"  Base enhance: {result['metrics']['total_time_ms']}ms")
    _info(f"  Depth estimation: {result['metrics']['depth_estimation_ms']}ms")
    _info(f"  VFX: {result['metrics']['vfx_ms']}ms")


def example_list_presets():
    """Example: List all available presets."""
    _info("=== Available VFX Presets ===")
    
    for name, config in VFX_PRESETS.items():
        _info(f"\n{name}:")
        _info(f"  Description: {config['description']}")
        _info(f"  Bloom: {config.get('bloom_intensity', 0)}")
        _info(f"  Fog: {config.get('fog_density', 0)}")
        if 'lut_default' in config:
            _info(f"  Default LUT: {config['lut_default']}")


def example_cli_usage():
    """Example: Show CLI usage."""
    _info("=== CLI Usage Examples ===")
    
    _info("\n1. Single image with VFX:")
    _info("   python realize_v8_unified_cli_extension.py enhance-vfx \\")
    _info("       --input interior.jpg \\")
    _info("       --output enhanced.jpg \\")
    _info("       --base-preset signature_estate_agx \\")
    _info("       --vfx-preset cinematic_fog \\")
    _info("       --material-response")
    
    _info("\n2. Batch processing:")
    _info("   python realize_v8_unified_cli_extension.py batch-vfx \\")
    _info("       --input renders/ \\")
    _info("       --output finals/ \\")
    _info("       --base-preset signature_estate \\")
    _info("       --vfx-preset dramatic_dof \\")
    _info("       --jobs 4")
    
    _info("\n3. With custom LUT:")
    _info("   python realize_v8_unified_cli_extension.py enhance-vfx \\")
    _info("       --input interior.jpg \\")
    _info("       --output enhanced.jpg \\")
    _info("       --vfx-preset subtle_estate \\")
    _info("       --lut 02_Location_Aesthetic/California/Montecito_Golden_Hour_HDR.cube")


def main():
    """Run examples."""
    _info("VFX Extension Examples for Transformation Portal\n")
    
    example_list_presets()
    _info("\n" + "="*60 + "\n")
    
    example_cli_usage()
    _info("\n" + "="*60 + "\n")
    
    # Uncomment to run single image example (requires actual image file)
    # example_single_image()


if __name__ == "__main__":
    main()
