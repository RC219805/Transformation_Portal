#!/usr/bin/env python3
"""Quick validation script for Materials V2/V3 configuration.

Verifies that the production_ultra_materials preset properly enables
both Materials V2 and V3 engines with correct configuration.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset


def validate_materials_config():
    """Validate Materials V2/V3 configuration."""
    print("=" * 70)
    print("Materials V2/V3 Configuration Validation")
    print("=" * 70)

    # Create config with production_ultra_materials preset
    print("\n1. Creating config with PRODUCTION_ULTRA_MATERIALS preset...")
    cfg = PipelineConfig(preset=Preset.PRODUCTION_ULTRA_MATERIALS)
    print("   ✓ Config created successfully")

    # Check Materials V2
    print("\n2. Validating Materials V2...")
    assert cfg.materials_v2 is not None, "FAIL: Materials V2 config is None"
    print(f"   ✓ Config block exists: {cfg.materials_v2 is not None}")

    assert cfg.materials_v2.enabled is True, "FAIL: Materials V2 not enabled"
    print(f"   ✓ Enabled: {cfg.materials_v2.enabled}")

    print(f"   ✓ Backend: {cfg.materials_v2.backend}")
    print(f"   ✓ Confidence threshold: {cfg.materials_v2.confidence.confidence_threshold}")

    # Check material thresholds
    thresholds = cfg.materials_v2.confidence.material_thresholds
    required_materials = ["wood", "metal", "glass", "fabric", "stone"]
    for material in required_materials:
        assert material in thresholds, f"FAIL: Missing threshold for {material}"
    print(f"   ✓ Material thresholds: {len(thresholds)} materials configured")

    # Check Materials V3
    print("\n3. Validating Materials V3...")
    assert cfg.materials_v3 is not None, "FAIL: Materials V3 config is None"
    print(f"   ✓ Config block exists: {cfg.materials_v3 is not None}")

    assert cfg.materials_v3.enabled is True, "FAIL: Materials V3 not enabled"
    print(f"   ✓ Enabled: {cfg.materials_v3.enabled}")

    print(f"   ✓ Backend: {cfg.materials_v3.backend}")
    print(f"   ✓ Taxonomy: {cfg.materials_v3.taxonomy}")
    print(f"   ✓ Max megapixels: {cfg.materials_v3.max_megapixels}")

    # Check segmentation config
    print("\n4. Validating Segmentation Configuration...")
    assert cfg.segmentation.backend in ["segformer", "auto"], f"FAIL: Invalid segmentation backend: {cfg.segmentation.backend}"
    print(f"   ✓ Backend: {cfg.segmentation.backend}")
    print(f"   ✓ Input size: {cfg.segmentation.input_long_side}px")
    print(f"   ✓ Allow downloads: {cfg.segmentation.allow_downloads}")

    # Check model path fields exist
    assert hasattr(cfg.segmentation, "segformer_model_path"), "FAIL: Missing segformer_model_path field"
    assert hasattr(cfg.segmentation, "sam_model_path"), "FAIL: Missing sam_model_path field"
    assert hasattr(cfg.segmentation, "efficientsam_model_path"), "FAIL: Missing efficientsam_model_path field"
    print("   ✓ Model path fields present")

    # Check MPS safety
    print("\n5. Validating MPS Safety Configuration...")
    assert cfg.phase2 is not None, "FAIL: Phase2 config missing"
    assert cfg.phase2.tile_based_upscaling is True, "FAIL: Tiled upscaling not enabled"
    assert cfg.phase2.upscale_tile_size <= 2048, f"FAIL: Tile size too large for MPS: {cfg.phase2.upscale_tile_size}"
    print(f"   ✓ Tiled upscaling: {cfg.phase2.tile_based_upscaling}")
    print(f"   ✓ Tile size: {cfg.phase2.upscale_tile_size}px")
    print(f"   ✓ Overlap: {cfg.phase2.upscale_overlap}px")

    # Check validation runs without errors
    print("\n6. Testing Configuration Validation...")
    try:
        cfg._validate_materials_config()
        print("   ✓ Validation passed (no errors)")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
        return False

    # Test config fingerprinting
    print("\n7. Testing Configuration Fingerprinting...")
    fp1 = cfg._cfg_fingerprint()
    cfg.materials_v2.confidence.confidence_threshold = 0.99
    fp2 = cfg._cfg_fingerprint()
    assert fp1 != fp2, "FAIL: Fingerprint doesn't change with config"
    print(f"   ✓ Fingerprint includes Materials V2: {fp1 != fp2}")

    # Generate sample report metadata
    print("\n8. Sample Report Metadata...")
    report_materials_v2 = {
        "enabled": bool(cfg.materials_v2 and cfg.materials_v2.enabled),
        "backend": cfg.materials_v2.backend if cfg.materials_v2 else None,
        "confidence_threshold": cfg.materials_v2.confidence.confidence_threshold if cfg.materials_v2 else None,
        "material_thresholds": cfg.materials_v2.confidence.material_thresholds if cfg.materials_v2 else None,
    }

    report_materials_v3 = {
        "enabled": bool(cfg.materials_v3 and cfg.materials_v3.enabled),
        "taxonomy": str(cfg.materials_v3.taxonomy) if cfg.materials_v3 else None,
        "backend": cfg.materials_v3.backend if cfg.materials_v3 else None,
    }

    print(f"   Materials V2: {json.dumps(report_materials_v2, indent=2)}")
    print(f"   Materials V3: {json.dumps(report_materials_v3, indent=2)}")

    # Success!
    print("\n" + "=" * 70)
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 70)
    print("\nMaterials V2/V3 are properly configured and ready for use.")
    print("Use preset: production_ultra_materials")
    print("\nExample usage:")
    print("  lux-depth-v2 \\")
    print("    --input sample.tif \\")
    print("    --output-dir ./output \\")
    print("    --preset production_ultra_materials")
    print("")

    return True


if __name__ == "__main__":
    success = validate_materials_config()
    sys.exit(0 if success else 1)
