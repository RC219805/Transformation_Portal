#!/usr/bin/env python3
"""Quick smoke test for Materials V3 pipeline integration."""

from pathlib import Path
from lux_depth_v2.config import PipelineConfig
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.materials_v3 import MaterialsV3Config, MaterialTaxonomy, RefinementStrategy

def test_materials_v3_integration():
    """Test that Materials V3 integrates correctly with pipeline."""
    
    print("=" * 80)
    print("Materials V3 Pipeline Integration Smoke Test")
    print("=" * 80)
    
    # Test 1: Disabled by default
    print("\n[TEST 1] Materials V3 disabled by default...")
    cfg = PipelineConfig(output_dir=Path("/tmp/test"))
    pipe = LuxPipelineV2(cfg)
    assert pipe.materials_v3_engine is None, "Materials V3 should be None when disabled"
    print("✓ PASS: Materials V3 is None when disabled")
    
    # Test 2: Can be enabled
    print("\n[TEST 2] Materials V3 can be enabled...")
    cfg2 = PipelineConfig(output_dir=Path("/tmp/test"))
    cfg2.materials_v3 = MaterialsV3Config()
    cfg2.materials_v3.enabled = True
    cfg2.materials_v3.taxonomy = MaterialTaxonomy.BASE
    cfg2.materials_v3.refine_edges = RefinementStrategy.OFF
    
    pipe2 = LuxPipelineV2(cfg2)
    assert pipe2.materials_v3_engine is not None, "Materials V3 should exist when enabled"
    print("✓ PASS: Materials V3 engine created successfully")
    print(f"  - Taxonomy: {cfg2.materials_v3.taxonomy}")
    print(f"  - Refinement: {cfg2.materials_v3.refine_edges}")
    print(f"  - Max MP: {cfg2.materials_v3.max_megapixels}")
    
    # Test 3: Canary preset with pixel ops
    print("\n[TEST 3] Materials V3 canary preset with pixel ops...")
    cfg3 = PipelineConfig(output_dir=Path("/tmp/test"))
    cfg3.materials_v3 = MaterialsV3Config()
    cfg3.materials_v3.enabled = True
    cfg3.materials_v3.refine_edges = RefinementStrategy.CANARY
    cfg3.materials_v3.apply_pixel_ops = True
    cfg3.materials_v3.glass_response_enabled = True
    
    pipe3 = LuxPipelineV2(cfg3)
    assert pipe3.materials_v3_engine is not None
    assert pipe3.materials_v3_engine.config.apply_pixel_ops == True
    assert pipe3.materials_v3_engine.config.glass_response_enabled == True
    print("✓ PASS: Canary preset configured correctly")
    print(f"  - Pixel ops: {cfg3.materials_v3.apply_pixel_ops}")
    print(f"  - Glass response: {cfg3.materials_v3.glass_response_enabled}")
    
    print("\n" + "=" * 80)
    print("✅ All smoke tests passed!")
    print("=" * 80)
    print("\nMaterials V3 is successfully integrated into LuxPipelineV2.")
    print("Next steps:")
    print("  1. Auto-Preset v2 completion")
    print("  2. Stage 6 A/B validation with glass pixel ops")
    print("  3. Decision on promoting to default APEX")

if __name__ == "__main__":
    test_materials_v3_integration()
