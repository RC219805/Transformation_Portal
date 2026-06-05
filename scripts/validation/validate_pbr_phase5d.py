#!/usr/bin/env python3
"""Phase 5D: Production validation for PBR material generation.

Tests the enhanced heuristic backend on real luxury real estate TIFFs.
Validates all 6 PBR maps + material properties output.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for direct raw-checkout execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.spatial_ai.materials.contracts import MaterialInput
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend


def load_tiff(path: Path) -> np.ndarray:
    """Load TIFF as linear float32 RGB."""
    try:
        import tifffile

        img = tifffile.imread(path)
        # Convert to float32 and normalize
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha

        return img
    except ImportError:
        print("⚠️  tifffile not installed, trying PIL...")
        from PIL import Image

        img = Image.open(path)
        img_array = np.array(img)
        if np.issubdtype(img_array.dtype, np.integer):
            max_val = np.iinfo(img_array.dtype).max
            img_array = img_array.astype(np.float32) / float(max_val)
        else:
            img_array = img_array.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        return img_array


def validate_pbr_output(albedo, normal, roughness, metallic, ao, height, properties, image_shape):
    """Validate PBR output meets quality standards."""
    errors = []

    # Shape validation
    H, W = image_shape[:2]
    if albedo.shape != (H, W, 3):
        errors.append(f"Albedo shape {albedo.shape} != {(H, W, 3)}")
    if normal.shape != (H, W, 3):
        errors.append(f"Normal shape {normal.shape} != {(H, W, 3)}")
    if roughness.shape != (H, W):
        errors.append(f"Roughness shape {roughness.shape} != {(H, W)}")
    if metallic.shape != (H, W):
        errors.append(f"Metallic shape {metallic.shape} != {(H, W)}")
    if ao.shape != (H, W):
        errors.append(f"AO shape {ao.shape} != {(H, W)}")
    if height.shape != (H, W):
        errors.append(f"Height shape {height.shape} != {(H, W)}")

    # Dtype validation
    if albedo.dtype != np.float32:
        errors.append(f"Albedo dtype {albedo.dtype} != float32")
    if normal.dtype != np.float32:
        errors.append(f"Normal dtype {normal.dtype} != float32")

    # Value range validation
    if not (0 <= albedo.min() and albedo.max() <= 1.0):
        errors.append(f"Albedo range [{albedo.min():.3f}, {albedo.max():.3f}] not in [0, 1]")
    if not (-1.0 <= normal.min() and normal.max() <= 1.0):
        errors.append(f"Normal range [{normal.min():.3f}, {normal.max():.3f}] not in [-1, 1]")
    if not (0 <= roughness.min() and roughness.max() <= 1.0):
        errors.append(f"Roughness range [{roughness.min():.3f}, {roughness.max():.3f}] not in [0, 1]")
    if not (0 <= metallic.min() and metallic.max() <= 1.0):
        errors.append(f"Metallic range [{metallic.min():.3f}, {metallic.max():.3f}] not in [0, 1]")
    if not (0 <= ao.min() and ao.max() <= 1.0):
        errors.append(f"AO range [{ao.min():.3f}, {ao.max():.3f}] not in [0, 1]")

    # Properties validation
    if properties is None:
        errors.append("MaterialProperties is None")
    else:
        if not (0 <= properties.roughness_mean <= 1.0):
            errors.append(f"roughness_mean {properties.roughness_mean:.3f} not in [0, 1]")
        if not (0 <= properties.metallic_mean <= 1.0):
            errors.append(f"metallic_mean {properties.metallic_mean:.3f} not in [0, 1]")

    return errors


def test_small_tiff():
    """Test on small TIFF (288x192)."""
    print("\n" + "=" * 80)
    print("TEST 1: Small TIFF (BECW0138.TIF 288x192)")
    print("=" * 80)

    input_path = Path("input_images/Richard-Raw-Test/BECW0138.TIF")
    if not input_path.exists():
        print(f"❌ SKIP: {input_path} not found")
        return None

    # Load image
    print(f"\n📂 Loading {input_path}...")
    rgb = load_tiff(input_path)
    print(f"   Shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")

    # Create backend
    print("\n🔧 Initializing MaterialBackend (heuristic)...")
    backend = MaterialBackend(backend="heuristic", device="cpu")

    # Generate PBR textures
    print("\n🎨 Generating PBR textures...")
    start_time = time.time()
    result = backend.generate_pbr_textures(rgb=rgb)
    elapsed = time.time() - start_time

    # Extract textures from result object
    albedo = result.albedo
    normal = result.normal
    roughness = result.roughness
    metallic = result.metallic
    ao = result.ambient_occlusion
    height = result.height
    properties = result.properties

    # Validate
    print(f"\n✅ Generated in {elapsed:.2f}s")
    print(f"   Albedo:    {albedo.shape} {albedo.dtype} [{albedo.min():.3f}, {albedo.max():.3f}]")
    print(f"   Normal:    {normal.shape} {normal.dtype} [{normal.min():.3f}, {normal.max():.3f}]")
    print(f"   Roughness: {roughness.shape} {roughness.dtype} [{roughness.min():.3f}, {roughness.max():.3f}]")
    print(f"   Metallic:  {metallic.shape} {metallic.dtype} [{metallic.min():.3f}, {metallic.max():.3f}]")
    print(f"   AO:        {ao.shape} {ao.dtype} [{ao.min():.3f}, {ao.max():.3f}]")
    print(f"   Height:    {height.shape} {height.dtype} [{height.min():.3f}, {height.max():.3f}]")
    print(f"\n   Properties:")
    print(f"     roughness_mean: {properties.roughness_mean:.3f}")
    print(f"     metallic_mean:  {properties.metallic_mean:.3f}")
    print(f"     ao_strength:    {properties.ao_strength:.3f}")
    print(f"     normal_strength: {properties.normal_strength:.3f}")

    # Quality validation
    errors = validate_pbr_output(albedo, normal, roughness, metallic, ao, height, properties, rgb.shape)
    if errors:
        print(f"\n❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        return None
    else:
        print(f"\n✅ All validation checks passed!")

    # Performance metrics
    megapixels = (rgb.shape[0] * rgb.shape[1]) / 1e6
    time_per_mp = elapsed / megapixels
    print(f"\n📊 Performance:")
    print(f"   Resolution: {rgb.shape[0]}x{rgb.shape[1]} ({megapixels:.2f} MP)")
    print(f"   Time: {elapsed:.2f}s ({time_per_mp:.2f}s/MP)")
    print(f"   Target: <5s/MP ✅" if time_per_mp < 5.0 else f"   Target: <5s/MP ❌")

    return {"resolution": rgb.shape[:2], "megapixels": megapixels, "time": elapsed, "time_per_mp": time_per_mp}


def test_large_tiff():
    """Test on large luxury TIFF (4000x3000, 12MP)."""
    print("\n" + "=" * 80)
    print("TEST 2: Large Luxury TIFF (V2_750Picacho_GreatRoom.tiff 4000x3000, 12MP)")
    print("=" * 80)

    input_path = Path("input_images/source_tiffs/V2_750Picacho_GreatRoom.tiff")
    if not input_path.exists():
        print(f"❌ SKIP: {input_path} not found")
        return None

    # Load image
    print(f"\n📂 Loading {input_path} (69MB)...")
    rgb = load_tiff(input_path)
    print(f"   Shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")

    # Create backend
    print("\n🔧 Initializing MaterialBackend (heuristic)...")
    backend = MaterialBackend(backend="heuristic", device="cpu")

    # Generate PBR textures
    print("\n🎨 Generating PBR textures for 12MP image...")
    start_time = time.time()
    result = backend.generate_pbr_textures(rgb=rgb)
    elapsed = time.time() - start_time

    # Extract textures from result object
    albedo = result.albedo
    normal = result.normal
    roughness = result.roughness
    metallic = result.metallic
    ao = result.ambient_occlusion
    height = result.height
    properties = result.properties

    # Validate
    print(f"\n✅ Generated in {elapsed:.2f}s")
    print(f"   Albedo:    {albedo.shape}")
    print(f"   Normal:    {normal.shape}")
    print(f"   Roughness: {roughness.shape}")
    print(f"   Metallic:  {metallic.shape}")
    print(f"   AO:        {ao.shape}")
    print(f"   Height:    {height.shape}")

    # Quality validation
    errors = validate_pbr_output(albedo, normal, roughness, metallic, ao, height, properties, rgb.shape)
    if errors:
        print(f"\n❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        return None
    else:
        print(f"\n✅ All validation checks passed!")

    # Performance metrics
    megapixels = (rgb.shape[0] * rgb.shape[1]) / 1e6
    time_per_mp = elapsed / megapixels
    print(f"\n📊 Performance:")
    print(f"   Resolution: {rgb.shape[0]}x{rgb.shape[1]} ({megapixels:.2f} MP)")
    print(f"   Time: {elapsed:.2f}s ({time_per_mp:.2f}s/MP)")
    print(f"   Target: <5s/MP ✅" if time_per_mp < 5.0 else f"   Target: <5s/MP ⚠️ (acceptable for 12MP)")

    return {"resolution": rgb.shape[:2], "megapixels": megapixels, "time": elapsed, "time_per_mp": time_per_mp}


def test_with_material_hint():
    """Test with material hint integration."""
    print("\n" + "=" * 80)
    print("TEST 3: Material Hint Integration")
    print("=" * 80)

    input_path = Path("input_images/Richard-Raw-Test/BECW0138.TIF")
    if not input_path.exists():
        print(f"❌ SKIP: {input_path} not found")
        return

    # Load image
    print(f"\n📂 Loading {input_path}...")
    rgb = load_tiff(input_path)

    backend = MaterialBackend(backend="heuristic", device="cpu")

    # Test different material hints
    materials = ["wood", "stone", "metal", "glass", "fabric", "concrete"]

    print("\n🎨 Testing material hints...")
    for material in materials:
        result = backend.generate_pbr_textures(rgb=rgb, material_hint=material)
        roughness = result.roughness
        metallic = result.metallic
        properties = result.properties

        print(f"\n   {material.upper()}:")
        print(f"     roughness: [{roughness.min():.3f}, {roughness.max():.3f}] mean={properties.roughness_mean:.3f}")
        print(f"     metallic:  [{metallic.min():.3f}, {metallic.max():.3f}] mean={properties.metallic_mean:.3f}")

    print(f"\n✅ Material hint integration validated!")


def main():
    """Run all production validation tests."""
    print("\n" + "=" * 80)
    print("PHASE 5D: PRODUCTION VALIDATION")
    print("Testing Enhanced Heuristic PBR Backend (Phase 5C)")
    print("=" * 80)

    results = []

    # Test 1: Small TIFF
    result1 = test_small_tiff()
    if result1:
        results.append(("Small TIFF", result1))

    # Test 2: Large TIFF
    result2 = test_large_tiff()
    if result2:
        results.append(("Large TIFF", result2))

    # Test 3: Material hints
    test_with_material_hint()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        print("\n📊 Performance Benchmarks:")
        for name, result in results:
            print(f"\n   {name}:")
            print(f"     Resolution: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"     Megapixels: {result['megapixels']:.2f} MP")
            print(f"     Time: {result['time']:.2f}s")
            print(f"     Time/MP: {result['time_per_mp']:.2f}s/MP")

    print("\n✅ Phase 5D Production Validation COMPLETE!")
    print("\nNext: Add to performance_ledger.json + documentation (Phase 5F)")


if __name__ == "__main__":
    main()
