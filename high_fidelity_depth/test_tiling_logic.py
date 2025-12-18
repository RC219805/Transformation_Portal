#!/usr/bin/env python3
"""
Unit tests for tiling logic (no ML model required).

Tests:
1. Padding eliminates sliver tiles
2. Content-preserving reflect mode
3. Crop restores original dimensions
4. Weighted blend weights sum to 1.0
"""

import sys
from pathlib import Path

import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator, DepthConfig


def test_padding_logic():
    """Test that padding eliminates sliver tiles."""
    print("\n" + "="*80)
    print("TEST 1: Padding Logic")
    print("="*80)
    
    config = DepthConfig(tile_size=1024, overlap=192)
    estimator = HighFidelityDepthEstimator(config)
    
    test_dims = [
        (4001, 3001),
        (5999, 3599),
        (3000, 4000),
        (6000, 3600),
        (2048, 2048),
    ]
    
    all_passed = True
    
    for h, w in test_dims:
        # Create test image
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # Apply padding
        padded, crop_coords = estimator._pad_to_tile_geometry(image)
        h_pad, w_pad = padded.shape[:2]
        
        # Check that padded dimensions are valid
        stride = config.tile_size - config.overlap
        
        # Verify no sliver tiles would be created
        tiles_h = (h_pad - config.tile_size) // stride + 1
        tiles_w = (w_pad - config.tile_size) // stride + 1
        
        # Last tile should fit perfectly
        last_tile_y = (tiles_h - 1) * stride
        last_tile_x = (tiles_w - 1) * stride
        
        expected_h = last_tile_y + config.tile_size
        expected_w = last_tile_x + config.tile_size
        
        # Check dimensions
        if h_pad >= expected_h and w_pad >= expected_w:
            print(f"  ✓ {h}×{w} → {h_pad}×{w_pad}: OK (tiles: {tiles_h}×{tiles_w})")
        else:
            print(f"  ✗ {h}×{w} → {h_pad}×{w_pad}: FAIL")
            all_passed = False
        
        # Test crop restoration
        test_depth = np.random.rand(h_pad, w_pad).astype(np.float32)
        cropped = estimator._crop_to_original(test_depth, crop_coords)
        
        if cropped.shape == (h, w):
            print(f"    ✓ Crop restored original size: {cropped.shape}")
        else:
            print(f"    ✗ Crop failed: expected {(h, w)}, got {cropped.shape}")
            all_passed = False
    
    return all_passed


def test_reflect_padding():
    """Test that reflect padding is content-preserving."""
    print("\n" + "="*80)
    print("TEST 2: Reflect Padding (Content Preservation)")
    print("="*80)
    
    config = DepthConfig(tile_size=1024, overlap=192)
    estimator = HighFidelityDepthEstimator(config)
    
    # Create image with gradient
    h, w = 3000, 2000
    image = np.linspace(0, 255, w, dtype=np.uint8)
    image = np.tile(image[None, :], (h, 1))
    image = np.stack([image, image, image], axis=-1)
    
    # Apply padding
    padded, crop_coords = estimator._pad_to_tile_geometry(image)
    h_pad, w_pad = padded.shape[:2]
    
    # Check that padded region contains reflected content
    if w_pad > w:
        # Check right edge reflection - should have structure from reflection
        padded_right = padded[:h, w:, 0]  # Padding region on right
        
        # Reflected content should have structure (not black or constant)
        if np.std(padded_right) > 10:  # Should have structure
            print(f"  ✓ Right padding has structure (std={np.std(padded_right):.1f})")
        else:
            print(f"  ✗ Right padding lacks structure (std={np.std(padded_right):.1f})")
            return False
    
    if h_pad > h:
        # Check bottom edge reflection
        padded_bottom = padded[h:, :w, 0]  # Padding region on bottom
        
        if np.std(padded_bottom) > 10:
            print(f"  ✓ Bottom padding has structure (std={np.std(padded_bottom):.1f})")
        else:
            print(f"  ✗ Bottom padding lacks structure (std={np.std(padded_bottom):.1f})")
            return False
    
    print(f"  ✓ Reflect padding preserves content")
    return True


def test_blend_weights():
    """Test that blend weights normalize correctly in overlap regions."""
    print("\n" + "="*80)
    print("TEST 3: Weighted Overlap Blending")
    print("="*80)
    
    config = DepthConfig(tile_size=1024, overlap=192)
    estimator = HighFidelityDepthEstimator(config)
    
    # Create blend weight
    weight = estimator._create_blend_weight(config.tile_size, config.overlap)
    
    # Check weight properties
    assert weight.shape == (config.tile_size, config.tile_size), "Weight shape mismatch"
    assert weight.min() >= 0.0 and weight.max() <= 1.0, "Weight out of range"
    
    # Check taper at edges
    top_edge = weight[0, config.tile_size // 2]
    center = weight[config.tile_size // 2, config.tile_size // 2]
    
    print(f"  Edge weight: {top_edge:.3f}")
    print(f"  Center weight: {center:.3f}")
    
    if top_edge < center:
        print(f"  ✓ Edge taper verified (edge={top_edge:.3f} < center={center:.3f})")
    else:
        print(f"  ✗ Edge taper failed")
        return False
    
    # Simulate overlap regions in a 2x2 grid
    stride = config.tile_size - config.overlap
    
    # Create 2x2 grid
    h_grid = config.tile_size * 2 - config.overlap
    w_grid = config.tile_size * 2 - config.overlap
    weight_accum = np.zeros((h_grid, w_grid), dtype=np.float32)
    
    # Place 4 tiles in 2x2 configuration
    tiles = [(0, 0), (0, stride), (stride, 0), (stride, stride)]
    for y, x in tiles:
        y1 = y + config.tile_size
        x1 = x + config.tile_size
        weight_accum[y:y1, x:x1] += weight
    
    # Check non-overlapping center of first tile (should be ~1.0)
    center_region = weight_accum[config.tile_size//2:config.tile_size//2+10,
                                 config.tile_size//2:config.tile_size//2+10]
    center_mean = center_region.mean()
    
    # Check horizontal overlap region (should be ~1.0 after normalization)
    h_overlap = weight_accum[config.tile_size//2:config.tile_size//2+10,
                            stride:stride+10]
    h_overlap_mean = h_overlap.mean()
    
    # Check 4-way junction (should be ~1.0 after normalization)
    junction = weight_accum[stride:stride+10, stride:stride+10]
    junction_mean = junction.mean()
    
    print(f"  Non-overlap region: mean={center_mean:.3f}")
    print(f"  Horizontal overlap: mean={h_overlap_mean:.3f}")
    print(f"  4-way junction: mean={junction_mean:.3f}")
    
    # All should be approximately 1.0 (within tolerance)
    if (0.95 <= center_mean <= 1.05 and 
        0.95 <= h_overlap_mean <= 1.05 and
        0.95 <= junction_mean <= 1.05):
        print(f"  ✓ Blend weights normalize correctly everywhere")
        return True
    else:
        print(f"  ✗ Blend weight normalization failed")
        return False


def test_tile_extraction():
    """Test tile extraction with minimum size enforcement."""
    print("\n" + "="*80)
    print("TEST 4: Tile Extraction (No Slivers)")
    print("="*80)
    
    config = DepthConfig(tile_size=1024, overlap=192)
    estimator = HighFidelityDepthEstimator(config)
    
    test_dims = [
        (4001, 3001),
        (5999, 3599),
        (3000, 4000),
    ]
    
    all_passed = True
    min_size = 256
    
    for h, w in test_dims:
        # Create and pad image
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        padded, _ = estimator._pad_to_tile_geometry(image)
        
        # Extract tiles
        tiles = estimator._extract_tiles(padded)
        
        # Check for slivers
        sliver_count = 0
        for tile, y0, y1, x0, x1 in tiles:
            th, tw = tile.shape[:2]
            if th < min_size or tw < min_size:
                sliver_count += 1
                print(f"  ✗ Sliver tile: {th}×{tw} at ({y0},{x0})")
                all_passed = False
        
        if sliver_count == 0:
            print(f"  ✓ {h}×{w}: {len(tiles)} tiles, no slivers")
        else:
            print(f"  ✗ {h}×{w}: {sliver_count} sliver tiles detected")
    
    return all_passed


def main():
    """Run all unit tests."""
    print("="*80)
    print("SLIVER TILE ELIMINATION - UNIT TESTS")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Padding Logic", test_padding_logic()))
    results.append(("Reflect Padding", test_reflect_padding()))
    results.append(("Blend Weights", test_blend_weights()))
    results.append(("Tile Extraction", test_tile_extraction()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print("\n" + "="*80)
    print(f"Overall: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("✓✓✓ ALL UNIT TESTS PASSED ✓✓✓")
        return 0
    else:
        print(f"✗✗✗ {total - passed_count} TESTS FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
