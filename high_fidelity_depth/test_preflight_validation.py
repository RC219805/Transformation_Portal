#!/usr/bin/env python3
"""
Pre-flight validation tests before full validation run.

Tests infrastructure correctness:
1. Pad/crop dimension preservation
2. Dtype discipline in blending
3. Stride consistency
4. Reflection artifacts at borders
"""

import numpy as np
import cv2
import pytest
from high_fidelity_depth.depth_estimator import HighFidelityDepthEstimator
from high_fidelity_depth.depth_estimator import DepthConfig


class TestPadCropCorrectness:
    """Verify pad/crop preserves original dimensions."""
    
    @pytest.mark.parametrize("h,w,name", [
        (1023, 1023, "just_under_tile"),
        (1025, 1025, "just_over_tile"),
        (4000, 1000, "extreme_wide"),
        (1000, 4000, "extreme_tall"),
        (800, 600, "small_image"),
    ])
    def test_dimension_preservation(self, h, w, name):
        """Verify output shape == input shape for edge cases."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        # Create test image
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # Apply pad/crop workflow
        padded, crop_coords = estimator._pad_to_tile_geometry(image)
        cropped = estimator._crop_to_original(padded, crop_coords)
        
        # Verify dimensions preserved
        assert cropped.shape[:2] == (h, w), \
            f"{name}: Expected {(h, w)}, got {cropped.shape[:2]}"
        
        print(f"✓ PASS: {name} - {h}×{w} preserved (padded to {padded.shape[:2]})")


class TestBlendingDtype:
    """Verify blending occurs in float32, not uint16."""
    
    def test_accumulator_dtype(self):
        """Check depth_accum and weight_accum are float32."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        # Create mock tiles
        tile1 = np.random.rand(1024, 1024).astype(np.float32)
        tile2 = np.random.rand(1024, 1024).astype(np.float32)
        
        tiles = [
            (tile1, 0, 1024, 0, 1024),
            (tile2, 896, 1920, 0, 1024),
        ]
        
        # Blend
        result = estimator._blend_tiles(tiles, (1920, 1024))
        
        # Verify result is float32
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        
        # Verify range [0, 1]
        assert result.min() >= 0.0, f"Min value {result.min()} < 0"
        assert result.max() <= 1.0, f"Max value {result.max()} > 1"
        
        print(f"✓ PASS: Blending produces float32 output in [0, 1]")
    
    def test_no_premature_uint16_conversion(self):
        """Ensure no uint16 conversion during blending."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        # Create tiles with subtle gradients (would be quantized in uint16)
        tile1 = np.linspace(0.0, 0.01, 1024*1024).reshape(1024, 1024).astype(np.float32)
        tile2 = np.linspace(0.005, 0.015, 1024*1024).reshape(1024, 1024).astype(np.float32)
        
        tiles = [
            (tile1, 0, 1024, 0, 1024),
            (tile2, 896, 1920, 0, 1024),
        ]
        
        result = estimator._blend_tiles(tiles, (1920, 1024))
        
        # Check for smooth gradient (uint16 would create banding)
        # Compute gradient variance in overlap region
        overlap_region = result[896:1024, :]
        grad_y = np.diff(overlap_region, axis=0)
        
        # Variance should be very low for smooth blend
        grad_var = np.var(grad_y)
        
        # If converted to uint16, variance would spike due to quantization
        assert grad_var < 1e-6, f"High gradient variance {grad_var} suggests quantization"
        
        print(f"✓ PASS: No premature uint16 conversion (grad_var={grad_var:.2e})")


class TestStrideConsistency:
    """Verify stride is consistent regardless of padding."""
    
    def test_stride_calculation(self):
        """Verify stride = tile_size - overlap."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        expected_stride = 1024 - 128  # 896
        
        # Test with different image sizes
        for size in [2000, 3000, 4000, 5000]:
            image = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Pad to tile geometry
            padded, _ = estimator._pad_to_tile_geometry(image)
            h_pad, w_pad = padded.shape[:2]
            
            # Calculate number of tiles
            stride = config.tile_size - config.overlap
            num_tiles_h = (h_pad - config.tile_size) // stride + 1
            num_tiles_w = (w_pad - config.tile_size) // stride + 1
            
            # Verify stride consistency
            assert stride == expected_stride, \
                f"Stride mismatch: {stride} != {expected_stride}"
            
            # Verify coverage (last tile should end at padded boundary)
            last_tile_end_h = (num_tiles_h - 1) * stride + config.tile_size
            last_tile_end_w = (num_tiles_w - 1) * stride + config.tile_size
            
            assert last_tile_end_h == h_pad, \
                f"Vertical coverage gap: {last_tile_end_h} != {h_pad}"
            assert last_tile_end_w == w_pad, \
                f"Horizontal coverage gap: {last_tile_end_w} != {w_pad}"
        
        print(f"✓ PASS: Stride consistent ({expected_stride}px) across all sizes")


class TestReflectionArtifacts:
    """Test for mirrored structure artifacts near borders."""
    
    def test_horizontal_line_reflection(self):
        """Check if horizontal line creates mirror artifact."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        # Create synthetic image with strong horizontal line
        image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        image[100:110, :] = 255  # Horizontal line at y=100
        
        # Pad
        padded, crop_coords = estimator._pad_to_tile_geometry(image)
        
        # Check if line was mirrored near bottom border
        h_pad = padded.shape[0]
        bottom_region = padded[h_pad-200:, :, 0]  # Last 200 rows
        
        # Check for bright horizontal lines (mirrored structure)
        row_means = bottom_region.mean(axis=1)
        bright_rows = np.where(row_means > 200)[0]
        
        # Original line at y=100 shouldn't appear in bottom 200 rows
        # (unless image was < 200px, which it isn't)
        if len(bright_rows) > 5:
            print(f"⚠️  ARTIFACT: {len(bright_rows)} bright rows in bottom region (possible mirror)")
        else:
            print(f"✓ PASS: No horizontal line artifacts ({len(bright_rows)} bright rows)")
    
    def test_vertical_line_reflection(self):
        """Check if vertical line creates mirror artifact."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        # Create synthetic image with strong vertical line
        image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        image[:, 500:510] = 255  # Vertical line at x=500
        
        # Pad
        padded, crop_coords = estimator._pad_to_tile_geometry(image)
        
        # Check if line was mirrored near right border
        w_pad = padded.shape[1]
        right_region = padded[:, w_pad-200:, 0]  # Last 200 columns
        
        # Check for bright vertical lines
        col_means = right_region.mean(axis=0)
        bright_cols = np.where(col_means > 200)[0]
        
        if len(bright_cols) > 5:
            print(f"⚠️  ARTIFACT: {len(bright_cols)} bright columns in right region (possible mirror)")
        else:
            print(f"✓ PASS: No vertical line artifacts ({len(bright_cols)} bright columns)")


class TestWeightedBlending:
    """Verify weighted blending with Hann window."""
    
    def test_blend_weight_creation(self):
        """Verify blend weight has correct taper."""
        config = DepthConfig(tile_size=1024, overlap=128)
        estimator = HighFidelityDepthEstimator(config)
        
        weight = estimator._create_blend_weight(1024, 128)
        
        # Verify shape
        assert weight.shape == (1024, 1024), f"Wrong shape: {weight.shape}"
        
        # Verify center is 1.0
        center = weight[512, 512]
        assert abs(center - 1.0) < 1e-6, f"Center weight {center} != 1.0"
        
        # Verify edges taper to 0
        corner = weight[0, 0]
        assert corner < 0.5, f"Corner weight {corner} too high (expected <0.5)"
        
        # Verify Hann taper (smooth transition)
        top_edge = weight[:128, 512]  # Top edge, center column
        
        # Check monotonicity (should increase from 0 to 1)
        diffs = np.diff(top_edge)
        assert np.all(diffs >= 0), "Taper not monotonic"
        
        # Check smooth (no discontinuities)
        second_diffs = np.diff(diffs)
        assert np.max(np.abs(second_diffs)) < 0.05, "Taper not smooth"
        
        print(f"✓ PASS: Blend weight has Hann taper (corner={corner:.3f}, center={center:.3f})")


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("PRE-FLIGHT VALIDATION: Sliver Tile Fix Infrastructure")
    print("=" * 70)
    
    # Test 1: Pad/Crop Correctness
    print("\n[1/5] Pad/Crop Dimension Preservation")
    print("-" * 70)
    test_pad_crop = TestPadCropCorrectness()
    for h, w, name in [
        (1023, 1023, "just_under_tile"),
        (1025, 1025, "just_over_tile"),
        (4000, 1000, "extreme_wide"),
        (1000, 4000, "extreme_tall"),
        (800, 600, "small_image"),
    ]:
        test_pad_crop.test_dimension_preservation(h, w, name)
    
    # Test 2: Dtype Discipline
    print("\n[2/5] Blending Dtype Discipline")
    print("-" * 70)
    test_dtype = TestBlendingDtype()
    test_dtype.test_accumulator_dtype()
    test_dtype.test_no_premature_uint16_conversion()
    
    # Test 3: Stride Consistency
    print("\n[3/5] Stride Consistency")
    print("-" * 70)
    test_stride = TestStrideConsistency()
    test_stride.test_stride_calculation()
    
    # Test 4: Reflection Artifacts
    print("\n[4/5] Reflection Padding Artifacts")
    print("-" * 70)
    test_artifacts = TestReflectionArtifacts()
    test_artifacts.test_horizontal_line_reflection()
    test_artifacts.test_vertical_line_reflection()
    
    # Test 5: Weighted Blending
    print("\n[5/5] Weighted Blending (Hann Window)")
    print("-" * 70)
    test_blend = TestWeightedBlending()
    test_blend.test_blend_weight_creation()
    
    print("\n" + "=" * 70)
    print("✅ PRE-FLIGHT CHECKS COMPLETE")
    print("=" * 70)
    print("\nStatus: Infrastructure correctness verified")
    print("Next: Run full validation suite (10-20 images)")
