"""Tests for PBR integration in depth_canonical module.

These tests verify that the PBR module from lux_depth_v3 is correctly
integrated into the canonical depth pipeline.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from transformation_portal.depth_canonical import (
    DepthPipeline,
    UnifiedDepthConfig,
    ProcessingConfig,
    PBRConfig,
    generate_pbr_maps,
    write_pbr_maps,
)


def test_generate_pbr_maps_flat_depth_normal_is_up():
    """Flat depth surface should produce Z-up normals (blue: RGB = 128, 128, 255)."""
    # Create perfectly flat depth map
    depth = np.full((256, 256), 0.5, dtype=np.float32)

    config = PBRConfig(normal_strength=1.0, normal_blur_radius=0)
    normal_map, _, _ = generate_pbr_maps(depth, config)

    # Flat surface: gradients are zero, so normal = (0, 0, 1)
    # Encoded as RGB: (0+1)*127.5 = 127.5 ≈ 128 for X and Y
    # (1+1)*127.5 = 255 for Z

    # Check central region (avoid edge artifacts)
    center = normal_map[64:192, 64:192]

    # X and Y channels should be near 128 (neutral)
    assert np.abs(center[:, :, 0].mean() - 128) < 5, "X channel should be ~128 (neutral)"
    assert np.abs(center[:, :, 1].mean() - 128) < 5, "Y channel should be ~128 (neutral)"

    # Z channel should be near 255 (pointing up)
    assert center[:, :, 2].mean() > 250, "Z channel should be ~255 (up vector)"


@pytest.mark.parametrize("shape", [
    (256, 256),
    (512, 512),
    (128, 256),  # Non-square
    (100, 100),  # Small
    (1024, 768),  # HD-like
])
def test_generate_pbr_maps_output_shapes_match_input(shape):
    """CRITICAL: Output maps must have SAME dimensions as input depth."""
    h, w = shape
    depth = np.random.rand(h, w).astype(np.float32)

    config = PBRConfig(
        normal_blur_radius=3,
        roughness_blur_radius=5,
        ao_blur_radius=7,
    )

    normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config)

    # Check shapes
    assert normal_map.shape == (h, w, 3), f"Normal map shape mismatch for ({h}, {w})"
    assert roughness_map.shape == (h, w), f"Roughness map shape mismatch for ({h}, {w})"
    assert ao_map.shape == (h, w), f"AO map shape mismatch for ({h}, {w})"


def test_generate_pbr_maps_outputs_uint8():
    """All PBR maps should be uint8 in range [0, 255]."""
    depth = np.random.rand(128, 128).astype(np.float32)

    normal_map, roughness_map, ao_map = generate_pbr_maps(depth)

    # Check data types
    assert normal_map.dtype == np.uint8
    assert roughness_map.dtype == np.uint8
    assert ao_map.dtype == np.uint8

    # Check value ranges
    assert normal_map.min() >= 0 and normal_map.max() <= 255
    assert roughness_map.min() >= 0 and roughness_map.max() <= 255
    assert ao_map.min() >= 0 and ao_map.max() <= 255


def test_generate_pbr_maps_ao_independent_of_normal_strength():
    """CRITICAL: AO must be independent of normal_strength parameter."""
    depth = np.random.rand(256, 256).astype(np.float32)

    # Generate AO with different normal_strength values
    config_weak = PBRConfig(normal_strength=0.5, ao_strength=1.0, ao_blur_radius=5)
    config_strong = PBRConfig(normal_strength=2.0, ao_strength=1.0, ao_blur_radius=5)

    _, _, ao_weak = generate_pbr_maps(depth, config_weak)
    _, _, ao_strong = generate_pbr_maps(depth, config_strong)

    # AO maps should be IDENTICAL despite different normal_strength
    np.testing.assert_array_equal(
        ao_weak, ao_strong,
        err_msg="AO must be independent of normal_strength"
    )


def test_generate_pbr_maps_input_validation_2d():
    """Reject non-2D depth inputs."""
    depth_1d = np.random.rand(256)
    depth_3d = np.random.rand(256, 256, 3)

    with pytest.raises(ValueError, match="Depth must be 2D"):
        generate_pbr_maps(depth_1d)

    with pytest.raises(ValueError, match="Depth must be 2D"):
        generate_pbr_maps(depth_3d)


def test_generate_pbr_maps_input_validation_nan_inf():
    """Reject depth with NaN or Inf values."""
    depth_nan = np.full((256, 256), np.nan, dtype=np.float32)
    depth_inf = np.full((256, 256), np.inf, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or Inf"):
        generate_pbr_maps(depth_nan)

    with pytest.raises(ValueError, match="NaN or Inf"):
        generate_pbr_maps(depth_inf)


def test_write_pbr_maps_creates_pngs_atomically():
    """Verify write_pbr_maps creates PNGs atomically without temp file leaks."""
    depth = np.random.rand(128, 128).astype(np.float32)
    normal_map, roughness_map, ao_map = generate_pbr_maps(depth)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write PBR maps
        paths = write_pbr_maps(normal_map, roughness_map, ao_map, output_dir, "test_render")

        # Verify all maps were written
        assert "normal" in paths
        assert "roughness" in paths
        assert "ao" in paths

        # Verify files exist
        assert paths["normal"].exists()
        assert paths["roughness"].exists()
        assert paths["ao"].exists()

        # Verify no temp files remain
        temp_files = list(output_dir.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Found temp files: {temp_files}"

        # Verify file names
        assert paths["normal"].name == "test_render_normal.png"
        assert paths["roughness"].name == "test_render_roughness.png"
        assert paths["ao"].name == "test_render_ao.png"


def test_depth_pipeline_with_pbr_disabled():
    """Test DepthPipeline with PBR disabled."""
    config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=False)
        )
    )
    pipeline = DepthPipeline(config)

    depth = np.random.rand(256, 256).astype(np.float32)

    result = pipeline.process(depth_map=depth)

    # Depth should be stored
    assert result.depth_map is not None
    np.testing.assert_array_equal(result.depth_map, depth)

    # PBR maps should NOT be generated
    assert result.pbr_maps is None
    assert result.pbr_paths is None


def test_depth_pipeline_with_pbr_enabled():
    """Test DepthPipeline generates PBR maps when enabled."""
    config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=True, normal_strength=1.2)
        )
    )
    pipeline = DepthPipeline(config)

    depth = np.random.rand(256, 256).astype(np.float32)

    result = pipeline.process(depth_map=depth)

    # Depth should be stored
    assert result.depth_map is not None

    # PBR maps should be generated
    assert result.pbr_maps is not None
    assert "normal" in result.pbr_maps
    assert "roughness" in result.pbr_maps
    assert "ao" in result.pbr_maps

    # Check map types and shapes
    assert result.pbr_maps["normal"].shape == (256, 256, 3)
    assert result.pbr_maps["roughness"].shape == (256, 256)
    assert result.pbr_maps["ao"].shape == (256, 256)


def test_depth_pipeline_saves_pbr_maps_when_output_dir_provided():
    """Test DepthPipeline saves PBR maps to disk when output_dir is provided."""
    config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=True)
        )
    )
    pipeline = DepthPipeline(config)

    depth = np.random.rand(128, 128).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        result = pipeline.process(
            depth_map=depth,
            output_dir=output_dir,
            basename="test_image"
        )

        # PBR paths should be set
        assert result.pbr_paths is not None
        assert "normal" in result.pbr_paths
        assert "roughness" in result.pbr_paths
        assert "ao" in result.pbr_paths

        # Files should exist
        assert result.pbr_paths["normal"].exists()
        assert result.pbr_paths["roughness"].exists()
        assert result.pbr_paths["ao"].exists()

        # Verify correct filenames
        assert result.pbr_paths["normal"].name == "test_image_normal.png"
        assert result.pbr_paths["roughness"].name == "test_image_roughness.png"
        assert result.pbr_paths["ao"].name == "test_image_ao.png"


def test_depth_pipeline_requires_depth_map_in_phase1():
    """Test DepthPipeline raises error if depth_map not provided (Phase 1)."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    with pytest.raises(ValueError, match="depth_map must be provided"):
        pipeline.process()


def test_depth_pipeline_validates_depth_map_dimensionality():
    """Test DepthPipeline rejects non-2D depth maps."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    # 1D depth
    depth_1d = np.random.rand(256)
    with pytest.raises(ValueError, match="Depth map must be 2D"):
        pipeline.process(depth_map=depth_1d)

    # 3D depth
    depth_3d = np.random.rand(256, 256, 3)
    with pytest.raises(ValueError, match="Depth map must be 2D"):
        pipeline.process(depth_map=depth_3d)


def test_depth_pipeline_batch_processing():
    """Test DepthPipeline batch processing."""
    config = UnifiedDepthConfig(
        processing=ProcessingConfig(
            pbr=PBRConfig(enabled=True)
        )
    )
    pipeline = DepthPipeline(config)

    # Create batch of depth maps
    depths = [
        np.random.rand(128, 128).astype(np.float32),
        np.random.rand(128, 128).astype(np.float32),
        np.random.rand(128, 128).astype(np.float32),
    ]
    image_paths = [Path(f"image_{i}.jpg") for i in range(3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        results = pipeline.process_batch(
            image_paths=image_paths,
            output_dir=output_dir,
            depth_maps=depths
        )

        # Check results
        assert len(results) == 3

        for i, result in enumerate(results):
            assert result.pbr_maps is not None
            assert result.pbr_paths is not None

            # Files should exist
            assert result.pbr_paths["normal"].exists()
            assert result.pbr_paths["roughness"].exists()
            assert result.pbr_paths["ao"].exists()


def test_depth_pipeline_batch_requires_depth_maps():
    """Test batch processing requires images in Phase 2."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    # Phase 2: depth_maps optional, but images required
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Missing images should raise error
        with pytest.raises(ValueError, match="images parameter is required"):
            pipeline.process_batch(
                images=None,
                output_dir=output_dir,
                depth_maps=None
            )


def test_depth_pipeline_batch_length_mismatch():
    """Test batch processing detects length mismatch."""
    config = UnifiedDepthConfig()
    pipeline = DepthPipeline(config)

    image_paths = [Path("image1.jpg"), Path("image2.jpg")]
    depths = [np.random.rand(128, 128).astype(np.float32)]  # Only 1 depth map

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with pytest.raises(ValueError, match="Length mismatch"):
            pipeline.process_batch(
                image_paths=image_paths,
                output_dir=output_dir,
                depth_maps=depths
            )
