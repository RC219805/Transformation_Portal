"""Tests for input discovery hygiene filters."""

from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.input_discovery import DiscoveryConfig, discover_images


class TestInputDiscovery:
    """Test input discovery with hygiene filters."""

    def test_exclude_depth_artifacts(self, tmp_path):
        """Verify depth maps are excluded from input discovery."""
        # Create test files
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image_depth.png").touch()
        (tmp_path / "image_depthpro_depth16.png").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only RGB image should be discovered
        assert len(images) == 1
        assert images[0].name == "image.jpg"

    def test_exclude_pbr_artifacts(self, tmp_path):
        """Verify normal/roughness/ao maps are excluded."""
        # Create test files
        (tmp_path / "scene.png").touch()
        (tmp_path / "scene_normal.png").touch()
        (tmp_path / "scene_roughness.png").touch()
        (tmp_path / "scene_ao.png").touch()
        (tmp_path / "scene_pbr.png").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only RGB image should be discovered
        assert len(images) == 1
        assert images[0].name == "scene.png"

    def test_exclude_output_directories(self, tmp_path):
        """Verify output/ depth/ pbr/ directories are excluded."""
        # Create directories and files
        (tmp_path / "source.jpg").touch()

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "result.jpg").touch()

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        (depth_dir / "depth_map.png").touch()

        pbr_dir = tmp_path / "pbr"
        pbr_dir.mkdir()
        (pbr_dir / "normal.png").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only source image should be discovered
        assert len(images) == 1
        assert images[0].name == "source.jpg"

    def test_exclude_hidden_files(self, tmp_path):
        """Verify .DS_Store and .cache/ are excluded."""
        # Create test files
        (tmp_path / "visible.jpg").touch()
        (tmp_path / ".DS_Store").touch()
        (tmp_path / ".hidden.jpg").touch()

        # Create hidden directory
        hidden_dir = tmp_path / ".cache"
        hidden_dir.mkdir()
        (hidden_dir / "cached.jpg").touch()

        config = DiscoveryConfig(strict_mode=False, exclude_hidden=True)
        images = discover_images(tmp_path, config)

        # Only visible image should be discovered
        assert len(images) == 1
        assert images[0].name == "visible.jpg"

    def test_exclude_non_source(self, tmp_path):
        """Verify _non_source/ directories are excluded."""
        # Create test files
        (tmp_path / "source.jpg").touch()

        non_source_dir = tmp_path / "_non_source"
        non_source_dir.mkdir()
        (non_source_dir / "intermediate.jpg").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only source image should be discovered
        assert len(images) == 1
        assert images[0].name == "source.jpg"

    def test_strict_mode_fails_on_artifacts(self, tmp_path):
        """Verify strict mode raises on excluded artifacts."""
        # Create test files
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image_depth.png").touch()

        config = DiscoveryConfig(strict_mode=True)

        # Should raise ValueError in strict mode
        with pytest.raises(ValueError, match="Strict mode: 1 excluded artifacts found"):
            discover_images(tmp_path, config)

    def test_valid_images_discovered(self, tmp_path):
        """Verify normal JPEGs/TIFFs/PNGs are discovered."""
        # Create various valid formats
        (tmp_path / "photo1.jpg").touch()
        (tmp_path / "photo2.JPEG").touch()
        (tmp_path / "photo3.png").touch()
        (tmp_path / "photo4.tif").touch()
        (tmp_path / "photo5.TIFF").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # All 5 should be discovered
        assert len(images) == 5
        names = {img.name for img in images}
        assert names == {"photo1.jpg", "photo2.JPEG", "photo3.png", "photo4.tif", "photo5.TIFF"}

    def test_exclude_patterns_case_insensitive(self, tmp_path):
        """Verify _DEPTH and _Depth are also excluded (case-insensitive)."""
        # Create test files with various cases
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image_depth.png").touch()
        (tmp_path / "image_DEPTH.png").touch()
        (tmp_path / "image_Depth.png").touch()
        (tmp_path / "image_DEPTHPRO_DEPTH16.png").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only RGB image should be discovered
        assert len(images) == 1
        assert images[0].name == "image.jpg"

    def test_exclude_zone_artifacts(self, tmp_path):
        """Verify zone maps are excluded."""
        # Create test files
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image_zone.png").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only RGB image should be discovered
        assert len(images) == 1
        assert images[0].name == "image.jpg"

    def test_nested_directories(self, tmp_path):
        """Verify discovery works with nested valid directories."""
        # Create nested structure
        (tmp_path / "root.jpg").touch()

        subdir1 = tmp_path / "renders"
        subdir1.mkdir()
        (subdir1 / "render1.jpg").touch()
        (subdir1 / "render1_depth.png").touch()

        subdir2 = tmp_path / "photos"
        subdir2.mkdir()
        (subdir2 / "photo1.jpg").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Should find 3 valid images (root.jpg, render1.jpg, photo1.jpg)
        assert len(images) == 3
        names = {img.name for img in images}
        assert names == {"root.jpg", "render1.jpg", "photo1.jpg"}

    def test_manifests_and_logs_excluded(self, tmp_path):
        """Verify manifests/ and logs/ directories are excluded."""
        # Create test files
        (tmp_path / "source.jpg").touch()

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        (manifests_dir / "manifest.jpg").touch()

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "log.jpg").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only source image should be discovered
        assert len(images) == 1
        assert images[0].name == "source.jpg"

    def test_custom_extensions(self, tmp_path):
        """Verify custom extensions are respected."""
        # Create test files
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image.png").touch()
        (tmp_path / "image.webp").touch()

        config = DiscoveryConfig(strict_mode=False)

        # Only discover .jpg files
        images = discover_images(tmp_path, config, image_extensions=[".jpg"])

        assert len(images) == 1
        assert images[0].name == "image.jpg"

    def test_empty_directory(self, tmp_path):
        """Verify empty directory returns no images."""
        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        assert len(images) == 0

    def test_strict_mode_succeeds_with_clean_input(self, tmp_path):
        """Verify strict mode succeeds when no artifacts present."""
        # Create only valid files
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()

        config = DiscoveryConfig(strict_mode=True)

        # Should not raise
        images = discover_images(tmp_path, config)

        assert len(images) == 2

    def test_checkpoints_directory_excluded(self, tmp_path):
        """Verify checkpoints/ directory is excluded."""
        # Create test files
        (tmp_path / "source.jpg").touch()

        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        (checkpoints_dir / "checkpoint.jpg").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only source image should be discovered
        assert len(images) == 1
        assert images[0].name == "source.jpg"

    def test_v2_directory_excluded(self, tmp_path):
        """Verify v2/ output directory is excluded."""
        # Create test files
        (tmp_path / "source.jpg").touch()

        v2_dir = tmp_path / "v2"
        v2_dir.mkdir()
        (v2_dir / "enhanced.jpg").touch()

        config = DiscoveryConfig(strict_mode=False)
        images = discover_images(tmp_path, config)

        # Only source image should be discovered
        assert len(images) == 1
        assert images[0].name == "source.jpg"
