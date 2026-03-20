"""Unit tests for ArtifactManager.

Tests the artifact management logic extracted from orchestrator.py
as part of ADR-043 decomposition.

These tests verify:
1. Artifact type inference from paths
2. Artifact index building with hashes
3. Merkle root computation
4. Output key generation
5. Backward compatibility with orchestrator imports
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestArtifactManagerImports:
    """Test that imports work from both the new and legacy locations."""

    def test_import_from_artifact_manager(self):
        """Test that we can import from the new artifact_manager module."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            ArtifactManager,
            build_artifact_index,
            compute_artifact_merkle_root,
            infer_artifact_type,
            make_output_key,
            v2_log_filename,
        )

        assert callable(infer_artifact_type)
        assert callable(build_artifact_index)
        assert callable(compute_artifact_merkle_root)
        assert callable(make_output_key)
        assert callable(v2_log_filename)
        assert ArtifactManager is not None

    def test_backward_compatible_orchestrator_imports(self):
        """Test that legacy imports from orchestrator still work."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            _build_artifact_index,
            _compute_artifact_merkle_root,
            _infer_artifact_type,
            _v2_log_filename,
            make_output_key,
        )

        assert callable(_infer_artifact_type)
        assert callable(_build_artifact_index)
        assert callable(_compute_artifact_merkle_root)
        assert callable(_v2_log_filename)
        assert callable(make_output_key)


class TestInferArtifactType:
    """Test artifact type inference from paths."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("depth/image_depth.png", "depth_u16_png"),
            ("depth/image_depth.npy", "depth_float_npy"),
            ("depth/image_metadata.json", "depth_metadata"),
            ("depth/other.txt", "depth_aux"),
            ("segmentation/mask.npz", "segmentation_mask_npz"),
            ("segmentation/other.json", "segmentation_aux"),
            ("v2/image_report.json", "v2_report"),
            ("v2/image_enhanced.png", "v2_output"),
            ("manifests/batch_12345.json", "batch_manifest"),
            ("manifests/image_provenance.json", "provenance_sidecar"),
            ("manifests/image_combined.json", "combined_manifest"),
            ("manifests/other.json", "manifest_aux"),
            ("logs/v2_image.log", "v2_log"),
            ("pbr/image_normal.png", "pbr_normal"),
            ("pbr/image_roughness.png", "pbr_roughness"),
            ("pbr/ao_image.png", "pbr_ao"),
            ("pbr/image_ao.png", "pbr_ao"),
            ("pbr/image_height.png", "pbr_aux"),
            ("reconstruction/scene_reconstruction_report.json", "reconstruction_report"),
            ("reconstruction/scene_preflight.json", "reconstruction_preflight_json"),
            ("reconstruction/scene_diagnostics.json", "reconstruction_diagnostics_json"),
            ("reconstruction/scene_scene_manifest.json", "reconstruction_scene_manifest"),
            ("reconstruction/scene_manifest.json", "reconstruction_manifest_json"),
            ("reconstruction/debug/scene_manifest.json", "reconstruction_debug_scene_manifest_json"),
            ("reconstruction/debug/cameras.json", "reconstruction_debug_cameras_json"),
            ("reconstruction/debug/reprojection_preview.png", "reconstruction_debug_preview_png"),
            ("reconstruction/debug/image_overlay.png", "reconstruction_debug_overlay_png"),
            ("reconstruction/debug/other.txt", "reconstruction_debug_aux"),
            ("reconstruction/other.txt", "reconstruction_aux"),
            ("other/random.txt", "artifact"),
        ],
    )
    def test_infer_artifact_type(self, path: str, expected: str):
        """Test artifact type inference for various paths."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            infer_artifact_type,
        )

        assert infer_artifact_type(path) == expected


class TestV2LogFilename:
    """Test V2 log filename generation."""

    def test_simple_filename(self):
        """Test simple log filename without batch ID."""
        from transformation_portal.lux_depth_v3.artifact_manager import v2_log_filename

        result = v2_log_filename("image_jpg_12345678")
        assert result == "v2_image_jpg_12345678.log"

    def test_filename_with_batch_id(self):
        """Test log filename with batch ID."""
        from transformation_portal.lux_depth_v3.artifact_manager import v2_log_filename

        result = v2_log_filename("image_jpg_12345678", "batch_001")
        assert result == "v2_image_jpg_12345678__batch_001.log"

    def test_filename_sanitizes_batch_id(self):
        """Test that batch ID is sanitized."""
        from transformation_portal.lux_depth_v3.artifact_manager import v2_log_filename

        result = v2_log_filename("image", "2026-03-20_12:00:00")
        # Colons should be sanitized
        assert ":" not in result


class TestMakeOutputKey:
    """Test output key generation."""

    def test_preserves_directory_structure(self):
        """Test that directory structure is preserved."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        input_path = Path("/root/photos/scene1/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)
        assert "scene1" in str(key)

    def test_includes_extension_label(self):
        """Test that extension is included in key name."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        input_path = Path("/root/photos/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)
        assert "_jpg_" in str(key)

    def test_includes_hash_suffix(self):
        """Test that hash suffix is included."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        input_path = Path("/root/photos/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)
        parts = str(key).split("_")
        hash_suffix = parts[-1]
        assert len(hash_suffix) == 8
        assert all(c in "0123456789abcdef" for c in hash_suffix)

    def test_different_paths_produce_different_keys(self):
        """Test that different paths produce different keys."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        input_root = Path("/root/photos")
        key1 = make_output_key(Path("/root/photos/image1.jpg"), input_root)
        key2 = make_output_key(Path("/root/photos/image2.jpg"), input_root)

        assert key1 != key2

    def test_deterministic(self):
        """Test that same inputs produce same key."""
        from transformation_portal.lux_depth_v3.artifact_manager import make_output_key

        input_path = Path("/root/photos/image.jpg")
        input_root = Path("/root/photos")

        key1 = make_output_key(input_path, input_root)
        key2 = make_output_key(input_path, input_root)

        assert key1 == key2


class TestBuildArtifactIndex:
    """Test artifact index building."""

    def test_builds_index_from_existing_files(self, tmp_path: Path):
        """Test that index is built correctly from existing files."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            build_artifact_index,
        )

        # Create test files
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_file = depth_dir / "image_depth.png"
        depth_file.write_bytes(b"test depth data")

        index = build_artifact_index(tmp_path, [depth_file])

        assert len(index) == 1
        assert index[0]["artifact_type"] == "depth_u16_png"
        assert index[0]["relative_path"] == "depth/image_depth.png"
        assert "sha256" in index[0]
        assert len(index[0]["sha256"]) == 64
        assert index[0]["size_bytes"] == len(b"test depth data")

    def test_skips_missing_files(self, tmp_path: Path):
        """Test that missing files are skipped."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            build_artifact_index,
        )

        missing_file = tmp_path / "missing.png"
        index = build_artifact_index(tmp_path, [missing_file])

        assert len(index) == 0

    def test_index_is_sorted(self, tmp_path: Path):
        """Test that index is sorted by relative path."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            build_artifact_index,
        )

        # Create test files
        for subdir in ["v2", "depth", "manifests"]:
            d = tmp_path / subdir
            d.mkdir()
            (d / "file.png").write_bytes(b"data")

        paths = [
            tmp_path / "v2" / "file.png",
            tmp_path / "depth" / "file.png",
            tmp_path / "manifests" / "file.png",
        ]

        index = build_artifact_index(tmp_path, paths)

        assert len(index) == 3
        assert index[0]["relative_path"] == "depth/file.png"
        assert index[1]["relative_path"] == "manifests/file.png"
        assert index[2]["relative_path"] == "v2/file.png"


class TestComputeArtifactMerkleRoot:
    """Test merkle root computation."""

    def test_deterministic_merkle_root(self):
        """Test that merkle root is deterministic."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            compute_artifact_merkle_root,
        )

        index = [
            {"relative_path": "a.png", "sha256": "a" * 64},
            {"relative_path": "b.png", "sha256": "b" * 64},
        ]

        root1 = compute_artifact_merkle_root(index)
        root2 = compute_artifact_merkle_root(index)

        assert root1 == root2
        assert len(root1) == 64

    def test_different_artifacts_produce_different_roots(self):
        """Test that different artifacts produce different roots."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            compute_artifact_merkle_root,
        )

        index1 = [{"relative_path": "a.png", "sha256": "a" * 64}]
        index2 = [{"relative_path": "a.png", "sha256": "b" * 64}]

        root1 = compute_artifact_merkle_root(index1)
        root2 = compute_artifact_merkle_root(index2)

        assert root1 != root2

    def test_order_independent(self):
        """Test that insertion order doesn't affect root (sorted internally)."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            compute_artifact_merkle_root,
        )

        index1 = [
            {"relative_path": "a.png", "sha256": "a" * 64},
            {"relative_path": "b.png", "sha256": "b" * 64},
        ]
        index2 = [
            {"relative_path": "b.png", "sha256": "b" * 64},
            {"relative_path": "a.png", "sha256": "a" * 64},
        ]

        assert compute_artifact_merkle_root(index1) == compute_artifact_merkle_root(index2)

    def test_rejects_invalid_sha256(self):
        """Test that invalid sha256 raises error."""
        from transformation_portal.lux_depth_v3.artifact_manager import (
            compute_artifact_merkle_root,
        )

        index = [{"relative_path": "a.png", "sha256": "invalid"}]

        with pytest.raises(RuntimeError, match="Invalid artifact sha256"):
            compute_artifact_merkle_root(index)


class TestArtifactManagerClass:
    """Test the ArtifactManager class interface."""

    def test_manager_init(self, tmp_path: Path):
        """Test manager initialization."""
        from transformation_portal.lux_depth_v3.artifact_manager import ArtifactManager

        manager = ArtifactManager(tmp_path)
        assert manager.output_root == tmp_path.resolve()

    def test_manager_index_artifacts(self, tmp_path: Path):
        """Test manager index_artifacts method."""
        from transformation_portal.lux_depth_v3.artifact_manager import ArtifactManager

        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        test_file = depth_dir / "test.png"
        test_file.write_bytes(b"data")

        manager = ArtifactManager(tmp_path)
        index = manager.index_artifacts([test_file])

        assert len(index) == 1
        assert index[0]["artifact_type"] == "depth_u16_png"

    def test_manager_compute_merkle_root(self, tmp_path: Path):
        """Test manager compute_merkle_root method."""
        from transformation_portal.lux_depth_v3.artifact_manager import ArtifactManager

        manager = ArtifactManager(tmp_path)
        index = [{"relative_path": "a.png", "sha256": "a" * 64}]

        root = manager.compute_merkle_root(index)
        assert len(root) == 64

    def test_manager_generate_output_key(self, tmp_path: Path):
        """Test manager generate_output_key method."""
        from transformation_portal.lux_depth_v3.artifact_manager import ArtifactManager

        manager = ArtifactManager(tmp_path)
        input_path = tmp_path / "image.jpg"

        key = manager.generate_output_key(input_path, tmp_path)
        assert "_jpg_" in str(key)

    def test_manager_infer_type(self, tmp_path: Path):
        """Test manager infer_type method."""
        from transformation_portal.lux_depth_v3.artifact_manager import ArtifactManager

        manager = ArtifactManager(tmp_path)

        assert manager.infer_type("depth/image.png") == "depth_u16_png"
        assert manager.infer_type("v2/image.png") == "v2_output"
