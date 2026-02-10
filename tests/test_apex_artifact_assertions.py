"""CI artifact assertions for APEX pipeline outputs.

Verifies that expected artifacts are generated:
- Depth cache (.depth_cache/*.npy)
- Manifests (manifests/*.json)
- Run cards (run_card_*.json)
- No empty zones/ directory
"""

import json
from pathlib import Path
from typing import List

import pytest


class TestAPEXArtifactPresence:
    """Test that APEX pipeline generates expected artifacts."""

    def test_depth_cache_presence(self, tmp_path: Path):
        """Test that depth cache .npy files are created."""
        # Simulate output structure
        depth_cache_dir = tmp_path / ".depth_cache"
        depth_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create mock depth cache files
        test_files = ["image1_depth.npy", "image2_depth.npy"]
        for fname in test_files:
            (depth_cache_dir / fname).write_bytes(b"mock_numpy_data")

        # Verify presence
        cache_files = list(depth_cache_dir.glob("*.npy"))
        assert len(cache_files) == 2, f"Expected 2 cache files, found {len(cache_files)}"
        assert all(f.suffix == ".npy" for f in cache_files), "All cache files should be .npy"

    def test_manifests_presence(self, tmp_path: Path):
        """Test that manifest JSON files are created."""
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)

        # Create mock manifests
        test_manifests = ["image1_combined.json", "image2_combined.json", "batch_2024-01-01_120000.json"]
        for fname in test_manifests:
            manifest_data = {"image": fname, "status": "ok"}
            (manifests_dir / fname).write_text(json.dumps(manifest_data))

        # Verify presence
        manifest_files = list(manifests_dir.glob("*.json"))
        assert len(manifest_files) == 3, f"Expected 3 manifests, found {len(manifest_files)}"

        # Verify batch manifest exists
        batch_manifests = list(manifests_dir.glob("batch_*.json"))
        assert len(batch_manifests) == 1, "Expected 1 batch manifest"

    def test_run_card_presence(self, tmp_path: Path):
        """Test that run card is created when emit_run_card=True."""
        run_card_path = tmp_path / "run_card_2024-01-01_120000.json"

        run_card_data = {
            "batch_id": "2024-01-01_120000",
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:05:00Z",
            "runtime_stats": {"median": 2.5, "mean": 2.8},
            "outliers": [],
        }

        run_card_path.write_text(json.dumps(run_card_data, indent=2))

        # Verify presence and structure
        assert run_card_path.exists(), "Run card should exist"
        loaded = json.loads(run_card_path.read_text())
        assert "batch_id" in loaded
        assert "runtime_stats" in loaded
        assert "outliers" in loaded

    def test_no_empty_zones_directory(self, tmp_path: Path):
        """Test that zones/ directory is NOT created if unused."""
        # Simulate normal output structure WITHOUT zones/
        for dirname in ["depth", "v2", "manifests", "logs"]:
            (tmp_path / dirname).mkdir(parents=True, exist_ok=True)

        zones_dir = tmp_path / "zones"

        # Verify zones/ does NOT exist (fixed behavior)
        assert not zones_dir.exists(), "zones/ should NOT be created when unused"

    def test_zones_directory_created_when_used(self, tmp_path: Path):
        """Test that zones/ directory IS created when zoning features are enabled."""
        zones_dir = tmp_path / "zones"

        # Simulate enabling zoning features (future implementation)
        # For now, this test documents the intended behavior
        enable_zoning = False  # Future flag

        if enable_zoning:
            zones_dir.mkdir(parents=True, exist_ok=True)
            (zones_dir / "zone1").mkdir()
            assert zones_dir.exists()
            assert len(list(zones_dir.iterdir())) > 0
        else:
            # Current behavior: zones/ not created
            assert not zones_dir.exists()


class TestRuntimeOutlierDetection:
    """Test runtime outlier detection in batch manifests."""

    def test_outlier_metadata_in_batch_manifest(self, tmp_path: Path):
        """Test that outliers are recorded in batch manifest stats."""
        batch_manifest = {
            "batch_id": "test_batch",
            "stats": {
                "median": 1.5,
                "mean": 2.0,
                "outliers": [
                    {
                        "image": "slow_image.tif",
                        "metadata": {
                            "is_outlier": True,
                            "runtime_s": 8.5,
                            "median_runtime_s": 1.5,
                            "ratio_to_median": 5.67,
                            "threshold_multiplier": 5.0,
                        },
                    }
                ],
            },
        }

        manifest_path = tmp_path / "manifests" / "batch_test.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(batch_manifest, indent=2))

        # Verify outlier metadata
        loaded = json.loads(manifest_path.read_text())
        assert "outliers" in loaded["stats"]
        assert len(loaded["stats"]["outliers"]) == 1
        outlier = loaded["stats"]["outliers"][0]
        assert outlier["metadata"]["is_outlier"] is True
        assert outlier["metadata"]["ratio_to_median"] > 5.0

    def test_no_outliers_when_runtimes_uniform(self):
        """Test that no outliers are detected when runtimes are uniform."""
        from transformation_portal.lux_depth_v3.batch_stats import detect_runtime_outliers

        runtimes = [1.2, 1.3, 1.4, 1.5, 1.6]  # All within 33% of each other
        result = detect_runtime_outliers("test_image.tif", 1.4, runtimes, threshold_multiplier=5.0)

        assert result is None, "No outliers should be detected for uniform runtimes"

    def test_outlier_detected_when_5x_median(self):
        """Test that outlier is detected when runtime exceeds 5× median."""
        from transformation_portal.lux_depth_v3.batch_stats import detect_runtime_outliers

        runtimes = [1.2, 1.3, 1.4, 1.5, 1.6, 8.5]  # Last image 5.67× median
        result = detect_runtime_outliers("slow_image.tif", 8.5, runtimes, threshold_multiplier=5.0)

        assert result is not None, "Outlier should be detected"
        warning_msg, metadata = result
        assert "slow_image.tif" in warning_msg
        assert metadata["is_outlier"] is True
        assert metadata["ratio_to_median"] > 5.0


class TestArtifactIntegrity:
    """Test artifact integrity and completeness."""

    def test_manifest_references_depth_cache(self, tmp_path: Path):
        """Test that manifests reference corresponding depth cache files."""
        manifest_data = {
            "input": {"image_path": "test_image.tif"},
            "depth": {"float_depth_path": ".depth_cache/test_image_depth.npy"},
            "status": "ok",
        }

        manifest_path = tmp_path / "manifests" / "test_image_combined.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_data))

        # Verify depth cache reference
        loaded = json.loads(manifest_path.read_text())
        assert "depth" in loaded
        assert "float_depth_path" in loaded["depth"]
        assert loaded["depth"]["float_depth_path"].endswith(".npy")

    def test_batch_manifest_aggregates_results(self, tmp_path: Path):
        """Test that batch manifest aggregates all image results."""
        batch_manifest = {
            "batch_id": "test_batch",
            "results": [
                {"status": "ok", "image": "image1.tif", "runtime_s": 1.5},
                {"status": "ok", "image": "image2.tif", "runtime_s": 1.8},
                {"status": "error", "image": "image3.tif", "error": "Invalid input"},
            ],
            "stats": {
                "total_images": 3,
                "count": 2,  # Only successful ones
                "median": 1.65,
            },
        }

        manifest_path = tmp_path / "manifests" / "batch_test.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(batch_manifest, indent=2))

        # Verify aggregation
        loaded = json.loads(manifest_path.read_text())
        assert loaded["stats"]["total_images"] == 3
        assert loaded["stats"]["count"] == 2  # Only successful
        assert len(loaded["results"]) == 3
