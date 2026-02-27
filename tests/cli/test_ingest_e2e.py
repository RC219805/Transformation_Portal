"""Tests for the end-to-end RAW file ingest CLI command.

These tests validate the ingest e2e CLI command behavior including:
- Input discovery
- Dry-run mode
- JSON output format
- Phase execution flow
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestIngestE2EModule:
    """Tests for ingest_e2e module imports and basic functions."""

    def test_module_imports(self):
        """Test that ingest_e2e module can be imported."""
        from transformation_portal.cli.ingest_e2e import (
            E2ERunResult,
            PhaseResult,
            SUPPORTED_IMAGE_EXTENSIONS,
            SUPPORTED_RAW_EXTENSIONS,
            run_e2e_ingest,
        )

        assert callable(run_e2e_ingest)
        assert len(SUPPORTED_RAW_EXTENSIONS) > 0
        assert len(SUPPORTED_IMAGE_EXTENSIONS) > 0
        assert SUPPORTED_RAW_EXTENSIONS.issubset(SUPPORTED_IMAGE_EXTENSIONS)

    def test_cli_app_exists(self):
        """Test that CLI app is defined."""
        from transformation_portal.cli.ingest_e2e import app

        assert app is not None

    def test_supported_extensions(self):
        """Test that expected file extensions are supported."""
        from transformation_portal.cli.ingest_e2e import (
            SUPPORTED_IMAGE_EXTENSIONS,
            SUPPORTED_RAW_EXTENSIONS,
        )

        # RAW formats
        assert ".cr2" in SUPPORTED_RAW_EXTENSIONS
        assert ".cr3" in SUPPORTED_RAW_EXTENSIONS
        assert ".nef" in SUPPORTED_RAW_EXTENSIONS
        assert ".arw" in SUPPORTED_RAW_EXTENSIONS
        assert ".dng" in SUPPORTED_RAW_EXTENSIONS

        # Image formats
        assert ".tif" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".tiff" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS


class TestImageDiscovery:
    """Tests for image discovery functionality."""

    def test_discover_single_file(self, tmp_path: Path):
        """Test discovering a single image file."""
        from transformation_portal.cli.ingest_e2e import _discover_images

        # Create a test TIFF file
        test_file = tmp_path / "test.tiff"
        test_file.write_bytes(b"dummy tiff content")

        images = _discover_images(test_file, recursive=True)
        assert len(images) == 1
        assert images[0] == test_file

    def test_discover_directory(self, tmp_path: Path):
        """Test discovering images in a directory."""
        from transformation_portal.cli.ingest_e2e import _discover_images

        # Create test files
        (tmp_path / "image1.tiff").write_bytes(b"content1")
        (tmp_path / "image2.jpg").write_bytes(b"content2")
        (tmp_path / "image3.png").write_bytes(b"content3")
        (tmp_path / "not_image.txt").write_bytes(b"text")

        images = _discover_images(tmp_path, recursive=False)
        assert len(images) == 3
        assert all(p.suffix.lower() in {".tiff", ".jpg", ".png"} for p in images)

    def test_discover_recursive(self, tmp_path: Path):
        """Test recursive image discovery."""
        from transformation_portal.cli.ingest_e2e import _discover_images

        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.tiff").write_bytes(b"root")
        (subdir / "nested.tiff").write_bytes(b"nested")

        # Non-recursive
        images_flat = _discover_images(tmp_path, recursive=False)
        assert len(images_flat) == 1

        # Recursive
        images_recursive = _discover_images(tmp_path, recursive=True)
        assert len(images_recursive) == 2

    def test_discover_no_images(self, tmp_path: Path):
        """Test handling of directory with no images."""
        from transformation_portal.cli.ingest_e2e import _discover_images

        # Create only non-image files
        (tmp_path / "readme.txt").write_bytes(b"readme")
        (tmp_path / "script.py").write_bytes(b"code")

        images = _discover_images(tmp_path, recursive=True)
        assert len(images) == 0

    def test_discover_unsupported_file(self, tmp_path: Path):
        """Test that unsupported single file returns empty list."""
        from transformation_portal.cli.ingest_e2e import _discover_images

        test_file = tmp_path / "readme.txt"
        test_file.write_bytes(b"text content")

        images = _discover_images(test_file, recursive=True)
        assert len(images) == 0


class TestPhaseResult:
    """Tests for PhaseResult dataclass."""

    def test_phase_result_creation(self):
        """Test creating a PhaseResult."""
        from transformation_portal.cli.ingest_e2e import PhaseResult

        result = PhaseResult(
            phase="ingest",
            success=True,
            elapsed_seconds=1.5,
            items_processed=10,
            items_failed=0,
        )

        assert result.phase == "ingest"
        assert result.success is True
        assert result.elapsed_seconds == 1.5
        assert result.items_processed == 10
        assert result.items_failed == 0
        assert result.error is None

    def test_phase_result_with_error(self):
        """Test PhaseResult with error."""
        from transformation_portal.cli.ingest_e2e import PhaseResult

        result = PhaseResult(
            phase="depth",
            success=False,
            elapsed_seconds=0.5,
            error="Missing dependency",
        )

        assert result.success is False
        assert result.error == "Missing dependency"


class TestE2ERunResult:
    """Tests for E2ERunResult dataclass."""

    def test_e2e_result_creation(self):
        """Test creating an E2ERunResult."""
        from transformation_portal.cli.ingest_e2e import E2ERunResult, PhaseResult

        phase = PhaseResult(
            phase="ingest",
            success=True,
            elapsed_seconds=1.0,
            items_processed=5,
        )

        result = E2ERunResult(
            success=True,
            total_elapsed_seconds=1.5,
            phases=[phase],
            input_count=5,
            processed_count=5,
            failed_count=0,
            output_dir="/tmp/output",
            contract="legacy_linear_srgb",
        )

        assert result.success is True
        assert result.total_elapsed_seconds == 1.5
        assert len(result.phases) == 1
        assert result.contract == "legacy_linear_srgb"

    def test_e2e_result_to_dict(self):
        """Test E2ERunResult JSON serialization."""
        from transformation_portal.cli.ingest_e2e import E2ERunResult, PhaseResult

        phase = PhaseResult(
            phase="ingest",
            success=True,
            elapsed_seconds=1.0,
            items_processed=5,
        )

        result = E2ERunResult(
            success=True,
            total_elapsed_seconds=1.5,
            phases=[phase],
            input_count=5,
            processed_count=5,
            failed_count=0,
            output_dir="/tmp/output",
            contract="legacy_linear_srgb",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["contract"] == "legacy_linear_srgb"
        assert len(data["phases"]) == 1
        assert data["phases"][0]["phase"] == "ingest"

        # Verify JSON serializable
        json_str = json.dumps(data)
        assert '"success": true' in json_str


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_no_output_created(self, tmp_path: Path):
        """Test that dry-run does not create output files."""
        from transformation_portal.cli.ingest_e2e import run_e2e_ingest

        # Create test input
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.tiff").write_bytes(b"content")

        output_dir = tmp_path / "output"

        result = run_e2e_ingest(
            input_path=input_dir,
            output_dir=output_dir,
            dry_run=True,
        )

        assert result.success is True
        assert result.processed_count == 0  # Nothing actually processed
        assert not output_dir.exists() or not list(output_dir.iterdir())

    def test_dry_run_shows_plan(self, tmp_path: Path):
        """Test that dry-run shows execution plan."""
        from transformation_portal.cli.ingest_e2e import run_e2e_ingest

        # Create test input
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.tiff").write_bytes(b"content")

        result = run_e2e_ingest(
            input_path=input_dir,
            output_dir=tmp_path / "output",
            enable_depth=True,
            enable_evidence=True,
            dry_run=True,
        )

        assert result.success is True
        # Check phases are reported in plan
        phase_names = [p.phase for p in result.phases]
        assert "ingest" in phase_names
        assert "depth" in phase_names
        assert "evidence" in phase_names


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_directory(self, tmp_path: Path):
        """Test handling of empty input directory."""
        from transformation_portal.cli.ingest_e2e import run_e2e_ingest

        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        result = run_e2e_ingest(
            input_path=input_dir,
            output_dir=tmp_path / "output",
        )

        assert result.success is False
        assert "No supported images found" in result.error


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_ingest_app_registered(self):
        """Test that ingest app is registered in main CLI."""
        from transformation_portal.cli import ingest_app

        assert ingest_app is not None

    def test_ingest_in_exports(self):
        """Test that ingest_app is in __all__."""
        from transformation_portal import cli

        assert "ingest_app" in cli.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
