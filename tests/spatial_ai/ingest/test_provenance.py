"""Tests for provenance capture.

Tests cover:
- EXIF extraction from images
- Ingest metadata collection
- Provenance data assembly
- Sidecar JSON writing and loading
- File and array hashing
- Error handling

Architecture: ADR-023, Issue #890 Phase I
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.spatial_ai.ingest import CameraMetadata, ProvenanceCapture, ProvenanceData, ProvenanceError


class TestProvenanceCapture:
    """Test provenance capture functionality."""

    def test_capture_basic_provenance(self, tmp_path: Path):
        """Test basic provenance capture without EXIF."""
        # Create test image
        img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        img_path = tmp_path / "test.tiff"
        Image.fromarray(img).save(img_path)

        # Create test tensor
        tensor = np.random.rand(100, 100, 3).astype(np.float32)

        # Capture provenance
        capture = ProvenanceCapture()
        prov = capture.capture(
            source_path=img_path,
            tensor=tensor,
            gamma=1.0,
            bit_depth=32,
        )

        # Verify structure
        assert isinstance(prov, ProvenanceData)
        assert prov.ingest.source_file == str(img_path)
        assert prov.ingest.source_file_size_bytes == img_path.stat().st_size
        assert len(prov.ingest.source_file_hash_sha256) == 64
        assert prov.transform.gamma == 1.0
        assert prov.transform.bit_depth == 32
        assert prov.transform.dtype == "float32"
        assert len(prov.output.content_hash_sha256) == 64

    def test_provenance_hashes_deterministic(self, tmp_path: Path):
        """Test that hashes are deterministic (same input → same hash)."""
        # Create test image
        img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
        img_path = tmp_path / "test.tiff"
        Image.fromarray(img).save(img_path)

        # Create deterministic tensor
        tensor = np.ones((50, 50, 3), dtype=np.float32) * 0.5

        # Capture twice
        capture = ProvenanceCapture()
        prov1 = capture.capture(img_path, tensor, gamma=1.0, bit_depth=32)
        prov2 = capture.capture(img_path, tensor, gamma=1.0, bit_depth=32)

        # Hashes should match
        assert prov1.output.content_hash_sha256 == prov2.output.content_hash_sha256
        assert prov1.ingest.source_file_hash_sha256 == prov2.ingest.source_file_hash_sha256

    def test_hdr_detection(self, tmp_path: Path):
        """Test that HDR values >1.0 are detected."""
        img_path = tmp_path / "test.exr"
        img_path.write_text("dummy")  # Won't be read

        # HDR tensor
        tensor_hdr = np.random.rand(50, 50, 3).astype(np.float32) * 5.0
        capture = ProvenanceCapture()
        prov_hdr = capture.capture(img_path, tensor_hdr, gamma=1.0, bit_depth=32)
        assert prov_hdr.output.has_hdr_values is True

        # Non-HDR tensor
        tensor_sdr = np.random.rand(50, 50, 3).astype(np.float32) * 0.8
        prov_sdr = capture.capture(img_path, tensor_sdr, gamma=1.0, bit_depth=32)
        assert prov_sdr.output.has_hdr_values is False

    def test_write_and_load_sidecar(self, tmp_path: Path):
        """Test writing and loading provenance sidecar JSON."""
        # Create test data
        img_path = tmp_path / "test.tiff"
        img_path.write_text("dummy")

        tensor = np.random.rand(50, 50, 3).astype(np.float32)
        capture = ProvenanceCapture()
        prov = capture.capture(img_path, tensor, gamma=1.0, bit_depth=32)

        # Write sidecar
        sidecar_path = tmp_path / "test_provenance.json"
        capture.write_sidecar(prov, sidecar_path)

        # Verify file exists
        assert sidecar_path.exists()

        # Load and verify
        loaded = capture.load_sidecar(sidecar_path)
        assert loaded["ingest"]["source_file"] == str(img_path)
        assert loaded["transform"]["gamma"] == 1.0
        assert loaded["transform"]["bit_depth"] == 32
        assert "ADR-023" in loaded["adr_references"]

    def test_provenance_to_dict(self, tmp_path: Path):
        """Test provenance to_dict conversion."""
        img_path = tmp_path / "test.tiff"
        img_path.write_text("dummy")

        tensor = np.random.rand(50, 50, 3).astype(np.float32)
        capture = ProvenanceCapture()
        prov = capture.capture(
            img_path,
            tensor,
            gamma=1.0,
            bit_depth=32,
            notes="Test image for unit tests",
        )

        # Convert to dict
        prov_dict = prov.to_dict()

        # Verify structure
        assert "camera" in prov_dict
        assert "ingest" in prov_dict
        assert "transform" in prov_dict
        assert "output" in prov_dict
        assert "adr_references" in prov_dict
        assert prov_dict["notes"] == "Test image for unit tests"

        # Verify serializable to JSON
        json_str = json.dumps(prov_dict, indent=2)
        assert len(json_str) > 0


class TestCameraMetadata:
    """Test camera metadata extraction."""

    def test_empty_metadata_serialization(self):
        """Test that empty camera metadata serializes correctly."""
        camera = CameraMetadata()
        camera_dict = camera.to_dict()

        # Empty metadata should yield empty dict (no None values)
        assert camera_dict == {}

    def test_partial_metadata_serialization(self):
        """Test that partial camera metadata omits None values."""
        camera = CameraMetadata(
            make="Canon",
            model="EOS R5",
            iso=100,
            # Other fields None
        )
        camera_dict = camera.to_dict()

        # Only populated fields
        assert camera_dict == {"make": "Canon", "model": "EOS R5", "iso": 100}
        assert "lens_model" not in camera_dict  # None fields excluded


class TestProvenanceErrors:
    """Test provenance error handling."""

    def test_sidecar_write_error_handling(self, tmp_path: Path):
        """Test that sidecar write errors are handled properly."""
        img_path = tmp_path / "test.tiff"
        img_path.write_text("dummy")

        tensor = np.random.rand(50, 50, 3).astype(np.float32)
        capture = ProvenanceCapture()
        prov = capture.capture(img_path, tensor, gamma=1.0, bit_depth=32)

        # A regular file blocks the parent mkdir, so the write fails regardless
        # of the runner's uid (root would otherwise silently create system paths
        # via mkdir(parents=True)).
        blocking_file = tmp_path / "blocking_file"
        blocking_file.write_text("not a directory")
        invalid_path = blocking_file / "subdir" / "provenance.json"

        with pytest.raises(ProvenanceError, match="sidecar write"):
            capture.write_sidecar(prov, invalid_path)

    def test_sidecar_load_error_handling(self, tmp_path: Path):
        """Test that sidecar load errors are handled properly."""
        capture = ProvenanceCapture()

        # Try to load non-existent file
        nonexistent_path = tmp_path / "nonexistent.json"

        with pytest.raises(ProvenanceError, match="sidecar load"):
            capture.load_sidecar(nonexistent_path)

        # Try to load invalid JSON
        invalid_json_path = tmp_path / "invalid.json"
        invalid_json_path.write_text("not valid json{{{")

        with pytest.raises(ProvenanceError, match="sidecar load"):
            capture.load_sidecar(invalid_json_path)


class TestRAWMetadata:
    """Test RAW-specific metadata capture."""

    def test_raw_metadata_fields(self, tmp_path: Path):
        """Test that RAW-specific metadata fields are captured."""
        img_path = tmp_path / "test.cr2"
        img_path.write_text("dummy")

        tensor = np.random.rand(50, 50, 3).astype(np.float32)
        capture = ProvenanceCapture()
        prov = capture.capture(
            img_path,
            tensor,
            gamma=1.0,
            bit_depth=32,
            demosaic_method="AHD",
            white_balance_method="camera",
            color_matrix="sRGB",
        )

        # Verify RAW-specific fields
        assert prov.transform.demosaic_method == "AHD"
        assert prov.transform.white_balance_method == "camera"
        assert prov.transform.color_matrix == "sRGB"


class TestValueRangeCapture:
    """Test value range capture in output metadata."""

    def test_range_capture(self, tmp_path: Path):
        """Test that min/max value range is captured."""
        img_path = tmp_path / "test.tiff"
        img_path.write_text("dummy")

        # Known range tensor
        tensor = np.random.rand(50, 50, 3).astype(np.float32) * 2.5  # [0, 2.5]

        capture = ProvenanceCapture()
        prov = capture.capture(img_path, tensor, gamma=1.0, bit_depth=32)

        # Verify range
        assert prov.output.value_range_min >= 0.0
        assert prov.output.value_range_max <= 2.5
        assert prov.output.value_range_max > prov.output.value_range_min


# Pytest markers
pytestmark = [
    pytest.mark.unit,  # Fast unit tests
]
