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


class TestExifSubIfdExtraction:
    """Exposure, lens, date and GPS tags are read from their sub-IFDs (finding I-4)."""

    @staticmethod
    def _write_jpeg_with_exif(path: Path) -> None:
        from PIL.TiffImagePlugin import IFDRational

        exif = Image.Exif()
        exif[0x010F] = "ACME"  # Make (IFD0)
        exif[0x0110] = "Cam-1"  # Model (IFD0)
        exif_ifd = exif.get_ifd(0x8769)
        exif_ifd[0x829A] = IFDRational(1, 250)  # ExposureTime
        exif_ifd[0x829D] = IFDRational(28, 10)  # FNumber
        exif_ifd[0x8827] = 400  # ISOSpeedRatings
        exif_ifd[0x920A] = IFDRational(35, 1)  # FocalLength
        exif_ifd[0x9003] = "2026:01:02 03:04:05"  # DateTimeOriginal
        exif_ifd[0xA434] = "Lens 35mm"  # LensModel
        gps_ifd = exif.get_ifd(0x8825)
        gps_ifd[1] = "S"
        gps_ifd[2] = (IFDRational(48), IFDRational(30), IFDRational(0))
        gps_ifd[3] = "E"
        gps_ifd[4] = (IFDRational(2), IFDRational(15), IFDRational(0))
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, exif=exif)

    def test_exif_and_gps_sub_ifd_tags_are_extracted(self, tmp_path: Path):
        img_path = tmp_path / "exif.jpg"
        self._write_jpeg_with_exif(img_path)

        camera = ProvenanceCapture()._extract_exif(img_path)

        assert camera.make == "ACME"
        assert camera.model == "Cam-1"
        assert camera.lens_model == "Lens 35mm"
        assert camera.focal_length == pytest.approx(35.0)
        assert camera.aperture == pytest.approx(2.8)
        assert camera.iso == 400
        assert camera.datetime_original == "2026:01:02 03:04:05"
        assert camera.gps_latitude == pytest.approx(-48.5)
        assert camera.gps_longitude == pytest.approx(2.25)

    def test_ifd0_only_file_leaves_sub_ifd_fields_empty(self, tmp_path: Path):
        exif = Image.Exif()
        exif[0x010F] = "ACME"
        img_path = tmp_path / "ifd0.jpg"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path, exif=exif)

        camera = ProvenanceCapture()._extract_exif(img_path)

        assert camera.make == "ACME"
        assert camera.iso is None
        assert camera.gps_latitude is None

    @staticmethod
    def _write_jpeg(path: Path, exif_ifd: dict, gps_ifd: dict | None = None) -> None:
        exif = Image.Exif()
        exif[0x010F] = "ACME"  # Make (IFD0)
        exif.get_ifd(0x8769).update(exif_ifd)
        if gps_ifd:
            exif.get_ifd(0x8825).update(gps_ifd)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path, exif=exif)

    def test_shutter_speed_is_reported_as_a_fraction(self, tmp_path: Path):
        from PIL.TiffImagePlugin import IFDRational

        img_path = tmp_path / "shutter.jpg"
        self._write_jpeg(img_path, {0x829A: IFDRational(1, 250)})  # ExposureTime

        assert ProvenanceCapture()._extract_exif(img_path).shutter_speed == "1/250"

    def test_unknown_zero_over_zero_rationals_do_not_break_the_sidecar(self, tmp_path: Path):
        """0/0 "unknown" rationals become None, so the strict-JSON sidecar can still be written."""
        from PIL.TiffImagePlugin import IFDRational

        unknown = IFDRational(0, 0)
        img_path = tmp_path / "unknown.jpg"
        self._write_jpeg(
            img_path,
            {0x829A: unknown, 0x829D: unknown, 0x920A: unknown, 0xA434: "Manual lens"},
            {1: "N", 2: (unknown, unknown, unknown), 3: "E", 4: (unknown, unknown, unknown), 6: unknown},
        )
        capture = ProvenanceCapture()

        camera = capture._extract_exif(img_path)

        assert camera.lens_model == "Manual lens"
        assert camera.focal_length is None
        assert camera.aperture is None
        assert camera.shutter_speed is None
        assert camera.gps_latitude is None
        assert camera.gps_longitude is None
        assert camera.gps_altitude is None

        prov = capture.capture(
            source_path=img_path,
            tensor=np.zeros((8, 8, 3), dtype=np.float32),
            gamma=1.0,
            bit_depth=32,
            dtype="float32",
        )
        sidecar_path = tmp_path / "unknown.provenance.json"
        capture.write_sidecar(prov, sidecar_path)

        assert json.loads(sidecar_path.read_text(encoding="utf-8"))["camera"] == {
            "make": "ACME",
            "lens_model": "Manual lens",
        }

    def test_multi_valued_iso_does_not_drop_later_tags_or_gps(self, tmp_path: Path):
        from PIL.TiffImagePlugin import IFDRational

        img_path = tmp_path / "iso_pair.jpg"
        self._write_jpeg(
            img_path,
            {0x8827: (100, 200), 0xA434: "Lens 35mm"},  # ISOSpeedRatings with two values, LensModel
            {
                1: "N",
                2: (IFDRational(48), IFDRational(30), IFDRational(0)),
                3: "W",
                4: (IFDRational(2), IFDRational(15), IFDRational(0)),
            },
        )

        camera = ProvenanceCapture()._extract_exif(img_path)

        assert camera.iso == 100
        assert camera.lens_model == "Lens 35mm"
        assert camera.gps_latitude == pytest.approx(48.5)
        assert camera.gps_longitude == pytest.approx(-2.25)

    @pytest.mark.parametrize(("altitude_ref", "expected"), [(b"\x00", 430.0), (b"\x01", -430.0)])
    def test_gps_altitude_honours_the_sea_level_reference(self, tmp_path: Path, altitude_ref: bytes, expected: float):
        from PIL.TiffImagePlugin import IFDRational

        img_path = tmp_path / "altitude.jpg"
        self._write_jpeg(
            img_path,
            {0x8827: 100},
            {
                1: "N",
                2: (IFDRational(31), IFDRational(30), IFDRational(0)),
                3: "E",
                4: (IFDRational(35), IFDRational(28), IFDRational(0)),
                5: altitude_ref,  # GPSAltitudeRef
                6: IFDRational(430, 1),  # GPSAltitude
            },
        )

        assert ProvenanceCapture()._extract_exif(img_path).gps_altitude == pytest.approx(expected)


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
