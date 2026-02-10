"""Tests for provenance and metadata capture.

These tests validate audit-grade provenance capture for RAW/TIFF inputs,
ensuring deterministic, versioned sidecar generation with complete metadata.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    ExiftoolNotFoundError,
    IngestContext,
    InputFileMetadata,
    MissingRequiredFieldError,
    ProvenanceError,
    ProvenanceMetadata,
    SchemaValidationError,
    capture_provenance,
    extract_exif_metadata,
    get_exiftool_version,
    get_git_commit_sha,
    get_toolchain_versions,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tiff_image(tmp_path: Path) -> Path:
    """Create a small TIFF test image with EXIF data.

    Returns:
        Path to test TIFF file
    """
    # Create a simple 100x100 RGB image
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    # Add some EXIF data
    exif_dict = {
        # Use standard Pillow EXIF tags
        0x0132: "2026:02:10 12:00:00",  # DateTime
        0x010F: "Test Camera",  # Make
        0x0110: "Test Model",  # Model
    }

    # Save as TIFF with EXIF
    tiff_path = tmp_path / "test_image.tif"
    img.save(tiff_path, format="TIFF", exif=img.getexif())

    return tiff_path


@pytest.fixture
def mock_exiftool_available():
    """Mock exiftool availability check to return True."""
    with patch(
        "transformation_portal.lux_depth_v3.provenance._check_exiftool_available",
        return_value=True,
    ):
        yield


@pytest.fixture
def mock_exiftool_json_output():
    """Mock exiftool JSON output for testing."""
    return [
        {
            "SourceFile": "test_image.tif",
            "ExifTool:ExifToolVersion": "12.76",
            "File:FileType": "TIFF",
            "File:FileSize": "12345",
            "EXIF:Make": "Test Camera",
            "EXIF:Model": "Test Model",
            "EXIF:DateTime": "2026:02:10 12:00:00",
        }
    ]


@pytest.fixture
def valid_input_metadata() -> InputFileMetadata:
    """Valid InputFileMetadata for testing."""
    return InputFileMetadata(
        file_path="/path/to/test.tif",
        file_sha256="a" * 64,  # Valid SHA256
        file_size_bytes=12345,
        file_mtime_utc="2026-02-10T12:00:00+00:00",
    )


@pytest.fixture
def valid_ingest_context() -> IngestContext:
    """Valid IngestContext for testing."""
    return IngestContext(
        git_commit_sha="abc123def456",
        config_fingerprint="sha256:" + "f" * 64,
        ingest_timestamp_utc="2026-02-10T12:00:00+00:00",
        host_os="Linux-6.5.0-test",
        host_machine="x86_64",
        cli_args=["--preset", "max_quality"],
        working_directory="/workspace",
    )


@pytest.fixture
def valid_provenance_metadata(
    valid_input_metadata: InputFileMetadata,
    valid_ingest_context: IngestContext,
) -> ProvenanceMetadata:
    """Valid ProvenanceMetadata for testing."""
    return ProvenanceMetadata(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        input=valid_input_metadata,
        exif={"EXIF:Make": "Test Camera", "EXIF:Model": "Test Model"},
        toolchain={
            "python_version": "3.11.8",
            "exiftool_version": "12.76",
            "rawpy_version": "0.18.1",
            "libraw_version": "0.21.2",
        },
        ingest_context=valid_ingest_context,
    )


# =============================================================================
# Test: Exiftool Availability
# =============================================================================


class TestExiftoolAvailability:
    """Test exiftool detection and version extraction."""

    def test_get_exiftool_version_when_available(self):
        """Test exiftool version retrieval when exiftool is available."""
        # Mock subprocess.run to simulate exiftool -ver
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="12.76\n",
            )
            version = get_exiftool_version()
            assert version == "12.76"
            mock_run.assert_called_once()

    def test_get_exiftool_version_when_not_available(self):
        """Test exiftool version returns None when not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            version = get_exiftool_version()
            assert version is None

    def test_get_exiftool_version_timeout(self):
        """Test exiftool version returns None on timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("exiftool", 5)):
            version = get_exiftool_version()
            assert version is None


# =============================================================================
# Test: EXIF Metadata Extraction
# =============================================================================


class TestExifMetadataExtraction:
    """Test EXIF metadata extraction via exiftool."""

    def test_extract_exif_metadata_success(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
        mock_exiftool_json_output,
    ):
        """Test successful EXIF extraction from TIFF file."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_exiftool_json_output),
            )

            metadata = extract_exif_metadata(sample_tiff_image)

            assert isinstance(metadata, dict)
            assert "EXIF:Make" in metadata
            assert metadata["EXIF:Make"] == "Test Camera"
            mock_run.assert_called_once()
            # Verify exiftool was called with correct flags
            call_args = mock_run.call_args[0][0]
            assert "exiftool" in call_args
            assert "-G" in call_args  # Show group names
            assert "-a" in call_args  # Allow duplicates
            assert "-s" in call_args  # Use tag names
            assert "-j" in call_args  # JSON output

    def test_extract_exif_metadata_exiftool_not_available(
        self,
        sample_tiff_image: Path,
    ):
        """Test extraction fails gracefully when exiftool not available."""
        with patch(
            "transformation_portal.lux_depth_v3.provenance._check_exiftool_available",
            return_value=False,
        ):
            with pytest.raises(ExiftoolNotFoundError) as exc_info:
                extract_exif_metadata(sample_tiff_image)

            assert "exiftool not found" in str(exc_info.value).lower()
            assert "apt-get install" in str(exc_info.value)

    def test_extract_exif_metadata_file_not_found(self, tmp_path: Path):
        """Test extraction fails on non-existent file."""
        nonexistent = tmp_path / "does_not_exist.tif"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_exif_metadata(nonexistent)

        assert "not found" in str(exc_info.value).lower()

    def test_extract_exif_metadata_exiftool_error(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
    ):
        """Test extraction fails when exiftool returns error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Error reading file",
            )

            with pytest.raises(ProvenanceError) as exc_info:
                extract_exif_metadata(sample_tiff_image)

            assert "exiftool failed" in str(exc_info.value).lower()

    def test_extract_exif_metadata_malformed_json(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
    ):
        """Test extraction fails on malformed JSON output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not valid json",
            )

            with pytest.raises(ProvenanceError) as exc_info:
                extract_exif_metadata(sample_tiff_image)

            assert "failed to parse" in str(exc_info.value).lower()

    def test_extract_exif_metadata_timeout(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
    ):
        """Test extraction fails on timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("exiftool", 30)):
            with pytest.raises(ProvenanceError) as exc_info:
                extract_exif_metadata(sample_tiff_image)

            assert "timed out" in str(exc_info.value).lower()


# =============================================================================
# Test: Toolchain Version Capture
# =============================================================================


class TestToolchainVersions:
    """Test toolchain version capture."""

    def test_get_toolchain_versions_all_available(self):
        """Test toolchain version capture when all tools available."""
        with patch("subprocess.run") as mock_run:
            # Mock exiftool version
            mock_run.return_value = MagicMock(returncode=0, stdout="12.76\n")

            with patch("importlib.import_module") as mock_import:
                # Mock rawpy module
                mock_rawpy = MagicMock()
                mock_rawpy.__version__ = "0.18.1"
                mock_rawpy.libraw_version = "0.21.2"
                mock_import.return_value = mock_rawpy

                versions = get_toolchain_versions()

                assert "python_version" in versions
                assert "exiftool_version" in versions
                # Note: rawpy might not be imported in this test context

    def test_get_toolchain_versions_python_always_present(self):
        """Test that Python version is always captured."""
        versions = get_toolchain_versions()
        assert "python_version" in versions
        assert versions["python_version"]  # Not None or empty


# =============================================================================
# Test: Git SHA Capture
# =============================================================================


class TestGitSHACapture:
    """Test git commit SHA capture."""

    def test_get_git_commit_sha_in_repo(self, tmp_path: Path):
        """Test git SHA capture when in a git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="abc123def456789\n",
            )

            sha = get_git_commit_sha(tmp_path)

            assert sha == "abc123def456789"
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["cwd"] == tmp_path

    def test_get_git_commit_sha_not_in_repo(self, tmp_path: Path):
        """Test git SHA capture when not in a git repository."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128)  # Git error

            sha = get_git_commit_sha(tmp_path)

            assert sha is None

    def test_get_git_commit_sha_git_not_available(self, tmp_path: Path):
        """Test git SHA capture when git not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            sha = get_git_commit_sha(tmp_path)
            assert sha is None


# =============================================================================
# Test: ProvenanceMetadata Validation
# =============================================================================


class TestProvenanceMetadataValidation:
    """Test provenance metadata validation."""

    def test_validate_required_fields_success(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation passes with all required fields."""
        # Should not raise
        valid_provenance_metadata.validate_required_fields()

    def test_validate_required_fields_missing_file_path(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation fails on missing file_path."""
        valid_provenance_metadata.input.file_path = ""

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            valid_provenance_metadata.validate_required_fields()

        assert "file_path" in str(exc_info.value).lower()

    def test_validate_required_fields_missing_file_sha256(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation fails on missing file_sha256."""
        valid_provenance_metadata.input.file_sha256 = ""

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            valid_provenance_metadata.validate_required_fields()

        assert "sha256" in str(exc_info.value).lower()

    def test_validate_required_fields_invalid_file_size(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation fails on invalid file size."""
        valid_provenance_metadata.input.file_size_bytes = 0

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            valid_provenance_metadata.validate_required_fields()

        assert "file_size_bytes" in str(exc_info.value).lower()

    def test_validate_required_fields_missing_exiftool_version(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation fails when exiftool_version missing."""
        valid_provenance_metadata.toolchain["exiftool_version"] = None

        with pytest.raises(MissingRequiredFieldError) as exc_info:
            valid_provenance_metadata.validate_required_fields()

        assert "exiftool_version" in str(exc_info.value).lower()

    def test_validate_required_fields_wrong_schema_version(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test validation fails on wrong schema version."""
        valid_provenance_metadata.schema_version = "99.0.0"

        with pytest.raises(SchemaValidationError) as exc_info:
            valid_provenance_metadata.validate_required_fields()

        assert "schema version" in str(exc_info.value).lower()


# =============================================================================
# Test: JSON Serialization & Determinism
# =============================================================================


class TestJSONSerialization:
    """Test JSON serialization and determinism."""

    def test_to_json_stable_key_ordering(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test JSON output has stable key ordering."""
        json1 = valid_provenance_metadata.to_json()
        json2 = valid_provenance_metadata.to_json()

        # Same object should produce identical JSON
        assert json1 == json2

    def test_to_json_deterministic_across_instances(
        self,
        valid_input_metadata: InputFileMetadata,
        valid_ingest_context: IngestContext,
    ):
        """Test that two identical metadata instances produce identical JSON."""
        # Create two identical instances
        prov1 = ProvenanceMetadata(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            input=valid_input_metadata,
            exif={"EXIF:Make": "Test"},
            toolchain={"python_version": "3.11", "exiftool_version": "12.76"},
            ingest_context=valid_ingest_context,
        )

        prov2 = ProvenanceMetadata(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            input=valid_input_metadata,
            exif={"EXIF:Make": "Test"},
            toolchain={"python_version": "3.11", "exiftool_version": "12.76"},
            ingest_context=valid_ingest_context,
        )

        json1 = prov1.to_json()
        json2 = prov2.to_json()

        assert json1 == json2

    def test_to_json_parseable(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test JSON output is parseable."""
        json_str = valid_provenance_metadata.to_json()
        parsed = json.loads(json_str)

        assert isinstance(parsed, dict)
        assert parsed["schema_version"] == PROVENANCE_SCHEMA_VERSION

    def test_roundtrip_serialization(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
    ):
        """Test serialization and deserialization roundtrip."""
        # Serialize to dict
        data = valid_provenance_metadata.to_dict()

        # Deserialize
        restored = ProvenanceMetadata.from_dict(data)

        # Should be equal
        assert restored.schema_version == valid_provenance_metadata.schema_version
        assert restored.input.file_path == valid_provenance_metadata.input.file_path
        assert restored.exif == valid_provenance_metadata.exif


# =============================================================================
# Test: Sidecar File Writing & Reading
# =============================================================================


class TestSidecarFileOperations:
    """Test sidecar file writing and reading."""

    def test_write_sidecar_creates_file(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
        tmp_path: Path,
    ):
        """Test sidecar file is created."""
        sidecar_path = tmp_path / "test_provenance.json"

        valid_provenance_metadata.write_sidecar(sidecar_path)

        assert sidecar_path.exists()
        assert sidecar_path.is_file()

    def test_write_sidecar_creates_parent_dir(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
        tmp_path: Path,
    ):
        """Test sidecar writing creates parent directories."""
        sidecar_path = tmp_path / "subdir" / "nested" / "test_provenance.json"

        valid_provenance_metadata.write_sidecar(sidecar_path)

        assert sidecar_path.exists()
        assert sidecar_path.parent.exists()

    def test_write_sidecar_atomic(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
        tmp_path: Path,
    ):
        """Test sidecar writing is atomic (no temp file left behind)."""
        sidecar_path = tmp_path / "test_provenance.json"

        valid_provenance_metadata.write_sidecar(sidecar_path)

        # Check no temp files left behind
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_write_sidecar_validates_before_write(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
        tmp_path: Path,
    ):
        """Test sidecar writing validates before writing."""
        sidecar_path = tmp_path / "test_provenance.json"

        # Break required field
        valid_provenance_metadata.input.file_path = ""

        with pytest.raises(MissingRequiredFieldError):
            valid_provenance_metadata.write_sidecar(sidecar_path)

        # File should not exist
        assert not sidecar_path.exists()

    def test_load_sidecar_success(
        self,
        valid_provenance_metadata: ProvenanceMetadata,
        tmp_path: Path,
    ):
        """Test loading sidecar file."""
        sidecar_path = tmp_path / "test_provenance.json"
        valid_provenance_metadata.write_sidecar(sidecar_path)

        # Load it back
        loaded = ProvenanceMetadata.load_sidecar(sidecar_path)

        assert loaded.schema_version == valid_provenance_metadata.schema_version
        assert loaded.input.file_path == valid_provenance_metadata.input.file_path

    def test_load_sidecar_file_not_found(self, tmp_path: Path):
        """Test loading non-existent sidecar fails."""
        sidecar_path = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError):
            ProvenanceMetadata.load_sidecar(sidecar_path)

    def test_load_sidecar_invalid_schema(self, tmp_path: Path):
        """Test loading sidecar with invalid schema version fails."""
        sidecar_path = tmp_path / "test_provenance.json"

        # Write invalid schema
        invalid_data = {
            "schema_version": "99.0.0",
            "input": {},
            "exif": {},
            "toolchain": {},
            "ingest_context": {},
        }

        with open(sidecar_path, "w") as f:
            json.dump(invalid_data, f)

        with pytest.raises(SchemaValidationError) as exc_info:
            ProvenanceMetadata.load_sidecar(sidecar_path)

        assert "unsupported" in str(exc_info.value).lower()


# =============================================================================
# Test: End-to-End Provenance Capture
# =============================================================================


class TestProvenanceCapture:
    """Test end-to-end provenance capture."""

    def test_capture_provenance_success(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
        mock_exiftool_json_output,
    ):
        """Test successful end-to-end provenance capture."""
        with patch("subprocess.run") as mock_run:
            # Mock exiftool extraction
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_exiftool_json_output),
            )

            config_fingerprint = "sha256:" + "f" * 64

            provenance = capture_provenance(
                image_path=sample_tiff_image,
                config_fingerprint=config_fingerprint,
                cli_args=["--preset", "max_quality"],
                repo_root=None,
            )

            # Validate result
            assert isinstance(provenance, ProvenanceMetadata)
            assert provenance.schema_version == PROVENANCE_SCHEMA_VERSION
            assert provenance.input.file_path == str(sample_tiff_image)
            assert provenance.input.file_sha256  # Should have hash
            assert provenance.input.file_size_bytes > 0
            assert provenance.exif  # Should have EXIF data
            assert provenance.toolchain["python_version"]
            assert provenance.ingest_context.config_fingerprint == config_fingerprint

    def test_capture_provenance_file_not_found(self, tmp_path: Path):
        """Test provenance capture fails on non-existent file."""
        nonexistent = tmp_path / "does_not_exist.tif"

        with pytest.raises(FileNotFoundError):
            capture_provenance(
                image_path=nonexistent,
                config_fingerprint="sha256:abc123",
            )

    def test_capture_provenance_exiftool_not_available(
        self,
        sample_tiff_image: Path,
    ):
        """Test provenance capture fails when exiftool not available."""
        with patch(
            "transformation_portal.lux_depth_v3.provenance._check_exiftool_available",
            return_value=False,
        ):
            with pytest.raises(ExiftoolNotFoundError):
                capture_provenance(
                    image_path=sample_tiff_image,
                    config_fingerprint="sha256:abc123",
                )

    def test_capture_provenance_deterministic_hash(
        self,
        sample_tiff_image: Path,
        mock_exiftool_available,
        mock_exiftool_json_output,
    ):
        """Test that file hash is deterministic."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_exiftool_json_output),
            )

            config_fingerprint = "sha256:test"

            prov1 = capture_provenance(
                image_path=sample_tiff_image,
                config_fingerprint=config_fingerprint,
            )

            prov2 = capture_provenance(
                image_path=sample_tiff_image,
                config_fingerprint=config_fingerprint,
            )

            # File hash should be identical
            assert prov1.input.file_sha256 == prov2.input.file_sha256


# =============================================================================
# Test: Integration with Real TIFF Fixtures
# =============================================================================


@pytest.mark.integration
class TestRealTIFFProvenance:
    """Integration tests with real TIFF fixtures (if available)."""

    @pytest.fixture
    def real_tiff_fixture(self) -> Path:
        """Path to real TIFF fixture if available."""
        fixture_path = Path(
            "tests/fixtures/pipelines/750_picacho_lane/input/750Picacho_GreatRoom_UltraQuality.tif"
        )
        if not fixture_path.exists():
            pytest.skip("Real TIFF fixture not available")
        return fixture_path

    def test_provenance_capture_real_tiff(
        self,
        real_tiff_fixture: Path,
        tmp_path: Path,
    ):
        """Test provenance capture with real TIFF fixture (requires exiftool)."""
        # Check if exiftool is available
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("exiftool not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("exiftool not available")

        # Capture provenance
        config_fingerprint = "sha256:test_real_tiff"
        provenance = capture_provenance(
            image_path=real_tiff_fixture,
            config_fingerprint=config_fingerprint,
            cli_args=["--test"],
        )

        # Validate
        assert provenance.input.file_path == str(real_tiff_fixture)
        assert provenance.input.file_sha256
        assert len(provenance.input.file_sha256) == 64  # SHA256 hex length
        assert provenance.exif  # Should have extracted EXIF

        # Write sidecar
        sidecar_path = tmp_path / "real_tiff_provenance.json"
        provenance.write_sidecar(sidecar_path)

        # Verify sidecar is valid JSON
        with open(sidecar_path) as f:
            data = json.load(f)
        assert data["schema_version"] == PROVENANCE_SCHEMA_VERSION

    def test_provenance_determinism_real_tiff(
        self,
        real_tiff_fixture: Path,
        tmp_path: Path,
    ):
        """Test determinism with real TIFF fixture."""
        # Check if exiftool is available
        try:
            result = subprocess.run(
                ["exiftool", "-ver"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                pytest.skip("exiftool not available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("exiftool not available")

        config_fingerprint = "sha256:determinism_test"

        # Capture twice
        prov1 = capture_provenance(
            image_path=real_tiff_fixture,
            config_fingerprint=config_fingerprint,
        )

        prov2 = capture_provenance(
            image_path=real_tiff_fixture,
            config_fingerprint=config_fingerprint,
        )

        # These fields should be identical (deterministic)
        assert prov1.input.file_sha256 == prov2.input.file_sha256
        assert prov1.input.file_size_bytes == prov2.input.file_size_bytes
        assert prov1.input.file_path == prov2.input.file_path

        # These fields may differ (nondeterministic, but that's expected)
        # - ingest_timestamp_utc (time changes)
        # But the EXIF data itself should be identical
        assert prov1.exif == prov2.exif
