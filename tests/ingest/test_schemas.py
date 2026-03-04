"""Test suite for ingest contract schemas.

Tests:
- Schema validation (valid and invalid inputs)
- Schema version enforcement
- Required fields enforcement
- Type validation
- Deterministic serialization
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transformation_portal.ingest.schemas import (
    ExifMetadata,
    FileIntegrity,
    HostEnvironment,
    IngestManifest,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
    ToolchainVersion,
)


class TestToolchainVersion:
    """Tests for ToolchainVersion schema."""

    def test_valid_minimal(self):
        """Test minimal valid ToolchainVersion."""
        version = ToolchainVersion(
            name="exiftool",
            version="12.50",
        )
        assert version.name == "exiftool"
        assert version.version == "12.50"
        assert version.path is None

    def test_valid_with_path(self):
        """Test ToolchainVersion with path."""
        version = ToolchainVersion(
            name="exiftool",
            version="12.50",
            path="/usr/local/bin/exiftool",
        )
        assert version.path == "/usr/local/bin/exiftool"

    def test_immutable(self):
        """Test that ToolchainVersion is immutable."""
        version = ToolchainVersion(name="test", version="1.0")
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            version.name = "modified"


class TestFileIntegrity:
    """Tests for FileIntegrity schema."""

    def test_valid_sha256(self):
        """Test valid SHA256 hash."""
        integrity = FileIntegrity(
            sha256="a" * 64,
            size_bytes=1024,
            path="/input/test.cr2",
        )
        assert integrity.sha256 == "a" * 64
        assert integrity.size_bytes == 1024

    def test_sha256_normalized_to_lowercase(self):
        """Test SHA256 is normalized to lowercase."""
        integrity = FileIntegrity(
            sha256="ABCD" * 16,
            size_bytes=1024,
            path="/input/test.cr2",
        )
        assert integrity.sha256 == ("abcd" * 16)

    def test_invalid_sha256_length(self):
        """Test invalid SHA256 length."""
        with pytest.raises(ValueError, match="Invalid SHA256 hash"):
            FileIntegrity(
                sha256="a" * 32,  # Too short
                size_bytes=1024,
                path="/input/test.cr2",
            )

    def test_invalid_sha256_characters(self):
        """Test invalid SHA256 characters."""
        with pytest.raises(ValueError, match="Invalid SHA256 hash"):
            FileIntegrity(
                sha256="g" * 64,  # 'g' is not hex
                size_bytes=1024,
                path="/input/test.cr2",
            )


class TestIngestTimestamps:
    """Tests for IngestTimestamps schema."""

    def test_valid_iso_timestamps(self):
        """Test valid ISO 8601 timestamps."""
        timestamps = IngestTimestamps(
            ingest_start="2026-02-10T12:00:00+00:00",
            ingest_end="2026-02-10T12:05:00+00:00",
            exiftool_extract_duration_sec=2.5,
        )
        assert timestamps.ingest_start == "2026-02-10T12:00:00+00:00"
        assert timestamps.exiftool_extract_duration_sec == 2.5

    def test_iso_timestamp_with_z_suffix(self):
        """Test ISO 8601 with Z suffix."""
        timestamps = IngestTimestamps(
            ingest_start="2026-02-10T12:00:00Z",
            ingest_end="2026-02-10T12:05:00Z",
        )
        assert timestamps.ingest_start == "2026-02-10T12:00:00Z"

    def test_invalid_timestamp_format(self):
        """Test invalid timestamp format (malformed)."""
        with pytest.raises(ValueError, match="Invalid ISO 8601 timestamp"):
            IngestTimestamps(
                ingest_start="2026-02-10 12:00:00",  # Space instead of T
                ingest_end="2026-02-10T12:05:00+00:00",
            )

    def test_naive_timestamp_rejected(self):
        """Test that naive timestamps (no timezone) are rejected.

        Contract requires UTC with timezone. Naive timestamps like
        '2026-02-10T12:00:00' (no Z or offset) must be rejected.
        """
        with pytest.raises(ValueError, match="(Naive timestamp|Invalid ISO 8601)"):
            IngestTimestamps(
                ingest_start="2026-02-10T12:00:00",  # Missing timezone
                ingest_end="2026-02-10T12:05:00+00:00",
            )


class TestExifMetadata:
    """Tests for ExifMetadata schema."""

    def test_normalizes_focal_length_with_mm_suffix(self):
        """Test focal length string values are normalized before validation."""
        metadata = ExifMetadata(
            all_tags={"EXIF:FocalLength": "4.5 mm"},
            focal_length="4.5 mm",
        )
        assert metadata.focal_length == pytest.approx(4.5)

    def test_normalizes_bit_depth_triplet_string(self):
        """Test bit depth string triplets normalize to a single integer."""
        metadata = ExifMetadata(
            all_tags={"EXIF:BitsPerSample": "8 8 8"},
            bit_depth="8 8 8",
        )
        assert metadata.bit_depth == 8

    def test_invalid_focal_length_string_still_rejected(self):
        """Test malformed focal length strings fail validation."""
        with pytest.raises(ValueError, match="Invalid focal_length"):
            ExifMetadata(
                all_tags={"EXIF:FocalLength": "mm"},
                focal_length="mm",
            )


class TestPipelineConfig:
    """Tests for PipelineConfig schema."""

    def test_valid_config_sha256(self):
        """Test valid config SHA256."""
        config = PipelineConfig(
            config_sha256="b" * 64,
            cli_args=["--preset", "luxury"],
            preset="luxury",
        )
        assert config.config_sha256 == "b" * 64
        assert config.cli_args == ["--preset", "luxury"]

    def test_config_sha256_normalized(self):
        """Test config SHA256 normalized to lowercase."""
        config = PipelineConfig(
            config_sha256="BCDE" * 16,
        )
        assert config.config_sha256 == ("bcde" * 16)


class TestProvenanceSidecar:
    """Tests for ProvenanceSidecar schema."""

    def test_valid_minimal_sidecar(self):
        """Test minimal valid ProvenanceSidecar."""
        sidecar = ProvenanceSidecar(
            file_integrity=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            exif=ExifMetadata(
                all_tags={"File:FileType": "CR2"},
            ),
            toolchain=[
                ToolchainVersion(name="exiftool", version="12.50"),
            ],
            host=HostEnvironment(
                hostname="test-host",
                os="Linux",
                os_version="5.10.0",
                python_version="3.11.0",
                arch="x86_64",
            ),
            timestamps=IngestTimestamps(
                ingest_start="2026-02-10T12:00:00+00:00",
                ingest_end="2026-02-10T12:05:00+00:00",
            ),
            pipeline_config=PipelineConfig(
                config_sha256="b" * 64,
            ),
            run_id="550e8400-e29b-41d4-a716-446655440000",
        )

        assert sidecar.schema_version == "1.0.2"
        assert sidecar.file_integrity.sha256 == "a" * 64
        assert sidecar.run_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_invalid_schema_version(self):
        """Test invalid schema version."""
        with pytest.raises(ValueError, match="Input should be '1.0.2'"):
            ProvenanceSidecar(
                schema_version="2.0.0",  # Unsupported
                file_integrity=FileIntegrity(
                    sha256="a" * 64,
                    size_bytes=1024,
                    path="/input/test.cr2",
                ),
                exif=ExifMetadata(all_tags={}),
                toolchain=[],
                host=HostEnvironment(
                    hostname="test",
                    os="Linux",
                    os_version="5.10.0",
                    python_version="3.11.0",
                    arch="x86_64",
                ),
                timestamps=IngestTimestamps(
                    ingest_start="2026-02-10T12:00:00+00:00",
                    ingest_end="2026-02-10T12:05:00+00:00",
                ),
                pipeline_config=PipelineConfig(config_sha256="b" * 64),
                run_id="test-run",
            )

    def test_git_commit_validation(self):
        """Test git commit SHA validation."""
        # Valid git commit
        sidecar = ProvenanceSidecar(
            file_integrity=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            exif=ExifMetadata(all_tags={}),
            toolchain=[],
            host=HostEnvironment(
                hostname="test",
                os="Linux",
                os_version="5.10.0",
                python_version="3.11.0",
                arch="x86_64",
            ),
            timestamps=IngestTimestamps(
                ingest_start="2026-02-10T12:00:00+00:00",
                ingest_end="2026-02-10T12:05:00+00:00",
            ),
            pipeline_config=PipelineConfig(config_sha256="b" * 64),
            git_commit="1234567890abcdef" * 2 + "12345678",  # 40 chars
            run_id="test-run",
        )
        assert len(sidecar.git_commit) == 40

        # Invalid git commit
        with pytest.raises(ValueError, match="Invalid git commit SHA"):
            ProvenanceSidecar(
                file_integrity=FileIntegrity(
                    sha256="a" * 64,
                    size_bytes=1024,
                    path="/input/test.cr2",
                ),
                exif=ExifMetadata(all_tags={}),
                toolchain=[],
                host=HostEnvironment(
                    hostname="test",
                    os="Linux",
                    os_version="5.10.0",
                    python_version="3.11.0",
                    arch="x86_64",
                ),
                timestamps=IngestTimestamps(
                    ingest_start="2026-02-10T12:00:00+00:00",
                    ingest_end="2026-02-10T12:05:00+00:00",
                ),
                pipeline_config=PipelineConfig(config_sha256="b" * 64),
                git_commit="invalid-git-commit",
                run_id="test-run",
            )

    def test_deterministic_json_serialization(self):
        """Test deterministic JSON serialization."""
        sidecar = ProvenanceSidecar(
            file_integrity=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            exif=ExifMetadata(
                all_tags={"camera": "Canon", "iso": 400},
            ),
            toolchain=[
                ToolchainVersion(name="exiftool", version="12.50"),
            ],
            host=HostEnvironment(
                hostname="test-host",
                os="Linux",
                os_version="5.10.0",
                python_version="3.11.0",
                arch="x86_64",
            ),
            timestamps=IngestTimestamps(
                ingest_start="2026-02-10T12:00:00+00:00",
                ingest_end="2026-02-10T12:05:00+00:00",
            ),
            pipeline_config=PipelineConfig(
                config_sha256="b" * 64,
            ),
            run_id="fixed-run-id",
        )

        # Serialize twice
        json1 = sidecar.to_json_deterministic()
        json2 = sidecar.to_json_deterministic()

        # Should be identical
        assert json1 == json2

        # Should have sorted keys
        data = json.loads(json1)
        keys = list(data.keys())
        assert keys == sorted(keys)


class TestIngestManifest:
    """Tests for IngestManifest schema."""

    def test_valid_manifest(self):
        """Test valid IngestManifest."""
        manifest = IngestManifest(
            input_file=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            status="success",
            provenance_sidecar_path="/output/test_provenance.json",
            ingest_duration_sec=5.5,
        )

        assert manifest.schema_version == "1.0.2"
        assert manifest.status == "success"
        assert manifest.ingest_duration_sec == 5.5

    def test_invalid_status(self):
        """Test invalid status value."""
        with pytest.raises(ValueError, match="Invalid status"):
            IngestManifest(
                input_file=FileIntegrity(
                    sha256="a" * 64,
                    size_bytes=1024,
                    path="/input/test.cr2",
                ),
                status="invalid-status",  # Not in allowed set
                provenance_sidecar_path="/output/test_provenance.json",
                ingest_duration_sec=5.5,
            )

    def test_manifest_with_error(self):
        """Test manifest with error status."""
        manifest = IngestManifest(
            input_file=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            status="error",
            error_message="File corrupted",
            provenance_sidecar_path="/output/test_provenance.json",
            ingest_duration_sec=0.5,
        )

        assert manifest.status == "error"
        assert manifest.error_message == "File corrupted"
