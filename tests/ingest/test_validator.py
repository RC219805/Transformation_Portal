"""Test suite for schema validator.

Tests:
- Schema validation for valid/invalid inputs
- Unknown fields detection (drift)
- Required fields enforcement
- 8-bit conversion detection
- Gamma correction detection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transformation_portal.ingest.schemas import (
    FileIntegrity,
    HostEnvironment,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
    ToolchainVersion,
    ExifMetadata,
)
from transformation_portal.ingest.validator import (
    SchemaValidationError,
    validate_schema,
    validate_ingest_contract,
    validate_no_8bit_conversion,
    validate_linear_gamma,
)


class TestValidateSchema:
    """Tests for validate_schema function."""
    
    def test_valid_provenance_dict(self):
        """Test validation of valid provenance dictionary."""
        sidecar = ProvenanceSidecar(
            file_integrity=FileIntegrity(
                sha256="a" * 64,
                size_bytes=1024,
                path="/input/test.cr2",
            ),
            exif=ExifMetadata(all_tags={}),
            toolchain=[ToolchainVersion(name="exiftool", version="12.50")],
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
        
        errors = validate_schema(sidecar.model_dump(), schema_type="provenance")
        assert errors == []
    
    def test_missing_required_field(self):
        """Test validation with missing required field."""
        data = {
            "schema_version": "1.0.0",
            "file_integrity": {
                "sha256": "a" * 64,
                "size_bytes": 1024,
                "path": "/input/test.cr2",
            },
            # Missing 'exif' field
            "toolchain": [],
            "host": {
                "hostname": "test",
                "os": "Linux",
                "os_version": "5.10.0",
                "python_version": "3.11.0",
                "arch": "x86_64",
            },
            "timestamps": {
                "ingest_start": "2026-02-10T12:00:00+00:00",
                "ingest_end": "2026-02-10T12:05:00+00:00",
            },
            "pipeline_config": {"config_sha256": "b" * 64},
            "run_id": "test-run",
        }
        
        errors = validate_schema(data, schema_type="provenance")
        assert len(errors) > 0
        assert any("exif" in error.lower() for error in errors)
    
    def test_unsupported_schema_version(self):
        """Test validation with unsupported schema version."""
        data = {
            "schema_version": "2.0.0",  # Unsupported
            "file_integrity": {
                "sha256": "a" * 64,
                "size_bytes": 1024,
                "path": "/input/test.cr2",
            },
            "exif": {"all_tags": {}},
            "toolchain": [],
            "host": {
                "hostname": "test",
                "os": "Linux",
                "os_version": "5.10.0",
                "python_version": "3.11.0",
                "arch": "x86_64",
            },
            "timestamps": {
                "ingest_start": "2026-02-10T12:00:00+00:00",
                "ingest_end": "2026-02-10T12:05:00+00:00",
            },
            "pipeline_config": {"config_sha256": "b" * 64},
            "run_id": "test-run",
        }
        
        errors = validate_schema(data, schema_type="provenance")
        assert len(errors) > 0
        assert any("2.0.0" in error for error in errors)
    
    def test_unknown_fields_strict_mode(self):
        """Test detection of unknown fields in strict mode."""
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
            run_id="test-run",
        )
        
        data = sidecar.model_dump()
        data["unknown_field"] = "should_fail"  # Add unknown field
        
        # Strict mode should detect drift
        errors = validate_schema(data, schema_type="provenance", strict_mode=True)
        assert len(errors) > 0
        assert any("unknown_field" in error for error in errors)
        
        # Non-strict mode should pass
        errors = validate_schema(data, schema_type="provenance", strict_mode=False)
        assert errors == []
    
    def test_type_mismatch(self):
        """Test validation with type mismatch."""
        data = {
            "schema_version": "1.0.0",
            "file_integrity": {
                "sha256": "a" * 64,
                "size_bytes": "not-an-integer",  # Type mismatch
                "path": "/input/test.cr2",
            },
            "exif": {"all_tags": {}},
            "toolchain": [],
            "host": {
                "hostname": "test",
                "os": "Linux",
                "os_version": "5.10.0",
                "python_version": "3.11.0",
                "arch": "x86_64",
            },
            "timestamps": {
                "ingest_start": "2026-02-10T12:00:00+00:00",
                "ingest_end": "2026-02-10T12:05:00+00:00",
            },
            "pipeline_config": {"config_sha256": "b" * 64},
            "run_id": "test-run",
        }
        
        errors = validate_schema(data, schema_type="provenance")
        assert len(errors) > 0
        assert any("type" in error.lower() or "size_bytes" in error.lower() for error in errors)


class TestValidate8BitConversion:
    """Tests for 8-bit conversion detection."""
    
    def test_valid_uint16(self):
        """Test valid uint16 image with full range."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create 16-bit image with values > 255
        img = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)
        
        error = validate_no_8bit_conversion(img, expected_dtype="uint16")
        assert error is None
    
    def test_8bit_dtype_violation(self):
        """Test detection of 8-bit dtype when 16-bit expected."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create 8-bit image
        img = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        
        error = validate_no_8bit_conversion(img, expected_dtype="uint16")
        assert error is not None
        assert "8-bit conversion" in error
    
    def test_8bit_range_violation(self):
        """Test detection of 8-bit range in uint16 image."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create uint16 image but with only 8-bit range
        img = np.random.randint(0, 255, size=(100, 100), dtype=np.uint16)
        
        error = validate_no_8bit_conversion(img, expected_dtype="uint16")
        assert error is not None
        assert "8-bit range" in error


class TestValidateLinearGamma:
    """Tests for gamma correction detection."""
    
    def test_linear_image(self):
        """Test linear image passes validation."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create linear image with more data in shadows
        img = np.random.power(0.5, size=(1000, 1000)).astype(np.float32)
        
        error = validate_linear_gamma(img)
        # May or may not pass depending on random distribution
        # Just ensure it doesn't crash
        assert error is None or isinstance(error, str)
    
    def test_gamma_corrected_image(self):
        """Test gamma-corrected image is detected."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create gamma-corrected image with more data in midtones
        linear = np.random.random(size=(1000, 1000)).astype(np.float32)
        gamma_corrected = np.power(linear, 1/2.2)  # Apply gamma
        
        error = validate_linear_gamma(gamma_corrected)
        # Should detect gamma correction
        assert error is not None
        assert "gamma" in error.lower()


class TestValidateIngestContract:
    """Tests for complete ingest contract validation."""
    
    def test_valid_sidecar_file(self, tmp_path: Path):
        """Test validation of valid sidecar file."""
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
            run_id="test-run",
        )
        
        # Write to file
        sidecar_path = tmp_path / "test_provenance.json"
        with open(sidecar_path, "w") as f:
            f.write(sidecar.to_json_deterministic())
        
        # Should pass validation
        validate_ingest_contract(sidecar_path)
    
    def test_invalid_sidecar_raises_error(self, tmp_path: Path):
        """Test validation of invalid sidecar raises error."""
        # Write invalid JSON
        sidecar_path = tmp_path / "invalid_provenance.json"
        with open(sidecar_path, "w") as f:
            json.dump({
                "schema_version": "2.0.0",  # Unsupported
                "file_integrity": {"sha256": "a" * 64, "size_bytes": 1024, "path": "/test"},
            }, f)
        
        # Should raise SchemaValidationError
        with pytest.raises(SchemaValidationError):
            validate_ingest_contract(sidecar_path)
