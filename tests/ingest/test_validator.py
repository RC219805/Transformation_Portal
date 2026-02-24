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

from transformation_portal.ingest.errors import (
    BitDepthViolation,
    GammaViolation,
    IngestExitCode,
    OtherIngestFailure,
    SchemaDriftFailure,
    SchemaValidationFailure,
    aggregate_errors,
    aggregate_exit_code,
    exit_code_priority,
)
from transformation_portal.ingest.schemas import (
    ExifMetadata,
    FileIntegrity,
    HostEnvironment,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
    ToolchainVersion,
)
from transformation_portal.ingest.validator import (
    EXIT_8BIT_CONVERSION,
    EXIT_GAMMA_VIOLATION,
    EXIT_OTHER_FAILURE,
    EXIT_SCHEMA_DRIFT,
    EXIT_SCHEMA_VALIDATION_FAILED,
    EXIT_SUCCESS,
    SchemaValidationError,
    aggregate_exit_codes,
    classify_validation_error,
    classify_validation_errors,
    validate_ingest_contract,
    validate_linear_gamma,
    validate_no_8bit_conversion,
    validate_schema,
    validate_schema_errors,
)


class TestExitCodeAggregation:
    """Tests for centralized ingest exit-code classification and aggregation."""

    def test_classify_validation_error_prefers_structured_schema_drift(self):
        class StructuredError:
            error_type = "schema_drift"

            def __str__(self):
                return "ignored due to structured metadata"

        assert classify_validation_error(StructuredError()) == EXIT_SCHEMA_DRIFT

    def test_classify_validation_errors_fallbacks_to_schema_validation(self):
        errors = ["field foo is invalid", "missing required bar"]
        assert classify_validation_errors(errors) == EXIT_SCHEMA_VALIDATION_FAILED

    def test_aggregate_exit_codes_applies_contract_severity_precedence(self):
        exit_codes = [
            EXIT_SUCCESS,
            EXIT_SCHEMA_VALIDATION_FAILED,
            EXIT_8BIT_CONVERSION,
            EXIT_GAMMA_VIOLATION,
            EXIT_SCHEMA_DRIFT,
            EXIT_OTHER_FAILURE,
        ]
        assert aggregate_exit_codes(exit_codes) == EXIT_SCHEMA_DRIFT

    def test_aggregate_exit_codes_returns_success_for_empty_or_success_only(self):
        assert aggregate_exit_codes([]) == EXIT_SUCCESS
        assert aggregate_exit_codes([EXIT_SUCCESS, EXIT_SUCCESS]) == EXIT_SUCCESS

    def test_aggregate_exit_codes_unknown_values_collapse_to_other_failure(self):
        assert aggregate_exit_codes([999]) == EXIT_OTHER_FAILURE


class TestTypedIngestErrors:
    """Tests for typed ingest-domain error hierarchy and aggregation."""

    def test_ingest_error_populates_exception_args(self):
        error = SchemaValidationFailure("schema issue")
        assert error.args == ("schema issue",)

    def test_ingest_error_uses_identity_equality(self):
        first_error = SchemaValidationFailure("schema issue")
        second_error = SchemaValidationFailure("schema issue")
        assert first_error != second_error

    def test_ingest_error_repr_is_stable_and_compact(self):
        error = SchemaDriftFailure("drift issue")
        assert repr(error) == "SchemaDriftFailure(exit_code=4, priority=40, message='drift issue')"

    def test_aggregate_errors_uses_priority_not_exit_code_magnitude(self):
        errors = [
            OtherIngestFailure("fallback failure"),  # exit code 5, lowest priority
            SchemaValidationFailure("schema issue"),  # exit code 1, mid priority
            SchemaDriftFailure("drift issue"),  # exit code 4, highest priority
        ]
        dominant_error = aggregate_errors(errors)
        assert isinstance(dominant_error, SchemaDriftFailure)
        assert dominant_error is not None
        assert dominant_error.exit_code == IngestExitCode.SCHEMA_DRIFT

    def test_aggregate_exit_code_empty_returns_success(self):
        assert aggregate_errors([]) is None
        assert aggregate_exit_code([]) == IngestExitCode.SUCCESS

    def test_exit_code_priority_includes_success(self):
        assert exit_code_priority(IngestExitCode.SUCCESS) < exit_code_priority(IngestExitCode.OTHER_FAILURE)

    def test_validate_schema_errors_returns_typed_errors(self):
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
        payload = sidecar.model_dump()
        payload["unknown_field"] = "drift"

        typed_errors = validate_schema_errors(payload, schema_type="provenance")
        assert typed_errors
        assert all(hasattr(error, "exit_code") for error in typed_errors)
        assert any(isinstance(error, SchemaDriftFailure) for error in typed_errors)

    def test_compatibility_wrapper_matches_typed_aggregation(self):
        typed_errors = [
            GammaViolation("gamma"),
            BitDepthViolation("8-bit"),
            SchemaValidationFailure("schema"),
        ]
        typed_exit_code = int(aggregate_exit_code(typed_errors))
        compat_exit_code = classify_validation_errors([error.message for error in typed_errors])
        assert compat_exit_code == typed_exit_code


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
        """Test detection of unknown fields (schema drift).

        As of v1.0.0, all schemas use ConfigDict(extra="forbid"),
        so unknown fields are ALWAYS rejected by Pydantic.
        """
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

        # Unknown fields are ALWAYS rejected (via Pydantic extra="forbid")
        errors = validate_schema(data, schema_type="provenance", strict_mode=True)
        assert len(errors) > 0
        assert any("unknown_field" in error or "Extra inputs" in error for error in errors)

        # Even in non-strict mode, extra fields are rejected
        # (strict_mode is now a legacy parameter)
        errors = validate_schema(data, schema_type="provenance", strict_mode=False)
        assert len(errors) > 0
        assert any("unknown_field" in error or "Extra inputs" in error for error in errors)

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
        """Test linear image passes validation (deterministic fixture)."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        # Create deterministic linear-like histogram
        # Linear images have more pixels in shadows (first 3 bins > 20%)
        img = np.zeros((100, 100), dtype=np.float32)
        img[:30, :] = 0.1  # 30% of pixels in shadows (bins 0-1)
        img[30:60, :] = 0.3  # 30% in lower-mids
        img[60:80, :] = 0.6  # 20% in upper-mids
        img[80:, :] = 0.9  # 20% in highlights

        error = validate_linear_gamma(img)
        # Must pass: shadow_ratio = 60% (well above 20% threshold)
        assert error is None

    def test_gamma_corrected_image(self):
        """Test gamma-corrected image is detected (deterministic fixture)."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        # Create deterministic gamma-corrected histogram
        # Gamma images have few pixels in shadows (first 3 bins < 15%)
        img = np.zeros((100, 100), dtype=np.float32)
        img[:10, :] = 0.1  # Only 10% in shadows
        img[10:50, :] = 0.5  # 40% in midtones (gamma shift)
        img[50:80, :] = 0.7  # 30% in upper-mids
        img[80:, :] = 0.95  # 20% in highlights

        error = validate_linear_gamma(img)
        # Must fail: shadow_ratio = 10% (below 15% = 20% - 5% tolerance)
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
            json.dump(
                {
                    "schema_version": "2.0.0",  # Unsupported
                    "file_integrity": {"sha256": "a" * 64, "size_bytes": 1024, "path": "/test"},
                },
                f,
            )

        # Should raise SchemaValidationError
        with pytest.raises(SchemaValidationError):
            validate_ingest_contract(sidecar_path)
