"""Unit tests for lux_depth_v2 schemas module.

Tests for versioned output schemas (ImageReport, RunCard, ServiceError, PresetMetadata).
"""

import json
from pathlib import Path
import tempfile
import pytest

from lux_depth_v2.schemas import (
    ImageReport,
    RunCard,
    ServiceError,
    PresetMetadata,
    StageResult,
    ProcessingStatus,
    StageName,
    validate_schema_version,
    load_image_report,
    load_run_card,
    SCHEMA_VERSION,
    PIPELINE_VERSION,
)


class TestSchemaVersionValidation:
    """Tests for schema version validation."""

    def test_validate_schema_version_valid(self):
        """Test validation with correct major version."""
        data = {"schema_version": "2.0.0"}
        assert validate_schema_version(data, required_major=2) is True

    def test_validate_schema_version_valid_different_minor(self):
        """Test validation accepts different minor versions."""
        data = {"schema_version": "2.5.1"}
        assert validate_schema_version(data, required_major=2) is True

    def test_validate_schema_version_invalid_major(self):
        """Test validation rejects wrong major version."""
        data = {"schema_version": "3.0.0"}
        assert validate_schema_version(data, required_major=2) is False

    def test_validate_schema_version_missing(self):
        """Test validation fails when schema_version is missing."""
        data = {}
        assert validate_schema_version(data) is False

    def test_validate_schema_version_invalid_format(self):
        """Test validation handles invalid version format."""
        data = {"schema_version": "invalid"}
        assert validate_schema_version(data) is False


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self):
        """Test creating a StageResult."""
        stage = StageResult(
            stage=StageName.DEPTH_INFERENCE.value,
            status=ProcessingStatus.SUCCESS.value,
            elapsed_ms=150.5,
        )
        assert stage.stage == "depth_inference"
        assert stage.status == "success"
        assert stage.elapsed_ms == 150.5
        assert stage.error is None
        assert stage.warnings == []

    def test_stage_result_with_error(self):
        """Test StageResult with error information."""
        stage = StageResult(
            stage=StageName.UPSCALING.value,
            status=ProcessingStatus.FAILED.value,
            elapsed_ms=50.0,
            error="Out of memory",
            warnings=["Low memory warning"],
        )
        assert stage.error == "Out of memory"
        assert len(stage.warnings) == 1

    def test_stage_result_to_dict(self):
        """Test StageResult serialization to dict."""
        stage = StageResult(
            stage=StageName.POST_PROCESSING.value,
            status=ProcessingStatus.SUCCESS.value,
            elapsed_ms=200.0,
        )
        data = stage.to_dict()
        assert isinstance(data, dict)
        assert data["stage"] == "post_processing"
        assert data["status"] == "success"


class TestImageReport:
    """Tests for ImageReport schema."""

    def test_image_report_defaults(self):
        """Test ImageReport with default values."""
        report = ImageReport()
        assert report.schema_version == SCHEMA_VERSION
        assert report.pipeline_version == PIPELINE_VERSION
        assert report.status == ProcessingStatus.SKIPPED.value
        assert report.stages == []

    def test_image_report_full(self):
        """Test ImageReport with all fields populated."""
        stage = StageResult(
            stage=StageName.DEPTH_INFERENCE.value,
            status=ProcessingStatus.SUCCESS.value,
            elapsed_ms=100.0,
        )

        report = ImageReport(
            image_path="test.jpg",
            status=ProcessingStatus.SUCCESS.value,
            output_master16="test_master16.tif",
            output_upscaled16="test_upscaled16.tif",
            output_marketing="test_marketing.png",
            elapsed_ms=500.0,
            stages=[stage],
            preset="interior_luxury",
            device="cuda",
            upscale_factor=4,
        )

        assert report.image_path == "test.jpg"
        assert report.status == "success"
        assert report.output_master16 == "test_master16.tif"
        assert len(report.stages) == 1

    def test_image_report_json_roundtrip(self):
        """Test ImageReport JSON serialization and deserialization."""
        original = ImageReport(
            image_path="test.jpg",
            status=ProcessingStatus.SUCCESS.value,
            output_master16="test_master16.tif",
            preset="photo_realistic",
        )

        # Serialize to JSON
        json_str = original.to_json()
        data = json.loads(json_str)

        # Verify schema version is present
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["pipeline_version"] == PIPELINE_VERSION
        assert data["image_path"] == "test.jpg"

    def test_image_report_save_load(self):
        """Test saving and loading ImageReport from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            # Create and save report
            original = ImageReport(
                image_path="test.jpg",
                status=ProcessingStatus.SUCCESS.value,
                output_master16="test_master16.tif",
            )
            original.save(report_path)

            # Load report
            loaded = load_image_report(report_path)
            assert loaded.image_path == original.image_path
            assert loaded.status == original.status
            assert loaded.schema_version == SCHEMA_VERSION


class TestRunCard:
    """Tests for RunCard schema."""

    def test_run_card_defaults(self):
        """Test RunCard with default values."""
        card = RunCard()
        assert card.schema_version == SCHEMA_VERSION
        assert card.pipeline_version == PIPELINE_VERSION
        assert card.device == "cpu"
        assert card.preset == "photo_realistic"
        assert card.input_count == 0

    def test_run_card_full(self):
        """Test RunCard with all fields populated."""
        card = RunCard(
            run_id="test-run-123",
            device="cuda",
            preset="interior_luxury",
            execution_mode="batch",
            input_count=10,
            input_dir="/path/to/input",
            output_dir="/path/to/output",
            artifacts=["image1.tif", "image2.tif"],
            total_elapsed_ms=5000.0,
            success_count=8,
            failed_count=2,
            errors=["Error processing image3.jpg"],
            warnings=["Low memory warning"],
        )

        assert card.run_id == "test-run-123"
        assert card.input_count == 10
        assert card.success_count == 8
        assert card.failed_count == 2

    def test_run_card_json_roundtrip(self):
        """Test RunCard JSON serialization."""
        original = RunCard(
            run_id="test-123",
            input_count=5,
            success_count=5,
        )

        json_str = original.to_json()
        data = json.loads(json_str)

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["run_id"] == "test-123"

    def test_run_card_save_load(self):
        """Test saving and loading RunCard from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = Path(tmpdir) / "run_card.json"

            original = RunCard(
                run_id="test-run-456",
                input_count=3,
                success_count=3,
            )
            original.save(card_path)

            loaded = load_run_card(card_path)
            assert loaded.run_id == original.run_id
            assert loaded.schema_version == SCHEMA_VERSION


class TestServiceError:
    """Tests for ServiceError schema."""

    def test_service_error_minimal(self):
        """Test ServiceError with minimal fields."""
        error = ServiceError(
            error_code="INVALID_INPUT",
            message="Invalid file format",
        )
        assert error.error_code == "INVALID_INPUT"
        assert error.message == "Invalid file format"
        assert error.hint is None
        assert error.request_id is None

    def test_service_error_full(self):
        """Test ServiceError with all fields."""
        error = ServiceError(
            error_code="PROCESSING_FAILED",
            message="Image processing failed",
            hint="Check input image format and size",
            request_id="req-123",
            details={"error": "Out of memory"},
        )
        assert error.error_code == "PROCESSING_FAILED"
        assert error.hint == "Check input image format and size"
        assert error.request_id == "req-123"
        assert error.details["error"] == "Out of memory"

    def test_service_error_json(self):
        """Test ServiceError JSON serialization."""
        error = ServiceError(
            error_code="FILE_TOO_LARGE",
            message="File exceeds size limit",
            hint="Maximum size is 100MB",
            request_id="req-456",
        )

        json_str = error.to_json()
        data = json.loads(json_str)

        assert data["error_code"] == "FILE_TOO_LARGE"
        assert data["message"] == "File exceeds size limit"
        assert data["hint"] == "Maximum size is 100MB"


class TestPresetMetadata:
    """Tests for PresetMetadata schema."""

    def test_preset_metadata_creation(self):
        """Test creating PresetMetadata."""
        metadata = PresetMetadata(
            name="test_preset",
            display_name="Test Preset",
            description="Test description",
            intended_use="Testing",
            quality_tier="standard",
            stability="stable",
        )
        assert metadata.name == "test_preset"
        assert metadata.display_name == "Test Preset"
        assert metadata.quality_tier == "standard"
        assert metadata.stability == "stable"

    def test_preset_metadata_with_performance(self):
        """Test PresetMetadata with performance data."""
        metadata = PresetMetadata(
            name="test_preset",
            display_name="Test Preset",
            description="Test description",
            intended_use="Testing",
            quality_tier="max",
            stability="stable",
            performance={"throughput_img_hr": "100-200", "memory_gb": "4-6"},
            parameters={"exposure": 0.0, "contrast": 1.05},
        )
        assert metadata.performance["throughput_img_hr"] == "100-200"
        assert metadata.parameters["contrast"] == 1.05

    def test_preset_metadata_to_dict(self):
        """Test PresetMetadata serialization."""
        metadata = PresetMetadata(
            name="test_preset",
            display_name="Test Preset",
            description="Test description",
            intended_use="Testing",
            quality_tier="apex",
            stability="canary",
        )
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test_preset"
        assert data["quality_tier"] == "apex"


class TestSchemaCompatibility:
    """Tests for backward compatibility."""

    def test_load_image_report_incompatible_version(self):
        """Test loading ImageReport with incompatible schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            # Create report with incompatible version
            data = {
                "schema_version": "1.0.0",  # Incompatible major version
                "image_path": "test.jpg",
            }
            report_path.write_text(json.dumps(data))

            # Should raise ValueError
            with pytest.raises(ValueError, match="Incompatible schema version"):
                load_image_report(report_path)

    def test_load_run_card_incompatible_version(self):
        """Test loading RunCard with incompatible schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = Path(tmpdir) / "run_card.json"

            data = {
                "schema_version": "3.0.0",  # Incompatible major version
                "run_id": "test",
            }
            card_path.write_text(json.dumps(data))

            with pytest.raises(ValueError, match="Incompatible schema version"):
                load_run_card(card_path)
