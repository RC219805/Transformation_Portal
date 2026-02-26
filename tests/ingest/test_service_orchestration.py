"""Unit tests for ingest orchestration service wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.ingest.errors import IngestExitCode, OtherIngestFailure
from transformation_portal.ingest.metadata_service import BatchExtractResult
from transformation_portal.ingest.metadata_service import ExtractResult as CoreExtractResult
from transformation_portal.ingest.metadata_service import MetadataExtractionService as CoreMetadataExtractionService
from transformation_portal.ingest.metadata_service import ValidateResult as CoreValidateResult
from transformation_portal.ingest.service import (
    MetadataExtractionOrchestrationService,
    MetadataExtractionService,
    ServiceRunRequest,
)


class _StubCoreService:
    def __init__(self) -> None:
        self.extract_requests = []
        self.batch_requests = []
        self.validate_requests = []
        self.extract_result: CoreExtractResult | None = None
        self.batch_result: BatchExtractResult | None = None
        self.validate_result: CoreValidateResult | None = None

    def extract(self, request):  # pragma: no cover - simple spy wrapper
        self.extract_requests.append(request)
        if self.extract_result is None:
            raise AssertionError("extract_result not configured")
        return self.extract_result

    def batch_extract(self, request):  # pragma: no cover - simple spy wrapper
        self.batch_requests.append(request)
        if self.batch_result is None:
            raise AssertionError("batch_result not configured")
        return self.batch_result

    def validate(self, request):  # pragma: no cover - simple spy wrapper
        self.validate_requests.append(request)
        if self.validate_result is None:
            raise AssertionError("validate_result not configured")
        return self.validate_result


def test_service_run_request_defaults() -> None:
    request = ServiceRunRequest(command="extract")

    assert request.input_path is None
    assert request.input_paths == ()
    assert request.output_dir is None
    assert request.machine_mode is False
    assert request.strict is True
    assert request.args == {}


def test_orchestration_service_uses_metadata_service_by_default() -> None:
    service = MetadataExtractionService()

    assert isinstance(service.metadata_service, CoreMetadataExtractionService)


def test_orchestration_service_alias_points_to_primary_service() -> None:
    assert MetadataExtractionOrchestrationService is MetadataExtractionService


def test_run_extract_delegates_to_core_service(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr2"
    input_path.touch()
    stub = _StubCoreService()
    stub.extract_result = CoreExtractResult(
        path=input_path,
        success=True,
        output_path=tmp_path / "sample.provenance.json",
        elapsed_seconds=0.1,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    request = ServiceRunRequest(
        command="extract",
        input_path=input_path,
        args={"preset": "default", "cli_args": ["--json"]},
    )

    result = service.run(request)

    assert result.success is True
    assert result.exit_code == int(IngestExitCode.SUCCESS)
    assert len(stub.extract_requests) == 1
    assert stub.extract_requests[0].input_path == input_path


def test_run_extract_preserves_core_default_config_and_normalizes_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr2"
    input_path.touch()
    output_path = tmp_path / "sample.provenance.json"
    output_dir = tmp_path / "sidecars"
    stub = _StubCoreService()
    stub.extract_result = CoreExtractResult(
        path=input_path,
        success=True,
        output_path=output_path,
        elapsed_seconds=0.1,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    result = service.run(
        ServiceRunRequest(
            command="extract",
            input_path=str(input_path),
            output_dir=str(output_dir),
            args={"output_path": str(output_path)},
        )
    )

    assert result.success is True
    delegated = stub.extract_requests[0]
    assert delegated.input_path == input_path
    assert delegated.output_path == output_path
    assert delegated.output_dir == output_dir
    assert delegated.config_dict is None


def test_run_extract_cli_args_none_normalizes_to_empty_sequence(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr2"
    input_path.touch()
    stub = _StubCoreService()
    stub.extract_result = CoreExtractResult(
        path=input_path,
        success=True,
        output_path=tmp_path / "sample.provenance.json",
        elapsed_seconds=0.1,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    result = service.run(
        ServiceRunRequest(
            command="extract",
            input_path=input_path,
            args={"cli_args": None},
        )
    )

    assert result.success is True
    delegated = stub.extract_requests[0]
    assert delegated.cli_args == []


def test_run_extract_with_invalid_cli_args_type_returns_other_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cr2"
    input_path.touch()
    service = MetadataExtractionService()

    result = service.run(
        ServiceRunRequest(
            command="extract",
            input_path=input_path,
            args={"cli_args": 123},
        )
    )

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert payload.get("extract_result") is None
    assert payload.get("sidecar") is None
    assert "cli_args for extract command must be a sequence or None, got int" in str(payload.get("error", ""))


def test_run_extract_with_invalid_input_type_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="extract", input_path=123))  # type: ignore[arg-type]

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert "must be str or Path, got int" in str(payload.get("error", ""))


def test_run_extract_missing_input_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="extract", input_path=None))

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert payload.get("extract_result") is None
    assert payload.get("sidecar") is None
    assert "Input path required for extract command" in str(payload.get("error", ""))


def test_run_extract_batch_missing_directory_returns_setup_failure(tmp_path: Path) -> None:
    service = MetadataExtractionService()
    missing_dir = tmp_path / "missing"

    result = service.run(ServiceRunRequest(command="extract-batch", input_path=missing_dir))

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    batch_result = payload.get("batch_result")
    assert batch_result is not None
    assert batch_result.summary_counts["total"] == 0
    assert batch_result.summary_counts["failure"] == 0


def test_run_extract_batch_missing_input_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="extract-batch", input_path=None))

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    batch_result = payload.get("batch_result")
    assert batch_result is not None
    assert "Input directory required for extract-batch command" in str(batch_result.dominant_error)


def test_run_extract_batch_preserves_core_default_config_and_normalizes_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    image_path = input_dir / "a.cr2"
    image_path.touch()
    output_dir = tmp_path / "sidecars"

    summary = {
        "total": 1,
        "success": 1,
        "failure": 0,
        "by_exit_code": {
            code.name: 0 for code in sorted(IngestExitCode, key=lambda code: code.value) if code != IngestExitCode.SUCCESS
        },
    }
    stub = _StubCoreService()
    stub.batch_result = BatchExtractResult(
        items=[],
        total_elapsed=0.01,
        summary_counts=summary,
        dominant_error=None,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    result = service.run(
        ServiceRunRequest(
            command="extract-batch",
            input_path=str(input_dir),
            input_paths=[str(image_path)],
            output_dir=str(output_dir),
        )
    )

    assert result.success is True
    delegated = stub.batch_requests[0]
    assert delegated.output_dir == output_dir
    assert delegated.input_root == input_dir
    assert list(delegated.input_paths) == [image_path]
    assert delegated.config_dict is None


def test_run_extract_batch_with_invalid_input_paths_fails_without_silent_drop(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    service = MetadataExtractionService()

    result = service.run(
        ServiceRunRequest(
            command="extract-batch",
            input_path=input_dir,
            input_paths=[input_dir / "valid.cr2", 7],  # type: ignore[list-item]
        )
    )

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    batch_result = payload.get("batch_result")
    assert batch_result is not None
    assert "input_paths must contain only str or Path values" in str(batch_result.dominant_error)


def test_run_extract_batch_with_invalid_cli_args_type_returns_other_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    image_path = input_dir / "a.cr2"
    image_path.touch()
    service = MetadataExtractionService()

    result = service.run(
        ServiceRunRequest(
            command="extract-batch",
            input_path=input_dir,
            input_paths=[image_path],
            args={"cli_args": 123},
        )
    )

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert payload.get("batch_result") is None
    assert "cli_args for extract-batch command must be a sequence or None, got int" in str(payload.get("error", ""))


def test_run_extract_batch_with_invalid_input_type_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="extract-batch", input_path=123))  # type: ignore[arg-type]

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    batch_result = payload.get("batch_result")
    assert batch_result is not None
    assert "must be str or Path, got int" in str(batch_result.dominant_error)


def test_run_validate_propagates_dominant_error_exit_code(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "missing.provenance.json"
    error = OtherIngestFailure("missing sidecar")
    stub = _StubCoreService()
    stub.validate_result = CoreValidateResult(
        success=False,
        errors=[error],
        dominant_error=error,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    result = service.run(
        ServiceRunRequest(
            command="validate",
            input_path=sidecar_path,
            strict=True,
        )
    )

    assert result.success is False
    assert result.exit_code == int(error.exit_code)
    assert len(stub.validate_requests) == 1


def test_run_validate_sidecar_read_failure_preserves_failure_semantics(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "missing.provenance.json"
    stub = _StubCoreService()
    stub.validate_result = CoreValidateResult(
        success=True,
        errors=[],
        dominant_error=None,
    )
    service = MetadataExtractionService(metadata_service=stub)  # type: ignore[arg-type]

    result = service.run(
        ServiceRunRequest(
            command="validate",
            input_path=sidecar_path,
            strict=True,
        )
    )

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)


def test_run_validate_with_invalid_input_type_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="validate", input_path=123))  # type: ignore[arg-type]

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert "must be str or Path, got int" in str(payload.get("error", ""))


def test_run_validate_missing_input_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="validate", input_path=None))

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
    payload = result.payload or {}
    assert payload.get("validate_result") is None
    assert payload.get("sidecar_data") is None
    assert "Sidecar path required for validate command" in str(payload.get("error", ""))


def test_run_unsupported_command_returns_other_failure() -> None:
    service = MetadataExtractionService()

    result = service.run(ServiceRunRequest(command="not-a-real-command"))

    assert result.success is False
    assert result.exit_code == int(IngestExitCode.OTHER_FAILURE)
