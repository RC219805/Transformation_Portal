"""Scaffold tests for ingest orchestration service skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.ingest.metadata_service import MetadataExtractionService
from transformation_portal.ingest.service import MetadataExtractionOrchestrationService, ServiceRunRequest


def test_service_run_request_defaults() -> None:
    request = ServiceRunRequest(command="extract")

    assert request.input_path is None
    assert request.input_paths == ()
    assert request.output_dir is None
    assert request.machine_mode is False
    assert request.strict is True
    assert request.args == {}


def test_orchestration_service_uses_metadata_service_by_default() -> None:
    service = MetadataExtractionOrchestrationService()

    assert isinstance(service.metadata_service, MetadataExtractionService)


def test_orchestration_service_run_is_explicit_placeholder(tmp_path: Path) -> None:
    service = MetadataExtractionOrchestrationService()
    request = ServiceRunRequest(
        command="extract",
        input_path=tmp_path / "sample.cr2",
    )

    with pytest.raises(NotImplementedError, match="orchestration skeleton"):
        service.run(request)
