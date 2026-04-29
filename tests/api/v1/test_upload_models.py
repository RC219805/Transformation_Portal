"""Unit tests for the upload staging response models (Phase 1.2 PR E).

These tests verify the typed models in
``transformation_portal.api.v1.uploads`` accept the wire shapes produced by
``StagedUploadResult.to_response_data()`` in
``src/transformation_portal/ingest/upload_staging.py``. They complement the
end-to-end route tests in ``tests/test_app_orchestrator_contract_http.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformation_portal.api.v1 import (
    UploadArtifacts,
    UploadStagingData,
    UploadStagingEnvelope,
    UploadSummary,
)

# Reference shape from StagedUploadResult.to_response_data().
# Filenames and paths mirror the constants in upload_staging.py:
#   BASELINE_MANIFEST_FILENAME = "baseline_manifest.tp.meta.baseline_manifest.v1.json"
#   CAPTURE_METADATA_FILENAME  = "capture_metadata.tp.meta.capture.v1.json"
#   UPLOAD_RECEIPT_FILENAME    = "upload_receipt.tp.orchestrator.upload_staging.v1.json"
# portal_dir is built as batch_root / "_portal" (upload_staging.py:472).
_BATCH = "upload_1700000000_abcd1234"
_PORTAL = f"/uploads/batches/{_BATCH}/_portal"
_SAMPLE_RESPONSE_DATA: dict = {
    "batch_id": _BATCH,
    "input_dir": f"/uploads/batches/{_BATCH}/input",
    "metadata_dir": _PORTAL,
    "artifacts": {
        "baseline_manifest_path": f"{_PORTAL}/baseline_manifest.tp.meta.baseline_manifest.v1.json",
        "capture_metadata_path": f"{_PORTAL}/capture_metadata.tp.meta.capture.v1.json",
        "upload_receipt_path": f"{_PORTAL}/upload_receipt.tp.orchestrator.upload_staging.v1.json",
    },
    "received_at_epoch_seconds": 1700000000.0,
    "summary": {
        "file_count": 3,
        "total_bytes": 2097152,
        "capture_metadata_enabled": False,
        "capture_metadata_record_count": 0,
        "top_level_roots": ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
        "warnings": [],
    },
}


# ---------------------------------------------------------------------------
# UploadArtifacts
# ---------------------------------------------------------------------------


class TestUploadArtifacts:
    def test_valid_shape_validates(self) -> None:
        arts = UploadArtifacts(**_SAMPLE_RESPONSE_DATA["artifacts"])
        assert arts.baseline_manifest_path.endswith("baseline_manifest.tp.meta.baseline_manifest.v1.json")
        assert arts.capture_metadata_path.endswith("capture_metadata.tp.meta.capture.v1.json")
        assert arts.upload_receipt_path.endswith("upload_receipt.tp.orchestrator.upload_staging.v1.json")

    def test_extra_keys_are_preserved(self) -> None:
        arts = UploadArtifacts(
            baseline_manifest_path="/a",
            capture_metadata_path="/b",
            upload_receipt_path="/c",
            future_artifact="/d",
        )
        dumped = arts.model_dump(mode="json")
        assert dumped["future_artifact"] == "/d"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            UploadArtifacts(
                baseline_manifest_path="/a",
                capture_metadata_path="/b",
                # upload_receipt_path omitted
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# UploadSummary
# ---------------------------------------------------------------------------


class TestUploadSummary:
    def test_valid_shape_validates(self) -> None:
        summary = UploadSummary(**_SAMPLE_RESPONSE_DATA["summary"])
        assert summary.file_count == 3
        assert summary.total_bytes == 2097152
        assert summary.capture_metadata_enabled is False
        assert summary.top_level_roots == ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
        assert summary.warnings == []

    def test_with_warnings_validates(self) -> None:
        summary = UploadSummary(
            file_count=1,
            total_bytes=512,
            capture_metadata_enabled=True,
            capture_metadata_record_count=1,
            top_level_roots=["img.jpg"],
            warnings=["duplicate_filename_detected"],
        )
        assert summary.warnings == ["duplicate_filename_detected"]

    def test_extra_keys_are_preserved(self) -> None:
        data = dict(_SAMPLE_RESPONSE_DATA["summary"], future_stat=99)
        summary = UploadSummary(**data)
        dumped = summary.model_dump(mode="json")
        assert dumped["future_stat"] == 99


# ---------------------------------------------------------------------------
# UploadStagingData — full payload
# ---------------------------------------------------------------------------


class TestUploadStagingData:
    def test_full_response_data_shape_validates(self) -> None:
        data = UploadStagingData(**_SAMPLE_RESPONSE_DATA)
        assert data.batch_id == "upload_1700000000_abcd1234"
        assert data.received_at_epoch_seconds == 1700000000.0
        assert isinstance(data.artifacts, UploadArtifacts)
        assert isinstance(data.summary, UploadSummary)

    def test_model_dump_round_trip(self) -> None:
        data = UploadStagingData(**_SAMPLE_RESPONSE_DATA)
        dumped = data.model_dump(mode="json")
        assert dumped["batch_id"] == _SAMPLE_RESPONSE_DATA["batch_id"]
        assert dumped["artifacts"]["baseline_manifest_path"] == (_SAMPLE_RESPONSE_DATA["artifacts"]["baseline_manifest_path"])
        assert dumped["summary"]["file_count"] == 3

    def test_extra_top_level_keys_are_preserved(self) -> None:
        extended = dict(_SAMPLE_RESPONSE_DATA, future_field="v2")
        data = UploadStagingData(**extended)
        dumped = data.model_dump(mode="json")
        assert dumped["future_field"] == "v2"

    def test_missing_batch_id_raises(self) -> None:
        incomplete = {k: v for k, v in _SAMPLE_RESPONSE_DATA.items() if k != "batch_id"}
        with pytest.raises(ValidationError):
            UploadStagingData(**incomplete)


# ---------------------------------------------------------------------------
# UploadStagingEnvelope — full envelope round-trip
# ---------------------------------------------------------------------------


class TestUploadStagingEnvelope:
    def test_envelope_round_trip(self) -> None:
        payload = UploadStagingEnvelope(
            **{
                "schema": "tp.orchestrator.upload_staging.v1",
                "success": True,
                "data": _SAMPLE_RESPONSE_DATA,
                "error": None,
            }
        )
        dumped = payload.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.upload_staging.v1"
        assert dumped["success"] is True
        assert dumped["data"]["batch_id"] == _SAMPLE_RESPONSE_DATA["batch_id"]
        assert dumped["data"]["summary"]["file_count"] == 3

    def test_envelope_validates_upload_staging_data_shape(self) -> None:
        payload = UploadStagingEnvelope(
            **{
                "schema": "tp.orchestrator.upload_staging.v1",
                "success": True,
                "data": _SAMPLE_RESPONSE_DATA,
                "error": None,
            }
        )
        assert isinstance(payload.data, UploadStagingData)
        assert payload.data.batch_id == _SAMPLE_RESPONSE_DATA["batch_id"]

        invalid_data = {k: v for k, v in _SAMPLE_RESPONSE_DATA.items() if k != "batch_id"}
        with pytest.raises(ValidationError):
            UploadStagingEnvelope(
                **{
                    "schema": "tp.orchestrator.upload_staging.v1",
                    "success": True,
                    "data": invalid_data,
                    "error": None,
                }
            )
