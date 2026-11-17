"""Typed upload staging and exact request-observation envelopes.

POST /v1/uploads/staging preserves tp.orchestrator.upload_staging.v1 and
its existing extensible nested payload. GET /v1/uploads/staging/receipts/
{request_id} adds tp.orchestrator.upload_receipt.v1. Its closed observation
contains upload paths only after an exact retained completion is verified.
The GET handler constructs this model before producing its JSON response.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from transformation_portal.api.v1.envelopes import ApiEnvelope


class UploadArtifacts(BaseModel):
    """Nested ``artifacts`` dict from ``StagedUploadResult.to_response_data``.

    All three values are absolute path strings produced by
    ``stage_upload_batch`` and written into the batch directory.
    """

    model_config = ConfigDict(extra="allow")

    baseline_manifest_path: str
    capture_metadata_path: str
    upload_receipt_path: str


class UploadSummary(BaseModel):
    """Nested ``summary`` dict from ``StagedUploadResult.to_response_data``.

    Mirrors the ``summary`` sub-dict exactly: counts, flags, root paths,
    and any advisory warnings emitted during staging.
    """

    model_config = ConfigDict(extra="allow")

    file_count: int
    total_bytes: int
    capture_metadata_enabled: bool
    capture_metadata_record_count: int
    top_level_roots: list[str]
    warnings: list[str]


class UploadStagingData(BaseModel):
    """Payload for ``tp.orchestrator.upload_staging.v1``.

    Mirrors the top-level dict returned by
    ``StagedUploadResult.to_response_data`` (upload_staging.py:154).
    ``input_dir`` and ``metadata_dir`` are absolute path strings;
    ``received_at_epoch_seconds`` is a Unix timestamp float.

    ``extra="allow"`` at the top level lets future additions to
    ``to_response_data`` pass through without a model bump.
    """

    model_config = ConfigDict(extra="allow")

    batch_id: str
    input_dir: str
    metadata_dir: str
    artifacts: UploadArtifacts
    received_at_epoch_seconds: float
    summary: UploadSummary


UploadStagingEnvelope = ApiEnvelope[UploadStagingData]
"""Convenience alias for the typed envelope wrapping ``UploadStagingData``."""

__all__ = [
    "UploadArtifacts",
    "UploadReceiptData",
    "UploadReceiptEnvelope",
    "UploadStagingData",
    "UploadStagingEnvelope",
    "UploadSummary",
]


class UploadReceiptData(BaseModel):
    """A request-correlated observation; missing evidence is not rejection."""

    model_config = ConfigDict(extra="forbid")
    request_id: str
    status: Literal["pending", "ready", "expired", "unknown"]
    receipt: UploadStagingData | None = None

    @model_validator(mode="after")
    def validate_ready_receipt(self) -> UploadReceiptData:
        if (self.status == "ready") != (self.receipt is not None):
            raise ValueError("Only a ready upload observation may contain its receipt")
        return self


UploadReceiptEnvelope = ApiEnvelope[UploadReceiptData]
