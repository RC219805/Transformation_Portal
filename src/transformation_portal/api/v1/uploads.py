"""Typed response models for the orchestrator's upload staging route.

One route gets a ``response_model=`` annotation in PR E of the Phase 1.2
sequence:

- ``POST /v1/uploads/staging`` → ``ApiEnvelope[UploadStagingData]``
  (``tp.orchestrator.upload_staging.v1``)

The route handler returns ``JSONResponse(_api_envelope(...))`` directly, so
``response_model`` is **OpenAPI-only** — no runtime serialisation by FastAPI.
The wire shape is produced by ``StagedUploadResult.to_response_data()`` in
``src/transformation_portal/ingest/upload_staging.py:154``; the models here
mirror that output field-for-field.

The nested ``artifacts`` and ``summary`` dicts have stable, closed shapes
in the current implementation. They are modelled as explicit sub-classes
(rather than ``dict[str, Any]``) because ``to_response_data`` constructs them
from typed dataclass attributes — their keys are unlikely to churn.
``extra="allow"`` is still set on all three models so future additions to
``to_response_data`` don't require an immediate model bump.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

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
    "UploadStagingData",
    "UploadStagingEnvelope",
    "UploadSummary",
]
