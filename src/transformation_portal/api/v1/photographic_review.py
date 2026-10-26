"""Closed, additive photographic review requests; no physical paths are accepted."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactPath = Annotated[str, Field(min_length=1, max_length=1024)]
CorrelationId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")]


class ReviewMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_version: int = Field(ge=0)


class InspectionPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    view: Literal["recorded", "clip_srgb"]
    policy_sha256: Digest | None = None


class ReviewDecisionRequest(ReviewMutation):
    artifact_path: ArtifactPath
    sha256: Digest
    source_identity_sha256: Digest | None = None
    status: Literal["selected", "needs_revision", "approved"]
    note: str = Field(default="", max_length=2000)
    inspection_presentation: InspectionPresentation | None = None


class LightingGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1, max_length=120)
    reference_sha256: Digest
    members: list[Digest] = Field(min_length=1, max_length=32)


class LightingGroupsRequest(ReviewMutation):
    groups: list[LightingGroup] = Field(min_length=1, max_length=32)


class RevisionRequest(ReviewMutation):
    artifact_path: ArtifactPath
    sha256: Digest
    source_identity_sha256: Digest | None = None
    scene_id: str = Field(min_length=1, max_length=120)
    recipe: dict[str, Any]
    reference_sha256: Digest | None = None
    preview_only: bool = True
    presentation_policy_sha256: Digest | None = None


class ApprovedSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_path: ArtifactPath
    sha256: Digest
    source_identity_sha256: Digest | None = None


class DeliveryRequest(ReviewMutation):
    request_correlation_id: CorrelationId | None = None
    property_name: str = Field(min_length=1, max_length=120)
    presets: list[str | dict[str, Any]] = Field(min_length=1, max_length=6)
    approved_sha256: list[Digest] = Field(default_factory=list, max_length=32)
    approved_sources: list[ApprovedSource] | None = Field(default=None, min_length=1, max_length=32)
    scene_labels: dict[
        Annotated[str, Field(min_length=1, max_length=120)], Annotated[str, Field(min_length=1, max_length=120)]
    ] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def require_approved_selection(self) -> Self:
        if not self.approved_sha256 and not self.approved_sources:
            raise ValueError("Delivery requires an explicit approved source selection")
        return self


class DeliveryHistorySource(BaseModel):
    """Public identity only; no source bindings or physical paths."""

    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_path: ArtifactPath
    sha256: Digest
    source_identity_sha256: Digest
    presentation_policy_sha256: Digest | None = None


class DeliveryHistoryPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_name: Literal["tp.photo.export_preset.v1"] = Field(alias="schema")
    name: str
    format: str
    intended_use: str
    long_edge: int | None
    jpeg_quality: int
    sharpen_amount: float
    render: str
    shoulder: float


class DeliveryHistoryIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    property_name: str = Field(min_length=1, max_length=120)
    sources: list[DeliveryHistorySource] = Field(min_length=1, max_length=32)
    scene_labels: dict[str, str] = Field(max_length=32)
    presets: list[DeliveryHistoryPreset] = Field(min_length=1, max_length=6)


class DeliveryHistoryFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    sha256: Digest
    size_bytes: int = Field(gt=0)
    url: str


class DeliveryHistoryActor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    username: str | None = None
    role: str | None = None
    assurance: str | None = None


class DeliveryHistoryRender(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_name: Literal["tp.lux.sdr_render_receipt.v1", "tp.photo.presentation_render.v1"] = Field(alias="schema")
    mode: Literal["perceptual_srgb", "soft_srgb", "clip_srgb"]
    shoulder: float = Field(ge=0.1, le=0.95)
    operation: Literal["render_sdr", "preserve_recorded_display"]
    presentation_policy_sha256: Digest | None = None


class DeliveryHistoryDerivative(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_master_sha256: Digest
    source_identity_sha256: Digest
    presentation_policy_sha256: Digest | None = None
    derivative_sha256: Digest
    preset: str = Field(min_length=1, max_length=32)
    width: int = Field(ge=1, le=65536)
    height: int = Field(ge=1, le=65536)
    bit_depth: int = Field(ge=1, le=64)
    color_domain: Literal["extended_linear_srgb", "source_encoding_preserved", "extended_encoded_srgb", "display_encoded_srgb"]
    recipe: DeliveryHistoryPreset
    operation: Literal["byte_preserve", "analytic_extended_srgb_transfer", "render_sdr", "preserve_recorded_display"]
    render: DeliveryHistoryRender | None


class DeliveryHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=128)
    created_at: float | None
    actor: DeliveryHistoryActor
    approval_version: int | None = Field(default=None, ge=0)
    request_correlation_id: CorrelationId | None = None
    request_fingerprint_schema: Literal["tp.photo.delivery_intent.v1"] | None = None
    request_fingerprint_sha256: Digest | None = None
    intent: DeliveryHistoryIntent | None = None
    sources: list[DeliveryHistorySource] = Field(max_length=32)
    applied_derivatives_schema: Literal["tp.photo.delivery_applied.v1"] | None = None
    applied_derivatives: list[DeliveryHistoryDerivative] | None = Field(default=None, max_length=224)
    approval_at_creation: Literal["recorded", "unknown"]
    current_recorded_approval: Literal["approved", "withdrawn", "unknown"]
    source_validation: Literal["not_checked"] = "not_checked"
    file_validation: Literal["not_checked"] = "not_checked"
    files: list[DeliveryHistoryFile] = Field(max_length=5)


class DeliveryHistoryPage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state_version: int = Field(ge=0)
    snapshot_count: int = Field(ge=0, le=64)
    items: list[DeliveryHistoryEntry] = Field(max_length=64)
    next_cursor: str | None
