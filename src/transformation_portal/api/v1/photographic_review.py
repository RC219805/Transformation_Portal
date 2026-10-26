"""Closed, additive photographic review requests; no physical paths are accepted."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactPath = Annotated[str, Field(min_length=1, max_length=1024)]


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
