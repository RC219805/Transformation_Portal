"""Schema constants for offline APEX model-family characterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

REPORT_SCHEMA_VERSION = "apex_model_family_characterization_report.v1"
SUMMARY_SCHEMA_VERSION = "apex_redacted_summary.v1"
MATRIX_SCHEMA_VERSION = "apex_family_matrix.v1"
INPUT_DIGEST_VERSION = "input_digest.v1"
TOOL_VERSION = "characterize_apex_model_families/0.1.0"

SPEC_STATUS_OK = "ok"
SPEC_STATUS_NAME_MISMATCH = "name_mismatch"
SPEC_STATUS_INVALID = "invalid_family_spec"

GOVERNANCE_COMMERCIAL_READY = "commercial_ready"
GOVERNANCE_RESEARCH_ONLY = "research_only"
GOVERNANCE_LICENSE_BLOCKED = "license_blocked"
GOVERNANCE_UNKNOWN = "unknown"

OBSERVATION_NOT_RUN = "not_run"
OBSERVATION_MOCKED = "mocked"
OBSERVATION_OBSERVED_LOCAL = "observed_local"
OBSERVATION_EVIDENCE_MISSING = "evidence_missing"

SOURCE_MOCK_V1 = "mock_v1"
SOURCE_REDACTED_SUMMARY_V1 = "redacted_summary_v1"

BLOCKER_SPEC_INVALID = "spec_invalid"
BLOCKER_OBSERVATION_MISSING = "observation_missing"
BLOCKER_LICENSE_BLOCKED = "license_blocked"

GROUP_BLOCKER_ONLY_ONE_MEMBER = "only_one_member"
GROUP_BLOCKER_MEMBER_NOT_COMPARABLE = "member_not_comparable"

ALLOWED_DEPTH_BACKENDS = frozenset({"da3", "depth_pro"})
ALLOWED_SEGMENTATION_BACKENDS = frozenset({"efficientsam", "sam2"})
ALLOWED_QUALITY_TIERS = frozenset({"apex", "premium"})
ALLOWED_OBSERVATION_STATUSES = frozenset(
    {OBSERVATION_NOT_RUN, OBSERVATION_MOCKED, OBSERVATION_OBSERVED_LOCAL, OBSERVATION_EVIDENCE_MISSING},
)
ALLOWED_SOURCES = frozenset({SOURCE_MOCK_V1, SOURCE_REDACTED_SUMMARY_V1})
COMPARABLE_OBSERVATION_STATUSES = frozenset({OBSERVATION_MOCKED, OBSERVATION_OBSERVED_LOCAL})
CLOSED_COMPARISON_BLOCKERS = frozenset(
    {BLOCKER_SPEC_INVALID, BLOCKER_OBSERVATION_MISSING, BLOCKER_LICENSE_BLOCKED},
)
CLOSED_GROUP_BLOCKERS = frozenset({GROUP_BLOCKER_ONLY_ONE_MEMBER, GROUP_BLOCKER_MEMBER_NOT_COMPARABLE})

DEPTH_FAMILY_TOKENS = {"da3": "da3", "depth_pro": "depthpro"}
SEGMENTATION_FAMILY_TOKENS = {"efficientsam": "efficientsam", "sam2": "sam2"}

FAMILY_SPEC_KEYS = frozenset(
    {
        "candidate_family",
        "depth_backend",
        "segmentation_backend",
        "materials_version",
        "quality_tier",
        "pbr_enabled",
        "v2_enabled",
    },
)
FAMILY_SPEC_REQUIRED_KEYS = frozenset({"depth_backend", "segmentation_backend", "quality_tier"})
OBSERVATION_KEYS = frozenset(
    {
        "candidate_family",
        "source",
        "status",
        "fallback_used",
        "runtime_ms",
        "promotion_verdict",
        "metric_contract",
        "mask_evidence_status",
    },
)
FAMILY_FILE_TOP_LEVEL_KEYS = frozenset({"schema_version", "default_governance", "families"})
FAMILY_FILE_GOVERNANCE_KEYS = frozenset({"non_commercial_ok", "accept_depth_pro_license"})

RECONCILIATION_INVARIANTS = (
    "family_count_matches_rows",
    "spec_counts_partition_family_count",
    "observation_counts_partition_family_count",
    "governance_counts_partition_family_count",
    "comparable_count_matches_rule",
)


class FamilySpecDict(TypedDict):
    candidate_family: str
    depth_backend: str
    segmentation_backend: str
    materials_version: int
    quality_tier: str
    pbr_enabled: bool
    v2_enabled: bool


@dataclass(frozen=True)
class FamilySpec:
    """Declarative APEX family identity."""

    depth_backend: str
    segmentation_backend: str
    quality_tier: str
    materials_version: int = 3
    pbr_enabled: bool = False
    v2_enabled: bool = False
    candidate_family: str | None = None

    def to_report_dict(self) -> FamilySpecDict:
        return {
            "candidate_family": self.candidate_family or canonical_family_name(self),
            "depth_backend": self.depth_backend,
            "segmentation_backend": self.segmentation_backend,
            "materials_version": self.materials_version,
            "quality_tier": self.quality_tier,
            "pbr_enabled": self.pbr_enabled,
            "v2_enabled": self.v2_enabled,
        }


def canonical_family_name(spec: FamilySpec | dict[str, Any]) -> str:
    """Return the canonical APEX model-family label."""

    if isinstance(spec, FamilySpec):
        materials_version = spec.materials_version
        depth_backend = spec.depth_backend
        segmentation_backend = spec.segmentation_backend
        quality_tier = spec.quality_tier
        pbr_enabled = spec.pbr_enabled
        v2_enabled = spec.v2_enabled
    else:
        materials_version = int(spec.get("materials_version", 3))
        depth_backend = str(spec.get("depth_backend", ""))
        segmentation_backend = str(spec.get("segmentation_backend", ""))
        quality_tier = str(spec.get("quality_tier", ""))
        pbr_enabled = bool(spec.get("pbr_enabled", False))
        v2_enabled = bool(spec.get("v2_enabled", False))

    depth_token = DEPTH_FAMILY_TOKENS[depth_backend]
    seg_token = SEGMENTATION_FAMILY_TOKENS[segmentation_backend]
    suffix = ""
    if pbr_enabled:
        suffix += "_pbr"
    if v2_enabled:
        suffix += "_v2"
    return f"materials_v{materials_version}_{depth_token}_{seg_token}_{quality_tier}{suffix}"
