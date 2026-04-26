"""Validation for redacted APEX observation summaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from transformation_portal.evals.apex_model_family_schema import SUMMARY_SCHEMA_VERSION

ALLOWED_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "source_evidence_sha256",
        "candidate_family",
        "depth_backend",
        "segmentation_backend",
        "quality_tier",
        "pbr_enabled",
        "v2_enabled",
        "fallback_used",
        "runtime_ms",
        "promotion_verdict",
        "metric_contract",
        "mask_evidence_status",
        "outside_mask_delta_status",
        "seam_halo_score_status",
        "applied_ops_count",
        "blocked_ops_count",
        "canonical_asset_count",
        "case_count",
        "passing_case_count",
        "metrics_valid_count",
    },
)
REQUIRED_SUMMARY_KEYS = frozenset({"schema_version", "source_evidence_sha256", "candidate_family"})

FORBIDDEN_KEY_PARTS = (
    "path",
    "file",
    "filename",
    "sourcefile",
    "directory",
    "working_directory",
    "host",
    "exif",
    "iptc",
    "xmp",
    "gps",
    "serial",
    "creator",
    "copyright",
    "artist",
    "owner",
    "camera",
    "lens",
)
RAW_ARTIFACT_SUFFIXES = (".npz", ".npy", ".tif", ".tiff", ".png")
RAW_JSON_NAME_PATTERNS = (
    "run_card_",
    "batch_",
    "_combined.json",
    "_combined_provenance.json",
    "_provenance.json",
)
PATH_VALUE_PATTERNS = (
    re.compile(r"(^|[\"' ])/(Users|Volumes)/"),
    re.compile(r"^[A-Za-z]:\\"),
    re.compile(r"\.(tif|tiff|npz|npy|png)$", re.IGNORECASE),
    re.compile(r"(run_card_|batch_|_combined\.json|_combined_provenance\.json|_provenance\.json)", re.IGNORECASE),
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def reject_raw_summary_path(path: Path) -> None:
    """Reject direct raw artifact/run-card paths before file ingestion."""

    name = path.name
    lower_name = name.lower()
    if lower_name.endswith(RAW_ARTIFACT_SUFFIXES) or any(pattern in lower_name for pattern in RAW_JSON_NAME_PATTERNS):
        raise ValueError(f"Observed local inputs must use {SUMMARY_SCHEMA_VERSION}, not raw artifact {path}")


def _reject_forbidden_value(value: Any, *, key: str) -> None:
    if isinstance(value, Mapping):
        raise ValueError(f"Redacted summary field {key!r} must not contain nested objects")
    if isinstance(value, list):
        for item in value:
            _reject_forbidden_value(item, key=key)
        return
    if not isinstance(value, str):
        return
    for pattern in PATH_VALUE_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"Redacted summary field {key!r} contains path-like or raw artifact value")


def validate_redacted_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an allowlisted APEX redacted summary payload."""

    keys = set(payload)
    missing = sorted(REQUIRED_SUMMARY_KEYS - keys)
    if missing:
        raise ValueError(f"Redacted summary missing required field(s): {', '.join(missing)}")
    unknown = sorted(keys - ALLOWED_SUMMARY_KEYS)
    if unknown:
        raise ValueError(f"Redacted summary contains unsupported field(s): {', '.join(unknown)}")

    for key in keys:
        normalized_key = key.lower()
        if any(part in normalized_key for part in FORBIDDEN_KEY_PARTS):
            raise ValueError(f"Redacted summary contains forbidden metadata field {key!r}")
        _reject_forbidden_value(payload[key], key=key)

    if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"Redacted summary schema_version must be {SUMMARY_SCHEMA_VERSION!r}")
    source_hash = str(payload.get("source_evidence_sha256") or "")
    if not HEX_SHA256_RE.match(source_hash):
        raise ValueError("Redacted summary source_evidence_sha256 must be a lowercase SHA-256 digest")

    return {key: payload[key] for key in sorted(keys)}
