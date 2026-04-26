"""Offline APEX model-family characterization report builder."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformation_portal.evals.apex_model_family_schema import (
    ALLOWED_DEPTH_BACKENDS,
    ALLOWED_OBSERVATION_STATUSES,
    ALLOWED_QUALITY_TIERS,
    ALLOWED_SEGMENTATION_BACKENDS,
    BLOCKER_LICENSE_BLOCKED,
    BLOCKER_OBSERVATION_MISSING,
    BLOCKER_SPEC_INVALID,
    COMPARABLE_OBSERVATION_STATUSES,
    FAMILY_FILE_GOVERNANCE_KEYS,
    FAMILY_FILE_TOP_LEVEL_KEYS,
    FAMILY_SPEC_KEYS,
    FAMILY_SPEC_REQUIRED_KEYS,
    GOVERNANCE_COMMERCIAL_READY,
    GOVERNANCE_LICENSE_BLOCKED,
    GOVERNANCE_RESEARCH_ONLY,
    GOVERNANCE_UNKNOWN,
    GROUP_BLOCKER_MEMBER_NOT_COMPARABLE,
    GROUP_BLOCKER_ONLY_ONE_MEMBER,
    INPUT_DIGEST_VERSION,
    MATRIX_SCHEMA_VERSION,
    OBSERVATION_EVIDENCE_MISSING,
    OBSERVATION_KEYS,
    OBSERVATION_MOCKED,
    OBSERVATION_NOT_RUN,
    OBSERVATION_OBSERVED_LOCAL,
    RECONCILIATION_INVARIANTS,
    REPORT_SCHEMA_VERSION,
    SOURCE_MOCK_V1,
    SOURCE_REDACTED_SUMMARY_V1,
    SPEC_STATUS_INVALID,
    SPEC_STATUS_NAME_MISMATCH,
    SPEC_STATUS_OK,
    SUMMARY_SCHEMA_VERSION,
    TOOL_VERSION,
    FamilySpec,
    canonical_family_name,
)
from transformation_portal.evals.apex_redacted_summary_schema import (
    reject_raw_summary_path,
    validate_redacted_summary,
)
from transformation_portal.ingest.canonical_json import canonicalize_json, dump_json

NOW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ApexModelFamilyError(ValueError):
    """Base error for model-family report construction."""


class DuplicateFamilyError(ApexModelFamilyError):
    """Raised when family labels are duplicated."""


class ObservationBindingError(ApexModelFamilyError):
    """Raised when observations cannot be bound to family rows."""


class ObservationValidationError(ApexModelFamilyError):
    """Raised when an observation or redacted summary is invalid."""


class ReconciliationError(ApexModelFamilyError):
    """Raised when report self-checks fail."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using deterministic canonical JSON bytes."""

    return canonicalize_json(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_now(value: str) -> str:
    if not NOW_RE.match(str(value or "")):
        raise ValueError("--now must use UTC format YYYY-MM-DDTHH:MM:SSZ")
    return str(value)


def parse_bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} must be boolean-like, got {value!r}")


def parse_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    return parsed


def parse_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _parse_key_values(value: str, *, label: str, item_separator: str, allowed_keys: frozenset[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_part in str(value or "").split(item_separator):
        part = raw_part.strip()
        if not part:
            continue
        key, separator, raw_value = part.partition("=")
        if separator != "=":
            raise ValueError(f"Invalid {label} spec {value!r}; expected key=value pairs")
        normalized_key = key.strip()
        if normalized_key not in allowed_keys:
            raise ValueError(f"Unsupported {label} field {normalized_key!r}")
        if normalized_key in parsed:
            raise ValueError(f"Duplicate {label} field {normalized_key!r}")
        parsed[normalized_key] = raw_value.strip()
    return parsed


def parse_family_spec(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse a family spec into a normalized report dictionary."""

    if isinstance(value, Mapping):
        raw = {str(key): value[key] for key in value}
        unknown = sorted(set(raw) - FAMILY_SPEC_KEYS)
        if unknown:
            raise ValueError(f"Unsupported family field(s): {', '.join(unknown)}")
    else:
        raw = _parse_key_values(str(value), label="family", item_separator=",", allowed_keys=FAMILY_SPEC_KEYS)

    missing = sorted(FAMILY_SPEC_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"Family spec missing required field(s): {', '.join(missing)}")

    spec = {
        "depth_backend": _normalize_token(raw["depth_backend"]),
        "segmentation_backend": _normalize_token(raw["segmentation_backend"]),
        "materials_version": parse_int(raw.get("materials_version", 3), field="materials_version"),
        "quality_tier": _normalize_token(raw["quality_tier"]),
        "pbr_enabled": parse_bool(raw.get("pbr_enabled", False), field="pbr_enabled"),
        "v2_enabled": parse_bool(raw.get("v2_enabled", False), field="v2_enabled"),
    }
    errors = _spec_errors(spec)
    if not errors:
        expected_name = canonical_family_name(FamilySpec(**spec))
    else:
        expected_name = ""
    provided_name = _normalize_token(raw.get("candidate_family", ""))
    spec["candidate_family"] = provided_name or expected_name
    return spec


def _spec_errors(spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if spec.get("depth_backend") not in ALLOWED_DEPTH_BACKENDS:
        errors.append("unsupported_depth_backend")
    if spec.get("segmentation_backend") not in ALLOWED_SEGMENTATION_BACKENDS:
        errors.append("unsupported_segmentation_backend")
    if spec.get("quality_tier") not in ALLOWED_QUALITY_TIERS:
        errors.append("unsupported_quality_tier")
    if int(spec.get("materials_version", 0)) <= 0:
        errors.append("invalid_materials_version")
    return errors


def spec_validation_for(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors = _spec_errors(spec)
    expected_name = None
    if not errors:
        expected_name = canonical_family_name(spec)
        if spec.get("candidate_family") != expected_name:
            errors.append("candidate_family_name_mismatch")
            return {
                "status": SPEC_STATUS_NAME_MISMATCH,
                "errors": errors,
                "expected_candidate_family": expected_name,
            }
    return {
        "status": SPEC_STATUS_OK if not errors else SPEC_STATUS_INVALID,
        "errors": errors,
        "expected_candidate_family": expected_name,
    }


def parse_matrix_spec(value: str) -> list[dict[str, Any]]:
    raw = _parse_key_values(str(value), label="matrix", item_separator=";", allowed_keys=FAMILY_SPEC_KEYS)
    if "candidate_family" in raw:
        raise ValueError("Matrix specs must not include candidate_family; it is derived per row")
    missing = sorted(FAMILY_SPEC_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"Matrix spec missing required field(s): {', '.join(missing)}")

    dimensions = {
        key: [item.strip() for item in str(raw.get(key, "")).split(",") if item.strip()]
        for key in ("depth_backend", "segmentation_backend", "quality_tier", "materials_version", "pbr_enabled", "v2_enabled")
    }
    dimensions["materials_version"] = dimensions["materials_version"] or ["3"]
    dimensions["pbr_enabled"] = dimensions["pbr_enabled"] or ["false"]
    dimensions["v2_enabled"] = dimensions["v2_enabled"] or ["false"]

    specs = []
    for depth, seg, tier, version, pbr, v2 in product(
        dimensions["depth_backend"],
        dimensions["segmentation_backend"],
        dimensions["quality_tier"],
        dimensions["materials_version"],
        dimensions["pbr_enabled"],
        dimensions["v2_enabled"],
    ):
        specs.append(
            parse_family_spec(
                {
                    "depth_backend": depth,
                    "segmentation_backend": seg,
                    "quality_tier": tier,
                    "materials_version": version,
                    "pbr_enabled": pbr,
                    "v2_enabled": v2,
                }
            )
        )
    return specs


def load_family_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Family file must be a JSON object")
    unknown = sorted(set(payload) - FAMILY_FILE_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Family file contains unsupported top-level field(s): {', '.join(unknown)}")
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(f"Family file schema_version must be {MATRIX_SCHEMA_VERSION!r}")
    default_governance = payload.get("default_governance", {})
    if not isinstance(default_governance, dict):
        raise ValueError("Family file default_governance must be an object")
    unknown_governance = sorted(set(default_governance) - FAMILY_FILE_GOVERNANCE_KEYS)
    if unknown_governance:
        raise ValueError(f"Family file default_governance contains unsupported field(s): {', '.join(unknown_governance)}")
    families = payload.get("families", [])
    if not isinstance(families, list):
        raise ValueError("Family file families must be a list")
    specs = [parse_family_spec(item) for item in families]
    governance = {
        "non_commercial_ok": parse_bool(default_governance.get("non_commercial_ok", False), field="non_commercial_ok"),
        "accept_depth_pro_license": parse_bool(
            default_governance.get("accept_depth_pro_license", False),
            field="accept_depth_pro_license",
        ),
    }
    return specs, governance


def parse_mock_observation(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = {str(key): value[key] for key in value}
        unknown = sorted(set(raw) - OBSERVATION_KEYS)
        if unknown:
            raise ObservationValidationError(f"Unsupported observation field(s): {', '.join(unknown)}")
    else:
        raw = _parse_key_values(str(value), label="observation", item_separator=",", allowed_keys=OBSERVATION_KEYS)
    if "candidate_family" not in raw:
        raise ObservationValidationError("Observation spec missing candidate_family")
    source = str(raw.get("source") or SOURCE_MOCK_V1)
    if source != SOURCE_MOCK_V1:
        raise ObservationValidationError("--observation only accepts source=mock_v1")
    status = _normalize_token(raw.get("status", OBSERVATION_MOCKED))
    if status != OBSERVATION_MOCKED:
        raise ObservationValidationError("source=mock_v1 observations must use status=mocked")
    observation = _default_observation()
    observation.update(
        {
            "status": OBSERVATION_MOCKED,
            "source": SOURCE_MOCK_V1,
            "fallback_used": _optional_bool(raw.get("fallback_used"), field="fallback_used"),
            "runtime_ms": _optional_float(raw.get("runtime_ms"), field="runtime_ms"),
            "promotion_verdict": raw.get("promotion_verdict") or None,
            "metric_contract": raw.get("metric_contract") or None,
            "mask_evidence_status": raw.get("mask_evidence_status") or None,
        }
    )
    return {"candidate_family": _normalize_token(raw["candidate_family"]), "observation": observation}


def parse_redacted_summary_binding(value: str | Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = frozenset({"candidate_family", "path"})
    if isinstance(value, Mapping):
        raw = {str(key): value[key] for key in value}
        unknown = sorted(set(raw) - allowed_keys)
        if unknown:
            raise ObservationValidationError(f"Unsupported redacted-summary field(s): {', '.join(unknown)}")
    else:
        raw = _parse_key_values(str(value), label="redacted-summary", item_separator=",", allowed_keys=allowed_keys)
    if "candidate_family" not in raw or "path" not in raw:
        raise ObservationValidationError("Redacted summary binding requires candidate_family and path")
    path = Path(str(raw["path"]))
    reject_raw_summary_path(path)
    payload = validate_redacted_summary(json.loads(path.read_text(encoding="utf-8")))
    candidate_family = _normalize_token(raw["candidate_family"])
    if payload["candidate_family"] != candidate_family:
        raise ObservationValidationError("Redacted summary candidate_family does not match binding")
    summary_sha256 = sha256_file(path)
    observation = _default_observation()
    observation.update(
        {
            "status": OBSERVATION_OBSERVED_LOCAL,
            "source": SOURCE_REDACTED_SUMMARY_V1,
            "fallback_used": payload.get("fallback_used"),
            "runtime_ms": payload.get("runtime_ms"),
            "promotion_verdict": payload.get("promotion_verdict"),
            "metric_contract": payload.get("metric_contract"),
            "mask_evidence_status": payload.get("mask_evidence_status"),
            "evidence_ref": {
                "summary_sha256": summary_sha256,
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "originating_bundle_schema": "apex_evidence_bundle.v1",
                "source_evidence_sha256": payload["source_evidence_sha256"],
            },
            "summary": {key: payload[key] for key in sorted(payload) if key not in {"schema_version", "candidate_family"}},
        }
    )
    return {"candidate_family": candidate_family, "observation": observation}


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return parse_bool(value, field=field)


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return parse_float(value, field=field)


def _default_observation() -> dict[str, Any]:
    return {
        "status": OBSERVATION_NOT_RUN,
        "source": None,
        "fallback_used": None,
        "runtime_ms": None,
        "promotion_verdict": None,
        "metric_contract": None,
        "mask_evidence_status": None,
    }


def _governance(spec: Mapping[str, Any], *, non_commercial_ok: bool, accept_depth_pro_license: bool) -> dict[str, Any]:
    if spec.get("depth_backend") == "depth_pro":
        acknowledged = bool(non_commercial_ok and accept_depth_pro_license)
        return {
            "status": GOVERNANCE_RESEARCH_ONLY if acknowledged else GOVERNANCE_LICENSE_BLOCKED,
            "license_tier": "research_non_commercial",
            "non_commercial_ok": bool(non_commercial_ok),
            "accept_depth_pro_license": bool(accept_depth_pro_license),
            "blocked_reason": None if acknowledged else "depth_pro_license_acknowledgments_required",
        }
    if spec.get("depth_backend") == "da3":
        return {
            "status": GOVERNANCE_COMMERCIAL_READY,
            "license_tier": "commercial_safe",
            "non_commercial_ok": bool(non_commercial_ok),
            "accept_depth_pro_license": bool(accept_depth_pro_license),
            "blocked_reason": None,
        }
    return {
        "status": GOVERNANCE_UNKNOWN,
        "license_tier": "unknown",
        "non_commercial_ok": bool(non_commercial_ok),
        "accept_depth_pro_license": bool(accept_depth_pro_license),
        "blocked_reason": "unsupported_depth_backend",
    }


def _comparison_blockers(
    spec_validation: Mapping[str, Any], governance: Mapping[str, Any], observation: Mapping[str, Any]
) -> list[str]:
    blockers = []
    if spec_validation.get("status") != SPEC_STATUS_OK:
        blockers.append(BLOCKER_SPEC_INVALID)
    if observation.get("status") not in COMPARABLE_OBSERVATION_STATUSES:
        blockers.append(BLOCKER_OBSERVATION_MISSING)
    if governance.get("status") == GOVERNANCE_LICENSE_BLOCKED:
        blockers.append(BLOCKER_LICENSE_BLOCKED)
    return blockers


def _binding_candidate_family(value: str | Mapping[str, Any], *, label: str) -> str:
    if isinstance(value, Mapping):
        raw = {str(key): value[key] for key in value}
    else:
        raw = _parse_key_values(
            str(value), label=label, item_separator=",", allowed_keys=frozenset({"candidate_family", "path"})
        )
    candidate_family = raw.get("candidate_family")
    if not candidate_family:
        raise ObservationBindingError(f"{label} binding missing candidate_family")
    return _normalize_token(candidate_family)


def _bind_observations(
    family_names: set[str],
    observations: Iterable[str | Mapping[str, Any]] | None,
    redacted_summaries: Iterable[str | Mapping[str, Any]] | None,
    *,
    allow_observation_invalid: bool = False,
) -> dict[str, dict[str, Any]]:
    bound: dict[str, dict[str, Any]] = {}
    for raw in observations or ():
        parsed = parse_mock_observation(raw)
        _bind_one_observation(bound, family_names, parsed)
    for raw in redacted_summaries or ():
        try:
            parsed = parse_redacted_summary_binding(raw)
        except (OSError, ObservationValidationError, ValueError) as exc:
            if not allow_observation_invalid:
                raise ObservationValidationError(str(exc)) from exc
            candidate_family = _binding_candidate_family(raw, label="redacted-summary")
            parsed = {
                "candidate_family": candidate_family,
                "observation": {
                    **_default_observation(),
                    "status": OBSERVATION_EVIDENCE_MISSING,
                    "source": SOURCE_REDACTED_SUMMARY_V1,
                    "error_code": _observation_error_code(exc),
                },
            }
        _bind_one_observation(bound, family_names, parsed)
    return bound


def _bind_one_observation(bound: dict[str, dict[str, Any]], family_names: set[str], parsed: Mapping[str, Any]) -> None:
    candidate_family = str(parsed["candidate_family"])
    if candidate_family not in family_names:
        raise ObservationBindingError(f"Observation references unknown candidate_family {candidate_family!r}")
    if candidate_family in bound:
        raise ObservationBindingError(f"Duplicate observation for candidate_family {candidate_family!r}")
    bound[candidate_family] = dict(parsed["observation"])


def _observation_error_code(exc: BaseException) -> str:
    message = str(exc).lower()
    if "raw artifact" in message:
        return "raw_artifact_rejected"
    if "path-like" in message:
        return "path_like_value_rejected"
    if "missing required" in message:
        return "required_key_missing"
    if "unsupported field" in message:
        return "unsupported_field"
    if "schema_version" in message:
        return "schema_version_invalid"
    return "redacted_summary_invalid"


def _summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    family_count = len(rows)
    return {
        "family_count": family_count,
        "spec_valid_count": sum(1 for row in rows if row["spec_validation"]["status"] == SPEC_STATUS_OK),
        "spec_invalid_count": sum(1 for row in rows if row["spec_validation"]["status"] != SPEC_STATUS_OK),
        "not_run_count": sum(1 for row in rows if row["observation"]["status"] == OBSERVATION_NOT_RUN),
        "mocked_count": sum(1 for row in rows if row["observation"]["status"] == OBSERVATION_MOCKED),
        "observed_local_count": sum(1 for row in rows if row["observation"]["status"] == OBSERVATION_OBSERVED_LOCAL),
        "evidence_missing_count": sum(1 for row in rows if row["observation"]["status"] == OBSERVATION_EVIDENCE_MISSING),
        "commercial_ready_count": sum(1 for row in rows if row["governance"]["status"] == GOVERNANCE_COMMERCIAL_READY),
        "research_only_count": sum(1 for row in rows if row["governance"]["status"] == GOVERNANCE_RESEARCH_ONLY),
        "license_blocked_count": sum(1 for row in rows if row["governance"]["status"] == GOVERNANCE_LICENSE_BLOCKED),
        "governance_unknown_count": sum(1 for row in rows if row["governance"]["status"] == GOVERNANCE_UNKNOWN),
        "comparable_count": sum(1 for row in rows if row["comparable"]),
    }


def _self_check(rows: list[dict[str, Any]], summary: Mapping[str, int]) -> dict[str, Any]:
    failures = []
    family_count = int(summary["family_count"])
    checks = {
        "family_count_matches_rows": family_count == len(rows),
        "spec_counts_partition_family_count": summary["spec_valid_count"] + summary["spec_invalid_count"] == family_count,
        "observation_counts_partition_family_count": (
            summary["not_run_count"]
            + summary["mocked_count"]
            + summary["observed_local_count"]
            + summary["evidence_missing_count"]
            == family_count
        ),
        "governance_counts_partition_family_count": (
            summary["commercial_ready_count"]
            + summary["research_only_count"]
            + summary["license_blocked_count"]
            + summary["governance_unknown_count"]
            == family_count
        ),
        "comparable_count_matches_rule": summary["comparable_count"] == sum(1 for row in rows if row["comparable"]),
    }
    for invariant in RECONCILIATION_INVARIANTS:
        if not checks[invariant]:
            failures.append({"invariant": invariant})
    return {"status": "ok" if not failures else "failed", "failures": failures}


def _comparison_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes = {
        "segmentation_axis": ("materials_version", "depth_backend", "quality_tier", "pbr_enabled", "v2_enabled"),
        "depth_axis": ("materials_version", "segmentation_backend", "quality_tier", "pbr_enabled", "v2_enabled"),
        "pbr_axis": ("materials_version", "depth_backend", "segmentation_backend", "quality_tier", "v2_enabled"),
        "v2_axis": ("materials_version", "depth_backend", "segmentation_backend", "quality_tier", "pbr_enabled"),
        "tier_axis": ("materials_version", "depth_backend", "segmentation_backend", "pbr_enabled", "v2_enabled"),
    }
    groups = []
    for axis, fixed_fields in axes.items():
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            spec = row["family_spec"]
            key = tuple(spec[field] for field in fixed_fields)
            buckets.setdefault(key, []).append(row)
        for key, members in sorted(buckets.items(), key=lambda item: tuple(str(part) for part in item[0])):
            member_names = sorted(row["family_spec"]["candidate_family"] for row in members)
            blockers = []
            if len(members) < 2:
                blockers.append(GROUP_BLOCKER_ONLY_ONE_MEMBER)
            if any(not row["comparable"] for row in members):
                blockers.append(GROUP_BLOCKER_MEMBER_NOT_COMPARABLE)
            groups.append(
                {
                    "axis": axis,
                    "fixed": {field: value for field, value in zip(fixed_fields, key)},
                    "members": member_names,
                    "comparable": not blockers,
                    "blocking_reasons": blockers,
                }
            )
    return groups


def collect_family_specs(
    *,
    families: Iterable[str | Mapping[str, Any]] | None = None,
    matrices: Iterable[str] | None = None,
    family_files: Iterable[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    specs = [parse_family_spec(item) for item in families or ()]
    for matrix in matrices or ():
        specs.extend(parse_matrix_spec(matrix))
    file_governance = {"non_commercial_ok": False, "accept_depth_pro_license": False}
    for family_file in family_files or ():
        file_specs, governance = load_family_file(Path(family_file))
        specs.extend(file_specs)
        file_governance.update(governance)
    return specs, file_governance


def _input_digest_payload(
    *,
    specs: list[dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
    non_commercial_ok: bool,
    accept_depth_pro_license: bool,
    output_format: str,
) -> dict[str, Any]:
    observation_bindings = []
    for candidate_family, observation in sorted(observations.items()):
        evidence_ref = observation.get("evidence_ref") if isinstance(observation.get("evidence_ref"), dict) else {}
        observation_bindings.append(
            {
                "candidate_family": candidate_family,
                "status": observation.get("status"),
                "source": observation.get("source"),
                "summary_sha256": evidence_ref.get("summary_sha256"),
            }
        )
    return {
        "schema_version": INPUT_DIGEST_VERSION,
        "tool_version": TOOL_VERSION,
        "families": sorted(specs, key=lambda item: item.get("candidate_family", "")),
        "observation_bindings": observation_bindings,
        "governance": {
            "non_commercial_ok": bool(non_commercial_ok),
            "accept_depth_pro_license": bool(accept_depth_pro_license),
        },
        "format": output_format,
    }


def build_apex_model_family_characterization_report(
    *,
    family_specs: Iterable[str | Mapping[str, Any]],
    output_path: Path | str,
    observations: Iterable[str | Mapping[str, Any]] | None = None,
    redacted_summaries: Iterable[str | Mapping[str, Any]] | None = None,
    non_commercial_ok: bool = False,
    accept_depth_pro_license: bool = False,
    output_format: str = "json",
    created_at: str = "1970-01-01T00:00:00Z",
    allow_observation_invalid: bool = False,
) -> dict[str, Any]:
    """Build and persist an offline APEX model-family characterization report."""

    created_at = validate_now(created_at)
    specs = [parse_family_spec(spec) for spec in family_specs]
    if not specs:
        raise ValueError("At least one family spec is required")
    family_names = [str(spec.get("candidate_family") or "") for spec in specs]
    duplicate_names = sorted(name for name, count in Counter(family_names).items() if count > 1)
    if duplicate_names:
        raise DuplicateFamilyError(f"Duplicate candidate_family value(s): {', '.join(duplicate_names)}")

    observations_by_family = _bind_observations(
        set(family_names),
        observations,
        redacted_summaries,
        allow_observation_invalid=allow_observation_invalid,
    )
    rows = []
    for spec in specs:
        spec_validation = spec_validation_for(spec)
        governance = _governance(
            spec,
            non_commercial_ok=non_commercial_ok,
            accept_depth_pro_license=accept_depth_pro_license,
        )
        observation = observations_by_family.get(spec["candidate_family"], _default_observation())
        blockers = _comparison_blockers(spec_validation, governance, observation)
        rows.append(
            {
                "family_spec": dict(spec),
                "spec_validation": spec_validation,
                "governance": governance,
                "observation": observation,
                "comparable": not blockers,
                "comparison_blockers": blockers,
            }
        )
    rows.sort(key=lambda row: row["family_spec"]["candidate_family"])
    summary = _summary(rows)
    self_check = _self_check(rows, summary)
    digest_payload = _input_digest_payload(
        specs=specs,
        observations=observations_by_family,
        non_commercial_ok=non_commercial_ok,
        accept_depth_pro_license=accept_depth_pro_license,
        output_format=output_format,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "report_mode": "offline",
        "created_at": created_at,
        "input_digest": {"schema_version": INPUT_DIGEST_VERSION, "sha256": sha256_bytes(canonical_json_bytes(digest_payload))},
        "families": rows,
        "summary": summary,
        "comparison_groups": _comparison_groups(rows),
        "self_check": self_check,
        "notes": [
            "This report does not run models.",
            "This report does not replace apex_evidence_bundle.v1.",
            "Real artifacts remain external and uncommitted.",
        ],
    }
    if self_check["status"] != "ok":
        raise ReconciliationError("APEX model-family report self-check failed")
    write_report(report, Path(output_path), output_format=output_format)
    return report


def write_report(report: Mapping[str, Any], output_path: Path, *, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        with output_path.open("w", encoding="utf-8") as handle:
            dump_json(report, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    elif output_format == "markdown":
        output_path.write_text(render_markdown(report), encoding="utf-8")
    else:
        raise ValueError("output_format must be json or markdown")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# APEX Model-Family Characterization",
        "",
        f"schema_version: {report['schema_version']}",
        f"created_at: {report['created_at']}",
        "",
        "| candidate_family | depth_backend | segmentation_backend | quality_tier | pbr_enabled | v2_enabled | governance.status | observation.status | comparable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(report.get("families", []), key=lambda item: item["family_spec"]["candidate_family"]):
        spec = row["family_spec"]
        lines.append(
            "| {candidate_family} | {depth_backend} | {segmentation_backend} | {quality_tier} | {pbr_enabled} | {v2_enabled} | {governance_status} | {observation_status} | {comparable} |".format(
                candidate_family=spec["candidate_family"],
                depth_backend=spec["depth_backend"],
                segmentation_backend=spec["segmentation_backend"],
                quality_tier=spec["quality_tier"],
                pbr_enabled=str(spec["pbr_enabled"]).lower(),
                v2_enabled=str(spec["v2_enabled"]).lower(),
                governance_status=row["governance"]["status"],
                observation_status=row["observation"]["status"],
                comparable=str(row["comparable"]).lower(),
            )
        )
    return "\n".join(line.rstrip() for line in lines) + "\n"
