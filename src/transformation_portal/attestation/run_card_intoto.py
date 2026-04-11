"""in-toto Statement + DSSE helpers for Lux run-card attestations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from transformation_portal.ingest.canonical_json import canonicalize_json

from .dsse import DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, build_dsse_envelope, decode_dsse_payload, validate_dsse_envelope_surface
from .run_card_detached import RUN_CARD_V2_SCHEMA_URI, compute_run_card_sha256, validate_run_card_v2_surface

IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
RUN_CARD_PREDICATE_TYPE = (
    "https://rc219805.github.io/Transformation_Portal/docs/schemas/attestation/lux-depth-run-card-predicate.v1.schema.json"
)


def _string_field(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _is_sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _subject_digest(name: str, sha256: str) -> dict[str, Any]:
    return {
        "name": name,
        "digest": {
            "sha256": sha256,
        },
    }


def build_run_card_subjects(
    *,
    run_card_path: Path,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
) -> list[dict[str, Any]]:
    """Build the in-toto subject list for the run card and committed artifacts."""
    validate_run_card_v2_surface(run_card_payload)
    subjects = [_subject_digest(run_card_path.name, compute_run_card_sha256(run_card_bytes))]
    artifact_index = run_card_payload.get("artifact_index")
    if not isinstance(artifact_index, Sequence):
        raise ValueError("run_card.artifact_index must be a list")
    for artifact in artifact_index:
        if not isinstance(artifact, Mapping):
            raise ValueError("run_card.artifact_index entries must be objects")
        subjects.append(
            _subject_digest(
                _string_field(artifact.get("relative_path"), field="artifact.relative_path"),
                _string_field(artifact.get("sha256"), field="artifact.sha256"),
            )
        )
    return subjects


def build_run_card_predicate(
    *,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
    release_assessment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact Lux run-card predicate payload."""
    validate_run_card_v2_surface(run_card_payload)
    artifact_tree = run_card_payload["artifact_tree"]
    predicate: dict[str, Any] = {
        "run_card_schema": RUN_CARD_V2_SCHEMA_URI,
        "run_card_sha256": compute_run_card_sha256(run_card_bytes),
        "batch_id": run_card_payload["batch_id"],
        "artifact_tree": {
            "algorithm": artifact_tree["algorithm"],
            "leaf_format": artifact_tree["leaf_format"],
            "leaf_count": artifact_tree["leaf_count"],
            "root_sha256": artifact_tree["root_sha256"],
        },
        "backend_selection": {
            key: value
            for key, value in dict(run_card_payload.get("backend_selection", {})).items()
            if key in {"requested", "resolved", "device", "model_id", "logical_backend", "resolved_engine"}
        },
        "config_fingerprint_sha256": run_card_payload["config_fingerprint"]["sha256"],
        "git_revision": dict(run_card_payload.get("git_revision", {})),
    }
    if release_assessment is not None:
        release_assessment_bytes = canonicalize_json(dict(release_assessment))
        predicate["release_assessment"] = {
            "sha256": hashlib.sha256(release_assessment_bytes).hexdigest(),
            "status": release_assessment.get("status"),
        }
    return predicate


def build_run_card_statement(
    *,
    run_card_path: Path,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
    release_assessment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the in-toto Statement payload for a Lux run card."""
    return {
        "_type": IN_TOTO_STATEMENT_TYPE,
        "subject": build_run_card_subjects(
            run_card_path=run_card_path,
            run_card_payload=run_card_payload,
            run_card_bytes=run_card_bytes,
        ),
        "predicateType": RUN_CARD_PREDICATE_TYPE,
        "predicate": build_run_card_predicate(
            run_card_payload=run_card_payload,
            run_card_bytes=run_card_bytes,
            release_assessment=release_assessment,
        ),
    }


def canonical_run_card_statement_bytes(
    *,
    run_card_path: Path,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
    release_assessment: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize the Statement under the repo canonical JSON profile."""
    return canonicalize_json(
        build_run_card_statement(
            run_card_path=run_card_path,
            run_card_payload=run_card_payload,
            run_card_bytes=run_card_bytes,
            release_assessment=release_assessment,
        )
    )


def build_run_card_dsse_envelope(
    *,
    run_card_path: Path,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
    key_id: str,
    signature_bytes: bytes,
    release_assessment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the DSSE envelope for the run-card Statement."""
    statement_bytes = canonical_run_card_statement_bytes(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
        release_assessment=release_assessment,
    )
    return build_dsse_envelope(
        payload=statement_bytes,
        payload_type=DSSE_IN_TOTO_JSON_PAYLOAD_TYPE,
        key_id=key_id,
        signature_bytes=signature_bytes,
    )


def decode_run_card_statement_from_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Decode and parse the in-toto Statement carried in a DSSE envelope."""
    validate_dsse_envelope_surface(envelope)
    payload_type = envelope.get("payloadType")
    if payload_type != DSSE_IN_TOTO_JSON_PAYLOAD_TYPE:
        raise ValueError(f"DSSE envelope payloadType must be {DSSE_IN_TOTO_JSON_PAYLOAD_TYPE!r}")
    try:
        statement = json.loads(decode_dsse_payload(envelope).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("DSSE envelope payload is not valid JSON") from exc
    if not isinstance(statement, dict):
        raise ValueError("DSSE envelope payload root must be a JSON object")
    return statement


def validate_run_card_statement_binding(
    statement: Mapping[str, Any],
    *,
    run_card_path: Path,
    run_card_payload: Mapping[str, Any],
    run_card_bytes: bytes,
) -> None:
    """Validate that an in-toto Statement binds to the given run card."""
    if statement.get("_type") != IN_TOTO_STATEMENT_TYPE:
        raise ValueError(f"Statement _type must be {IN_TOTO_STATEMENT_TYPE!r}")
    if statement.get("predicateType") != RUN_CARD_PREDICATE_TYPE:
        raise ValueError(f"Statement predicateType must be {RUN_CARD_PREDICATE_TYPE!r}")
    expected_statement = build_run_card_statement(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    if statement.get("subject") != expected_statement["subject"]:
        raise ValueError("Statement does not bind to this run card: subject mismatch")
    predicate = statement.get("predicate")
    if not isinstance(predicate, Mapping):
        raise ValueError("Statement predicate must be an object")
    expected_predicate = expected_statement["predicate"]
    for field in (
        "run_card_schema",
        "run_card_sha256",
        "batch_id",
        "artifact_tree",
        "backend_selection",
        "config_fingerprint_sha256",
        "git_revision",
    ):
        if predicate.get(field) != expected_predicate[field]:
            raise ValueError(f"Statement does not bind to this run card: predicate {field} mismatch")
    if "release_assessment" in predicate:
        release_assessment = predicate["release_assessment"]
        if not isinstance(release_assessment, Mapping):
            raise ValueError("Statement predicate release_assessment must be an object when present")
        sha256_value = release_assessment.get("sha256")
        if not _is_sha256_digest(sha256_value):
            raise ValueError("Statement predicate release_assessment.sha256 must be a sha256 digest")
        if "status" not in release_assessment:
            raise ValueError("Statement predicate release_assessment.status must be present")
        status_value = release_assessment["status"]
        if status_value is not None and not isinstance(status_value, str):
            raise ValueError("Statement predicate release_assessment.status must be a string or null")
