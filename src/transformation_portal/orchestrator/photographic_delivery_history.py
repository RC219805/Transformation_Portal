"""Bounded, redacted delivery history and server-owned export intent identity.

History observes committed metadata only. It cannot establish source-byte or
file availability, and its correlation/fingerprint fields do not grant POST
idempotency or permission to retry an export with a newer review version.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Literal, Sequence
from urllib.parse import quote

from transformation_portal.api.v1.photographic_review import (
    DeliveryHistoryActor,
    DeliveryHistoryDerivative,
    DeliveryHistoryEntry,
    DeliveryHistoryFile,
    DeliveryHistoryIntent,
    DeliveryHistoryPage,
    DeliveryHistoryPreset,
    DeliveryHistoryRender,
    DeliveryHistorySource,
    DeliveryRequest,
)
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.photographic_review_access import ReviewContext, ReviewInputError
from transformation_portal.orchestrator.photographic_review_store import ReviewConflict

INTENT_SCHEMA: Literal["tp.photo.delivery_intent.v1"] = "tp.photo.delivery_intent.v1"
CURSOR_SCHEMA = "tp.photo.delivery_cursor.v1"
APPLIED_SCHEMA: Literal["tp.photo.delivery_applied.v1"] = "tp.photo.delivery_applied.v1"
MAX_DELIVERIES = 64
PRESET_FIELDS = (
    "schema",
    "name",
    "format",
    "intended_use",
    "long_edge",
    "jpeg_quality",
    "sharpen_amount",
    "render",
    "shoulder",
)


def _hash(value: Any) -> str:
    return hashlib.sha256(canonicalize_json(value)).hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["artifact_path"], row["sha256"], row.get("source_identity_sha256", row["sha256"])


def _public_intent(value: dict[str, Any]) -> DeliveryHistoryIntent:
    """Allowlist known semantics while preserving future fields in storage."""
    return DeliveryHistoryIntent(
        property_name=value["property_name"],
        sources=[
            DeliveryHistorySource(
                artifact_path=row["artifact_path"],
                sha256=row["sha256"],
                source_identity_sha256=row["source_identity_sha256"],
                presentation_policy_sha256=row.get("presentation_policy_sha256"),
            )
            for row in value["sources"]
        ],
        scene_labels=value["scene_labels"],
        presets=[DeliveryHistoryPreset.model_validate({key: row[key] for key in PRESET_FIELDS}) for row in value["presets"]],
    )


def applied_derivatives(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Record actual export operations after execution, separate from intent."""
    result = []
    for row in inventory["files"]:
        preset = DeliveryHistoryPreset.model_validate({key: row["recipe"][key] for key in PRESET_FIELDS})
        receipt = row.get("render")
        render = None
        if receipt is not None:
            operation = (
                "preserve_recorded_display" if receipt.get("operation") == "preserve_recorded_display" else "render_sdr"
            )
            render = DeliveryHistoryRender.model_validate(
                {
                    "schema": receipt["schema"],
                    "mode": receipt["recipe"]["mode"],
                    "shoulder": receipt["recipe"]["shoulder"],
                    "operation": operation,
                    "presentation_policy_sha256": receipt.get("presentation_policy_sha256"),
                }
            )
            if (receipt["schema"] == "tp.photo.presentation_render.v1") != (operation == "preserve_recorded_display"):
                raise ValueError("Applied delivery render operation disagrees with its receipt")
        elif preset.format == "source" and row["recipe"].get("companion") is None:
            operation = "byte_preserve"
        elif preset.format == "tiff32" or row["recipe"].get("companion") == "analytic_extended_srgb_transfer_no_clipping":
            operation = "analytic_extended_srgb_transfer"
        else:
            raise ValueError("Applied delivery lacks its render evidence")
        result.append(
            DeliveryHistoryDerivative.model_validate(
                {
                    "source_master_sha256": row["source_master_sha256"],
                    "source_identity_sha256": row["source_identity_sha256"],
                    "presentation_policy_sha256": row.get("presentation_policy_sha256"),
                    "derivative_sha256": row["sha256"],
                    "preset": row["preset"],
                    "width": row["width"],
                    "height": row["height"],
                    "bit_depth": row["bit_depth"],
                    "color_domain": row["color_domain"],
                    "recipe": preset,
                    "operation": operation,
                    "render": render,
                }
            ).model_dump(mode="json", by_alias=True)
        )
    if len(result) > 224:
        raise ValueError("Applied delivery evidence exceeds its source/preset bound")
    return result


def _public_derivative(row: dict[str, Any]) -> DeliveryHistoryDerivative:
    fields = (
        "source_master_sha256",
        "source_identity_sha256",
        "presentation_policy_sha256",
        "derivative_sha256",
        "preset",
        "width",
        "height",
        "bit_depth",
        "color_domain",
        "operation",
    )
    render = row.get("render")
    return DeliveryHistoryDerivative.model_validate(
        {
            **{key: row.get(key) for key in fields},
            "recipe": DeliveryHistoryPreset.model_validate({key: row["recipe"][key] for key in PRESET_FIELDS}),
            "render": (
                DeliveryHistoryRender.model_validate(
                    {key: render.get(key) for key in ("schema", "mode", "shoulder", "operation", "presentation_policy_sha256")}
                )
                if render is not None
                else None
            ),
        }
    )


def delivery_intent(
    context: ReviewContext,
    payload: DeliveryRequest,
    refs: Sequence[dict[str, Any]],
    presets: Sequence[Any],
) -> tuple[dict[str, Any], str]:
    """Freeze the actual ordered sources, recorded presentation and recipes."""
    public = DeliveryHistoryIntent(
        property_name=payload.property_name,
        sources=[
            DeliveryHistorySource(
                artifact_path=ref["artifact_path"],
                sha256=ref["sha256"],
                source_identity_sha256=ref.get("source_identity_sha256", ref["sha256"]),
                presentation_policy_sha256=ref.get("presentation_policy_sha256"),
            )
            for ref in refs
        ],
        scene_labels={
            ref["scene_id"]: payload.scene_labels.get(ref["scene_id"], ref.get("label", ref["scene_id"])) for ref in refs
        },
        presets=[DeliveryHistoryPreset.model_validate(preset.to_payload()) for preset in presets],
    ).model_dump(mode="json", by_alias=True)
    digest = _hash(
        {
            "schema": INTENT_SCHEMA,
            "scope": {"tenant_id": context.tenant_id, "job_id": context.job_id, "actor": context.actor},
            "expected_version": payload.expected_version,
            **public,
        }
    )
    return public, digest


def _entry(context: ReviewContext, state: dict[str, Any], saved: dict[str, Any]) -> DeliveryHistoryEntry:
    """Construct an allowlisted DTO, never return arbitrary persisted objects."""
    approvals = saved.get("approvals", [])
    approved = {_identity(row) for row in state["decisions"] if row["status"] == "approved"}
    recorded = bool(approvals) and {row["sha256"] for row in approvals} == set(saved.get("approved_sha256", []))
    current: Literal["approved", "withdrawn", "unknown"] = (
        "unknown" if not recorded else "approved" if all(_identity(row) in approved for row in approvals) else "withdrawn"
    )
    actor = saved.get("actor", {})
    known_intent = saved.get("request_fingerprint_schema") == INTENT_SCHEMA
    known_applied = saved.get("applied_derivatives_schema") == APPLIED_SCHEMA
    intent = _public_intent(saved["intent"]) if known_intent and saved.get("intent") is not None else None
    identity = saved["id"]
    files = []
    for row in saved.get("files", []):
        filename = row["filename"]
        if not isinstance(filename, str) or any(char in filename for char in ("/", "\\", "\r", "\n")):
            raise ValueError("Invalid saved delivery filename")
        files.append(
            DeliveryHistoryFile(
                filename=filename,
                content_type=row["content_type"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                url=f"/v1/jobs/{quote(context.job_id, safe='')}/photographic-review/files/{quote(identity, safe='')}/{quote(filename, safe='')}",
            )
        )
    return DeliveryHistoryEntry(
        id=identity,
        created_at=saved.get("created_at"),
        actor=DeliveryHistoryActor(**{key: actor.get(key) for key in ("username", "role", "assurance")}),
        approval_version=saved.get("approval_version"),
        request_correlation_id=saved.get("request_correlation_id"),
        request_fingerprint_schema=INTENT_SCHEMA if known_intent else None,
        request_fingerprint_sha256=saved.get("request_fingerprint_sha256") if known_intent else None,
        intent=intent,
        sources=(
            intent.sources
            if intent is not None
            else [
                DeliveryHistorySource(
                    artifact_path=row["artifact_path"],
                    sha256=row["sha256"],
                    source_identity_sha256=row.get("source_identity_sha256", row["sha256"]),
                )
                for row in approvals
            ]
        ),
        applied_derivatives_schema=APPLIED_SCHEMA if known_applied else None,
        applied_derivatives=[_public_derivative(row) for row in saved["applied_derivatives"]] if known_applied else None,
        approval_at_creation="recorded" if recorded else "unknown",
        current_recorded_approval=current,
        files=files,
    )


def history_page(
    context: ReviewContext,
    state: dict[str, Any],
    *,
    limit: int = 20,
    cursor: str | None = None,
    request_correlation_id: str | None = None,
) -> dict[str, Any]:
    """Page an append-only committed prefix; newer deliveries never shift it.

    The cursor is a context-bound validated selector, not an authorization
    token. All requests still cross the normal tenant/job authentication seam.
    Snapshot membership is fixed; recorded approval observations remain live.
    """
    deliveries = state["deliveries"]
    if type(limit) is not int or not 1 <= limit <= MAX_DELIVERIES or len(deliveries) > MAX_DELIVERIES:
        raise ReviewInputError("Delivery history limit must be in [1,64]")
    scope = _hash({"job_id": context.job_id, "tenant_id": context.tenant_id, "actor": context.actor})
    count, offset = len(deliveries), 0
    if cursor is not None:
        try:
            if len(cursor) > 1024:
                raise ValueError("Oversized cursor")
            raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            selection = json.loads(raw)
            if (
                not isinstance(selection, dict)
                or canonicalize_json(selection) != raw
                or set(selection) != {"schema", "scope", "count", "prefix_sha256", "offset", "correlation"}
                or selection["schema"] != CURSOR_SCHEMA
                or selection["scope"] != scope
                or selection["correlation"] != request_correlation_id
                or type(selection["count"]) is not int
                or not 0 <= selection["count"] <= MAX_DELIVERIES
                or type(selection["offset"]) is not int
                or not 0 <= selection["offset"] <= MAX_DELIVERIES
            ):
                raise ValueError("Invalid cursor")
            count, offset = selection["count"], selection["offset"]
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ReviewInputError("Delivery history cursor is invalid for this request") from exc
        if count > len(deliveries) or selection["prefix_sha256"] != _hash([row["id"] for row in deliveries[:count]]):
            raise ReviewConflict("Delivery history snapshot changed; refresh history")
    selected = [
        row
        for row in reversed(deliveries[:count])
        if request_correlation_id is None or row.get("request_correlation_id") == request_correlation_id
    ]
    if offset > len(selected):
        raise ReviewInputError("Delivery history cursor offset exceeds its snapshot")
    end = min(offset + limit, len(selected))
    next_cursor = None
    if end < len(selected):
        next_cursor = (
            base64.urlsafe_b64encode(
                canonicalize_json(
                    {
                        "schema": CURSOR_SCHEMA,
                        "scope": scope,
                        "count": count,
                        "prefix_sha256": _hash([row["id"] for row in deliveries[:count]]),
                        "offset": end,
                        "correlation": request_correlation_id,
                    }
                )
            )
            .decode("ascii")
            .rstrip("=")
        )
    return DeliveryHistoryPage(
        state_version=state["state_version"],
        snapshot_count=len(selected),
        items=[_entry(context, state, row) for row in selected[offset:end]],
        next_cursor=next_cursor,
    ).model_dump(mode="json", by_alias=True)
