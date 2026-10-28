"""Metadata-only, actor-bound photographic scene pages.

The index is a navigation projection. Its recorded decisions never authorize
review or delivery, and native composite authority requires selected discovery.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from copy import deepcopy
from typing import Any

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewContext, ReviewInputError

JsonObject = dict[str, Any]
SCENE_LIMIT = 64
PAGE_LIMIT = 16
DECISION_FIELDS = ("artifact_path", "sha256", "source_identity_sha256", "status", "note", "actor", "created_at")


def catalogue_changed() -> ReviewAccessError:
    return ReviewAccessError("Photographic catalogue changed; refresh the scene list", code="CATALOGUE_CHANGED")


def recorded_decision(row: JsonObject) -> JsonObject:
    return {key: deepcopy(row[key]) for key in DECISION_FIELDS if key in row}


def scene_snapshot(context: ReviewContext, references: list[JsonObject], state: JsonObject) -> tuple[str, list[JsonObject]]:
    """Hash authorized metadata, never source files or image headers."""
    revisions = [entry["variant"] for entry in state.get("revisions", [])]
    refs = [*references, *revisions]
    scenes: dict[str, JsonObject] = {}
    for ref in refs:
        scene_id = ref["scene_id"]
        if scene_id not in scenes:
            if len(scenes) >= SCENE_LIMIT:
                raise ReviewInputError("Photographic scene navigation is bounded to 64 scenes")
            scenes[scene_id] = {
                "scene_id": scene_id,
                "label": ref.get("label", scene_id),
                "review_status": "unreviewed",
                "recorded_decision": None,
                "source_validation": "not_checked",
                "variant_count": 0,
                "variant_paths": [],
                "related_artifacts": [],
            }
        scene = scenes[scene_id]
        if not ref.get("related_only"):
            scene["variant_count"] += 1
            scene["variant_paths"].append(ref["artifact_path"])
        role = ref.get("summary_role") or ("master" if ref.get("role") != "original" else None)
        if role is not None:
            scene["related_artifacts"].append(
                {
                    "artifact_path": ref["artifact_path"],
                    "role": role,
                    "sha256": ref.get("sha256"),
                    "size_bytes": ref.get("size_bytes"),
                }
            )
    for scene_id, scene in scenes.items():
        variants = [ref for ref in refs if ref["scene_id"] == scene_id and not ref.get("related_only")]
        revised = [ref for ref in revisions if ref["scene_id"] == scene_id]
        candidates = revised[-1:] or [ref for ref in variants if ref.get("role") != "original"]
        decisions = [
            row
            for row in state.get("decisions", [])
            if any(ref["artifact_path"] == row["artifact_path"] and ref.get("sha256") == row["sha256"] for ref in candidates)
        ]
        if decisions:
            decision = max(decisions, key=lambda row: row.get("created_at", 0))
            scene["recorded_decision"] = recorded_decision(decision)
            if decision["status"] in {"approved", "needs_revision"}:
                scene["review_status"] = decision["status"]
    identity = hashlib.sha256(
        canonicalize_json(
            {
                "schema": "tp.photo.catalogue_snapshot.v1",
                "job_id": context.job_id,
                "tenant_id": context.tenant_id,
                "actor": context.actor,
                "binding_identity": context.binding_identity,
                "index": references,
                "review_state": state,
            }
        )
    ).hexdigest()
    return identity, list(scenes.values())


def scene_page(
    context: ReviewContext,
    references: list[JsonObject],
    state: JsonObject,
    secret: bytes,
    session: str,
    *,
    limit: int = 8,
    status: str = "all",
    sort: str = "scene_id",
    cursor: str | None = None,
) -> JsonObject:
    if (
        type(limit) is not int
        or not 1 <= limit <= PAGE_LIMIT
        or status not in {"all", "unreviewed", "needs_revision", "approved"}
    ):
        raise ReviewInputError("Scene page limit or review filter is invalid")
    if sort not in {"scene_id", "label"}:
        raise ReviewInputError("Scene sort must be scene_id or label")
    identity, scenes = scene_snapshot(context, references, state)
    filtered = [scene for scene in scenes if status == "all" or scene["review_status"] == status]
    filtered.sort(key=lambda scene: (scene[sort], scene["scene_id"]))
    offset = 0
    if cursor is not None:
        try:
            if not isinstance(cursor, str) or not 1 <= len(cursor) <= 1024:
                raise ValueError("size")
            encoded, signature = cursor.split(".")
            raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {"session", "identity", "offset", "limit", "status", "sort"}:
                raise ValueError("shape")
            if value["session"] != session:
                raise catalogue_changed()
            expected = hmac.new(secret, raw, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature")
            if (value["identity"], value["limit"], value["status"], value["sort"]) != (identity, limit, status, sort):
                raise catalogue_changed()
            offset = value["offset"]
            if type(offset) is not int or not 0 < offset < len(filtered):
                raise ValueError("offset")
        except ReviewAccessError:
            raise
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ReviewInputError("Scene cursor is invalid; refresh the scene list") from exc
    next_offset = offset + limit
    next_cursor = None
    if next_offset < len(filtered):
        raw = canonicalize_json(
            {"session": session, "identity": identity, "offset": next_offset, "limit": limit, "status": status, "sort": sort}
        )
        next_cursor = (
            base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") + "." + hmac.new(secret, raw, hashlib.sha256).hexdigest()
        )
    sources = [*references, *(entry["variant"] for entry in state.get("revisions", []))]
    current_finals = {
        ref["scene_id"]: ref["artifact_path"]
        for ref in sources
        if not ref.get("related_only") and ref.get("role") not in {"original", "working_master"}
    }
    source_scenes = {}
    for ref in sources:
        if ref.get("related_only"):
            continue
        role = ref.get("role", "final")
        if role not in {"original", "working_master"}:
            role = "final" if current_finals.get(ref["scene_id"]) == ref["artifact_path"] else "previous_revision"
        source_scenes[ref["artifact_path"]] = ref["scene_id"], ref.get("label", ref["scene_id"]), role
    decisions = []
    for row in state.get("decisions", []):
        if row["artifact_path"] in source_scenes:
            scene_id, label, role = source_scenes[row["artifact_path"]]
            decisions.append({**recorded_decision(row), "scene_id": scene_id, "label": label, "role": role})
    return {
        "schema": "tp.photo.scene_page.v1",
        "state_version": state["state_version"],
        "catalogue_identity_sha256": identity,
        "scenes": filtered[offset:next_offset],
        "next_cursor": next_cursor,
        "total_scenes": len(scenes),
        "total_filtered": len(filtered),
        "limit": limit,
        "filter": status,
        "sort": sort,
        "decisions": decisions,
        "groups": deepcopy(state.get("groups", [])),
    }
