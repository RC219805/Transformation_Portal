"""Fenced adapter for completed V4 artifacts; no admission or scheduling authority."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from transformation_portal.core.execution_plan_v3 import parse_photography_plan
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.evidence import MAX_EVIDENCE_BYTES, verify_execution_evidence_v2
from transformation_portal.lux_depth_v4.io import directory_path, snapshot
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result


def publication_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Reserve the complete possible inventory without decoding input pixels.

    Alpha and backend confidence are unknown during resolution, so both reserve
    a slot. Calibration and preview outputs follow the admitted plan exactly.
    """
    paths = ["execution-plan.json", "execution-evidence.json"]
    for item in payload["inputs"]:
        names = [
            "source-master.npy",
            "master.npy",
            "native-depth.npy",
            "depth-valid.npy",
            "relative-depth.npy",
            "aligned-depth-valid.npy",
            "alpha.npy",
            "depth-confidence.npy",
            "delivery.tif",
            "photograph.json",
        ]
        if "calibration" in item.get("companions", {}):
            names.extend(("metric-depth-m.npy", "aligned-metric-depth-m.npy"))
        if "materials_v4" in payload["configuration"]:
            names.append("materials-baseline.npy")
        if payload["configuration"]["preview_maps"]:
            names.extend(("preview-normal.npy", "preview-roughness.npy", "preview-ao.npy"))
        paths.extend(f"{item['id']}/{name}" for name in names)
    return tuple(paths)


def validate_publication_plan(payload: Mapping[str, Any], limits: GenerationPublicationLimits) -> None:
    """Reject unpublishable managed work before backend initialization.

    Limits are part of the canonical plan, not mutable execution configuration.
    A changed publisher requires fresh preparation and admission.
    """
    _validate_publication_plan(payload, limits)


def _validate_publication_plan(
    payload: Mapping[str, Any], limits: GenerationPublicationLimits, *, profile: Any = None
) -> None:
    """Reuse bounded admission for private versioned photographic inventories."""
    carried = payload.get("publication")
    if carried is None:
        raise ValueError("Managed publication requires prepare(request, publisher=publisher)")
    if GenerationPublicationLimits.from_payload(carried) != limits:
        raise ValueError("Publisher limits changed after managed preparation; prepare and admit a new plan")
    resources = payload["resources"]
    if resources["max_output_bytes"] > limits.max_total_bytes:
        raise ValueError("Managed output budget exceeds publisher total-byte limit")
    paths = (publication_paths if profile is None else profile.publication_paths)(payload)
    if len(paths) > limits.max_files:
        raise ValueError(f"Managed batch reserves {len(paths)} artifacts, exceeding publisher limit {limits.max_files}")
    # Every array is at most RGB float32 on the master or padded proxy grid.
    # TIFF reserves RGBA uint16 plus metadata; JSON is bounded by its verifier.
    proxy_edge = ((payload["configuration"]["target_size"] + 13) // 14) * 14
    maximum_file = min(
        resources["max_output_bytes"],
        max(
            resources["max_pixels"] * 12 + 256,
            proxy_edge * proxy_edge * 12 + 256,
            resources["max_pixels"] * 8 + 1024 * 1024,
            MAX_EVIDENCE_BYTES,
            0 if profile is None else profile.additional_file_bound,
        ),
    )
    if maximum_file > limits.max_file_bytes:
        raise ValueError("Managed per-file bound exceeds publisher file-byte limit; reduce max_pixels or output budget")
    # Dispatch IDs are bounded to 64 ASCII characters; generation IDs to 32 hex
    # characters. Reserve the publisher's full 256-character MIME field bound.
    manifest = {
        "schema": "tp.artifact.generation.v1",
        "job_id": "j" * 64,
        "tenant_id": "t" * 64,
        "generation_id": "g" * 32,
        "files": [
            {
                "path": path,
                "storage_path": f"generations/{'g' * 32}/{path}",
                "size_bytes": maximum_file,
                "sha256": "f" * 64,
                "content_type": "\u0000" * 256,
            }
            for path in paths
        ],
    }
    if len(canonicalize_json(manifest)) > limits.max_manifest_bytes:
        raise ValueError("Managed artifact inventory exceeds publisher manifest-byte limit")


async def publish_result(result: LuxDepthV4Result, *, publisher: GenerationPublisher, fence: DispatchFence) -> dict[str, Any]:
    """Publish only through an already admitted, plan-bound dispatch fence.

    GenerationPublisher rechecks expected hashes on its own staged snapshots;
    the existing record store remains the sole lease/visibility commit authority.
    """
    return await _publish_result(result, publisher=publisher, fence=fence)


async def _publish_result(
    result: LuxDepthV4Result, *, publisher: GenerationPublisher, fence: DispatchFence, profile: Any = None
) -> dict[str, Any]:
    """Versioned verification/reservation reuse the same dispatch visibility fence."""
    arguments = await asyncio.to_thread(_prepare_publication, result, publisher=publisher, fence=fence, profile=profile)
    return await publisher.publish(fence, **arguments)


def _prepare_publication(
    result: LuxDepthV4Result, *, publisher: GenerationPublisher, fence: DispatchFence, profile: Any = None
) -> dict[str, Any]:
    """Keep bounded file verification and numerical reconstruction off the event loop."""
    root = directory_path(result.output_root)
    if str(root) != fence.output_root or Path(result.evidence_path) != root / "execution-evidence.json":
        raise ValueError("Completed V4 output root does not match the dispatch fence")
    verifier = verify_execution_evidence_v2 if profile is None else profile.verify_evidence
    verified = verifier(root, expected_plan_sha256=result.plan_fingerprint_sha256)
    evidence = verified.to_payload()
    inventory = {record.path: root / record.path for record in verified.artifacts}
    if len(result.artifact_paths) != len(inventory) or set(result.artifact_paths) != set(inventory):
        raise ValueError("V4 result artifact inventory differs from verified completion")
    if (
        result.input_count != len(evidence["inputs"])
        or result.depth_cache_hits != evidence["cache"]["hits"]
        or result.depth_cache_misses != evidence["cache"]["misses"]
    ):
        raise ValueError("V4 result summary differs from verified completion")
    plan_bytes, plan_snapshot = snapshot(root, root / "execution-plan.json", maximum_bytes=MAX_EVIDENCE_BYTES)
    # Dispatch binds the complete admitted bytes, including the fingerprint
    # field; evidence separately binds the semantic fingerprint within them.
    # Bind the verified record too: it becomes the publisher's staging hash.
    verified_plan = next(record for record in verified.artifacts if record.path == "execution-plan.json")
    if plan_snapshot["sha256"] != fence.locator.plan_digest or verified_plan.sha256 != fence.locator.plan_digest:
        raise ValueError("Completed photographic plan bytes do not match the admitted dispatch plan")
    plan = (parse_photography_plan if profile is None else profile.parse_plan)(plan_bytes)
    if plan.plan_fingerprint_sha256 != result.plan_fingerprint_sha256:
        raise ValueError("Managed publication plan changed after completion verification")
    payload = plan.to_payload()
    limits = publisher.limits
    (validate_publication_plan if profile is None else profile.validate_publication_plan)(payload, limits)
    reserved = (publication_paths if profile is None else profile.publication_paths)(payload)
    if not set(inventory).issubset(reserved):
        raise ValueError("Managed output inventory exceeds the prepared publication reservation")
    if any(record.size_bytes > limits.max_file_bytes for record in verified.artifacts):
        raise ValueError("Managed output exceeds publisher file-byte limit")
    expected = {record.path: {"size_bytes": record.size_bytes, "sha256": record.sha256} for record in verified.artifacts}
    return {
        "files": inventory,
        "state": "succeeded",
        "exit_code": 0,
        "artifacts": {
            "schema": "tp.lux.delivery.v2" if profile is None else profile.delivery_schema,
            "paths": list(inventory),
            "execution_evidence": "execution-evidence.json",
        },
        "run_summary": {
            "pipeline": "lux_depth_v4" if profile is None else profile.pipeline,
            "plan_schema": plan.schema,
            "plan_fingerprint_sha256": result.plan_fingerprint_sha256,
            "input_count": result.input_count,
            "production_acceptance": "pending",
        },
        "expected_file_integrity": expected,
    }
