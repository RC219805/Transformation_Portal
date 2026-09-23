"""Bounded, fenced publication of complete managed V6 photographic generations."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Mapping

from transformation_portal.core.execution_plan_v5 import ExecutionPlanV5
from transformation_portal.lux_depth_v4.publication import _validate_publication_plan
from transformation_portal.lux_depth_v5.publication import publication_paths as inference_paths
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence


def publication_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Reserve both complete stage namespaces, including optional alpha/metric."""
    paths = ["execution-plan.json", "execution-evidence.json", "v6/plan.json", "v6/evidence.json"]
    paths.extend(f"source-v5/{path}" for path in inference_paths(payload["inference"]))
    for item in payload["inference"]["inputs"]:
        names = [
            "baseline.npy",
            "master.npy",
            "display.npy",
            "alpha.npy",
            "delivery.tif",
            "preview.png",
            "photograph.json",
            "native-depth.npy",
            "native-numeric-valid.npy",
            "native-support.npy",
            "native-sky.npy",
            "relative-depth.npy",
            "depth-valid.npy",
            "depth-support.npy",
            "depth-support-score.npy",
            "depth-relative.tif",
            "depth-preview.png",
            "depth-preview-valid.png",
            "depth.json",
        ]
        if "calibration" in item.get("companions", {}):
            names.append("metric-depth-m.npy")
        paths.extend(f"v6/{item['id']}/{name}" for name in names)
    return tuple(paths)


class _ReservationProfile:
    publication_paths = staticmethod(publication_paths)
    additional_file_bound = 16 * 1024 * 1024 + 4096


def validate_publication_plan(payload: Mapping[str, Any], limits: GenerationPublicationLimits) -> None:
    """Reject unpublishable composite inventories before initializing inference."""
    _validate_publication_plan(
        {**payload, "configuration": payload["inference"]["configuration"]}, limits, profile=_ReservationProfile
    )


def _publication_arguments(plan_bytes: bytes, publisher: GenerationPublisher, fence: DispatchFence) -> dict[str, Any]:
    from .managed import verify_managed_evidence

    plan = ExecutionPlanV5(plan_bytes)
    if hashlib.sha256(plan_bytes).hexdigest() != fence.locator.plan_digest:
        raise ValueError("Managed V6 publication plan differs from its exact dispatch fence")
    payload = plan.to_payload()
    validate_publication_plan(payload, publisher.limits)
    verified = verify_managed_evidence(Path(fence.output_root), expected_plan_bytes=plan_bytes)
    if str(verified.output_root) != fence.output_root:
        raise ValueError("Managed V6 publication root differs from its dispatch fence")
    files = {record.path: verified.output_root / record.path for record in verified.artifacts}
    if not set(files).issubset(publication_paths(payload)):
        raise ValueError("Managed V6 publication inventory exceeds its frozen reservation")
    if any(record.size_bytes > publisher.limits.max_file_bytes for record in verified.artifacts):
        raise ValueError("Managed V6 artifact exceeds publisher per-file limit")
    return {
        "files": files,
        "state": "succeeded",
        "exit_code": 0,
        "artifacts": {"schema": "tp.lux.delivery.v4", "paths": sorted(files), "execution_evidence": "execution-evidence.json"},
        "run_summary": {
            "pipeline": "lux_depth_v6",
            "plan_schema": plan.schema,
            "plan_fingerprint_sha256": plan.plan_fingerprint_sha256,
            "input_count": len(payload["inference"]["inputs"]),
            "production_acceptance": "not_established",
            "photographic_outputs": ["16-bit sRGB TIFF", "draft sRGB PNG"],
            "depth_artifacts": True,
        },
        "expected_file_integrity": {
            record.path: {"size_bytes": record.size_bytes, "sha256": record.sha256} for record in verified.artifacts
        },
    }


async def _publish_admitted_result(
    plan_bytes: bytes, *, publisher: GenerationPublisher, fence: DispatchFence
) -> dict[str, Any]:
    """Verify off the event loop, then let the existing fence commit visibility."""
    arguments = await asyncio.to_thread(_publication_arguments, plan_bytes, publisher, fence)
    return await publisher.publish(fence, **arguments)
