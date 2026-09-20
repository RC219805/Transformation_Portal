"""Fenced adapter for completed V4 artifacts; no admission or scheduling authority."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformation_portal.lux_depth_v4.evidence import verify_execution_evidence_v2
from transformation_portal.lux_depth_v4.io import directory_path
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result


async def publish_result(result: LuxDepthV4Result, *, publisher: GenerationPublisher, fence: DispatchFence) -> dict[str, Any]:
    """Publish only through an already admitted, plan-bound dispatch fence.

    GenerationPublisher rechecks expected hashes on its own staged snapshots;
    the existing record store remains the sole lease/visibility commit authority.
    """
    root = directory_path(result.output_root)
    if str(root) != fence.output_root or Path(result.evidence_path) != root / "execution-evidence.json":
        raise ValueError("Completed V4 output root does not match the dispatch fence")
    if result.plan_fingerprint_sha256 != fence.locator.plan_digest:
        raise ValueError("Completed V4 plan does not match the admitted dispatch plan")
    verified = verify_execution_evidence_v2(root, expected_plan_sha256=result.plan_fingerprint_sha256)
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
    expected = {record.path: {"size_bytes": record.size_bytes, "sha256": record.sha256} for record in verified.artifacts}
    return await publisher.publish(
        fence,
        inventory,
        state="succeeded",
        exit_code=0,
        artifacts={"schema": "tp.lux.delivery.v2", "paths": list(inventory), "execution_evidence": "execution-evidence.json"},
        run_summary={
            "pipeline": "lux_depth_v4",
            "plan_schema": "tp.execution.plan.v2",
            "plan_fingerprint_sha256": result.plan_fingerprint_sha256,
            "input_count": result.input_count,
            "production_acceptance": "pending",
        },
        expected_file_integrity=expected,
    )
