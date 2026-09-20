"""V5 reservations and semantic verification on the existing publication fence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from transformation_portal.core.execution_plan_v4 import parse_plan_v4
from transformation_portal.lux_depth_v4.publication import _publish_result, _validate_publication_plan
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence

if TYPE_CHECKING:
    from transformation_portal.lux_depth_v4.evidence import VerifiedExecutionEvidenceV2
    from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result


def publication_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Reserve optional ICC/alpha and all admitted derivatives before inference."""
    paths = ["execution-plan.json", "execution-evidence.json"]
    for item in payload["inputs"]:
        names = [
            "source-master.npy",
            "master.npy",
            "native-depth.npy",
            "depth-valid.npy",
            "native-numeric-valid.npy",
            "native-support.npy",
            "native-sky.npy",
            "relative-depth.npy",
            "aligned-depth-valid.npy",
            "aligned-depth-support.npy",
            "depth-support-score.npy",
            "depth-baseline.npy",
            "delivery.tif",
            "photograph.json",
            "alpha.npy",
            "source-icc.npy",
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
    _validate_publication_plan(payload, limits, profile=_PublicationProfile)


class _PublicationProfile:
    pipeline = "lux_depth_v5"
    delivery_schema = "tp.lux.delivery.v3"
    additional_file_bound = 16 * 1024 * 1024 + 4096
    publication_paths = staticmethod(publication_paths)
    parse_plan = staticmethod(parse_plan_v4)
    validate_publication_plan = staticmethod(validate_publication_plan)

    @staticmethod
    def verify_evidence(root: Path, *, expected_plan_sha256: str) -> VerifiedExecutionEvidenceV2:
        from .evidence import verify_execution_evidence_v3

        return verify_execution_evidence_v3(root, expected_plan_sha256=expected_plan_sha256)


async def publish_result(result: LuxDepthV4Result, *, publisher: GenerationPublisher, fence: DispatchFence) -> dict[str, Any]:
    """Publish only semantically verified artifacts under the already admitted fence."""
    return await _publish_result(result, publisher=publisher, fence=fence, profile=_PublicationProfile)
