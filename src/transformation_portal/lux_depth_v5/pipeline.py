"""V5 depth semantics on V4's governed discovery, executor, and resource lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from transformation_portal.core.depth_evidence import DepthEvidence, build_depth_evidence
from transformation_portal.core.execution_identity_v5 import materialize_stage_identity
from transformation_portal.core.execution_plan_v4 import parse_plan_v4
from transformation_portal.lux_depth_v4.pipeline import LuxDepthV4Result, _run
from transformation_portal.stage_graph.stage import StageContext

from .backend import DA3Session
from .lifecycle import PreparedLuxExecutionV5, authorize_model, validate_prepared_bindings
from .photography import AlignedDepth, align_depth, enhance_master_v5

if TYPE_CHECKING:
    from transformation_portal.orchestrator.artifact_store.generation import GenerationPublicationLimits, GenerationPublisher


@dataclass(frozen=True)
class LuxDepthV5Result(LuxDepthV4Result):
    """Completed candidate; managed visibility still requires verified publication."""


def read_depth(
    context: StageContext, source_sha256: str, companion: Mapping[str, Any] | None, configuration: dict
) -> DepthEvidence:
    wire = context.artifacts["depth.depth"]
    if not isinstance(wire, dict) or set(wire) != {"native_depth", "sky_mask", "sky_available", "precision", "recipe"}:
        raise ValueError("Invalid V5 cached native evidence")
    if (
        type(wire["sky_available"]) is not bool
        or wire["precision"] != configuration["depth"]["precision"]
        or wire["recipe"] != "tp.da3.explicit_precision_sky.v1"
    ):
        raise ValueError("Cached depth recipe differs from the admitted inference")
    if wire["sky_available"] != (wire["sky_mask"] is not None):
        raise ValueError("Cached sky availability disagrees with its carrier")
    return build_depth_evidence(
        wire["native_depth"],
        wire["sky_mask"],
        context.artifacts["preprocess.proxy"],
        source_sha256,
        companion=companion,
        precision=wire["precision"],
    )


def _align(context: StageContext, evidence: DepthEvidence, configuration: dict) -> AlignedDepth:
    return align_depth(
        evidence,
        context.artifacts["preprocess.master"],
        context.artifacts["preprocess.proxy"],
        refinement=configuration["depth"]["refinement"],
    )


class _ExecutionProfile:
    cache_namespace = "identity-v5"
    evidence_schema = "tp.lux.execution.evidence.v3"
    failure_schema = "tp.lux.execution.failure.v3"
    result_type = LuxDepthV5Result
    session_type = DA3Session
    parse_plan = staticmethod(parse_plan_v4)
    validate_prepared_bindings = staticmethod(validate_prepared_bindings)
    authorize_model = staticmethod(authorize_model)
    read_depth = staticmethod(read_depth)
    identity = staticmethod(materialize_stage_identity)

    @staticmethod
    def browser_preview(master: Any, input_id: str) -> tuple[bytes, dict[str, Any]]:
        from .preview import encode_preview

        return encode_preview(master, relative_path=f"{input_id}/preview.png")

    @staticmethod
    def validate_publication_plan(payload: dict, limits: Any) -> None:
        from .publication import validate_publication_plan

        return validate_publication_plan(payload, limits)

    @staticmethod
    def depth_wire(context: StageContext, session: DA3Session, configuration: dict) -> dict:
        arrays, response = session.compute(context.artifacts["preprocess.proxy"].pixels)
        if (
            response.get("native_semantics") != "da3_metric_uncalibrated"
            or response.get("precision") != configuration["depth"]["precision"]
            or response.get("inference_recipe") != "tp.da3.explicit_precision_sky.v1"
        ):
            raise RuntimeError("Worker returned an unauthorized depth recipe")
        return {
            "depth.depth": {
                "native_depth": arrays["native_depth"],
                "sky_mask": arrays.get("sky_mask"),
                "sky_available": response["sky_available"],
                "precision": response["precision"],
                "recipe": response["inference_recipe"],
            }
        }

    @staticmethod
    def enhance(context: StageContext, evidence: DepthEvidence, configuration: dict) -> dict[str, Any]:
        aligned = _align(context, evidence, configuration)
        master, response = enhance_master_v5(
            context.artifacts["preprocess.master"],
            aligned,
            strength=configuration["strength"],
            clarity=configuration["clarity"],
        )
        return {
            "enhance.master": master,
            "enhance.depth_baseline": master,
            "enhance.depth_response": response,
            "enhance.aligned": aligned,
        }

    @staticmethod
    def output_products(
        context: StageContext, evidence: DepthEvidence, configuration: dict, input_id: str
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        aligned = context.artifacts["enhance.aligned"]
        if type(aligned) is not AlignedDepth or aligned.evidence_content_hash != evidence.content_hash():
            raise ValueError("Output alignment differs from the bound native evidence")
        arrays = {
            "native-numeric-valid.npy": evidence.numeric_valid,
            "native-support.npy": evidence.support_mask,
            "native-sky.npy": np.zeros(evidence.shape, dtype=bool) if evidence.sky_mask is None else evidence.sky_mask,
            "depth-baseline.npy": context.artifacts["enhance.depth_baseline"].pixels,
            "relative-depth.npy": aligned.relative_depth,
            "aligned-depth-valid.npy": aligned.valid_mask,
            "aligned-depth-support.npy": aligned.support_mask,
            "depth-support-score.npy": aligned.support_confidence,
        }
        source_icc = context.artifacts["preprocess.master"].source_icc
        if source_icc is not None:
            if len(source_icc) > 16 * 1024 * 1024:
                raise ValueError("Source ICC profile exceeds V5 evidence byte bound")
            arrays["source-icc.npy"] = np.frombuffer(source_icc, dtype=np.uint8)
        if aligned.metric_map_m is not None:
            arrays["aligned-metric-depth-m.npy"] = aligned.metric_map_m
        descriptor = {
            "schema": "tp.lux.photograph.v3" if "browser_preview" in configuration else "tp.lux.photograph.v2",
            "aligned_depth": {
                "evidence": aligned.to_payload(),
                "content_sha256": aligned.content_hash(),
                "relative_path": f"{input_id}/relative-depth.npy",
                "validity_path": f"{input_id}/aligned-depth-valid.npy",
                "support_path": f"{input_id}/aligned-depth-support.npy",
                "support_score_path": f"{input_id}/depth-support-score.npy",
                "metric_path": f"{input_id}/aligned-metric-depth-m.npy" if aligned.metric_map_m is not None else None,
            },
            "depth_baseline": context.artifacts["enhance.depth_baseline"].to_payload(),
            "depth_response": context.artifacts["enhance.depth_response"],
        }
        return arrays, descriptor

    @staticmethod
    def preview_maps(evidence: DepthEvidence) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        from .photography import generate_preview_maps_v5

        return generate_preview_maps_v5(evidence)


def run(
    prepared: PreparedLuxExecutionV5,
    *,
    cancellation: Callable[[], bool] | None = None,
    publisher: GenerationPublisher | None = None,
    publication_limits: GenerationPublicationLimits | None = None,
    managed_process_group: bool = False,
) -> LuxDepthV5Result:
    validate_prepared_bindings(prepared)
    return _run(
        prepared,
        cancellation=cancellation,
        publisher=publisher,
        profile=_ExecutionProfile,
        publication_limits=publication_limits,
        managed_process_group=managed_process_group,
    )
