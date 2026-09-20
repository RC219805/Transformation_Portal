"""Closed LuxDepthV5 plans; older photography plans remain independently executable."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

import jsonschema

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import _validate_json_values, digest_payload, photography_nodes
from transformation_portal.core.execution_plan_v3 import materials_photography_nodes, parse_photography_plan
from transformation_portal.ingest.canonical_json import canonicalize_json

PLAN_SCHEMA = "tp.execution.plan.v4"


def depth_photography_nodes(configuration: Mapping[str, Any], *, companions: bool = False) -> list[dict[str, Any]]:
    """Bind raw inference separately from calibrated and photographic derivatives."""
    factory = materials_photography_nodes if "materials_v4" in configuration else photography_nodes
    nodes = factory(configuration, companions=companions)
    # Calibration companions must never activate the legacy material compositor.
    nodes[0]["inputs"].pop("materials", None)
    nodes[2]["inputs"].pop("materials", None)
    if "materials_v4" not in configuration:
        nodes[2]["outputs"].pop("materials", None)
        nodes[3]["inputs"].pop("materials", None)
    nodes[1].update(stage="tp.stage.lux.depth.v3", inputs={"proxy": "preprocess.proxy"})
    nodes[1]["outputs"] = {"depth": "tp.depth.native.v1"}
    nodes[1]["configuration"] = {
        "native_semantics": "da3_metric_uncalibrated",
        "synthetic_fallback": False,
        "precision": configuration["depth"]["precision"],
        "sky_mask_policy": "da3_mono_sky_ge_0_3",
    }
    nodes[2]["stage"] = "tp.stage.lux.enhance.v3"
    nodes[2]["configuration"]["depth"] = copy.deepcopy(configuration["depth"])
    nodes[2]["outputs"].update(
        depth_baseline="tp.image.master.v1", depth_response="tp.depth.response.v1", aligned="tp.depth.aligned.v1"
    )
    nodes[3]["stage"] = "tp.stage.lux.output.v3"
    nodes[3]["inputs"].update(
        depth_baseline="enhance.depth_baseline", depth_response="enhance.depth_response", aligned="enhance.aligned"
    )
    nodes[3]["configuration"]["depth"] = copy.deepcopy(configuration["depth"])
    if companions:
        nodes[2]["inputs"]["calibration"] = "$calibration"
        nodes[3]["inputs"]["calibration"] = "$calibration"
    return nodes


def legacy_validation_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Structural reuse of existing validators; this is never execution authority."""
    projected = copy.deepcopy(dict(payload))
    materials = "materials_v4" in projected["configuration"]
    projected["schema"] = "tp.execution.plan.v3" if materials else "tp.execution.plan.v2"
    projected["pipeline"] = "lux_depth_v4"
    projected["configuration"].pop("depth")
    factory = materials_photography_nodes if materials else photography_nodes
    projected["nodes"] = factory(
        projected["configuration"], companions=any("companions" in item for item in projected["inputs"])
    )
    projected.pop("plan_fingerprint_sha256", None)
    projected["plan_fingerprint_sha256"] = digest_payload(projected)
    return projected


def validate_plan_v4(payload: Mapping[str, Any]) -> None:
    _validate_json_values(payload)
    try:
        decode_bounded_json_object(canonicalize_json(payload))
        schema = decode_bounded_json_object(
            files("transformation_portal.schemas.execution").joinpath("plan.v4.schema.json").read_bytes()
        )
        jsonschema.Draft202012Validator(schema).validate(payload)
        if any("materials" in item.get("companions", {}) for item in payload["inputs"]):
            raise ValueError("V5 requires Materials V4 evidence, not legacy companion material masks")
        parse_photography_plan(canonicalize_json(legacy_validation_projection(payload)))
    except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
        raise ExecutionPlanError(f"Invalid execution plan v4: {exc}") from exc
    if payload["nodes"] != depth_photography_nodes(
        payload["configuration"], companions=any("companions" in item for item in payload["inputs"])
    ):
        raise ExecutionPlanError("V5 nodes differ from the closed depth evidence graph")
    unsigned = dict(payload)
    if unsigned.pop("plan_fingerprint_sha256") != digest_payload(unsigned):
        raise ExecutionPlanError("Execution plan v4 fingerprint mismatch")


@dataclass(frozen=True)
class ExecutionPlanV4:
    """Canonical immutable carrier for the opt-in depth successor."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes:
            raise ExecutionPlanError("Plan carrier must be bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        validate_plan_v4(payload)
        if canonicalize_json(payload) != self.canonical_bytes:
            raise ExecutionPlanError("Plan carrier is not canonical JSON")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExecutionPlanV4:
        _validate_json_values(payload)
        return cls(canonicalize_json(dict(payload)))

    @property
    def schema(self) -> str:
        return PLAN_SCHEMA

    @property
    def plan_fingerprint_sha256(self) -> str:
        return self.to_payload()["plan_fingerprint_sha256"]

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)

    def to_canonical_json(self) -> str:
        return self.canonical_bytes.decode("utf-8")


def parse_plan_v4(data: bytes | str) -> ExecutionPlanV4:
    return ExecutionPlanV4.from_payload(decode_bounded_json_object(data))
