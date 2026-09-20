"""Opt-in Materials V4 extension of the closed core photography plan family.

V2 remains unchanged. V3 binds source-aligned material evidence and the complete
response policy, while retaining the same executor and native depth authority.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

import jsonschema

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import (
    ExecutionPlanV2,
    _validate_json_values,
    digest_payload,
    photography_nodes,
    require_digest,
    validate_plan_v2,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

PLAN_SCHEMA = "tp.execution.plan.v3"


def materials_photography_nodes(configuration: Mapping[str, Any], *, companions: bool = False) -> list[dict[str, Any]]:
    """Freeze the one reviewed graph; V3 never runs two material compositors."""
    nodes = photography_nodes(configuration, companions=companions)
    nodes[0]["inputs"].pop("materials", None)
    nodes[0]["inputs"]["materials_v4"] = "$materials_v4"
    nodes[2]["stage"] = "tp.stage.lux.enhance.v2"
    nodes[2]["inputs"].pop("materials", None)
    nodes[2]["inputs"]["materials_v4"] = "$materials_v4"
    nodes[2]["outputs"]["materials"] = "tp.materials.execution.v1"
    nodes[2]["outputs"]["materials_baseline"] = "tp.image.master.v1"
    nodes[2]["configuration"]["materials_v4"] = copy.deepcopy(configuration["materials_v4"])
    nodes[3]["inputs"]["materials"] = "enhance.materials"
    nodes[3]["inputs"]["materials_baseline"] = "enhance.materials_baseline"
    return nodes


def validate_plan_v3(payload: Mapping[str, Any]) -> None:
    """Validate the closed schema, legacy invariants, and material bindings."""
    _validate_json_values(payload)
    try:
        decode_bounded_json_object(canonicalize_json(payload))
        schema = decode_bounded_json_object(
            files("transformation_portal.schemas.execution").joinpath("plan.v3.schema.json").read_bytes()
        )
        jsonschema.Draft202012Validator(schema).validate(payload)
    except (ValueError, TypeError, jsonschema.ValidationError) as exc:
        raise ExecutionPlanError(f"Invalid execution plan v3: {exc}") from exc

    from transformation_portal.lux_depth_v4.materials import validate_frozen_record
    from transformation_portal.materials_v4.engine import ResponsePolicy

    configuration = payload["configuration"]
    try:
        policy = ResponsePolicy.from_payload(configuration["materials_v4"])
    except (TypeError, ValueError) as exc:
        raise ExecutionPlanError(f"Invalid material response policy: {exc}") from exc
    if policy.to_payload() != configuration["materials_v4"]:
        raise ExecutionPlanError("Material response policy must be normalized")
    if policy.max_working_bytes > payload["resources"]["memory_mib"] * 1024 * 1024:
        raise ExecutionPlanError("Material tile memory exceeds the execution memory budget")
    has_companions = any("companions" in item for item in payload["inputs"])
    if payload["nodes"] != materials_photography_nodes(configuration, companions=has_companions):
        raise ExecutionPlanError("V3 nodes differ from the closed material photography graph")
    count = 0
    for item in payload["inputs"]:
        if "materials" in item.get("companions", {}):
            raise ExecutionPlanError("V3 forbids combining legacy and V4 material compositors")
        if "materials_v4" in item:
            count += 1
            try:
                validate_frozen_record(item["materials_v4"], payload["resources"])
            except (TypeError, ValueError) as exc:
                raise ExecutionPlanError(f"Invalid material evidence binding: {exc}") from exc
            if item["materials_v4"]["source_sha256"] != item["sha256"]:
                raise ExecutionPlanError("Material evidence source differs from selected input")
            calibration = item.get("companions", {}).get("calibration")
            if calibration is not None and item["materials_v4"]["shape"] != [calibration["height"], calibration["width"]]:
                raise ExecutionPlanError("Material evidence and calibration geometry differ")
    if not count:
        raise ExecutionPlanError("V3 requires at least one material evidence binding")
    # Reuse every V2 cross-field invariant on an explicit structural projection.
    # This projection is validation only and is never an executable carrier.
    legacy = copy.deepcopy(dict(payload))
    legacy["schema"] = "tp.execution.plan.v2"
    legacy.pop("materials_manifest")
    legacy["configuration"].pop("materials_v4")
    for item in legacy["inputs"]:
        item.pop("materials_v4", None)
    legacy["nodes"] = photography_nodes(legacy["configuration"], companions=has_companions)
    legacy.pop("plan_fingerprint_sha256")
    legacy["plan_fingerprint_sha256"] = digest_payload(legacy)
    validate_plan_v2(legacy)
    from transformation_portal.lux_depth_v4.companions import _portable_path

    manifest = payload["materials_manifest"]
    _portable_path(manifest["path"])
    require_digest(manifest["sha256"])
    if type(manifest["size_bytes"]) is not int or manifest["size_bytes"] > min(
        payload["resources"]["max_input_bytes"], 1024 * 1024
    ):
        raise ExecutionPlanError("Material manifest exceeds the declared byte budget")
    unsigned = dict(payload)
    observed = unsigned.pop("plan_fingerprint_sha256")
    if require_digest(observed) != digest_payload(unsigned):
        raise ExecutionPlanError("Execution plan v3 fingerprint mismatch")


@dataclass(frozen=True)
class ExecutionPlanV3:
    """Canonical, immutable plan carrier for the explicit Materials V4 path."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes:
            raise ExecutionPlanError("Plan carrier must be bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        validate_plan_v3(payload)
        if canonicalize_json(payload) != self.canonical_bytes:
            raise ExecutionPlanError("Plan carrier is not canonical JSON")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExecutionPlanV3:
        validate_plan_v3(payload)
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


PhotographyPlan = ExecutionPlanV2 | ExecutionPlanV3


def parse_photography_plan(data: bytes | str) -> PhotographyPlan:
    """Reject all plans outside the two explicitly supported photography versions."""
    payload = decode_bounded_json_object(data)
    if payload.get("schema") == PLAN_SCHEMA:
        return ExecutionPlanV3.from_payload(payload)
    return ExecutionPlanV2.from_payload(payload)
