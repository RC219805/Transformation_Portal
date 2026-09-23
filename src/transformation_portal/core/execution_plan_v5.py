"""Closed composite authority for governed inference and LuxDepthV6 finishing.

The embedded V4 plan is the sole model/input authority. The outer plan freezes
finishing before inference; resulting pixel hashes are bound after execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

import jsonschema

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import _validate_json_values, digest_payload
from transformation_portal.core.execution_plan_v4 import ExecutionPlanV4
from transformation_portal.ingest.canonical_json import canonicalize_json

PLAN_SCHEMA = "tp.execution.plan.v5"
MAX_PLAN_BYTES = 1024 * 1024
MAX_COMPLETION_BYTES = 16 * 1024 * 1024
ENVELOPE_RESERVE = MAX_PLAN_BYTES + MAX_COMPLETION_BYTES


def stage_output_budget(total: int) -> int:
    """Reserve both complete retained stages and the outer authority records."""
    if type(total) is not int or total <= ENVELOPE_RESERVE + 1:
        raise ExecutionPlanError("V6 total output budget cannot reserve both stages and completion")
    return (total - ENVELOPE_RESERVE) // 2


@dataclass(frozen=True)
class ExecutionPlanV5:
    """Immutable admitted raw-input-to-V6 generation; never an executable command."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes or len(self.canonical_bytes) > MAX_PLAN_BYTES:
            raise ExecutionPlanError("V6 managed plan requires bounded canonical bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        _validate_json_values(payload)
        schema = decode_bounded_json_object(
            files("transformation_portal.schemas.execution").joinpath("plan.v5.schema.json").read_bytes()
        )
        try:
            jsonschema.Draft202012Validator(schema).validate(payload)
            # JSON Schema accepts integral floats as integers; runtime budgets
            # require exact integer authority at both composite and child levels.
            for section in ("resources", "publication"):
                for name, value in payload[section].items():
                    if type(value) is not int:
                        raise ValueError(f"V6 {section} field {name!r} must be an exact integer")
            upstream = ExecutionPlanV4.from_payload(payload["inference"]).to_payload()
            if "materials_manifest" in upstream or "materials_v4" in upstream["configuration"]:
                raise ValueError("V6 cannot replay applied Materials responses; omit materials inputs")
            expected_resources = {**upstream["resources"], "max_output_bytes": payload["resources"]["max_output_bytes"]}
            if payload["resources"] != expected_resources or payload["resources"]["max_pixels"] > 100_000_000:
                raise ValueError("V6 composite resource policy differs from its inference plan")
            if upstream["resources"]["max_output_bytes"] != stage_output_budget(payload["resources"]["max_output_bytes"]):
                raise ValueError("V6 inference output budget differs from its reserved stage budget")
            if payload["publication"] != upstream.get("publication"):
                raise ValueError("V6 publisher limits differ between admitted stages")
            if payload["resources"]["max_output_bytes"] > payload["publication"]["max_total_bytes"]:
                raise ValueError("V6 generation exceeds its publisher byte budget")
            unsigned = dict(payload)
            if unsigned.pop("plan_fingerprint_sha256") != digest_payload(unsigned):
                raise ValueError("V6 composite fingerprint mismatch")
            if canonicalize_json(payload) != self.canonical_bytes:
                raise ValueError("V6 composite plan must be canonical JSON")
        except (KeyError, TypeError, ValueError, jsonschema.ValidationError) as exc:
            raise ExecutionPlanError(f"Invalid execution plan v5: {exc}") from exc

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExecutionPlanV5:
        _validate_json_values(payload)
        return cls(canonicalize_json(dict(payload)))

    @property
    def schema(self) -> str:
        return PLAN_SCHEMA

    @property
    def plan_fingerprint_sha256(self) -> str:
        return self.to_payload()["plan_fingerprint_sha256"]

    @property
    def inference_plan(self) -> ExecutionPlanV4:
        return ExecutionPlanV4.from_payload(self.to_payload()["inference"])

    def to_payload(self) -> dict[str, Any]:
        return decode_bounded_json_object(self.canonical_bytes)


def parse_plan_v5(data: bytes | str) -> ExecutionPlanV5:
    return ExecutionPlanV5.from_payload(decode_bounded_json_object(data))
