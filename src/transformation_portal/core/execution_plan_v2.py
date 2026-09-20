"""Closed, immutable photography plans in the shared execution contract family.

Parsing establishes structure and integrity, never permission to execute code.
Only the Lux domain boundary may authorize model/runtime selection.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any, Mapping

import jsonschema

from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object
from transformation_portal.ingest.canonical_json import canonicalize_json

PLAN_SCHEMA = "tp.execution.plan.v2"
MAX_INPUTS = 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def digest_payload(payload: Any) -> str:
    """Hash the repository's canonical JSON representation."""
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def require_digest(value: Any) -> str:
    """Reject placeholder and malformed identities."""
    if not isinstance(value, str) or not _DIGEST.fullmatch(value) or value == "0" * 64:
        raise ExecutionPlanError("Expected a non-placeholder SHA-256 digest")
    return value


def photography_nodes(configuration: Mapping[str, Any], *, companions: bool = False) -> list[dict[str, Any]]:
    """Describe the closed photography graph, with explicit consumed artifacts."""
    nodes: list[dict[str, Any]] = [
        {
            "id": "preprocess",
            "stage": "tp.stage.lux.preprocess.v2",
            "inputs": {"source": "$input"},
            "outputs": {"master": "tp.image.master.v1", "proxy": "tp.image.proxy.v1"},
            "configuration": {
                "input_color": configuration["input_color"],
                "target_size": configuration["target_size"],
                "orientation": "canonical",
                "master_space": "linear_srgb",
                "proxy_space": "srgb",
                "geometry": "resize_then_edge_pad",
            },
        },
        {
            "id": "depth",
            "stage": "tp.stage.lux.depth.v2",
            "inputs": {"proxy": "preprocess.proxy"},
            "outputs": {"depth": "tp.depth.artifact.v2"},
            "configuration": {"native_semantics": "da3_metric_uncalibrated", "synthetic_fallback": False},
        },
        {
            "id": "enhance",
            "stage": "tp.stage.lux.enhance.v1",
            "inputs": {"master": "preprocess.master", "depth": "depth.depth", "proxy": "preprocess.proxy"},
            "outputs": {"master": "tp.image.master.v1"},
            "configuration": {"strength": configuration["strength"], "clarity": configuration["clarity"]},
        },
        {
            "id": "output",
            "stage": "tp.stage.lux.output.v2",
            "inputs": {
                "master": "enhance.master",
                "source_master": "preprocess.master",
                "depth": "depth.depth",
                "proxy": "preprocess.proxy",
            },
            "outputs": {"delivery": "image/tiff", "master": "application/x-npy", "evidence": "application/json"},
            "configuration": {"bit_depth": 16, "color_space": "srgb", "preview_maps": configuration["preview_maps"]},
        },
    ]
    if companions:
        nodes[0]["inputs"].update(calibration="$calibration", materials="$materials")
        nodes[1]["inputs"]["calibration"] = "$calibration"
        nodes[2]["inputs"]["materials"] = "$materials"
        nodes[2]["outputs"]["materials"] = "tp.lux.materials.application.v1"
        nodes[3]["inputs"]["materials"] = "enhance.materials"
    return nodes


def _validate_json_values(value: Any, *, depth: int = 0) -> None:
    """Reject coercible Python objects and non-finite values before hashing."""
    if depth > 24:
        raise ExecutionPlanError("Execution plan exceeds maximum nesting depth")
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ExecutionPlanError("Execution plan object keys must be strings")
        for item in value.values():
            _validate_json_values(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_values(item, depth=depth + 1)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ExecutionPlanError("Execution plan numbers must be finite")
    elif value is not None and type(value) not in (str, bool, int):
        raise ExecutionPlanError("Execution plan values must be explicit JSON types")


def validate_plan_v2(payload: Mapping[str, Any]) -> None:
    """Validate the packaged schema and all cross-field authority invariants."""
    _validate_json_values(payload)
    # Apply the same body/string/integer limits to Python and serialized callers.
    try:
        decode_bounded_json_object(canonicalize_json(payload))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ExecutionPlanError("Execution plan cannot be serialized as bounded canonical JSON") from exc
    schema = decode_bounded_json_object(
        files("transformation_portal.schemas.execution").joinpath("plan.v2.schema.json").read_bytes()
    )
    try:
        jsonschema.Draft202012Validator(schema).validate(payload)
    except jsonschema.ValidationError as exc:
        raise ExecutionPlanError(f"Invalid execution plan v2: {exc.message}") from exc
    for name, value in payload["resources"].items():
        if type(value) is not int:
            raise ExecutionPlanError(f"Resource {name!r} must be an exact integer")
    if type(payload["configuration"]["target_size"]) is not int:
        raise ExecutionPlanError("target_size must be an exact integer")
    if payload["model"]["revision"] == "0" * 40:
        raise ExecutionPlanError("Model revision cannot be a placeholder")
    has_companions = any("companions" in item for item in payload["inputs"])
    if has_companions != ("companions_manifest" in payload):
        raise ExecutionPlanError("Companion inputs require an exact manifest receipt")
    if has_companions:
        receipt = payload["companions_manifest"]
        path = receipt["path"]
        parts = PurePosixPath(path)
        if (
            not path
            or path == "."
            or parts.is_absolute()
            or ".." in parts.parts
            or parts.as_posix() != path
            or "\\" in path
            or ":" in path
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in path)
            or any(part.endswith((".", " ")) for part in parts.parts)
        ):
            raise ExecutionPlanError("Companion manifest must have a portable confined path")
        require_digest(receipt["sha256"])
        if type(receipt["size_bytes"]) is not int or receipt["size_bytes"] > payload["resources"]["max_input_bytes"]:
            raise ExecutionPlanError("Companion manifest exceeds its declared resource budget")
    expected_nodes = photography_nodes(
        payload["configuration"], companions=any("companions" in item for item in payload["inputs"])
    )
    if canonicalize_json(payload["nodes"]) != canonicalize_json(expected_nodes):
        raise ExecutionPlanError("Plan nodes do not match the closed photography input/output contracts")
    seen: set[str] = set()
    previous = ""
    for index, item in enumerate(payload["inputs"]):
        path = item["path"]
        parts = PurePosixPath(path)
        key = unicodedata.normalize("NFC", path).casefold()
        if (
            not path
            or path == "."
            or path != parts.as_posix()
            or parts.is_absolute()
            or ".." in parts.parts
            or "\\" in path
            or ":" in path
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in path)
            or any(part.endswith((".", " ")) for part in parts.parts)
            or key in seen
            or (index and path <= previous)
            or item["id"] != f"input-{index:04d}"
        ):
            raise ExecutionPlanError("Input selection is not a sorted, portable, unique relative inventory")
        if type(item["size_bytes"]) is not int or item["size_bytes"] > payload["resources"]["max_input_bytes"]:
            raise ExecutionPlanError("Input byte size exceeds its declared resource budget")
        seen.add(key)
        previous = path
        require_digest(item["sha256"])
        companion = item.get("companions")
        if companion is not None:
            if companion["path"] != path or companion["source_sha256"] != item["sha256"]:
                raise ExecutionPlanError("Companion evidence must bind the exact prepared source")
            calibration = companion.get("calibration")
            if calibration is not None:
                if any(type(calibration[name]) is not int for name in ("width", "height")):
                    raise ExecutionPlanError("Calibration geometry must use exact integers")
                if calibration["width"] * calibration["height"] > payload["resources"]["max_pixels"]:
                    raise ExecutionPlanError("Calibration geometry exceeds its pixel budget")
                if not 0 <= calibration["cx"] < calibration["width"] or not 0 <= calibration["cy"] < calibration["height"]:
                    raise ExecutionPlanError("Calibration principal point must lie on its declared source grid")
                if (
                    not calibration["source"].strip()
                    or calibration["source"].strip() != calibration["source"]
                    or calibration["source"].casefold() in {"estimated", "unknown", "unavailable"}
                    or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in calibration["source"])
                ):
                    raise ExecutionPlanError("Calibration requires explicit measured provenance")
            materials = companion.get("materials")
            if materials is not None:
                if set(materials["confidences"]) - set(materials["masks"]):
                    raise ExecutionPlanError("Material confidence requires a corresponding mask")
                mask_paths: set[str] = set()
                mask_shapes: set[tuple[int, ...]] = set()
                mask_bytes = 0
                for mask in materials["masks"].values():
                    mask_path = mask["path"]
                    parts = PurePosixPath(mask_path)
                    mask_key = unicodedata.normalize("NFC", mask_path).casefold()
                    if (
                        not mask_path
                        or mask_path == "."
                        or parts.is_absolute()
                        or ".." in parts.parts
                        or parts.as_posix() != mask_path
                        or "\\" in mask_path
                        or ":" in mask_path
                        or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in mask_path)
                        or any(part.endswith((".", " ")) for part in parts.parts)
                        or mask_key in mask_paths
                    ):
                        raise ExecutionPlanError("Material paths must be portable, confined, and unique")
                    mask_paths.add(mask_key)
                    mask_shapes.add(tuple(mask["shape"]))
                    mask_bytes += mask["size_bytes"]
                    require_digest(mask["sha256"])
                    if type(mask["size_bytes"]) is not int or mask["size_bytes"] > payload["resources"]["max_input_bytes"]:
                        raise ExecutionPlanError("Material bytes exceed the declared resource budget")
                    if (
                        any(type(size) is not int for size in mask["shape"])
                        or math.prod(mask["shape"]) > payload["resources"]["max_pixels"]
                    ):
                        raise ExecutionPlanError("Material geometry exceeds the declared pixel budget")
                if mask_bytes > payload["resources"]["max_input_bytes"]:
                    raise ExecutionPlanError("Combined material bytes exceed the declared resource budget")
                if len(mask_shapes) != 1 or (
                    calibration is not None and next(iter(mask_shapes)) != (calibration["height"], calibration["width"])
                ):
                    raise ExecutionPlanError("Companion geometry must describe one canonical master")
    unsigned = dict(payload)
    observed = unsigned.pop("plan_fingerprint_sha256")
    if require_digest(observed) != digest_payload(unsigned):
        raise ExecutionPlanError("Execution plan v2 fingerprint mismatch")


@dataclass(frozen=True)
class ExecutionPlanV2:
    """Canonical bytes are the immutable carrier; projections are fresh copies."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes:
            raise ExecutionPlanError("Plan carrier must be bytes")
        payload = decode_bounded_json_object(self.canonical_bytes)
        validate_plan_v2(payload)
        if canonicalize_json(payload) != self.canonical_bytes:
            raise ExecutionPlanError("Plan carrier is not canonical JSON")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExecutionPlanV2:
        validate_plan_v2(payload)
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


def parse_execution_plan(data: bytes | str) -> Any:
    """Version-dispatch within the one core-owned plan family."""
    from transformation_portal.core.execution_plan import parse_execution_plan_json

    payload = decode_bounded_json_object(data)
    if payload.get("schema") == PLAN_SCHEMA:
        return ExecutionPlanV2.from_payload(payload)
    return parse_execution_plan_json(data)
