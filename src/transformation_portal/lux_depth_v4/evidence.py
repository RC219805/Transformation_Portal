"""Closed V4 completion validation before any managed-generation visibility."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import jsonschema

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import ExecutionPlanV2, require_digest
from transformation_portal.depth.backends.da3_runtime_identity import DA3RuntimeIdentityEvidence
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.io import directory_path, pinned_directory, snapshot

MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _decode_evidence(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ValueError("Completion evidence exceeds byte limit")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate completion evidence field")
            result[key] = value
        return result

    def nonfinite(_value: str) -> NoReturn:
        raise ValueError("Nonfinite completion evidence number")

    try:
        result = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=nonfinite)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid completion JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Completion evidence must be an object")
    pending = [(result, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 64:
            raise ValueError("Completion evidence exceeds nesting limit")
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return result


@dataclass(frozen=True)
class VerifiedArtifact:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedExecutionEvidenceV2:
    output_root: Path
    canonical_bytes: bytes
    plan_fingerprint_sha256: str
    artifacts: tuple[VerifiedArtifact, ...]

    def to_payload(self) -> dict[str, Any]:
        return _decode_evidence(self.canonical_bytes)


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or ":" in value
        or "\x00" in value
    ):
        raise ValueError("Execution artifact path must be canonical and confined")
    return value


def _inventory(root: Path) -> set[str]:
    observed = set()

    def unreadable(error: OSError) -> None:
        raise error

    for current, directories, names in os.walk(root, followlinks=False, onerror=unreadable):
        directory = Path(current)
        if any((directory / name).is_symlink() for name in directories):
            raise ValueError("Execution output contains a linked directory")
        for name in names:
            path = directory / name
            if path.is_symlink():
                raise ValueError("Execution output contains a linked artifact")
            observed.add(path.relative_to(root).as_posix())
            if len(observed) > 32769:
                raise ValueError("Execution output exceeds artifact count bound")
    return observed


def verify_execution_evidence_v2(output_root: Path, *, expected_plan_sha256: str) -> VerifiedExecutionEvidenceV2:
    """Rehash every completed output and bind its coverage to the exact canonical plan.

    Verification establishes artifact integrity and execution provenance, never
    photographic quality or production approval. No models are loaded here.
    """
    require_digest(expected_plan_sha256)
    root = directory_path(output_root)
    with pinned_directory(root):
        raw, completion = snapshot(root, root / "execution-evidence.json", maximum_bytes=MAX_EVIDENCE_BYTES)
        evidence = _decode_evidence(raw)
        schema = json.loads(files("transformation_portal.schemas.execution").joinpath("evidence.v2.schema.json").read_text())
        try:
            jsonschema.Draft202012Validator(schema).validate(evidence)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Invalid V4 completion evidence: {exc.message}") from exc
        if canonicalize_json(evidence) != raw or evidence["plan_fingerprint_sha256"] != expected_plan_sha256:
            raise ValueError("Completion evidence is noncanonical or does not match the expected plan")
        plan_bytes, _ = snapshot(root, root / "execution-plan.json", maximum_bytes=MAX_EVIDENCE_BYTES)
        plan = ExecutionPlanV2(plan_bytes)
        payload = plan.to_payload()
        if plan.plan_fingerprint_sha256 != expected_plan_sha256 or evidence["inputs"] != payload["inputs"]:
            raise ValueError("Completion input coverage differs from the canonical plan")
        runtime = DA3RuntimeIdentityEvidence.from_mapping(evidence["worker_runtime"])
        backend = runtime.to_mapping()["backend_identity"]
        if (
            not runtime.cacheable
            or backend["model_canonical_key"] != payload["model"]["canonical_key"]
            or backend["model_repo_id"] != payload["model"]["repo_id"]
            or backend["model_lock_revision"] != payload["model"]["revision"]
            or backend["actual_device"] != payload["device"]
        ):
            raise ValueError("Completion worker identity differs from the governed plan")
        inputs = {item["id"]: item for item in payload["inputs"]}
        executions = evidence["executions"]
        if [record["input_id"] for record in executions] != list(inputs):
            raise ValueError("Completion must cover every prepared input exactly once, in order")
        hits = sum(record["depth_cache_hit"] for record in executions)
        cache = evidence["cache"]
        if cache["hits"] != hits or cache["misses"] != len(inputs) - hits or not cache["enabled"] and hits:
            raise ValueError("Completion cache summary differs from executed inputs")
        declared: dict[str, dict[str, Any]] = {}
        verified = []
        total_bytes = completion["size_bytes"]
        for artifact in evidence["artifacts"]:
            relative = _relative(artifact["path"])
            if relative in declared or relative == "execution-evidence.json":
                raise ValueError("Duplicate or recursive completion artifact")
            declared[relative] = artifact
            if artifact["kind"] == "plan":
                if relative != "execution-plan.json" or artifact["input_id"] is not None:
                    raise ValueError("Invalid canonical plan artifact")
            elif artifact["input_id"] not in inputs or not relative.startswith(f"{artifact['input_id']}/"):
                raise ValueError("Artifact belongs to an unprepared input")
            _, observed = snapshot(root, root / relative, maximum_bytes=artifact["size_bytes"], retain_bytes=False)
            if any(observed[key] != artifact[key] for key in ("path", "size_bytes", "sha256")):
                raise ValueError("Published artifact differs from completion evidence")
            total_bytes += artifact["size_bytes"]
            verified.append(VerifiedArtifact(relative, artifact["size_bytes"], artifact["sha256"]))
        if declared.get("execution-plan.json", {}).get("kind") != "plan":
            raise ValueError("Completion inventory omits its canonical plan")
        for input_id, source in inputs.items():
            required = {
                "source-master.npy": "array",
                "master.npy": "array",
                "native-depth.npy": "array",
                "depth-valid.npy": "array",
                "aligned-depth-valid.npy": "array",
                "relative-depth.npy": "array",
                "delivery.tif": "image",
                "photograph.json": "descriptor",
            }
            if any(declared.get(f"{input_id}/{name}", {}).get("kind") != kind for name, kind in required.items()):
                raise ValueError("Completion inventory omits a required photographic artifact")
            descriptor_bytes, _ = snapshot(root, root / input_id / "photograph.json", maximum_bytes=MAX_EVIDENCE_BYTES)
            descriptor = decode_bounded_json_object(descriptor_bytes)
            if descriptor.get("schema") != "tp.lux.photograph.v1" or descriptor.get("input_id") != input_id:
                raise ValueError("Photograph descriptor has invalid source binding")
            for field in ("source", "master", "depth"):
                if not isinstance(descriptor.get(field), dict) or descriptor[field].get("source_sha256") != source["sha256"]:
                    raise ValueError("Photograph descriptor refers to another source")
            calibration = source.get("companions", {}).get("calibration")
            depth = descriptor["depth"]
            master_shape = descriptor["master"].get("shape")
            if (
                not isinstance(master_shape, list)
                or len(master_shape) != 2
                or any(type(size) is not int or size <= 0 for size in master_shape)
                or master_shape != descriptor["source"].get("shape")
                or master_shape[0] * master_shape[1] > payload["resources"]["max_pixels"]
            ):
                raise ValueError("Photograph master geometry differs from its source or pixel budget")
            expected_alignment = {
                "shape": master_shape,
                "validity_path": f"{input_id}/aligned-depth-valid.npy",
                "relative_path": f"{input_id}/relative-depth.npy",
                "metric_path": f"{input_id}/aligned-metric-depth-m.npy" if calibration is not None else None,
                "invalid_value": 0,
                "interpolation": "valid_weighted_bilinear_nearest_validity",
            }
            if descriptor.get("aligned_depth") != expected_alignment:
                raise ValueError("Photograph aligned depth must bind its master geometry and verified validity artifact")
            if depth.get("has_metric_depth") is not (calibration is not None):
                raise ValueError("Photograph metric depth differs from its admitted calibration")
            if calibration is not None:
                carried = depth.get("calibration")
                if not isinstance(carried, dict) or carried.get("input_intrinsics") != calibration:
                    raise ValueError("Photograph calibration differs from the canonical plan")
                for name in ("metric-depth-m.npy", "aligned-metric-depth-m.npy"):
                    if declared.get(f"{input_id}/{name}", {}).get("kind") != "array":
                        raise ValueError("Completion inventory omits an admitted calibrated depth artifact")
            elif any(f"{input_id}/{name}" in declared for name in ("metric-depth-m.npy", "aligned-metric-depth-m.npy")):
                raise ValueError("Uncalibrated execution cannot publish metric depth artifacts")
            delivery = descriptor.get("delivery")
            if not isinstance(delivery, dict) or delivery.get("path") != f"{input_id}/delivery.tif":
                raise ValueError("Photograph delivery points outside its verified artifact")
        if total_bytes > payload["resources"]["max_output_bytes"]:
            raise ValueError("Completed generation exceeds its declared byte budget")
        if _inventory(root) != {*declared, "execution-evidence.json"}:
            raise ValueError("Output inventory differs from complete declared artifacts")
        verified.append(VerifiedArtifact("execution-evidence.json", completion["size_bytes"], completion["sha256"]))
    return VerifiedExecutionEvidenceV2(root, raw, expected_plan_sha256, tuple(verified))
