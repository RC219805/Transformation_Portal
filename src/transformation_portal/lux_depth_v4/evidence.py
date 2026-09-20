"""Closed V4 completion validation before any managed-generation visibility."""

from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import jsonschema
import numpy as np

from transformation_portal.core.execution_plan import decode_bounded_json_object
from transformation_portal.core.execution_plan_v2 import require_digest
from transformation_portal.core.execution_plan_v3 import parse_photography_plan
from transformation_portal.core.image_artifact import artifact_content_hash
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
        plan = parse_photography_plan(plan_bytes)
        payload = plan.to_payload()
        if evidence["plan_schema"] != plan.schema:
            raise ValueError("Completion plan schema differs from the canonical plan")
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
            if plan.schema == "tp.execution.plan.v3":
                try:
                    _verify_materials_receipt(root, input_id, descriptor, source, payload, declared)
                except (TypeError, KeyError, OverflowError) as exc:
                    raise ValueError("Malformed Materials V4 completion evidence") from exc
            delivery = descriptor.get("delivery")
            if not isinstance(delivery, dict) or delivery.get("path") != f"{input_id}/delivery.tif":
                raise ValueError("Photograph delivery points outside its verified artifact")
        if total_bytes > payload["resources"]["max_output_bytes"]:
            raise ValueError("Completed generation exceeds its declared byte budget")
        if _inventory(root) != {*declared, "execution-evidence.json"}:
            raise ValueError("Output inventory differs from complete declared artifacts")
        verified.append(VerifiedArtifact("execution-evidence.json", completion["size_bytes"], completion["sha256"]))
    return VerifiedExecutionEvidenceV2(root, raw, expected_plan_sha256, tuple(verified))


def _material_array(root: Path, relative: str, shape: tuple[int, ...], declared: dict) -> np.ndarray:
    """Read a closed float32 NPY carrier after checking geometry before allocation."""
    if declared.get(relative, {}).get("kind") != "array":
        raise ValueError("Materials verification requires its baseline, master, and declared alpha arrays")
    expected_bytes = math.prod(shape) * 4
    raw, observed = snapshot(root, root / relative, maximum_bytes=expected_bytes + 4096)
    if any(observed[key] != declared[relative][key] for key in ("path", "sha256", "size_bytes")):
        raise ValueError("Materials numeric artifact changed during verification")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        length_bytes = 2
        header_reader = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        length_bytes = 4
        header_reader = np.lib.format.read_array_header_2_0
    else:
        raise ValueError("Materials numeric evidence uses an unsupported NPY version")
    # NumPy checks max_header_size only after reading and decoding the header.
    # Reject the declared length before allowing that allocation.
    encoded_length = stream.read(length_bytes)
    header_size = int.from_bytes(encoded_length, "little")
    if len(encoded_length) != length_bytes or not 0 < header_size <= 4096:
        raise ValueError("Materials numeric evidence NPY header exceeds its byte budget")
    if stream.tell() + header_size > len(raw):
        raise ValueError("Materials numeric evidence NPY header is truncated")
    stream.seek(8)
    carried_shape, fortran, dtype = header_reader(stream, max_header_size=4096)
    if carried_shape != shape or fortran or dtype != np.dtype("float32") or len(raw) - stream.tell() != expected_bytes:
        raise ValueError("Materials numeric evidence has invalid shape, dtype, layout, or payload size")
    result = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    for start in range(0, shape[0], 128):
        if not np.isfinite(result[start : start + 128]).all():
            raise ValueError("Materials numeric evidence must be finite")
    return result


def _verify_materials_receipt(
    root: Path, input_id: str, descriptor: dict, source: dict, payload: dict, declared: dict
) -> None:
    """Bind the V3 response and independently measure its published material delta.

    Original segmentation masks are not published here, so this validates plan
    and evidence identities plus observed output statistics; it does not certify
    per-region semantic correctness or re-execute segmentation decisions.
    """
    from transformation_portal.materials_v4.contracts import MaterialEvidence
    from transformation_portal.materials_v4.engine import PreparedResponse, RegionResponse, ResponsePolicy
    from transformation_portal.materials_v4.operations import operation_contract_hash
    from transformation_portal.materials_v4.taxonomy import MATERIAL_LABELS

    receipt = descriptor.get("materials")
    receipt_keys = {
        "schema",
        "plan_sha256",
        "input_master_sha256",
        "output_master_sha256",
        "evidence_sha256",
        "evidence_status",
        "evidence_reason",
        "response_plan",
        "operations_sha256",
        "status",
        "reason",
        "changed_pixels",
        "max_abs_delta",
        "protected_changed_pixels",
        "protected_max_abs_delta",
        "mean_abs_delta",
        "regions",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_keys or receipt["schema"] != "tp.materials.execution.v1":
        raise ValueError("Materials V4 completion requires a complete execution receipt")
    response = receipt["response_plan"]
    plan_keys = {"schema", "source_sha256", "master_sha256", "evidence_sha256", "operations_sha256", "policy", "regions"}
    if not isinstance(response, dict) or set(response) != plan_keys or response["schema"] != "tp.materials.response_plan.v1":
        raise ValueError("Materials response plan has an invalid schema")
    policy = ResponsePolicy.from_payload(response["policy"])
    if policy.to_payload() != payload["configuration"]["materials_v4"] or response["policy"] != policy.to_payload():
        raise ValueError("Materials response policy differs from the canonical execution plan")
    if response["source_sha256"] != source["sha256"]:
        raise ValueError("Materials response refers to another photographic source")
    expected_evidence = source.get("materials_v4", {}).get("content_sha256")
    shape = tuple(descriptor["master"]["shape"])
    if expected_evidence is None:
        expected_evidence = MaterialEvidence(
            source["sha256"], shape, (), status="unavailable", reason="no_evidence_for_source"
        ).content_hash()
        if receipt["evidence_status"] != "unavailable" or receipt["evidence_reason"] != "no_evidence_for_source":
            raise ValueError("Absent material evidence must preserve its explicit abstention reason")
    if response["evidence_sha256"] != expected_evidence or receipt["evidence_sha256"] != expected_evidence:
        raise ValueError("Materials receipt evidence differs from the frozen source binding")
    if (
        response["operations_sha256"] != operation_contract_hash()
        or receipt["operations_sha256"] != response["operations_sha256"]
    ):
        raise ValueError("Materials operation contract differs from the verified implementation")
    if receipt["evidence_status"] not in {"available", "unavailable", "abstained", "degraded"}:
        raise ValueError("Materials evidence status is invalid")
    reason = receipt["evidence_reason"]
    if reason is not None and (not isinstance(reason, str) or not 1 <= len(reason) <= 512):
        raise ValueError("Materials evidence reason is invalid")
    if receipt["evidence_status"] != "available" and reason is None:
        raise ValueError("Unavailable material evidence requires its abstention reason")
    region_payloads, execution_regions = response["regions"], receipt["regions"]
    if not isinstance(region_payloads, list) or len(region_payloads) > 256 or not isinstance(execution_regions, list):
        raise ValueError("Materials response region inventory is invalid or exceeds its bound")
    if len(region_payloads) != len(execution_regions):
        raise ValueError("Materials executed regions differ from the response plan")
    if receipt["evidence_status"] in {"unavailable", "abstained"} and region_payloads:
        raise ValueError("Unavailable material evidence cannot claim region decisions")
    pixel_count = math.prod(shape)
    region_keys = {"region_id", "label", "status", "reason", "coverage_px", "resolved_coverage_px", "operation_id", "strength"}
    operations = {operation.label: operation for operation in policy.operations}
    abstentions = {
        "evidence_unavailable",
        "unsupported_material",
        "missing_semantic_confidence",
        "below_semantic_confidence",
        "supplied_confidence_disabled",
        "unsupported_provenance",
        "untrusted_inference",
        "calibration_producer_mismatch",
        "below_coverage_threshold",
        "below_resolved_coverage_threshold",
    }
    ids, decisions, total_changed = [], [], 0
    for region, executed in zip(region_payloads, execution_regions):
        if not isinstance(region, dict) or set(region) != region_keys:
            raise ValueError("Materials region decision has an invalid schema")
        if not isinstance(executed, dict) or set(executed) != region_keys | {"changed_pixels"}:
            raise ValueError("Materials region execution has an invalid schema")
        if {key: executed[key] for key in region_keys} != region:
            raise ValueError("Materials execution decision differs from the bound response plan")
        identifier = region["region_id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier) is None:
            raise ValueError("Materials region ID is invalid")
        ids.append(identifier)
        if region["label"] not in MATERIAL_LABELS:
            raise ValueError("Materials region label is not canonical")
        for name in ("coverage_px", "resolved_coverage_px"):
            if type(region[name]) is not int or not 0 <= region[name] <= pixel_count:
                raise ValueError("Materials region coverage is invalid")
        if region["resolved_coverage_px"] > region["coverage_px"]:
            raise ValueError("Resolved material coverage exceeds its original support")
        changed = executed["changed_pixels"]
        if type(changed) is not int or not 0 <= changed <= pixel_count:
            raise ValueError("Materials changed pixel count is invalid")
        total_changed += changed
        if region["status"] == "eligible":
            operation = operations.get(region["label"])
            if (
                receipt["evidence_status"] != "available"
                or operation is None
                or region["reason"] != "evidence_accepted"
                or region["operation_id"] != operation.operation_id
                or type(region["strength"]) not in {float, int}
                or region["strength"] != operation.strength
                or region["resolved_coverage_px"] < policy.min_coverage_px
            ):
                raise ValueError("Materials eligible decision violates the frozen response policy")
        elif (
            region["status"] != "abstained"
            or region["reason"] not in abstentions
            or region["operation_id"] is not None
            or region["strength"] is not None
            or changed
        ):
            raise ValueError("Materials abstention cannot claim an operation or changed pixels")
        decisions.append(RegionResponse(**region))
    if ids != sorted(set(ids)):
        raise ValueError("Materials region decisions must be unique and canonically ordered")
    prepared = PreparedResponse(
        response["source_sha256"],
        response["master_sha256"],
        response["evidence_sha256"],
        response["operations_sha256"],
        policy,
        tuple(decisions),
    )
    if prepared.to_payload() != response or prepared.content_hash() != receipt["plan_sha256"]:
        raise ValueError("Materials response plan fingerprint mismatch")
    master_payload = descriptor["master"]
    baseline_payload = descriptor.get("materials_baseline")
    master_keys = {
        "schema",
        "color_space",
        "alpha_mode",
        "shape",
        "source_sha256",
        "source_bit_depth",
        "source_icc_sha256",
        "metadata",
    }
    if (
        set(master_payload) != master_keys
        or master_payload["schema"] != "tp.image.master.v1"
        or master_payload["color_space"] != "linear_srgb"
        or type(master_payload["source_bit_depth"]) is not int
        or master_payload["source_bit_depth"] not in {8, 16, 32}
        or not isinstance(master_payload["metadata"], dict)
    ):
        raise ValueError("Materials master descriptor violates the photographic image contract")
    if master_payload["source_icc_sha256"] is not None:
        require_digest(master_payload["source_icc_sha256"])
    if baseline_payload != master_payload:
        raise ValueError("Materials response must preserve its baseline metadata, geometry, alpha, and source color identity")
    rgb_shape = (*shape, 3)
    if math.prod(rgb_shape) * 4 * 3 > payload["resources"]["memory_mib"] * 1024 * 1024:
        raise ValueError("Materials verification arrays exceed the declared memory budget")
    baseline = _material_array(root, f"{input_id}/materials-baseline.npy", rgb_shape, declared)
    final = _material_array(root, f"{input_id}/master.npy", rgb_shape, declared)
    alpha = None
    if master_payload.get("alpha_mode") is not None:
        if master_payload["alpha_mode"] != "straight":
            raise ValueError("Materials master alpha mode is invalid")
        alpha = _material_array(root, f"{input_id}/alpha.npy", shape, declared)
        if np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("Materials alpha is outside [0,1]")
    elif f"{input_id}/alpha.npy" in declared:
        raise ValueError("Materials master omits its published alpha identity")
    input_hash = artifact_content_hash(baseline_payload, {"pixels": baseline, "alpha": alpha})
    output_hash = artifact_content_hash(master_payload, {"pixels": final, "alpha": alpha})
    if (
        receipt["input_master_sha256"] != input_hash
        or response["master_sha256"] != input_hash
        or receipt["output_master_sha256"] != output_hash
    ):
        raise ValueError("Materials baseline or final master content differs from its bound identity")
    changed_pixels, absolute_sum, maximum = 0, 0.0, 0.0
    for y in range(0, shape[0], policy.tile_size):
        for x in range(0, shape[1], policy.tile_size):
            ys, xs = slice(y, y + policy.tile_size), slice(x, x + policy.tile_size)
            before, after = baseline[ys, xs], final[ys, xs]
            delta = np.abs(after.astype(np.float64) - before.astype(np.float64))
            changed_pixels += int(np.count_nonzero(np.any(delta != 0, axis=2)))
            absolute_sum += float(delta.sum(dtype=np.float64))
            maximum = max(maximum, float(delta.max(initial=0)))
            protected = np.any((before < 0) | (before > 1), axis=2)
            if alpha is not None:
                protected |= alpha[ys, xs] == 0
            if np.any(delta[protected] != 0):
                raise ValueError("Materials modified protected HDR or transparent baseline samples")
    for name in ("max_abs_delta", "mean_abs_delta", "protected_max_abs_delta"):
        value = receipt[name]
        if type(value) not in {float, int} or not math.isfinite(value) or value < 0:
            raise ValueError("Materials delta statistics must be finite nonnegative numbers")
    if (
        type(receipt["changed_pixels"]) is not int
        or receipt["changed_pixels"] != changed_pixels
        or total_changed != changed_pixels
        or receipt["max_abs_delta"] != maximum
        or receipt["mean_abs_delta"] != absolute_sum / math.prod(rgb_shape)
        or maximum > policy.max_abs_delta
        or type(receipt["protected_changed_pixels"]) is not int
        or receipt["protected_changed_pixels"] != 0
        or receipt["protected_max_abs_delta"] != 0
    ):
        raise ValueError("Materials statistics differ from independently measured final pixels or policy bounds")
    if receipt["status"] != ("applied" if changed_pixels else "abstained") or receipt["reason"] != (
        "pixels_changed" if changed_pixels else "no_pixel_change"
    ):
        raise ValueError("Materials execution status differs from measured output")
