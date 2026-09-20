"""Reference-bound depth measurements with independent-group uncertainty.

References are operator-supplied evidence, not certified truth. This evaluator
checks their declared identities and geometry; it never interprets RGB edges as
depth boundaries or promotes a model automatically. Array paths are confined to
the manifest directory and snapshotted before parsing without pickle support.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from transformation_portal.ingest.canonical_json import canonicalize_json

MANIFEST_SCHEMA = "tp.depth.quality.manifest.v1"
REPORT_SCHEMA = "tp.depth.quality.report.v1"
EVALUATOR_RECIPE = "reference_bound_depth_metrics_v2"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ARRAY_BYTES = 1024 * 1024 * 1024
MAX_PIXELS = 100_000_000
MAX_TOTAL_PIXELS = 100_000_000
_SEMANTICS = {"metric_distance_m": "m", "relative_distance": "arbitrary", "relative_inverse_depth": "arbitrary"}


class EvaluationError(ValueError):
    """Evidence is malformed, changed, unsafe, or semantically incompatible."""


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - (optional or set()):
        raise EvaluationError("Evidence object has missing or unknown fields")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise EvaluationError("Expected a non-placeholder SHA-256 digest")
    return value


def _name(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is None:
        raise EvaluationError("Identity names must be bounded portable identifiers")
    return value


def _integer(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise EvaluationError(f"Expected an integer in [{low},{high}]")
    return value


def _number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise EvaluationError("Expected a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise EvaluationError("Number exceeds finite range") from exc
    if not math.isfinite(result) or result < 0 or positive and result == 0:
        raise EvaluationError("Expected a finite nonnegative number (positive for thresholds)")
    return result


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, **extra}


def _snapshot(root: Path, relative: str, maximum: int) -> bytes:
    """Open every component without following links; hash callers' immutable bytes."""
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {".", ".."} for part in path.parts)
        or any(char in relative for char in ("\\", "\x00", ":"))
    ):
        raise EvaluationError("Array paths must be canonical relative paths without traversal")
    descriptors: list[int] = []
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for component in path.parts[:-1]:
            directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise EvaluationError("Evidence file is not regular or exceeds its byte budget")
        blocks = []
        remaining = before.st_size + 1
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        raw = b"".join(blocks)
        after = os.fstat(fd)
        if len(raw) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise EvaluationError("Evidence file changed while being read")
        return raw
    except OSError as exc:
        raise EvaluationError(f"Cannot read confined evidence file: {relative}") from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _load_array(root: Path, value: Any, shape: tuple[int, int], *, mask: bool = False) -> np.ndarray:
    record = _object(value, {"path", "sha256", "size_bytes"})
    _digest(record["sha256"])
    size = _integer(record["size_bytes"], 1, MAX_ARRAY_BYTES)
    if not isinstance(record["path"], str):
        raise EvaluationError("Array path must be text")
    raw = _snapshot(root, record["path"], size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise EvaluationError("Array content differs from its frozen hash or byte size")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise EvaluationError("Only bounded NPY v1/v2 arrays are supported")
        length_size = 2 if version == (1, 0) else 4
        encoded = stream.read(length_size)
        header_size = int.from_bytes(encoded, "little")
        if len(encoded) != length_size or not 0 < header_size <= 4096 or stream.tell() + header_size > len(raw):
            raise EvaluationError("NPY header exceeds byte budget or is truncated")
        stream.seek(8)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        carried_shape, fortran, dtype = reader(stream, max_header_size=4096)
    except (ValueError, TypeError, EOFError, OverflowError) as exc:
        raise EvaluationError("Invalid bounded NPY header") from exc
    allowed = {np.dtype("bool")} if mask else {np.dtype("float32"), np.dtype("float64")}
    if carried_shape != shape or fortran or dtype not in allowed:
        raise EvaluationError("Array geometry, dtype or layout differs from its declared grid")
    if len(raw) - stream.tell() != math.prod(shape) * dtype.itemsize:
        raise EvaluationError("NPY data length differs from its header")
    return np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)


def _settings(value: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "minimum_valid_pixels": 32,
        "minimum_boundary_pixels": 2,
        "minimum_ordinal_pairs": 2,
        "boundary_tolerance_px": 1.0,
        "relative_alignment": "scale_shift",
        "ordinal_tolerance": 0.0,
        "minimum_independent_groups": 5,
        "bootstrap_samples": 2000,
    }
    _object(value, set(), set(defaults))
    settings = {**defaults, **value}
    for key, low, high in (
        ("minimum_valid_pixels", 2, MAX_PIXELS),
        ("minimum_boundary_pixels", 1, MAX_PIXELS),
        ("minimum_ordinal_pairs", 1, 100_000),
        ("minimum_independent_groups", 2, 1000),
        ("bootstrap_samples", 100, 10_000),
    ):
        settings[key] = _integer(settings[key], low, high)
    settings["boundary_tolerance_px"] = _number(settings["boundary_tolerance_px"])
    if settings["boundary_tolerance_px"] > 32:
        raise EvaluationError("Boundary tolerance exceeds 32 source-grid pixels")
    settings["ordinal_tolerance"] = _number(settings["ordinal_tolerance"])
    if settings["relative_alignment"] not in ("none", "scale", "scale_shift"):
        raise EvaluationError("Unknown relative alignment recipe")
    return settings


def _bound_record(value: Any, source: str, grid_hash: str, *, prediction: bool) -> Mapping[str, Any]:
    required = {"source_sha256", "grid_sha256", "semantics", "units", "depth", "valid_mask"}
    required |= {"recipe_sha256", "boundary_threshold"} if prediction else {"boundaries", "ordinal_pairs", "provenance"}
    record = _object(value, required)
    if _digest(record["source_sha256"]) != source or _digest(record["grid_sha256"]) != grid_hash:
        raise EvaluationError("Depth source or grid differs from its declared scene")
    semantics = record["semantics"]
    if not isinstance(semantics, str) or semantics not in _SEMANTICS or record["units"] != _SEMANTICS[semantics]:
        raise EvaluationError("Depth semantics and units are incompatible")
    if prediction:
        _digest(record["recipe_sha256"])
        _number(record["boundary_threshold"], positive=True)
    else:
        provenance = _object(record["provenance"], {"kind", "method", "uncertainty_m"})
        if provenance["kind"] not in ("registered_depth", "surveyed_depth", "annotated_depth", "synthetic_depth"):
            raise EvaluationError("References must be depth evidence, never RGB-edge truth")
        if not isinstance(provenance["method"], str) or not provenance["method"].strip() or len(provenance["method"]) > 1000:
            raise EvaluationError("Reference provenance requires a bounded nonempty method")
        if semantics == "metric_distance_m" and record["depth"] is not None:
            _number(provenance["uncertainty_m"])
        elif provenance["uncertainty_m"] is not None:
            raise EvaluationError("Nonmetric reference uncertainty must not claim meters")
    return record


def _finite_depth(depth: np.ndarray, valid: np.ndarray, semantics: str) -> None:
    if not np.isfinite(depth[valid]).all():
        raise EvaluationError("Depth contains nonfinite samples marked valid")
    if semantics == "metric_distance_m" and np.any(depth[valid] <= 0):
        raise EvaluationError("Metric samples marked valid must be positive")


def _metric(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray, minimum: int) -> dict[str, Any]:
    count = int(valid.sum())
    if count < minimum:
        return _unavailable("insufficient_reference_overlap", valid_pixels=count)
    predicted = prediction[valid].astype(np.float64)
    target = reference[valid].astype(np.float64)
    difference = predicted - target
    ratio = np.maximum(predicted / target, target / predicted)
    return {
        "status": "measured",
        "valid_pixels": count,
        "abs_rel": float(np.mean(np.abs(difference) / target)),
        "rmse_m": float(np.sqrt(np.mean(difference**2))),
        "mae_m": float(np.mean(np.abs(difference))),
        "scale_bias_ratio": float(np.median(predicted / target)),
        "delta_1": float(np.mean(ratio < 1.25)),
        "delta_2": float(np.mean(ratio < 1.25**2)),
        "delta_3": float(np.mean(ratio < 1.25**3)),
        "alignment": "none",
    }


def _relative(
    predicted: np.ndarray, target: np.ndarray, valid: np.ndarray, semantics: str, target_semantics: str, settings: dict
) -> dict:
    count = int(valid.sum())
    if count < settings["minimum_valid_pixels"]:
        return _unavailable("insufficient_reference_overlap", valid_pixels=count)
    x, y = predicted[valid].astype(np.float64), target[valid].astype(np.float64)
    inverse = semantics == "relative_inverse_depth"
    if target_semantics == "metric_distance_m":
        y = 1.0 / y if inverse else y
    elif inverse != (target_semantics == "relative_inverse_depth"):
        return _unavailable("incompatible_relative_alignment_domains")
    if np.ptp(y) <= 1e-12:
        return _unavailable("reference_has_no_depth_variation", valid_pixels=count)
    method = settings["relative_alignment"]
    scale, shift = 1.0, 0.0
    if method == "scale_shift":
        variance = float(np.sum((x - x.mean()) ** 2))
        if variance <= 1e-20:
            return _unavailable("prediction_alignment_is_degenerate", valid_pixels=count)
        scale = float(np.sum((x - x.mean()) * (y - y.mean())) / variance)
        shift = float(y.mean() - scale * x.mean())
    elif method == "scale":
        denominator = float(np.dot(x, x))
        if denominator <= 1e-20:
            return _unavailable("prediction_alignment_is_degenerate", valid_pixels=count)
        scale = float(np.dot(x, y) / denominator)
    if scale <= 0 or not np.isfinite([scale, shift]).all():
        return _unavailable("alignment_would_reverse_depth_order")
    residual = x * scale + shift - y
    return {
        "status": "measured",
        "valid_pixels": count,
        "alignment": method,
        "alignment_domain": "inverse_distance" if inverse else "distance",
        "fit_support": "same_declared_valid_reference_overlap",
        "scale": scale,
        "shift": shift,
        "aligned_rmse": float(np.sqrt(np.mean(residual**2))),
        "aligned_nrmse": float(np.sqrt(np.mean(residual**2)) / np.std(y)),
        "aligned_mae": float(np.mean(np.abs(residual))),
        "metric_accuracy_claim": False,
    }


def depth_boundaries(depth: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    """Mark both endpoints of valid four-connected native-unit depth jumps."""
    edges = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        supported = valid[left] & valid[right]
        delta = np.zeros(supported.shape, dtype=np.float64)
        np.subtract(depth[left], depth[right], out=delta, where=supported)
        jumps = supported & (np.abs(delta) >= threshold)
        edges[left] |= jumps
        edges[right] |= jumps
    return edges


def _boundaries(predicted: np.ndarray, target: np.ndarray, settings: dict) -> dict:
    from scipy.ndimage import distance_transform_edt

    count, predicted_count = int(target.sum()), int(predicted.sum())
    if count < settings["minimum_boundary_pixels"]:
        return _unavailable("insufficient_reference_boundaries", reference_pixels=count)
    tolerance = settings["boundary_tolerance_px"]
    precision = recall = 0.0
    displacement: dict[str, Any] = _unavailable("no_predicted_boundaries")
    if predicted_count:
        to_target = distance_transform_edt(~target)[predicted]
        to_prediction = distance_transform_edt(~predicted)[target]
        precision = float(np.mean(to_target <= tolerance))
        recall = float(np.mean(to_prediction <= tolerance))
        displacement = {
            "status": "measured",
            "prediction_to_reference_mean_px": float(to_target.mean()),
            "reference_to_prediction_mean_px": float(to_prediction.mean()),
            "symmetric_mean_px": float((to_target.mean() + to_prediction.mean()) / 2),
            "prediction_to_reference_p95_px": float(np.percentile(to_target, 95)),
            "reference_to_prediction_p95_px": float(np.percentile(to_prediction, 95)),
        }
    return {
        "status": "measured",
        "reference_pixels": count,
        "predicted_pixels": predicted_count,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tolerance_px": tolerance,
        "distance": "euclidean_source_grid",
        "extraction": "both_endpoints_four_connected_absolute_depth_jump",
        "displacement": displacement,
    }


def _pairs(raw: Any, shape: tuple[int, int]) -> list[dict]:
    if not isinstance(raw, list) or len(raw) > 100_000:
        raise EvaluationError("Ordinal references require a bounded list")
    seen: set[tuple] = set()
    pairs = []
    for value in raw:
        pair = _object(value, {"a", "b", "relation"})
        points = []
        for key in ("a", "b"):
            if not isinstance(pair[key], list) or len(pair[key]) != 2:
                raise EvaluationError("Ordinal points must be [row,column]")
            points.append(tuple(_integer(coordinate, 0, bound - 1) for coordinate, bound in zip(pair[key], shape)))
        if points[0] == points[1] or tuple(sorted(points)) in seen:
            raise EvaluationError("Ordinal point pairs must be distinct and unique")
        seen.add(tuple(sorted(points)))
        if pair["relation"] not in ("nearer", "farther", "equal"):
            raise EvaluationError("Unknown ordinal relation")
        pairs.append({"a": points[0], "b": points[1], "relation": pair["relation"]})
    return pairs


def _ordinal(depth: np.ndarray, valid: np.ndarray, pairs: list[dict], semantics: str, settings: dict) -> dict:
    correct = support = 0
    for pair in pairs:
        if not valid[pair["a"]] or not valid[pair["b"]]:
            continue
        delta = float(depth[pair["a"]]) - float(depth[pair["b"]])
        if semantics == "relative_inverse_depth":
            delta = -delta
        result = "equal" if abs(delta) <= settings["ordinal_tolerance"] else "nearer" if delta < 0 else "farther"
        correct += result == pair["relation"]
        support += 1
    if support < settings["minimum_ordinal_pairs"]:
        return _unavailable("insufficient_valid_ordinal_pairs", valid_pairs=support, declared_pairs=len(pairs))
    return {
        "status": "measured",
        "valid_pairs": support,
        "declared_pairs": len(pairs),
        "coverage": support / len(pairs),
        "accuracy": correct / support,
        "tolerance_native_units": settings["ordinal_tolerance"],
    }


def _scene(raw: Any, root: Path, names: tuple[str, str], settings: dict) -> dict:
    scene = _object(
        raw, {"scene_id", "independence_group", "source_sha256", "grid", "evaluation_mask", "predictions", "reference"}
    )
    source = _digest(scene["source_sha256"])
    grid = _object(scene["grid"], {"height", "width", "coordinate_space"})
    shape = (_integer(grid["height"], 1, MAX_PIXELS), _integer(grid["width"], 1, MAX_PIXELS))
    if math.prod(shape) > MAX_PIXELS or grid["coordinate_space"] != "canonical_master":
        raise EvaluationError("Grid must be a bounded canonical-master raster")
    grid_hash = hashlib.sha256(canonicalize_json(grid)).hexdigest()
    domain = (
        np.ones(shape, dtype=bool)
        if scene["evaluation_mask"] is None
        else _load_array(root, scene["evaluation_mask"], shape, mask=True)
    )
    predictions = _object(scene["predictions"], set(names))
    reference = None if scene["reference"] is None else _bound_record(scene["reference"], source, grid_hash, prediction=False)
    target_depth = target_boundary = None
    reference_valid = np.zeros(shape, dtype=bool)
    pairs: list[dict] = []
    if reference is not None:
        reference_valid = _load_array(root, reference["valid_mask"], shape, mask=True) & domain
        if reference["depth"] is not None:
            target_depth = _load_array(root, reference["depth"], shape)
            _finite_depth(target_depth, reference_valid, reference["semantics"])
        if reference["boundaries"] is not None:
            target_boundary = _load_array(root, reference["boundaries"], shape, mask=True) & reference_valid
        pairs = _pairs(reference["ordinal_pairs"], shape)
    measurements = {}
    for name in names:
        prediction = _bound_record(predictions[name], source, grid_hash, prediction=True)
        depth = _load_array(root, prediction["depth"], shape)
        valid = _load_array(root, prediction["valid_mask"], shape, mask=True)
        _finite_depth(depth, valid, prediction["semantics"])
        overlap = valid & reference_valid
        unavailable = _unavailable("depth_reference_unavailable")
        metric, relative = unavailable.copy(), unavailable.copy()
        if target_depth is not None and reference is not None:
            metric = (
                _metric(depth, target_depth, overlap, settings["minimum_valid_pixels"])
                if prediction["semantics"] == reference["semantics"] == "metric_distance_m"
                else _unavailable("absolute_metric_requires_meter_predictions_and_reference")
            )
            relative = _relative(depth, target_depth, overlap, prediction["semantics"], reference["semantics"], settings)
        boundary = _unavailable("depth_boundary_reference_unavailable")
        if target_boundary is not None:
            if int(overlap.sum()) < settings["minimum_valid_pixels"]:
                boundary = _unavailable("insufficient_reference_overlap", valid_pixels=int(overlap.sum()))
            else:
                from scipy.ndimage import minimum_filter

                # Reference annotations and extracted prediction edges need
                # identical complete neighborhoods, including at sky/ROI holes.
                boundary_support = minimum_filter(overlap, size=3, mode="constant", cval=0)
                boundary_pixels = int(boundary_support.sum())
                if boundary_pixels < settings["minimum_valid_pixels"]:
                    boundary = _unavailable("insufficient_boundary_neighborhood_support", valid_pixels=boundary_pixels)
                else:
                    edges = depth_boundaries(depth, overlap, float(prediction["boundary_threshold"])) & boundary_support
                    boundary = _boundaries(edges, target_boundary & boundary_support, settings)
                boundary["support_policy"] = "complete_3x3_reference_prediction_overlap"
        measurements[name] = {
            "recipe_sha256": prediction["recipe_sha256"],
            "semantics": prediction["semantics"],
            "units": prediction["units"],
            "boundary_threshold_native_units": prediction["boundary_threshold"],
            "comparison_support_sha256": hashlib.sha256(overlap.tobytes()).hexdigest(),
            "reference_coverage": float(overlap.sum() / reference_valid.sum()) if reference_valid.any() else None,
            "metric": metric,
            "relative": relative,
            "boundary": boundary,
            "ordinal": _ordinal(depth, overlap, pairs, prediction["semantics"], settings),
        }
    return {
        "scene_id": _name(scene["scene_id"]),
        "independence_group": _name(scene["independence_group"]),
        "source_sha256": source,
        "grid": dict(grid),
        "grid_sha256": grid_hash,
        "reference_provenance": dict(reference["provenance"]) if reference is not None else None,
        "reference_valid_pixels": int(reference_valid.sum()),
        "predictions": measurements,
    }


def _comparison(scenes: list[dict], names: tuple[str, str], settings: dict, seed: str) -> dict:
    metrics = {
        "metric.abs_rel": "lower",
        "metric.rmse_m": "lower",
        "relative.aligned_nrmse": "lower",
        "boundary.f1": "higher",
        "ordinal.accuracy": "higher",
    }
    results = {}
    for key, direction in metrics.items():
        family, metric = key.split(".")
        groups: dict[str, list[float]] = {}
        excluded_support_pairs = 0
        for scene in scenes:
            supports = [scene["predictions"][name]["comparison_support_sha256"] for name in names]
            if supports[0] != supports[1]:
                excluded_support_pairs += 1
                continue
            baseline, candidate = (scene["predictions"][name][family] for name in names)
            if baseline["status"] != "measured" or candidate["status"] != "measured":
                continue
            if family == "relative" and baseline["alignment_domain"] != candidate["alignment_domain"]:
                continue
            groups.setdefault(scene["independence_group"], []).append(candidate[metric] - baseline[metric])
        if not groups:
            results[key] = _unavailable("no_comparable_reference_pairs", excluded_different_support=excluded_support_pairs)
            continue
        values = np.array([np.mean(groups[group]) for group in sorted(groups)], dtype=np.float64)
        uncertainty: dict[str, Any] = _unavailable(
            "insufficient_independent_groups", required=settings["minimum_independent_groups"]
        )
        if len(groups) >= settings["minimum_independent_groups"]:
            rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16))
            samples = np.empty(settings["bootstrap_samples"], dtype=np.float64)
            for index in range(len(samples)):
                samples[index] = np.mean(rng.choice(values, size=len(values), replace=True))
            uncertainty = {
                "status": "descriptive",
                "method": "paired_independence_group_percentile_bootstrap",
                "samples": len(samples),
                "confidence_level": 0.95,
                "mean_delta_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
                "independence": "operator_declared_groups_not_independently_certified",
            }
        results[key] = {
            "status": "measured",
            "preferred_direction": direction,
            "delta": "candidate_minus_baseline",
            "mean_delta": float(values.mean()),
            "independent_groups": len(groups),
            "paired_scenes": sum(map(len, groups.values())),
            "excluded_different_support": excluded_support_pairs,
            "aggregation": "equal_weight_group_means",
            "uncertainty": uncertainty,
        }
    return {"baseline": names[0], "candidate": names[1], "metrics": results, "promotion": "not_established"}


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Measure frozen named candidates without running models or assuming truth."""
    path = Path(manifest_path).absolute()
    raw = _snapshot(path.parent, path.name, MAX_MANIFEST_BYTES)

    def unique(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationError("Duplicate JSON fields are not permitted")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvaluationError(f"Nonfinite JSON number: {value}")

    try:
        payload = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvaluationError("Invalid evaluation manifest JSON") from exc
    manifest = _object(payload, {"schema", "baseline", "candidate", "scenes"}, {"settings"})
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise EvaluationError("Unsupported depth quality manifest schema")
    names = (_name(manifest["baseline"]), _name(manifest["candidate"]))
    if names[0] == names[1]:
        raise EvaluationError("Baseline and candidate names must be distinct")
    settings = _settings(manifest.get("settings", {}))
    if not isinstance(manifest["scenes"], list) or not 1 <= len(manifest["scenes"]) <= 1000:
        raise EvaluationError("Evaluation requires 1..1000 declared scenes")
    source_ids, scene_ids, scenes = set(), set(), []
    total_pixels = 0
    for scene in manifest["scenes"]:
        if not isinstance(scene, Mapping) or not isinstance(scene.get("grid"), Mapping):
            raise EvaluationError("Invalid scene/grid record")
        total_pixels += _integer(scene["grid"].get("height"), 1, MAX_PIXELS) * _integer(
            scene["grid"].get("width"), 1, MAX_PIXELS
        )
        if total_pixels > MAX_TOTAL_PIXELS:
            raise EvaluationError("Evaluation exceeds aggregate pixel budget")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                result = _scene(scene, path.parent, names, settings)
        except FloatingPointError as exc:
            raise EvaluationError("Measurements exceeded finite numeric range") from exc
        if result["source_sha256"] in source_ids or result["scene_id"] in scene_ids:
            raise EvaluationError("Duplicate source or scene cannot count as independent evidence")
        source_ids.add(result["source_sha256"])
        scene_ids.add(result["scene_id"])
        scenes.append(result)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    report = {
        "schema": REPORT_SCHEMA,
        "manifest_sha256": manifest_hash,
        "recipe": EVALUATOR_RECIPE,
        "recipe_sha256": hashlib.sha256(canonicalize_json({"recipe": EVALUATOR_RECIPE, "settings": settings})).hexdigest(),
        "settings": settings,
        "frozen_manifest": payload,
        "scenes": scenes,
        "comparison": _comparison(scenes, names, settings, manifest_hash),
        "reference_authority": "declared_provenance_and_array_bytes_verified_not_independent_truth_certification",
        "source_image_bytes_verified": False,
        "production_acceptance": "not_established",
    }
    try:
        canonicalize_json(report)
    except ValueError as exc:
        raise EvaluationError("Measurements exceeded finite numeric range") from exc
    return report


def write_report(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Write one immutable canonical report; never overwrite evidence or outputs."""
    report = evaluate_manifest(manifest_path)
    path = Path(output_path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise EvaluationError("Report destination must not contain symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonicalize_json(report))
    return report
