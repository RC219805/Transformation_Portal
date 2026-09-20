"""Depth quality evidence must reward correct geometry, not texture or declarations."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v5 import evaluation as ev

pytestmark = pytest.mark.unit


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    counter = 0

    def artifact(array):
        nonlocal counter
        counter += 1
        path = root / f"array-{counter}.npy"
        np.save(path, array, allow_pickle=False)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}

    grid = {"height": 16, "width": 16, "coordinate_space": "canonical_master"}
    source = "a" * 64
    common = {
        "source_sha256": source,
        "grid_sha256": hashlib.sha256(canonicalize_json(grid)).hexdigest(),
        "semantics": "metric_distance_m",
        "units": "m",
    }
    truth = np.broadcast_to(np.linspace(1, 5, 16, dtype=np.float32), (16, 16)).copy()
    validity = np.ones(truth.shape, dtype=bool)
    prediction = {
        **common,
        "recipe_sha256": "b" * 64,
        "depth": artifact(truth),
        "valid_mask": artifact(validity),
        "boundary_threshold": 0.5,
    }
    reference = {
        **common,
        "depth": artifact(truth),
        "valid_mask": artifact(validity),
        "boundaries": None,
        "ordinal_pairs": [
            {"a": [4, 3], "b": [4, 13], "relation": "nearer"},
            {"a": [9, 11], "b": [9, 2], "relation": "farther"},
        ],
        "provenance": {
            "kind": "synthetic_depth",
            "method": "Known analytic fixture, not physical camera evidence",
            "uncertainty_m": 0.0,
        },
    }
    scene = {
        "scene_id": "courtyard-01",
        "independence_group": "property-01",
        "source_sha256": source,
        "grid": grid,
        "evaluation_mask": None,
        "predictions": {"control": copy.deepcopy(prediction), "detail-candidate": copy.deepcopy(prediction)},
        "reference": reference,
    }
    manifest = {"schema": ev.MANIFEST_SCHEMA, "baseline": "control", "candidate": "detail-candidate", "scenes": [scene]}
    path = root / "manifest.json"

    def evaluate():
        path.write_bytes(canonicalize_json(manifest))
        return ev.evaluate_manifest(path)

    return root, artifact, truth, validity, manifest, path, evaluate


def _result(dataset, name="detail-candidate"):
    return dataset[-1]()["scenes"][0]["predictions"][name]


def test_identical_predictions_have_measured_zero_error_without_automatic_acceptance(dataset):
    report = dataset[-1]()
    result = report["scenes"][0]["predictions"]["detail-candidate"]
    assert result["metric"]["abs_rel"] == result["metric"]["rmse_m"] == 0
    assert result["metric"]["delta_1"] == 1
    assert result["relative"]["aligned_rmse"] == pytest.approx(0)
    assert result["ordinal"]["accuracy"] == 1
    assert result["boundary"]["status"] == "unavailable"
    comparison = report["comparison"]["metrics"]["metric.abs_rel"]
    assert comparison["mean_delta"] == 0
    assert comparison["uncertainty"]["reason"] == "insufficient_independent_groups"
    assert report["production_acceptance"] == report["comparison"]["promotion"] == "not_established"
    assert report["source_image_bytes_verified"] is False


def test_metric_scale_error_is_not_hidden_by_separately_fitted_relative_score(dataset):
    _, artifact, truth, _, manifest, _, _ = dataset
    manifest["scenes"][0]["predictions"]["detail-candidate"]["depth"] = artifact(truth * 2)
    result = _result(dataset)
    assert result["metric"]["abs_rel"] == 1
    assert result["metric"]["rmse_m"] == pytest.approx(np.sqrt(np.mean(truth**2)))
    assert result["metric"]["delta_1"] == 0
    assert result["metric"]["alignment"] == "none"
    assert result["relative"]["scale"] == pytest.approx(0.5)
    assert result["relative"]["shift"] == pytest.approx(0)
    assert result["relative"]["aligned_rmse"] == pytest.approx(0)
    assert result["relative"]["metric_accuracy_claim"] is False


def test_relative_inverse_alignment_uses_inverse_reference_domain(dataset):
    _, artifact, truth, _, manifest, _, _ = dataset
    prediction = manifest["scenes"][0]["predictions"]["detail-candidate"]
    prediction.update(semantics="relative_inverse_depth", units="arbitrary", depth=artifact(3 / truth + 7))
    result = _result(dataset)
    assert result["metric"]["status"] == "unavailable"
    assert result["relative"]["alignment_domain"] == "inverse_distance"
    assert result["relative"]["aligned_rmse"] < 1e-6
    assert result["ordinal"]["accuracy"] == 1
    report = dataset[-1]()
    assert report["comparison"]["metrics"]["relative.aligned_nrmse"]["status"] == "unavailable"


def test_wrong_order_cannot_be_repaired_with_negative_alignment_scale(dataset):
    _, artifact, truth, _, manifest, _, _ = dataset
    prediction = manifest["scenes"][0]["predictions"]["detail-candidate"]
    prediction.update(semantics="relative_distance", units="arbitrary", depth=artifact(-truth))
    result = _result(dataset)
    assert result["relative"]["reason"] == "alignment_would_reverse_depth_order"
    assert result["ordinal"]["accuracy"] == 0


def test_true_step_beats_noise_on_reference_boundaries(dataset):
    _, artifact, _, valid, manifest, _, _ = dataset
    truth = np.where(np.indices(valid.shape)[1] < 8, 1, 5).astype(np.float32)
    scene = manifest["scenes"][0]
    scene["reference"]["depth"] = artifact(truth)
    scene["reference"]["boundaries"] = artifact(ev.depth_boundaries(truth, valid, 0.5))
    scene["predictions"]["control"]["depth"] = artifact(
        np.random.default_rng(42).uniform(1, 5, valid.shape).astype(np.float32)
    )
    scene["predictions"]["detail-candidate"]["depth"] = artifact(truth)
    report = dataset[-1]()
    results = report["scenes"][0]["predictions"]
    correct, noise = results["detail-candidate"], results["control"]
    assert correct["boundary"]["f1"] == 1
    assert correct["boundary"]["displacement"]["symmetric_mean_px"] == 0
    assert noise["boundary"]["f1"] < 0.6
    assert noise["metric"]["rmse_m"] > correct["metric"]["rmse_m"]


def test_boundary_tolerance_and_displacement_are_source_grid_euclidean(dataset):
    _, artifact, _, valid, manifest, _, _ = dataset
    truth = np.where(np.indices(valid.shape)[1] < 8, 1, 5).astype(np.float32)
    shifted = np.where(np.indices(valid.shape)[1] < 9, 1, 5).astype(np.float32)
    scene = manifest["scenes"][0]
    scene["reference"]["boundaries"] = artifact(ev.depth_boundaries(truth, valid, 0.5))
    scene["predictions"]["detail-candidate"]["depth"] = artifact(shifted)
    manifest["settings"] = {"boundary_tolerance_px": 0}
    strict = _result(dataset)["boundary"]
    assert strict["f1"] == 0.5
    assert strict["displacement"]["symmetric_mean_px"] == 0.5
    manifest["settings"]["boundary_tolerance_px"] = 1
    assert _result(dataset)["boundary"]["f1"] == 1


def test_constant_prediction_is_not_perfect_for_missing_or_real_boundaries(dataset):
    _, artifact, truth, valid, manifest, _, _ = dataset
    scene = manifest["scenes"][0]
    scene["predictions"]["detail-candidate"]["depth"] = artifact(np.ones_like(truth))
    scene["reference"]["boundaries"] = artifact(np.zeros_like(valid))
    result = _result(dataset)
    assert result["boundary"]["status"] == "unavailable"
    assert result["relative"]["reason"] == "prediction_alignment_is_degenerate"
    boundary = np.zeros_like(valid)
    boundary[:, 8] = True
    scene["reference"]["boundaries"] = artifact(boundary)
    result = _result(dataset)
    assert result["boundary"]["f1"] == 0
    assert result["boundary"]["displacement"]["status"] == "unavailable"


def test_sky_and_unknown_regions_are_explicitly_excluded(dataset):
    _, artifact, truth, valid, manifest, _, _ = dataset
    scene = manifest["scenes"][0]
    domain = valid.copy()
    domain[:8] = False
    scene["evaluation_mask"] = artifact(domain)
    altered = truth.copy()
    altered[:8] = 1000
    scene["predictions"]["detail-candidate"]["depth"] = artifact(altered)
    result = _result(dataset)
    assert result["metric"]["rmse_m"] == 0
    assert result["metric"]["valid_pixels"] == 128
    assert result["ordinal"]["reason"] == "insufficient_valid_ordinal_pairs"


@pytest.mark.parametrize("mask_kind", ["evaluation", "reference", "prediction"])
def test_boundary_without_complete_neighborhood_is_unavailable(dataset, mask_kind):
    _, artifact, _, valid, manifest, _, _ = dataset
    scene = manifest["scenes"][0]
    depth = np.where(np.indices(valid.shape)[1] < 8, 1, 5).astype(np.float32)
    scene["reference"]["depth"] = artifact(depth)
    scene["reference"]["boundaries"] = artifact(ev.depth_boundaries(depth, valid, 0.5))
    for prediction in scene["predictions"].values():
        prediction["depth"] = artifact(depth)
    mask = valid.copy()
    mask[:, :8] = False
    if mask_kind == "evaluation":
        scene["evaluation_mask"] = artifact(mask)
    elif mask_kind == "reference":
        scene["reference"]["valid_mask"] = artifact(mask)
    else:
        scene["predictions"]["detail-candidate"]["valid_mask"] = artifact(mask)
    result = _result(dataset)
    assert result["metric"]["rmse_m"] == 0
    assert result["boundary"]["status"] == "unavailable"
    assert result["boundary"]["reason"] == "insufficient_reference_boundaries"
    assert result["boundary"]["reference_pixels"] == 0


def test_common_boundary_support_preserves_supported_exact_edges(dataset):
    _, artifact, _, valid, manifest, _, _ = dataset
    scene = manifest["scenes"][0]
    depth = np.where(np.indices(valid.shape)[1] < 8, 1, 5).astype(np.float32)
    scene["reference"]["boundaries"] = artifact(ev.depth_boundaries(depth, valid, 0.5))
    scene["predictions"]["detail-candidate"]["depth"] = artifact(depth)
    mask = valid.copy()
    mask[6:10, 6:10] = False
    scene["evaluation_mask"] = artifact(mask)
    boundary = _result(dataset)["boundary"]
    assert boundary["f1"] == 1
    assert boundary["displacement"]["symmetric_mean_px"] == 0
    assert boundary["predicted_pixels"] == boundary["reference_pixels"] == 16
    assert boundary["support_policy"] == "complete_3x3_reference_prediction_overlap"


def test_evaluator_recipe_identity_binds_algorithm_version_and_settings(dataset):
    first = dataset[-1]()
    assert first["recipe"] == "reference_bound_depth_metrics_v2"
    assert first["recipe_sha256"] != hashlib.sha256(canonicalize_json(first["settings"])).hexdigest()
    dataset[4]["settings"] = {"boundary_tolerance_px": 2}
    assert dataset[-1]()["recipe_sha256"] != first["recipe_sha256"]


def test_invalid_sentinels_do_not_create_boundary_edges():
    values = np.ones((16, 16), np.float32)
    valid = np.ones((16, 16), bool)
    valid[5:10, 5:10] = False
    values[~valid] = np.nan
    assert not ev.depth_boundaries(values, valid, 0.1).any()


def test_missing_references_remain_unavailable(dataset):
    dataset[4]["scenes"][0]["reference"] = None
    result = _result(dataset)
    assert all(result[key]["status"] == "unavailable" for key in ("metric", "relative", "boundary", "ordinal"))


def test_empty_prediction_validity_is_unavailable_not_perfect(dataset):
    _, artifact, _, valid, manifest, _, _ = dataset
    scene = manifest["scenes"][0]
    scene["predictions"]["detail-candidate"]["valid_mask"] = artifact(np.zeros_like(valid))
    scene["reference"]["boundaries"] = artifact(valid)
    result = _result(dataset)
    assert result["reference_coverage"] == 0
    assert all(result[key]["status"] == "unavailable" for key in ("metric", "relative", "boundary", "ordinal"))


def test_different_prediction_support_cannot_make_a_comparative_improvement(dataset):
    _, artifact, _, valid, manifest, _, _ = dataset
    valid = valid.copy()
    valid[:4] = False
    manifest["scenes"][0]["predictions"]["detail-candidate"]["valid_mask"] = artifact(valid)
    result = dataset[-1]()["comparison"]["metrics"]["metric.abs_rel"]
    assert result["status"] == "unavailable"
    assert result["excluded_different_support"] == 1


def test_boundary_and_ordinal_annotations_do_not_require_dense_depth(dataset):
    scene = dataset[4]["scenes"][0]
    scene["reference"]["depth"] = None
    scene["reference"]["provenance"]["uncertainty_m"] = None
    result = _result(dataset)
    assert result["metric"]["status"] == "unavailable"
    assert result["ordinal"]["accuracy"] == 1


def test_bootstrap_uses_independent_groups_and_is_deterministic(dataset):
    _, artifact, truth, _, manifest, _, _ = dataset
    original = manifest["scenes"][0]
    original["predictions"]["detail-candidate"]["depth"] = artifact(truth * 1.1)
    manifest["scenes"] = []
    for index in range(6):
        scene = copy.deepcopy(original)
        scene["scene_id"] = f"scene-{index}"
        scene["independence_group"] = f"group-{index // 2}"
        source = hashlib.sha256(str(index).encode()).hexdigest()
        scene["source_sha256"] = source
        scene["reference"]["source_sha256"] = source
        for prediction in scene["predictions"].values():
            prediction["source_sha256"] = source
        manifest["scenes"].append(scene)
    manifest["settings"] = {"minimum_independent_groups": 3, "bootstrap_samples": 100}
    first, second = dataset[-1](), dataset[-1]()
    assert canonicalize_json(first) == canonicalize_json(second)
    result = first["comparison"]["metrics"]["metric.abs_rel"]
    assert result["paired_scenes"] == 6
    assert result["independent_groups"] == 3
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["uncertainty"]["mean_delta_interval"] == pytest.approx([0.1, 0.1])


@pytest.mark.parametrize(
    "field,value", [("source_sha256", "c" * 64), ("grid_sha256", "d" * 64), ("units", "meters"), ("recipe_sha256", "0" * 64)]
)
def test_prediction_binding_failures_are_rejected(dataset, field, value):
    dataset[4]["scenes"][0]["predictions"]["detail-candidate"][field] = value
    with pytest.raises(ev.EvaluationError):
        dataset[-1]()


@pytest.mark.parametrize("field,value", [("kind", "rgb_edges"), ("method", ""), ("uncertainty_m", -1)])
def test_reference_requires_depth_provenance(dataset, field, value):
    dataset[4]["scenes"][0]["reference"]["provenance"][field] = value
    with pytest.raises(ev.EvaluationError):
        dataset[-1]()


@pytest.mark.parametrize("replacement", [np.zeros((16, 16), np.uint8), np.ones((8, 16), bool)])
def test_mask_dtype_and_geometry_are_exact(dataset, replacement):
    dataset[4]["scenes"][0]["predictions"]["detail-candidate"]["valid_mask"] = dataset[1](replacement)
    with pytest.raises(ev.EvaluationError, match="geometry, dtype"):
        dataset[-1]()


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_metric_samples_cannot_be_marked_valid(dataset, value):
    altered = dataset[2].copy()
    altered[2, 2] = value
    dataset[4]["scenes"][0]["predictions"]["detail-candidate"]["depth"] = dataset[1](altered)
    with pytest.raises(ev.EvaluationError, match="valid"):
        dataset[-1]()


def test_corrupt_array_is_rejected_before_metrics(dataset):
    root, _, _, _, manifest, _, _ = dataset
    path = root / manifest["scenes"][0]["predictions"]["detail-candidate"]["depth"]["path"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ev.EvaluationError, match="frozen hash"):
        dataset[-1]()


@pytest.mark.parametrize(
    "unsafe", ["../outside.npy", "/tmp/outside.npy", "sub/../file.npy", "sub//file.npy", "./file.npy", "sub\\file.npy"]
)
def test_array_paths_are_confined_and_canonical(dataset, unsafe):
    dataset[4]["scenes"][0]["predictions"]["detail-candidate"]["depth"]["path"] = unsafe
    with pytest.raises(ev.EvaluationError, match="canonical relative"):
        dataset[-1]()


@pytest.mark.parametrize("directory", [False, True])
def test_array_symlinks_are_not_followed(dataset, directory):
    root, _, _, _, manifest, _, _ = dataset
    record = manifest["scenes"][0]["predictions"]["detail-candidate"]["depth"]
    if directory:
        (root / "alias").symlink_to(root, target_is_directory=True)
        record["path"] = f"alias/{record['path']}"
    else:
        (root / "alias.npy").symlink_to(root / record["path"])
        record["path"] = "alias.npy"
    with pytest.raises(ev.EvaluationError, match="confined evidence"):
        dataset[-1]()


def test_npy_header_bomb_is_rejected_before_numpy_header_parser(dataset, monkeypatch):
    root, _, _, _, manifest, _, _ = dataset
    raw = b"\x93NUMPY\x02\x00" + (2**32 - 1).to_bytes(4, "little") + b"short"
    (root / "bomb.npy").write_bytes(raw)
    record = {"path": "bomb.npy", "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    manifest["scenes"][0]["predictions"]["detail-candidate"]["depth"] = record
    monkeypatch.setattr(
        np.lib.format, "read_array_header_2_0", lambda *args, **kwargs: pytest.fail("Unbounded header parser called")
    )
    with pytest.raises(ev.EvaluationError, match="bounded NPY header"):
        dataset[-1]()


def test_duplicate_scene_identity_does_not_inflate_uncertainty(dataset):
    dataset[4]["scenes"].append(copy.deepcopy(dataset[4]["scenes"][0]))
    with pytest.raises(ev.EvaluationError, match="Duplicate source or scene"):
        dataset[-1]()


def test_duplicate_json_fields_and_unknown_contract_fields_fail(dataset):
    _, _, _, _, manifest, path, _ = dataset
    path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(ev.EvaluationError, match="manifest JSON"):
        ev.evaluate_manifest(path)
    manifest["pretend_quality_passed"] = True
    with pytest.raises(ev.EvaluationError, match="unknown fields"):
        dataset[-1]()


def test_report_is_canonical_frozen_and_never_overwrites(dataset, tmp_path):
    dataset[-1]()
    path = tmp_path / "report.json"
    report = ev.write_report(dataset[5], path)
    assert path.read_bytes() == canonicalize_json(report)
    assert report["frozen_manifest"] == dataset[4]
    assert report["manifest_sha256"] == hashlib.sha256(dataset[5].read_bytes()).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        ev.write_report(dataset[5], path)


def test_cli_writes_report_and_corrupt_inputs_return_error(dataset, tmp_path):
    dataset[-1]()
    cli = Path(__file__).resolve().parents[2] / "scripts/validation/evaluate_lux_depth_v5.py"
    output = tmp_path / "result.json"
    command = [sys.executable, str(cli), "--manifest", str(dataset[5]), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_bytes())["schema"] == ev.REPORT_SCHEMA
    dataset[5].write_text("invalid")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 1
    assert "Depth evaluation incomplete" in completed.stderr
