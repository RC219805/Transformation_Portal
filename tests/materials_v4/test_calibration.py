"""Draft calibration validates observations and never manufactures edit authority."""

import hashlib
import sys
from copy import deepcopy
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.materials_v4 import calibration as calibration_module
from transformation_portal.materials_v4 import inference
from transformation_portal.materials_v4.__main__ import main
from transformation_portal.materials_v4.artifacts import load_evidence, write_evidence
from transformation_portal.materials_v4.calibration import (
    CALIBRATION_INPUT_SCHEMA,
    CalibrationModel,
    _isotonic,
    fit_calibration,
    load_calibration,
)
from transformation_portal.materials_v4.contracts import MaterialEvidence, MaterialsError, RegionEvidence
from transformation_portal.materials_v4.engine import ResponsePolicy, apply_response, plan_response
from transformation_portal.materials_v4.inference import InferenceConfig, infer_materials, inference_recipe, prepare_inference
from transformation_portal.materials_v4.proposal import Proposal
from transformation_portal.materials_v4.semantics import SemanticObservation

pytestmark = pytest.mark.unit


def _fit_metadata(*args, **kwargs):
    return calibration_module._fit_metadata(*args, **kwargs)


def decode_calibration_json_object(data):
    return calibration_module.decode_calibration_json_object(data)


def _write(path, payload):
    path.write_bytes(canonicalize_json(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def calibration_fixture(tmp_path, monkeypatch):
    """Small generated unit files prove contracts, not photographic acceptance."""
    sam, clip = tmp_path / "sam.pt", tmp_path / "clip.safetensors"
    sam.write_bytes(b"fixture-sam")
    clip.write_bytes(b"fixture-clip")

    class Checkpoint:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["fixture"]

    def safe_open(path, *, framework, device):
        assert path == str(clip) and framework == "pt" and device == "cpu"
        return Checkpoint()

    safetensors_stub = ModuleType("safetensors")
    safetensors_stub.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_stub)
    monkeypatch.setattr(inference, "SAM2_LARGE_SHA256", hashlib.sha256(sam.read_bytes()).hexdigest())
    monkeypatch.setattr(inference, "_runtime_payload", lambda device: {"actual_device": device, "version": "test"})
    master = ImageMaster(np.full((28, 28, 3), 0.5, np.float32), "a" * 64, 16)
    config = InferenceConfig(sam, clip, proxy_longest_side=28)
    recipe = inference_recipe(prepare_inference(master, config))
    source_root, data_root = tmp_path / "sources", tmp_path / "data"
    source_root.mkdir()
    data_root.mkdir()
    mask_path = data_root / "target.npy"
    np.save(mask_path, np.ones((28, 28), np.float32), allow_pickle=False)
    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    records, membership, bindings = [], [], []
    for index, (score, correct) in enumerate(zip([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [False, True, False, True, True, True])):
        source_path = source_root / f"source-{index}.png"
        Image.new("RGB", (28, 28), (index, 25, 50)).save(source_path)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        group, record_id, label = f"property-{index % 2}", f"record-{index}", "water" if correct else "sky"
        evidence = MaterialEvidence(
            source_sha,
            (28, 28),
            (
                RegionEvidence(
                    "region-0000",
                    "water",
                    np.ones((28, 28), bool),
                    score,
                    provenance="inferred",
                    score_type="uncalibrated_clip_ranking_v1",
                ),
            ),
            producer=recipe,
        )
        bundle = data_root / f"bundle-{index}"
        bundle.mkdir()
        write_evidence(evidence, bundle / "evidence.json")
        annotation = {
            "schema": "tp.materials.annotation.v1",
            "source_sha256": source_sha,
            "shape": [28, 28],
            "property_group": group,
            "label": label,
            "mask_path": "target.npy",
            "mask_sha256": mask_sha,
            "annotator": "unit-annotator",
            "reviewer": "unit-reviewer",
            "review_status": "reviewed",
        }
        annotation_sha = _write(data_root / f"annotation-{index}.json", annotation)
        membership.append(
            {"source_path": source_path.name, "source_sha256": source_sha, "property_group": group, "split": "calibration"}
        )
        records.append(
            {
                "record_id": record_id,
                "source_sha256": source_sha,
                "property_group": group,
                "split": "calibration",
                "true_label": label,
                "predicted_label": "water",
                "confidence": score,
                "predicted_mask": None,
                "target_mask": None,
            }
        )
        bindings.append(
            {
                "record_id": record_id,
                "evidence_path": f"bundle-{index}/evidence.json",
                "evidence_sha256": hashlib.sha256((bundle / "evidence.json").read_bytes()).hexdigest(),
                "region_id": "region-0000",
                "annotation_path": f"annotation-{index}.json",
                "annotation_sha256": annotation_sha,
            }
        )
    Image.new("RGB", (28, 28), "red").save(source_root / "sealed.png")
    membership.append(
        {
            "source_path": "sealed.png",
            "source_sha256": hashlib.sha256((source_root / "sealed.png").read_bytes()).hexdigest(),
            "property_group": "sealed-property",
            "split": "test",
        }
    )
    payload = {
        "schema": CALIBRATION_INPUT_SCHEMA,
        "evaluation": {
            "schema": "tp.materials.evaluation_input.v1",
            "records": records,
            "selection_sources": [record["source_sha256"] for record in records],
        },
        "split_membership": membership,
        "bindings": bindings,
    }
    return payload, source_root, data_root, master, config


def _fit(fixture, payload=None):
    original, source_root, data_root, *_ = fixture
    return fit_calibration(original if payload is None else payload, source_root=source_root, data_root=data_root)


def test_weighted_isotonic_pools_ties_and_violations_deterministically():
    pairs = [(0.2, True), (0.2, False), (0.3, False), (0.8, True)]
    assert _isotonic(pairs) == [[0.2, 0.3, 1 / 3, 3], [0.8, 0.8, 1.0, 1]]
    assert _isotonic(list(reversed(pairs))) == _isotonic(pairs)


def test_fitting_is_source_bound_deterministic_and_does_not_certify_acceptance(calibration_fixture):
    fitted = _fit(calibration_fixture)
    reordered = deepcopy(calibration_fixture[0])
    reordered["evaluation"]["records"].reverse()
    reordered["bindings"].reverse()
    reordered["split_membership"].reverse()
    assert _fit(calibration_fixture, reordered).to_payload() == fitted.to_payload()
    assert fitted.receipt.classes == ("water",)
    assert fitted.to_payload()["review_status"] == "unreviewed"
    assert fitted.fit["acceptance"] == "not_assessed"
    assert fitted.receipt.metrics["held_out_acceptance"] == "not_assessed"
    assert fitted.probability("water", 0.9) == 1.0
    assert fitted.probability("water", 0.6) == 0.5
    assert fitted.probability("water", 0.7) == 0.5
    assert fitted.probability("water", 0.1) is None
    assert fitted.probability("water", 0.99) is None
    assert fitted.probability("sky", 0.9) is None
    assert ResponsePolicy().trusted_calibration_sha256 == ()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("test_record", "calibration records only"),
        ("property_leak", "Property-group leak"),
        ("source_bytes", "source bytes differ"),
        ("score", "original uncalibrated inference"),
        ("reviewer", "distinct declared human"),
        ("annotation_hash", "Annotation bytes differ"),
        ("mask_bytes", "mask bytes differ"),
        ("membership", "source/property membership"),
        ("selection", "exactly the fitting sources"),
        ("duplicate_region", "Duplicate inference region"),
        ("all_correct", "No class has adequate"),
    ],
)
def test_invalid_calibration_provenance_and_split_inputs_fail_closed(calibration_fixture, mutation, match):
    payload = deepcopy(calibration_fixture[0])
    source_root, data_root = calibration_fixture[1:3]
    if mutation == "test_record":
        payload["evaluation"]["records"][0]["split"] = "validation"
        # Give this source its own group so split parsing reaches the fitter guard.
        payload["evaluation"]["records"][0]["property_group"] = "validation-property"
    elif mutation == "property_leak":
        payload["split_membership"][-1]["property_group"] = "property-0"
    elif mutation == "source_bytes":
        (source_root / "source-0.png").write_bytes(b"changed")
    elif mutation == "score":
        payload["evaluation"]["records"][0]["confidence"] = 0.4
    elif mutation in {"reviewer", "all_correct"}:
        for index in range(6 if mutation == "all_correct" else 1):
            path = data_root / f"annotation-{index}.json"
            from transformation_portal.core.execution_plan import decode_bounded_json_object

            annotation = decode_bounded_json_object(path.read_bytes())
            if mutation == "reviewer":
                annotation["reviewer"] = annotation["annotator"]
            else:
                annotation["label"] = "water"
                payload["evaluation"]["records"][index]["true_label"] = "water"
            payload["bindings"][index]["annotation_sha256"] = _write(path, annotation)
    elif mutation == "annotation_hash":
        payload["bindings"][0]["annotation_sha256"] = "0" * 64
    elif mutation == "mask_bytes":
        (data_root / "target.npy").write_bytes(b"changed")
    elif mutation == "membership":
        payload["split_membership"][0]["property_group"] = "other-property"
    elif mutation == "selection":
        payload["evaluation"]["selection_sources"].pop()
    elif mutation == "duplicate_region":
        record = deepcopy(payload["evaluation"]["records"][0])
        record["record_id"] = "duplicate"
        payload["evaluation"]["records"].append(record)
        binding = deepcopy(payload["bindings"][0])
        binding["record_id"] = "duplicate"
        payload["bindings"].append(binding)
    with pytest.raises(MaterialsError, match=match):
        _fit(calibration_fixture, payload)


def test_artifact_loading_checks_hashes_knots_and_never_accepts_reviewed_claim(calibration_fixture, tmp_path):
    fitted = _fit(calibration_fixture)
    path = tmp_path / "calibration.json"
    _write(path, fitted.to_payload())
    loaded, file_receipt = load_calibration(path)
    assert loaded.to_payload() == fitted.to_payload()
    assert file_receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    changed = fitted.to_payload()
    changed["review_status"] = "reviewed"
    _write(path, changed)
    with pytest.raises(MaterialsError, match="always unreviewed"):
        load_calibration(path)
    changed = fitted.to_payload()
    changed["fit"]["classes"]["water"]["knots"][0][2] = 0.9
    _write(path, changed)
    with pytest.raises(MaterialsError, match="does not bind"):
        load_calibration(path)


@pytest.mark.parametrize("alteration", ["producer_metadata", "unrelated_region", "renamed_region"])
def test_repackaged_region_cannot_inflate_fitting_support(calibration_fixture, alteration):
    payload = deepcopy(calibration_fixture[0])
    data_root = calibration_fixture[2]
    record = deepcopy(payload["evaluation"]["records"][0])
    binding = deepcopy(payload["bindings"][0])
    original = load_evidence(data_root / binding["evidence_path"], record["source_sha256"], (28, 28))
    if alteration == "producer_metadata":
        repackaged = replace(original, producer={**original.producer, "unrelated_note": "copied bundle"})
    elif alteration == "unrelated_region":
        extra_mask = np.zeros((28, 28), bool)
        extra_mask[0, 0] = True
        extra_region = replace(original.regions[0], region_id="unrelated", mask=extra_mask)
        repackaged = replace(original, regions=original.regions + (extra_region,))
    else:
        renamed = replace(original.regions[0], region_id="renamed")
        repackaged = replace(original, regions=(renamed,))
        binding["region_id"] = renamed.region_id
    assert repackaged.content_hash() != original.content_hash()
    bundle = data_root / "repackaged"
    bundle.mkdir()
    write_evidence(repackaged, bundle / "evidence.json")
    record["record_id"] = binding["record_id"] = "repackaged-record"
    binding["evidence_path"] = "repackaged/evidence.json"
    binding["evidence_sha256"] = hashlib.sha256((bundle / "evidence.json").read_bytes()).hexdigest()
    payload["evaluation"]["records"].append(record)
    payload["bindings"].append(binding)
    with pytest.raises(MaterialsError, match="Duplicate inference region"):
        _fit(calibration_fixture, payload)


def test_calibrated_inference_preserves_rejections_and_requires_explicit_policy_trust(
    calibration_fixture, tmp_path, monkeypatch
):
    fitted = _fit(calibration_fixture)
    path = tmp_path / "calibration.json"
    _write(path, fitted.to_payload())
    master, config = calibration_fixture[3:]
    prepared = prepare_inference(master, replace(config, calibration_artifact=path))
    monkeypatch.setattr(
        inference,
        "generate_proposals",
        lambda image, *args, **kwargs: (
            Proposal(np.ones(image.shape[:2], bool), 0.99, 0.98),
            Proposal(np.ones(image.shape[:2], bool), 0.99, 0.98),
        ),
    )
    monkeypatch.setattr(
        inference,
        "classify_proposals",
        lambda *args, **kwargs: (
            SemanticObservation("water", 0.9, 0.8, 0.8, None),
            SemanticObservation("unknown", 0.9, 0.1, 0.8, "below_absolute_similarity"),
        ),
    )
    evidence = infer_materials(master, prepared)
    assert evidence.calibration.content_hash() == fitted.receipt.content_hash()
    assert evidence.regions[0].semantic_confidence == 1.0
    assert evidence.regions[0].score_type == "calibrated_material_probability_v1"
    assert evidence.regions[1].score_type == "uncalibrated_clip_ranking_v1"
    policy = ResponsePolicy(min_coverage_px=1, min_confidence=0.0)
    plan = plan_response(master, evidence, policy)
    result, _ = apply_response(master, evidence, plan)
    assert all(region.status != "apply" for region in plan.regions)
    np.testing.assert_array_equal(result.pixels, master.pixels)
    assert evidence.producer["classifier_sha256"] == fitted.receipt.classifier_sha256


def test_calibration_identity_mismatch_and_mutation_fail_before_model_load(calibration_fixture, tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    _write(path, _fit(calibration_fixture).to_payload())
    master, config = calibration_fixture[3:]
    with pytest.raises(MaterialsError, match="recipe differs"):
        prepare_inference(master, replace(config, calibration_artifact=path, min_margin=0.1))
    prepared = prepare_inference(master, replace(config, calibration_artifact=path))
    path.write_bytes(b"changed")
    monkeypatch.setattr(
        inference, "generate_proposals", lambda *args, **kwargs: pytest.fail("Model loaded with changed calibration")
    )
    with pytest.raises(MaterialsError, match="Calibration artifact changed"):
        infer_materials(master, prepared)


def test_same_recipe_calibration_substitution_fails_before_model_load(calibration_fixture, tmp_path, monkeypatch):
    from transformation_portal.core.execution_plan import decode_bounded_json_object

    fitted = _fit(calibration_fixture)
    path = tmp_path / "calibration.json"
    _write(path, fitted.to_payload())
    master, config = calibration_fixture[3:]
    prepared = prepare_inference(master, replace(config, calibration_artifact=path))
    payload = deepcopy(calibration_fixture[0])
    annotation_path = calibration_fixture[2] / "annotation-0.json"
    annotation = decode_bounded_json_object(annotation_path.read_bytes())
    annotation["label"] = "water"
    payload["bindings"][0]["annotation_sha256"] = _write(annotation_path, annotation)
    payload["evaluation"]["records"][0]["true_label"] = "water"
    different_fit = _fit(calibration_fixture, payload)
    assert different_fit.fit["recipe"] == fitted.fit["recipe"]
    assert different_fit.to_payload() != fitted.to_payload()
    assert different_fit.probability("water", 0.6) != fitted.probability("water", 0.6)
    substituted = replace(prepared, calibration=different_fit)
    monkeypatch.setattr(
        inference, "generate_proposals", lambda *args, **kwargs: pytest.fail("Model loaded with substituted calibration")
    )
    with pytest.raises(MaterialsError, match="Prepared calibration differs from frozen artifact"):
        infer_materials(master, substituted)


def test_recipe_can_cross_sources_but_cannot_cross_inference_options(calibration_fixture):
    master, config = calibration_fixture[3:]
    prepared = prepare_inference(master, config)
    another = replace(prepared, source_sha256="b" * 64, master_sha256="c" * 64, proxy_sha256="d" * 64)
    assert inference_recipe(another) == inference_recipe(prepared)
    other_paths = replace(config, clip_checkpoint=config.clip_checkpoint.parent / "other.safetensors")
    assert inference_recipe(replace(prepared, config=other_paths)) == inference_recipe(prepared)
    assert inference_recipe(replace(prepared, config=replace(config, points_per_side=1))) != inference_recipe(prepared)


@pytest.mark.parametrize(
    "option,value",
    [
        ("device", "mps"),
        ("proxy_longest_side", 32),
        ("max_proposals", 32),
        ("classifier_batch_size", 4),
        ("min_top_probability", 0.6),
        ("min_similarity", 0.3),
        ("min_margin", 0.1),
        ("points_per_side", 8),
        ("points_per_batch", 2),
        ("max_mask_bytes", 256_000_000),
        ("max_proxy_mask_bytes", 32_000_000),
        ("max_candidate_bytes", 268_435_456),
    ],
)
def test_recipe_binds_each_mutable_proposal_and_classifier_setting(calibration_fixture, option, value):
    master, config = calibration_fixture[3:]
    prepared = prepare_inference(master, config)
    changed = replace(prepared, config=replace(config, **{option: value}))
    assert inference_recipe(changed) != inference_recipe(prepared)


def test_calibration_mutation_during_inference_cannot_publish_evidence(calibration_fixture, tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    _write(path, _fit(calibration_fixture).to_payload())
    master, config = calibration_fixture[3:]
    prepared = prepare_inference(master, replace(config, calibration_artifact=path))
    monkeypatch.setattr(inference, "generate_proposals", lambda *args, **kwargs: ())

    def mutate(*args, **kwargs):
        path.write_bytes(b"changed-during-inference")
        return ()

    monkeypatch.setattr(inference, "classify_proposals", mutate)
    with pytest.raises(MaterialsError, match="Calibration artifact changed"):
        infer_materials(master, prepared)


def test_annotation_mask_must_describe_the_predicted_region(calibration_fixture):
    payload = deepcopy(calibration_fixture[0])
    data_root = calibration_fixture[2]
    target = np.zeros((28, 28), np.float32)
    target[0, 0] = 1
    np.save(data_root / "target.npy", target, allow_pickle=False)
    from transformation_portal.core.execution_plan import decode_bounded_json_object

    annotation_path = data_root / "annotation-0.json"
    annotation = decode_bounded_json_object(annotation_path.read_bytes())
    annotation["mask_sha256"] = hashlib.sha256((data_root / "target.npy").read_bytes()).hexdigest()
    payload["bindings"][0]["annotation_sha256"] = _write(annotation_path, annotation)
    with pytest.raises(MaterialsError, match="at least half"):
        _fit(calibration_fixture, payload)


def test_fitting_cli_writes_an_unreviewed_artifact_and_rejects_overwrite(calibration_fixture, tmp_path, capsys):
    payload, source_root, data_root, *_ = calibration_fixture
    input_path, destination = tmp_path / "input.json", tmp_path / "fit"
    output = destination / "calibration.json"
    _write(input_path, payload)
    args = [
        "fit-calibration",
        "--input",
        str(input_path),
        "--source-root",
        str(source_root),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(destination),
    ]
    assert main(args) == 0
    assert load_calibration(output)[0].receipt.classes == ("water",)
    assert '"acceptance": "not_assessed"' in capsys.readouterr().out
    assert main(args) == 1


def _capacity_model(fixture, count):
    """Invented artifact rows test serialization capacity, never corpus adequacy."""
    original = _fit(fixture)
    fit = original.to_payload()["fit"]
    prototype = fit["annotation_provenance"][0]
    fit["annotation_provenance"] = [{**prototype, "record_id": f"capacity-{index}"} for index in range(count)]
    fit["classes"]["water"] = {
        "knots": [[0.5, 0.95, 0.5, count]],
        "support_records": count,
        "property_groups": ["property-0", "property-1"],
    }
    receipt = replace(original.receipt, artifact_sha256=hashlib.sha256(canonicalize_json(fit)).hexdigest())
    return fit, receipt


@pytest.mark.parametrize("count", [1025, 10_000])
def test_artifact_capacity_roundtrips_full_provenance_beyond_shared_metadata_budget(calibration_fixture, tmp_path, count):
    fit, receipt = _capacity_model(calibration_fixture, count)
    model = CalibrationModel(fit, receipt)
    encoded = canonicalize_json(model.to_payload())
    assert len(encoded) > 65_536
    if count == 10_000:
        assert len(encoded) > 1024 * 1024
    path = tmp_path / "capacity.json"
    path.write_bytes(encoded)
    loaded, identity = load_calibration(path)
    assert identity["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert loaded.to_payload() == model.to_payload()
    assert len(loaded.fit["annotation_provenance"]) == count
    assert loaded.receipt.artifact_sha256 == hashlib.sha256(canonicalize_json(loaded.to_payload()["fit"])).hexdigest()
    from transformation_portal.materials_v4.contracts import _metadata

    with pytest.raises(MaterialsError, match="metadata (sequence|byte)"):
        _metadata(fit)


def test_artifact_rejects_provenance_over_record_limit(calibration_fixture):
    fit, receipt = _capacity_model(calibration_fixture, 10_001)
    with pytest.raises(MaterialsError, match="sequence exceeds budget"):
        CalibrationModel(fit, receipt)


def test_full_artifact_envelope_counts_toward_byte_limit(calibration_fixture, monkeypatch):
    from transformation_portal.materials_v4 import calibration

    fitted = _fit(calibration_fixture)
    payload = fitted.to_payload()
    body_size = len(canonicalize_json(payload["fit"]))
    envelope_size = len(canonicalize_json(payload))
    assert body_size < envelope_size
    monkeypatch.setattr(calibration, "_MAX_JSON_BYTES", envelope_size - 1)
    with pytest.raises(MaterialsError, match="artifact exceeds byte budget"):
        CalibrationModel(payload["fit"], fitted.receipt)


def test_fit_metadata_is_deeply_immutable_and_preserves_all_fields(calibration_fixture):
    fit, receipt = _capacity_model(calibration_fixture, 1025)
    fit["annotation_provenance"][0]["elapsed_seconds"] = 0.25
    receipt = replace(receipt, artifact_sha256=hashlib.sha256(canonicalize_json(fit)).hexdigest())
    model = CalibrationModel(fit, receipt)
    expected = model.to_payload()
    fit["classes"]["water"]["knots"][0][2] = 0.9
    fit["annotation_provenance"].clear()
    assert model.to_payload() == expected
    assert model.fit["annotation_provenance"][0]["elapsed_seconds"] == 0.25
    with pytest.raises(TypeError):
        model.fit["classes"]["water"]["knots"][0][2] = 0.75
    returned = model.to_payload()
    returned["fit"]["annotation_provenance"].clear()
    assert model.to_payload() == expected
    assert model.receipt.artifact_sha256 == hashlib.sha256(canonicalize_json(expected["fit"])).hexdigest()


def test_fit_metadata_exact_canonical_byte_boundary_and_json_types():
    from transformation_portal.core.image_artifact import metadata_payload

    payload = {"unicode": "é\n", "values": [True, None, -3, 0.125], "timing": {"duration_seconds": 1.0}}
    encoded = canonicalize_json(payload)
    frozen = _fit_metadata(payload, maximum_bytes=len(encoded))
    assert canonicalize_json(metadata_payload(frozen)) == encoded
    with pytest.raises(MaterialsError, match="byte budget"):
        _fit_metadata(payload, maximum_bytes=len(encoded) - 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 2**63, -(2**63) - 1, object()])
def test_fit_metadata_rejects_nonfinite_or_unsupported_scalars(value):
    with pytest.raises(MaterialsError, match="bounded finite JSON"):
        _fit_metadata({"nested": [{"value": value}]})


def test_fit_metadata_rejects_cycles_depth_strings_and_nonstring_keys():
    cycle = []
    cycle.append(cycle)
    nested = None
    for _ in range(14):
        nested = [nested]
    for payload, match in [
        ({"cycle": cycle}, "nesting exceeds"),
        ({"nested": nested}, "nesting exceeds"),
        ({"long": "x" * 4097}, "string exceeds"),
        ({1: "nonstring"}, "keys exceed"),
        ({"surrogate": "\ud800"}, "valid UTF-8"),
    ]:
        with pytest.raises(MaterialsError, match=match):
            _fit_metadata(payload)


@pytest.mark.parametrize(
    "data",
    [
        b'{"key":1,"key":2}',
        b'{"nested":{"key":1,"key":2}}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e999}',
        b'{"value":9223372036854775808}',
        b'{"value":"\xff"}',
        b"[]",
        b'{"truncated":',
        b"",
    ],
)
def test_calibration_json_decoder_rejects_ambiguous_invalid_or_unbounded_values(data):
    with pytest.raises(MaterialsError):
        decode_calibration_json_object(data)


def test_calibration_json_exact_physical_limit_and_independent_core_limit():
    from transformation_portal.core.execution_plan import ExecutionPlanError, decode_bounded_json_object

    base = b'{"value":null}'
    data = base + b" " * (16 * 1024 * 1024 - len(base))
    assert decode_calibration_json_object(data) == {"value": None}
    with pytest.raises(MaterialsError, match="JSON exceeds byte budget"):
        decode_calibration_json_object(data + b" ")
    with pytest.raises(ExecutionPlanError):
        decode_bounded_json_object(data)


def test_fit_cli_accepts_bounded_input_larger_than_one_mib(calibration_fixture, tmp_path):
    payload, source_root, data_root, *_ = calibration_fixture
    input_path = tmp_path / "large-input.json"
    encoded = canonicalize_json(payload)
    input_path.write_bytes(encoded + b" " * (1024 * 1024 + 1 - len(encoded)))
    assert (
        main(
            [
                "fit-calibration",
                "--input",
                str(input_path),
                "--source-root",
                str(source_root),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(tmp_path / "large-fit"),
            ]
        )
        == 0
    )
    assert load_calibration(tmp_path / "large-fit" / "calibration.json")[0].fit["acceptance"] == "not_assessed"


@pytest.mark.parametrize("binding_field", ["evidence_path", "annotation_path"])
def test_fitting_paths_cannot_escape_data_root(calibration_fixture, binding_field):
    payload = deepcopy(calibration_fixture[0])
    payload["bindings"][0][binding_field] = "../outside.json"
    with pytest.raises(MaterialsError, match="relative path"):
        _fit(calibration_fixture, payload)


def test_calibration_loader_refuses_symlink_artifact(calibration_fixture, tmp_path):
    from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError

    path, link = tmp_path / "artifact.json", tmp_path / "link.json"
    _write(path, _fit(calibration_fixture).to_payload())
    link.symlink_to(path.name)
    with pytest.raises(ArtifactEvidenceError, match="opened safely"):
        load_calibration(link)
