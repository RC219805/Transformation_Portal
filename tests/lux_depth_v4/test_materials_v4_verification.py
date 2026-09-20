"""Closed V3 material receipt verification with real numeric output artifacts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v4.test_materials_v4 import material_case as material_case
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4 import evidence as verification
from transformation_portal.lux_depth_v4 import pipeline, prepare

pytestmark = pytest.mark.unit


@pytest.fixture
def verified_case(material_case, monkeypatch):
    """Stub runtime attestation only; actual master/receipt bytes are produced."""
    prepared = prepare(replace(material_case.request, strength=0.2))
    result = pipeline.run(prepared)
    plan = prepared.plan.to_payload()
    completion = json.loads(result.evidence_path.read_bytes())
    completion["worker_runtime"] = {
        "backend_identity": {
            "model_canonical_key": plan["model"]["canonical_key"],
            "model_repo_id": plan["model"]["repo_id"],
            "model_lock_revision": plan["model"]["revision"],
            "actual_device": plan["device"],
        }
    }
    monkeypatch.setattr(
        verification.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: copy.deepcopy(value)),
    )
    result.evidence_path.write_bytes(canonicalize_json(completion))
    return result, completion


def _verify(result):
    return verification.verify_execution_evidence_v2(result.output_root, expected_plan_sha256=result.plan_fingerprint_sha256)


def _rebind(result, completion, relative):
    path = result.output_root / relative
    row = next(row for row in completion["artifacts"] if row["path"] == relative)
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    result.evidence_path.write_bytes(canonicalize_json(completion))


def test_v3_verification_measures_material_delta_from_exact_enhanced_baseline(verified_case):
    result, _ = verified_case
    descriptor = json.loads((result.output_root / "input-0000/photograph.json").read_bytes())
    original = np.load(result.output_root / "input-0000/source-master.npy", allow_pickle=False)
    baseline = np.load(result.output_root / "input-0000/materials-baseline.npy", allow_pickle=False)
    final = np.load(result.output_root / "input-0000/master.npy", allow_pickle=False)
    assert not np.array_equal(original, baseline), "Nonzero global enhancement must make this distinction observable"
    actual = np.abs(final.astype(np.float64) - baseline.astype(np.float64))
    assert descriptor["materials"]["max_abs_delta"] == actual.max()
    assert descriptor["materials"]["changed_pixels"] == 640
    assert _verify(result).plan_fingerprint_sha256 == result.plan_fingerprint_sha256


def test_wrong_completion_plan_schema_rejected_even_when_both_versions_admitted(verified_case):
    result, completion = verified_case
    completion["plan_schema"] = "tp.execution.plan.v2"
    result.evidence_path.write_bytes(canonicalize_json(completion))
    with pytest.raises(ValueError, match="plan schema differs"):
        _verify(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "legacy_receipt",
        "missing_baseline_descriptor",
        "baseline_metadata",
        "policy",
        "source",
        "evidence",
        "operation",
        "plan_hash",
        "changed_count",
        "max_delta",
        "mean_delta",
        "status",
        "region_count",
        "protected_count",
    ],
)
def test_resigned_descriptor_cannot_forge_material_receipt(verified_case, mutation):
    result, completion = verified_case
    relative = "input-0000/photograph.json"
    path = result.output_root / relative
    descriptor = json.loads(path.read_bytes())
    receipt = descriptor["materials"]
    if mutation == "missing_receipt":
        del descriptor["materials"]
    elif mutation == "legacy_receipt":
        receipt["schema"] = "tp.lux.materials.application.v1"
    elif mutation == "missing_baseline_descriptor":
        del descriptor["materials_baseline"]
    elif mutation == "baseline_metadata":
        descriptor["materials_baseline"]["metadata"]["injected"] = True
    elif mutation == "policy":
        receipt["response_plan"]["policy"]["max_abs_delta"] = 0.1
    elif mutation == "source":
        receipt["response_plan"]["source_sha256"] = "f" * 64
    elif mutation == "evidence":
        receipt["evidence_sha256"] = receipt["response_plan"]["evidence_sha256"] = "f" * 64
    elif mutation == "operation":
        receipt["operations_sha256"] = receipt["response_plan"]["operations_sha256"] = "f" * 64
    elif mutation == "plan_hash":
        receipt["plan_sha256"] = "f" * 64
    elif mutation == "changed_count":
        receipt["changed_pixels"] = 0
    elif mutation == "max_delta":
        receipt["max_abs_delta"] = 0
    elif mutation == "mean_delta":
        receipt["mean_abs_delta"] = 0
    elif mutation == "status":
        receipt["status"] = "abstained"
    elif mutation == "region_count":
        receipt["regions"][0]["changed_pixels"] = 0
    else:
        receipt["protected_changed_pixels"] = 1
    path.write_bytes(canonicalize_json(descriptor))
    _rebind(result, completion, relative)
    with pytest.raises(ValueError):
        _verify(result)


@pytest.mark.parametrize("mutation", ["missing", "dtype", "huge_header", "pixels", "trailing_bytes"])
def test_baseline_numeric_carrier_must_match_geometry_and_bound_master(verified_case, mutation):
    result, completion = verified_case
    relative = "input-0000/materials-baseline.npy"
    path = result.output_root / relative
    if mutation == "missing":
        path.unlink()
        completion["artifacts"] = [row for row in completion["artifacts"] if row["path"] != relative]
        result.evidence_path.write_bytes(canonicalize_json(completion))
    else:
        if mutation == "dtype":
            np.save(path, np.zeros((32, 40, 3), dtype=np.uint16), allow_pickle=False)
        elif mutation == "huge_header":
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False, "shape": (10**9, 40, 3)})
            path.write_bytes(stream.getvalue())
        elif mutation == "pixels":
            pixels = np.load(path, allow_pickle=False)
            pixels[0, 0, 0] += 0.02
            np.save(path, pixels, allow_pickle=False)
        else:
            path.write_bytes(path.read_bytes() + b"ignored")
        _rebind(result, completion, relative)
    with pytest.raises(ValueError):
        _verify(result)


def test_unbound_source_in_v3_batch_has_verified_explicit_missing_evidence(material_case, monkeypatch):
    tifffile.imwrite(material_case.request.input_dir / "second.tif", material_case.pixels, photometric="rgb")
    prepared = prepare(replace(material_case.request, strength=0.2))
    result = pipeline.run(prepared)
    plan = prepared.plan.to_payload()
    completion = json.loads(result.evidence_path.read_bytes())
    completion["worker_runtime"] = {
        "backend_identity": {
            "model_canonical_key": plan["model"]["canonical_key"],
            "model_repo_id": plan["model"]["repo_id"],
            "model_lock_revision": plan["model"]["revision"],
            "actual_device": plan["device"],
        }
    }
    monkeypatch.setattr(
        verification.DA3RuntimeIdentityEvidence,
        "from_mapping",
        lambda value: SimpleNamespace(cacheable=True, to_mapping=lambda: value),
    )
    result.evidence_path.write_bytes(canonicalize_json(completion))
    _verify(result)
    second = json.loads((result.output_root / "input-0001/photograph.json").read_bytes())
    assert second["materials"]["evidence_reason"] == "no_evidence_for_source"
    assert second["materials"]["max_abs_delta"] == 0


@pytest.mark.parametrize("version, length_bytes", [((1, 0), 2), ((2, 0), 4)])
def test_numeric_header_length_is_bounded_before_numpy_header_read(tmp_path, monkeypatch, version, length_bytes):
    relative = "materials-baseline.npy"
    raw = np.lib.format.magic(*version) + (60000).to_bytes(length_bytes, "little") + b" " * 60000
    (tmp_path / relative).write_bytes(raw)
    declared = {
        relative: {
            "path": relative,
            "kind": "array",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    }

    def unexpected_header_read(*_args, **_kwargs):
        pytest.fail("Oversized header reached NumPy before its length was bounded")

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", unexpected_header_read)
    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", unexpected_header_read)
    # This file fits the RGB geometry's outer file budget, so only the explicit
    # 4 KiB header preflight prevents the large header read.
    with pytest.raises(ValueError, match="NPY header exceeds its byte budget"):
        verification._material_array(tmp_path, relative, (100, 100, 3), declared)
