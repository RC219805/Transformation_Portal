"""Tests for APEX evidence bundle and promotion gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.evals.apex_evidence_bundle import (
    APEX_MATERIALS_PIXEL_OPS_EMPTY,
    build_apex_evidence_bundle,
    parse_candidate_evidence,
)
from transformation_portal.evals.apex_visual import APEX_EVALSET_SCHEMA_VERSION, build_apex_eval_report, sha256_file

pytestmark = pytest.mark.unit


def _write_16bit(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(path)


def _write_evalset(root: Path, asset_path: Path, *, asset_id: str = "unit_image") -> Path:
    payload = {
        "schema_version": APEX_EVALSET_SCHEMA_VERSION,
        "evalset_id": "unit_canonical",
        "version": "v1",
        "description": "unit",
        "dataset_tier": "canonical_apex",
        "assets": [
            {
                "asset_id": asset_id,
                "asset_ref": str(asset_path.relative_to(root)),
                "sha256": sha256_file(asset_path),
                "asset_role": "canonical_apex_reference",
                "reference_path": str(asset_path.relative_to(root)),
                "canonical_bit_depth": 16,
                "canonical_format": "tiff",
                "canonical_scoring_eligible": True,
                "evaluate_at_native_resolution": True,
                "preserve_16bit_intermediates": True,
                "scene_type": "pool_exterior",
                "expected_materials": ["water", "stone"],
                "risk_zones": ["pool_edge"],
                "reject_if": ["halo"],
            }
        ],
    }
    evalset_dir = root / "evalsets" / "apex"
    evalset_dir.mkdir(parents=True, exist_ok=True)
    path = evalset_dir / "evalset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _materials_evidence(path: Path, *, applied_ops_count: int = 4, raw_authorized: bool = False) -> Path:
    payload = {
        "materials_v3_enabled": True,
        "pixel_ops_enabled": True,
        "masks_exist": True,
        "implemented_ops_exist": True,
        "applied_ops_count": applied_ops_count,
        "blocked_reason_counts": {"below_confidence_threshold": 1},
        "confidence_authority": {
            "raw_clip_similarity_authorized_pixel_ops": raw_authorized,
            "calibrated_score_type": "clip_softmax_margin_v1",
            "calibration_version": "unit",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _apex_report(tmp_path: Path, *, candidate: bool = True):
    image_path = tmp_path / "reference16.tif"
    candidate_path = tmp_path / "candidate16.tif"
    _write_16bit(image_path)
    _write_16bit(candidate_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    outputs = {"materials_v3": {"unit_image": candidate_path}} if candidate else {}
    return build_apex_eval_report(
        evalset_path,
        output_dir=tmp_path / "report",
        candidate_outputs=outputs,
        repo_root=tmp_path,
    )


def test_candidate_output_and_materials_telemetry_attach_to_bundle(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["schema_version"] == "apex_evidence_bundle.v1"
    assert bundle["promotion_verdict"] == "eligible"
    case = bundle["cases"][0]
    assert case["candidate_output"]["status"] == "present"
    assert case["materials_v3"]["status"] == "ok"
    assert case["materials_v3"]["applied_ops_count"] == 4
    assert case["materials_v3"]["confidence_authority"]["raw_clip_similarity_authorized_pixel_ops"] is False
    assert (tmp_path / "bundle" / "evidence_bundle.json").is_file()


def test_legacy_visible_delta_metrics_do_not_authorize_promotion(tmp_path):
    report = _apex_report(tmp_path)
    report["assets"][0]["candidates"][0] = {
        "candidate": "materials_v3",
        "status": "ok",
        "output_path": str(tmp_path / "candidate16.tif"),
        "metrics": {
            "ssim": 1.0,
            "lpips": None,
            "delta_e_proxy_mean_abs": 0.0,
            "delta_e_proxy_max_abs": 0.0,
        },
    }
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "invalid_metrics" in bundle["promotion_blocked_reasons"]
    assert bundle["cases"][0]["metrics_status"] == "invalid"


def test_legacy_missing_candidate_output_stays_missing_in_evidence_bundle(tmp_path):
    report = _apex_report(tmp_path)
    report["assets"][0]["candidates"][0] = {
        "candidate": "materials_v3",
        "status": "missing_candidate_output",
        "output_path": str(tmp_path / "missing_candidate16.tif"),
        "metrics": {},
    }
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "missing_candidate_output" in bundle["promotion_blocked_reasons"]
    case = bundle["cases"][0]
    assert case["candidate_output"]["status"] == "missing"
    assert case["candidate_output"]["path"] == str(tmp_path / "missing_candidate16.tif")
    assert case["metrics_status"] == "invalid"


def test_evidence_bundle_relative_output_uses_report_context(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir="nested_bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
    )

    expected = tmp_path / "report" / "nested_bundle" / "evidence_bundle.json"
    assert bundle["report_path"] == str(expected)
    assert expected.is_file()


def test_missing_materials_telemetry_blocks_materials_v3_promotion(tmp_path):
    report = _apex_report(tmp_path)

    bundle = build_apex_evidence_bundle(report, output_dir=tmp_path / "bundle", repo_root=tmp_path)

    assert bundle["promotion_verdict"] == "blocked"
    assert "missing_materials_v3_evidence" in bundle["promotion_blocked_reasons"]
    assert bundle["cases"][0]["materials_v3"]["status"] == "missing_evidence"


def test_run_scope_asset_ids_limit_promotion_scope(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        run_scope_asset_ids=["not_in_this_run"],
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "zero_canonical_eligible_assets" in bundle["promotion_blocked_reasons"]


def test_synthetic_data_blocks_promotion(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        synthetic_data=True,
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "synthetic_data" in bundle["promotion_blocked_reasons"]


def test_zero_eligible_canonical_assets_blocks_promotion(tmp_path):
    image_path = tmp_path / "delivery.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    payload = json.loads(evalset_path.read_text(encoding="utf-8"))
    payload["dataset_tier"] = "smoke_or_readiness"
    payload["assets"][0]["asset_role"] = "delivery_preview"
    payload["assets"][0]["canonical_bit_depth"] = 8
    payload["assets"][0]["canonical_format"] = "jpeg"
    payload["assets"][0]["sha256"] = sha256_file(image_path)
    evalset_path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    bundle = build_apex_evidence_bundle(report, output_dir=tmp_path / "bundle", repo_root=tmp_path)

    assert bundle["promotion_verdict"] == "blocked"
    assert "zero_canonical_eligible_assets" in bundle["promotion_blocked_reasons"]


def test_apex_materials_pixel_ops_empty_fails_apex_case(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json", applied_ops_count=0)

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert APEX_MATERIALS_PIXEL_OPS_EMPTY in bundle["promotion_blocked_reasons"]
    assert bundle["cases"][0]["case_verdict"] == "fail"
    assert bundle["cases"][0]["materials_v3"]["failure_code"] == APEX_MATERIALS_PIXEL_OPS_EMPTY


def test_raw_clip_similarity_cannot_authorize_pixel_ops(tmp_path):
    report = _apex_report(tmp_path)
    evidence = _materials_evidence(tmp_path / "materials.json", raw_authorized=True)

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "raw_clip_similarity_authorized_pixel_ops" in bundle["promotion_blocked_reasons"]


def test_parse_candidate_evidence_uses_candidate_asset_mapping():
    parsed = parse_candidate_evidence(["materials_v3:asset_1=output/evidence.json"])

    assert parsed == {"materials_v3": {"asset_1": Path("output/evidence.json")}}
