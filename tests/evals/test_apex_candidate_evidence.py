"""Tests for APEX evidence bundle and promotion gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# build_apex_eval_report calls load_16bit_tiff under the hood; without tifffile
# every reference read short-circuits to "unreadable_reference" with an empty
# metrics dict, which makes downstream `metrics["visible_delta"]` mutations
# KeyError. Skip the whole module rather than emit a misleading failure.
pytest.importorskip("tifffile")

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


def _write_multi_asset_evalset(root: Path, asset_paths: dict[str, Path]) -> Path:
    payload = {
        "schema_version": APEX_EVALSET_SCHEMA_VERSION,
        "evalset_id": "unit_canonical_multi",
        "version": "v1",
        "description": "unit multi",
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
            for asset_id, asset_path in asset_paths.items()
        ],
    }
    evalset_dir = root / "evalsets" / "apex_multi"
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


def _multi_apex_report(tmp_path: Path, *, candidate_asset_ids: set[str]):
    asset_paths = {
        "pool_image": tmp_path / "reference_pool16.tif",
        "kitchen_image": tmp_path / "reference_kitchen16.tif",
    }
    for asset_path in asset_paths.values():
        _write_16bit(asset_path)
    evalset_path = _write_multi_asset_evalset(tmp_path, asset_paths)
    outputs: dict[str, dict[str, Path]] = {"materials_v3": {}}
    for asset_id in candidate_asset_ids:
        candidate_path = tmp_path / f"{asset_id}_candidate16.tif"
        _write_16bit(candidate_path)
        outputs["materials_v3"][asset_id] = candidate_path
    return build_apex_eval_report(
        evalset_path,
        output_dir=tmp_path / "report",
        candidate_outputs=outputs,
        repo_root=tmp_path,
    )


def test_candidate_output_and_materials_telemetry_attach_to_bundle(tmp_path):
    report = _apex_report(tmp_path)
    report["assets"][0]["candidates"][0]["mask_evidence"] = {
        "status": "ok",
        "reported_path": "external/materials_v3_masks.npz",
        "format": "npz",
        "mask_count": 1,
        "union_shape": [8, 8],
        "union_nonzero_pixels": 16,
        "source": "candidate_mask",
    }
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
    assert case["mask_evidence"] == {
        "status": "ok",
        "reported_path": "external/materials_v3_masks.npz",
        "format": "npz",
        "mask_count": 1,
        "union_shape": [8, 8],
        "union_nonzero_pixels": 16,
        "source": "candidate_mask",
    }
    assert (tmp_path / "bundle" / "evidence_bundle.json").is_file()


def test_invalid_explicit_mask_evidence_blocks_promotion_via_invalid_metrics(tmp_path):
    report = _apex_report(tmp_path)
    candidate = report["assets"][0]["candidates"][0]
    candidate["status"] = "metrics_not_computed"
    candidate["metrics_authoritative"] = False
    candidate["mask_evidence"] = {
        "status": "invalid",
        "reason": "candidate_mask_dimension_mismatch",
        "reported_path": "external/materials_v3_masks.npz",
    }
    candidate["metrics"]["outside_mask_delta"] = {
        "status": "invalid_input",
        "reason": "candidate_mask_dimension_mismatch",
        "value": None,
        "comparison": {},
    }
    candidate["metrics"]["seam_halo_score"] = {
        "status": "invalid_input",
        "reason": "candidate_mask_dimension_mismatch",
        "value": None,
        "comparison": {},
    }
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    case = bundle["cases"][0]
    assert bundle["promotion_verdict"] == "blocked"
    assert "invalid_metrics" in bundle["promotion_blocked_reasons"]
    assert case["metrics_status"] == "invalid"
    assert case["mask_evidence"]["status"] == "invalid"
    assert case["metrics"]["outside_mask_delta"]["status"] != "mask_missing"


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


def test_unscoped_multi_asset_promotion_requires_all_canonical_candidate_outputs(tmp_path):
    report = _multi_apex_report(tmp_path, candidate_asset_ids={"pool_image"})
    evidence = _materials_evidence(tmp_path / "pool_materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"pool_image": evidence}},
        repo_root=tmp_path,
    )

    cases_by_asset = {case["asset_id"]: case for case in bundle["cases"]}
    assert bundle["promotion_verdict"] == "blocked"
    assert "missing_candidate_output" in bundle["promotion_blocked_reasons"]
    assert "invalid_metrics" in bundle["promotion_blocked_reasons"]
    assert cases_by_asset["pool_image"]["candidate_output"]["status"] == "present"
    assert cases_by_asset["kitchen_image"]["candidate_output"]["status"] == "missing"
    assert cases_by_asset["kitchen_image"]["metrics_status"] == "invalid"


def test_run_scope_requires_candidate_for_every_scoped_canonical_asset(tmp_path):
    report = _multi_apex_report(tmp_path, candidate_asset_ids={"pool_image"})
    evidence = _materials_evidence(tmp_path / "pool_materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"pool_image": evidence}},
        run_scope_asset_ids=["pool_image", "kitchen_image"],
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "missing_candidate_output" in bundle["promotion_blocked_reasons"]
    assert "invalid_metrics" in bundle["promotion_blocked_reasons"]


def test_run_scope_can_promote_complete_subset_without_unscoped_assets(tmp_path):
    report = _multi_apex_report(tmp_path, candidate_asset_ids={"pool_image"})
    evidence = _materials_evidence(tmp_path / "pool_materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"pool_image": evidence}},
        run_scope_asset_ids=["pool_image"],
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "eligible"
    assert bundle["promotion_blocked_reasons"] == []


def test_materials_v3_telemetry_required_for_every_scoped_candidate(tmp_path):
    report = _multi_apex_report(tmp_path, candidate_asset_ids={"pool_image", "kitchen_image"})
    evidence = _materials_evidence(tmp_path / "pool_materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"pool_image": evidence}},
        run_scope_asset_ids=["pool_image", "kitchen_image"],
        repo_root=tmp_path,
    )

    cases_by_asset = {case["asset_id"]: case for case in bundle["cases"]}
    assert bundle["promotion_verdict"] == "blocked"
    assert "missing_materials_v3_evidence" in bundle["promotion_blocked_reasons"]
    assert cases_by_asset["kitchen_image"]["materials_v3"]["status"] == "missing_evidence"


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


@pytest.mark.parametrize(
    "metric_status",
    [
        "invalid_input",
        "unsupported_bit_depth",
        "dimension_mismatch",
    ],
)
def test_invalid_metric_status_blocks_promotion(tmp_path, metric_status):
    report = _apex_report(tmp_path)
    report["assets"][0]["candidates"][0]["metrics"]["visible_delta"]["status"] = metric_status
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


def test_metric_without_explicit_v1_status_blocks_promotion(tmp_path):
    report = _apex_report(tmp_path)
    report["assets"][0]["candidates"][0]["metrics"]["visible_delta"] = {"value": 0.0}
    evidence = _materials_evidence(tmp_path / "materials.json")

    bundle = build_apex_evidence_bundle(
        report,
        output_dir=tmp_path / "bundle",
        candidate_evidence={"materials_v3": {"unit_image": evidence}},
        repo_root=tmp_path,
    )

    assert bundle["promotion_verdict"] == "blocked"
    assert "invalid_metrics" in bundle["promotion_blocked_reasons"]


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
