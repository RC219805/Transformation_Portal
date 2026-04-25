"""Tests for APEX visual quality evalset and depth benchmark helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transformation_portal.evals.apex_visual import (
    APEX_EVALSET_SCHEMA_VERSION,
    DepthBackendRunResult,
    build_apex_eval_report,
    build_depth_backend_benchmark_report,
    load_apex_evalset,
    sha256_file,
    visible_delta_metrics,
)

pytestmark = pytest.mark.unit

VALID_SOURCE_RAW_SHA = "a" * 64
VALID_RAW_SETTINGS_SHA = "b" * 64
VALID_ICC_SHA = "c" * 64


def _write_evalset(
    root: Path,
    asset_path: Path,
    *,
    sha256: str | None = None,
    dataset_tier: str = "smoke_or_readiness",
    evalset_overrides: dict | None = None,
    asset_overrides: dict | None = None,
) -> Path:
    evalset_dir = root / "evalsets" / "apex"
    evalset_dir.mkdir(parents=True, exist_ok=True)
    asset_payload = {
        "asset_id": "unit_image",
        "asset_ref": str(asset_path.relative_to(root)),
        "sha256": sha256 or sha256_file(asset_path),
        "scene_type": "pool_exterior",
        "expected_materials": ["water", "stone"],
        "risk_zones": ["water_glass_boundary"],
        "reject_if": ["haloing"],
        "manual_quality_score": None,
    }
    if asset_overrides:
        asset_payload.update(asset_overrides)
    payload = {
        "schema_version": APEX_EVALSET_SCHEMA_VERSION,
        "evalset_id": "unit_apex",
        "version": "v1",
        "description": "unit",
        "dataset_tier": dataset_tier,
        "assets": [asset_payload],
    }
    if evalset_overrides:
        payload.update(evalset_overrides)
    path = evalset_dir / "evalset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_apex_eval_asset_omits_absent_provenance_fields(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)

    evalset = load_apex_evalset(evalset_path, repo_root=tmp_path)
    asset_payload = evalset.assets[0].to_dict()
    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)
    report_asset = report["assets"][0]

    for key in (
        "source_raw_path",
        "source_raw_format",
        "source_raw_sha256",
        "raw_development_profile",
        "raw_development_settings_sha256",
        "canonical_icc_profile_name",
        "canonical_icc_profile_sha256",
        "working_color_space",
        "working_transfer_function",
    ):
        assert key not in asset_payload
        assert key not in report_asset


def test_apex_eval_asset_round_trips_provenance_fields(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={
            "source_raw_path": "source_raw/example.dng",
            "source_raw_format": ".DNG",
            "source_raw_sha256": VALID_SOURCE_RAW_SHA,
            "raw_development_profile": "Capture One Picacho APEX v1",
            "raw_development_settings_sha256": VALID_RAW_SETTINGS_SHA,
            "canonical_icc_profile_name": "ProPhoto RGB",
            "canonical_icc_profile_sha256": VALID_ICC_SHA,
            "working_color_space": "ProPhoto RGB",
            "working_transfer_function": "linear",
        },
    )

    evalset = load_apex_evalset(evalset_path, repo_root=tmp_path)
    asset_payload = evalset.assets[0].to_dict()
    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)
    report_asset = report["assets"][0]

    expected = {
        "source_raw_path": "source_raw/example.dng",
        "source_raw_format": "dng",
        "source_raw_sha256": VALID_SOURCE_RAW_SHA,
        "raw_development_profile": "Capture One Picacho APEX v1",
        "raw_development_settings_sha256": VALID_RAW_SETTINGS_SHA,
        "canonical_icc_profile_name": "ProPhoto RGB",
        "canonical_icc_profile_sha256": VALID_ICC_SHA,
        "working_color_space": "ProPhoto RGB",
        "working_transfer_function": "linear",
    }
    for key, value in expected.items():
        assert asset_payload[key] == value
        assert report_asset[key] == value
    assert report_asset["asset_status"]["status"] == "ready"


def _load_tool_module(script_name: str, module_name: str):
    script_path = Path(__file__).resolve().parents[2] / "tools" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apex_eval_report_tracks_ready_assets_and_candidate_metrics(tmp_path):
    image_path = tmp_path / "input.png"
    candidate_path = tmp_path / "candidate.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12] = 120
    Image.fromarray(image).save(image_path)
    Image.fromarray(image.copy()).save(candidate_path)
    evalset_path = _write_evalset(tmp_path, image_path)

    report = build_apex_eval_report(
        evalset_path,
        output_dir=tmp_path / "report",
        candidate_outputs={"identity": {"unit_image": candidate_path}},
        repo_root=tmp_path,
    )

    assert report["evalset"]["ready_asset_count"] == 1
    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    candidate = asset["candidates"][0]
    assert candidate["status"] in {"ok", "partial_metrics"}
    assert candidate["metrics"]["ssim"] == pytest.approx(1.0)
    if candidate["status"] == "partial_metrics":
        assert "lpips_unavailable" in candidate["metrics"]["metric_warnings"]
        assert candidate["metrics"]["lpips"] is None
    assert (tmp_path / "report" / "apex_eval_report.json").is_file()


def test_jpeg_delivery_asset_is_ready_but_not_canonical_scoring_eligible(tmp_path):
    image_path = tmp_path / "delivery.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path, format="JPEG")
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={
            "asset_role": "delivery_preview",
            "canonical_bit_depth": 8,
            "canonical_format": "jpeg",
            "canonical_color_space": "srgb",
            "canonical_scoring_eligible": True,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 1
    assert report["evalset"]["canonical_scoring_eligible_count"] == 0
    assert report["evalset"]["noncanonical_asset_count"] == 1
    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["asset_role"] == "delivery_preview"
    assert asset["reference_bit_depth"] == 8
    assert asset["reference_format"] == "jpeg"
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "noncanonical_dataset_tier"


def test_jpeg_cannot_be_promoted_to_canonical_apex_reference(tmp_path):
    image_path = tmp_path / "misdeclared_reference.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path, format="JPEG")
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="canonical_apex",
        asset_overrides={
            "asset_role": "canonical_apex_reference",
            "canonical_bit_depth": 8,
            "canonical_format": "jpeg",
            "canonical_color_space": "srgb",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "preserve_16bit_intermediates": True,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 1
    assert report["evalset"]["canonical_scoring_eligible_count"] == 0
    asset = report["assets"][0]
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "reference_bit_depth_below_16"


def test_misdeclared_8bit_tiff_is_not_canonical_scoring_eligible(tmp_path):
    image_path = tmp_path / "misdeclared_reference.tif"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path, format="TIFF")
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="canonical_apex",
        asset_overrides={
            "asset_role": "canonical_apex_reference",
            "canonical_bit_depth": 16,
            "canonical_format": "tiff",
            "canonical_color_space": "documented_source_profile",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "preserve_16bit_intermediates": True,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 1
    assert report["evalset"]["canonical_scoring_eligible_count"] == 0
    assert report["evalset"]["noncanonical_asset_count"] == 1
    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["declared_reference_bit_depth"] == 16
    assert asset["detected_reference_bit_depth"] == 8
    assert asset["reference_bit_depth"] == 8
    assert asset["reference_format"] == "tiff"
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "reference_bit_depth_below_16"


def test_missing_raw_source_path_does_not_affect_asset_status(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={
            "source_raw_path": "source_raw/missing_capture.dng",
            "source_raw_format": "dng",
            "source_raw_sha256": VALID_SOURCE_RAW_SHA,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 1
    assert report["evalset"]["missing_asset_count"] == 0
    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["source_raw_path"] == "source_raw/missing_capture.dng"
    assert asset["canonical_scoring_eligible"] is False


def test_raw_source_provenance_does_not_make_asset_canonical_scoring_eligible(tmp_path):
    image_path = tmp_path / "reference8.tif"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path, format="TIFF")
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="canonical_apex",
        asset_overrides={
            "asset_role": "canonical_apex_reference",
            "canonical_bit_depth": 8,
            "canonical_format": "tiff",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "preserve_16bit_intermediates": True,
            "source_raw_path": "source_raw/example.dng",
            "source_raw_format": "dng",
            "source_raw_sha256": VALID_SOURCE_RAW_SHA,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["source_raw_path"] == "source_raw/example.dng"
    assert asset["source_raw_format"] == "dng"
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "reference_bit_depth_below_16"


def test_rendering_asset_is_smoke_only_even_with_canonical_like_fields(tmp_path):
    image_path = tmp_path / "rendering.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="synthetic_smoke",
        asset_overrides={
            "asset_role": "synthetic_smoke",
            "canonical_bit_depth": 16,
            "canonical_format": "tiff",
            "canonical_color_space": "documented_source_profile",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "preserve_16bit_intermediates": True,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["dataset_tier"] == "synthetic_smoke"
    assert report["evalset"]["canonical_scoring_eligible_count"] == 0
    assert report["evalset"]["noncanonical_asset_count"] == 1
    asset = report["assets"][0]
    assert asset["asset_role"] == "synthetic_smoke"
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "noncanonical_dataset_tier"


def test_16bit_tiff_reference_is_canonical_scoring_eligible(tmp_path):
    image_path = tmp_path / "reference16.tif"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="canonical_apex",
        evalset_overrides={
            "canonical_bit_depth": 16,
            "canonical_format": "tiff",
            "canonical_color_space": "documented_source_profile",
        },
        asset_overrides={
            "asset_role": "canonical_apex_reference",
            "canonical_bit_depth": 16,
            "canonical_format": "tiff",
            "canonical_color_space": "documented_source_profile",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "preserve_16bit_intermediates": True,
        },
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 1
    assert report["evalset"]["canonical_scoring_eligible_count"] == 1
    assert report["evalset"]["noncanonical_asset_count"] == 0
    asset = report["assets"][0]
    assert asset["asset_role"] == "canonical_apex_reference"
    assert asset["reference_bit_depth"] == 16
    assert asset["reference_format"] == "tiff"
    assert asset["canonical_scoring_eligible"] is True
    assert asset["canonical_scoring_blocked_reason"] is None


def test_apex_eval_report_marks_missing_assets(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    image_path.unlink()

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 0
    assert report["evalset"]["canonical_scoring_eligible_count"] == 0
    assert report["evalset"]["missing_asset_count"] == 1
    assert report["assets"][0]["asset_status"]["status"] == "missing_asset"


@pytest.mark.parametrize("manual_score", [None, 0.0, 1.0])
def test_manual_quality_score_accepts_normalized_values(tmp_path, manual_score):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={"manual_quality_score": manual_score},
    )

    asset = load_apex_evalset(evalset_path, repo_root=tmp_path).assets[0]

    assert asset.manual_quality_score == manual_score


@pytest.mark.parametrize("manual_score", [-0.01, 1.01])
def test_manual_quality_score_rejects_out_of_range_values(tmp_path, manual_score):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={"manual_quality_score": manual_score},
    )

    with pytest.raises(ValueError, match="manual_quality_score"):
        load_apex_evalset(evalset_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("source_raw_sha256", "a" * 63),
        ("source_raw_sha256", "a" * 65),
        ("source_raw_sha256", "A" * 64),
        ("source_raw_sha256", ("a" * 63) + "g"),
        ("raw_development_settings_sha256", "b" * 63),
        ("canonical_icc_profile_sha256", "c" * 65),
    ],
)
def test_provenance_sha_fields_require_lowercase_64_hex(tmp_path, field_name, invalid_value):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={field_name: invalid_value},
    )

    with pytest.raises(ValueError, match=field_name):
        load_apex_evalset(evalset_path, repo_root=tmp_path)


def test_empty_provenance_sha_fields_normalize_to_absent(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        asset_overrides={
            "source_raw_sha256": "",
            "raw_development_settings_sha256": " ",
            "canonical_icc_profile_sha256": None,
        },
    )

    payload = load_apex_evalset(evalset_path, repo_root=tmp_path).assets[0].to_dict()

    assert "source_raw_sha256" not in payload
    assert "raw_development_settings_sha256" not in payload
    assert "canonical_icc_profile_sha256" not in payload


def test_run_apex_eval_returns_nonzero_for_non_ready_assets(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    image_path.unlink()
    output_dir = tmp_path / "report"

    module = _load_tool_module("run_apex_eval.py", "run_apex_eval_unit")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_apex_eval.py",
            "--evalset",
            str(evalset_path),
            "--output-dir",
            str(output_dir),
            "--emit-report",
            "off",
        ],
    )

    assert module.main() == 1
    assert (output_dir / "apex_eval_report.json").is_file()
    assert "non-ready assets: unit_image" in capsys.readouterr().out


def test_run_apex_eval_returns_stable_input_error_for_bad_candidate_output(tmp_path, monkeypatch, capsys):
    module = _load_tool_module("run_apex_eval.py", "run_apex_eval_bad_candidate_unit")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_apex_eval.py",
            "--evalset",
            str(tmp_path / "missing_evalset"),
            "--output-dir",
            str(tmp_path / "report"),
            "--candidate-output",
            "not-valid",
        ],
    )

    assert module.main() == 2
    captured = capsys.readouterr()
    assert "APEX eval error:" in captured.err
    assert "expected candidate:asset_id=path" in captured.err


def test_run_apex_eval_prints_resolved_report_path(tmp_path, monkeypatch, capsys):
    module = _load_tool_module("run_apex_eval.py", "run_apex_eval_resolved_path_unit")
    resolved_report = tmp_path / "resolved" / "apex_eval_report.json"

    def fake_build_report(*args, **kwargs):
        return {"report_path": str(resolved_report), "assets": []}

    monkeypatch.setattr(module, "build_apex_eval_report", fake_build_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_apex_eval.py",
            "--evalset",
            "evalsets/picacho_apex",
            "--output-dir",
            "relative-output",
        ],
    )

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == str(resolved_report)


def test_benchmark_depth_backends_returns_stable_input_error_for_missing_evalset(tmp_path, monkeypatch, capsys):
    module = _load_tool_module("benchmark_depth_backends.py", "benchmark_depth_backends_missing_evalset_unit")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_depth_backends.py",
            "--evalset",
            str(tmp_path / "missing_evalset"),
            "--backends",
            "da3-metric",
            "--output-dir",
            str(tmp_path / "benchmark"),
        ],
    )

    assert module.main() == 2
    captured = capsys.readouterr()
    assert "Depth backend benchmark error:" in captured.err
    assert "missing_evalset" in captured.err


def test_benchmark_depth_backends_rejects_empty_backend_list(tmp_path, monkeypatch, capsys):
    module = _load_tool_module("benchmark_depth_backends.py", "benchmark_depth_backends_empty_backends_unit")

    def fail_build_report(*args, **kwargs):
        raise AssertionError("empty backend list must fail before report generation")

    monkeypatch.setattr(module, "build_depth_backend_benchmark_report", fail_build_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_depth_backends.py",
            "--evalset",
            "evalsets/picacho_apex",
            "--backends",
            ",,,",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 2
    captured = capsys.readouterr()
    assert "--backends must include at least one backend id" in captured.err
    assert not captured.out


def test_benchmark_depth_backends_prints_resolved_report_path(tmp_path, monkeypatch, capsys):
    module = _load_tool_module("benchmark_depth_backends.py", "benchmark_depth_backends_resolved_path_unit")
    resolved_report = tmp_path / "resolved" / "depth_backend_comparison_report.json"

    def fake_build_report(*args, **kwargs):
        return {"report_path": str(resolved_report), "backends": []}

    monkeypatch.setattr(module, "build_depth_backend_benchmark_report", fake_build_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_depth_backends.py",
            "--evalset",
            "evalsets/picacho_apex",
            "--backends",
            "da3-metric",
            "--output-dir",
            "relative-output",
        ],
    )

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == str(resolved_report)


def test_visible_delta_metrics_marks_lpips_unavailable_as_partial(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(image).save(reference_path)
    Image.fromarray(image.copy()).save(candidate_path)
    monkeypatch.setattr("transformation_portal.evals.apex_visual._lpips_available", lambda: False)

    metrics = visible_delta_metrics(reference_path, candidate_path)

    assert metrics["status"] == "partial_metrics"
    assert metrics["lpips"] is None
    assert "lpips_unavailable" in metrics["metric_warnings"]
    assert metrics["ssim"] == pytest.approx(1.0)


def test_visible_delta_metrics_reports_unreadable_candidate(tmp_path):
    reference_path = tmp_path / "reference.png"
    candidate_path = tmp_path / "candidate.txt"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(reference_path)
    candidate_path.write_text("not an image", encoding="utf-8")

    metrics = visible_delta_metrics(reference_path, candidate_path)

    assert metrics["status"] == "unreadable_image"
    assert metrics["unreadable_role"] == "candidate"
    assert metrics["unreadable_path"] == str(candidate_path)
    assert metrics["ssim"] is None
    assert metrics["lpips"] is None


def test_load_apex_evalset_rejects_checksum_drift_in_report(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path, sha256="0" * 64)

    evalset = load_apex_evalset(evalset_path, repo_root=tmp_path)
    report = build_apex_eval_report(evalset.source_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["assets"][0]["asset_status"]["status"] == "checksum_mismatch"


def test_depth_backend_benchmark_blocks_depth_pro_without_license(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)

    report = build_depth_backend_benchmark_report(
        evalset_path,
        backends=["depth_pro"],
        quality_tier="apex",
        output_dir=tmp_path / "benchmark",
        repo_root=tmp_path,
    )

    backend = report["backends"][0]
    assert backend["backend"] == "depth_pro"
    assert backend["status"] == "license_blocked"


def test_depth_backend_benchmark_uses_mocked_runner(tmp_path):
    image_path = tmp_path / "reference16.tif"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(image_path)
    evalset_path = _write_evalset(
        tmp_path,
        image_path,
        dataset_tier="canonical_apex",
        asset_overrides={
            "asset_role": "canonical_apex_reference",
            "canonical_bit_depth": 16,
            "canonical_format": "tiff",
            "canonical_color_space": "documented_source_profile",
            "canonical_scoring_eligible": True,
            "evaluate_at_native_resolution": True,
            "allow_downsampled_model_inference": True,
            "preserve_16bit_intermediates": True,
            "source_raw_path": "source_raw/reference.dng",
            "source_raw_format": ".DNG",
            "source_raw_sha256": VALID_SOURCE_RAW_SHA,
            "raw_development_profile": "Capture One Picacho APEX v1",
            "raw_development_settings_sha256": VALID_RAW_SETTINGS_SHA,
            "canonical_icc_profile_name": "ProPhoto RGB",
            "canonical_icc_profile_sha256": VALID_ICC_SHA,
            "working_color_space": "ProPhoto RGB",
            "working_transfer_function": "linear",
        },
    )

    def runner(backend, asset, output_dir, quality_tier):
        assert backend == "da3_metric"
        assert asset.asset_id == "unit_image"
        assert output_dir == tmp_path / "benchmark"
        assert quality_tier == "apex"
        depth = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
        return DepthBackendRunResult(
            backend=backend,
            asset_id=asset.asset_id,
            status="success",
            runtime_ms=12.5,
            depth_map=depth,
            provenance={"model": "mock", "model_input": {"input_resolution": [1024, 768]}},
        )

    report = build_depth_backend_benchmark_report(
        evalset_path,
        backends=["da3-metric"],
        quality_tier="apex",
        output_dir=tmp_path / "benchmark",
        runner=runner,
        repo_root=tmp_path,
    )

    backend = report["backends"][0]
    assert backend["status"] == "ready"
    asset = backend["assets"][0]
    assert asset["status"] == "success"
    assert asset["metrics"]["runtime_ms"] == pytest.approx(12.5)
    assert asset["model_input"]["derived_from"] == str(image_path.relative_to(tmp_path))
    assert asset["model_input"]["input_bit_depth"] == 8
    assert asset["model_input"]["input_resolution"] == [1024, 768]
    assert asset["model_input"]["downsampled_for_inference"] is True
    assert "source_raw_path" not in asset["model_input"]
    assert "working_color_space" not in asset["model_input"]
    assert asset["evaluation_target"]["path"] == str(image_path.relative_to(tmp_path))
    assert asset["evaluation_target"]["bit_depth"] == 16
    assert asset["evaluation_target"]["evaluate_at_native_resolution"] is True
    assert asset["evaluation_target"]["source_raw_path"] == "source_raw/reference.dng"
    assert asset["evaluation_target"]["source_raw_format"] == "dng"
    assert asset["evaluation_target"]["source_raw_sha256"] == VALID_SOURCE_RAW_SHA
    assert asset["evaluation_target"]["raw_development_profile"] == "Capture One Picacho APEX v1"
    assert asset["evaluation_target"]["raw_development_settings_sha256"] == VALID_RAW_SETTINGS_SHA
    assert asset["evaluation_target"]["canonical_icc_profile_name"] == "ProPhoto RGB"
    assert asset["evaluation_target"]["canonical_icc_profile_sha256"] == VALID_ICC_SHA
    assert asset["evaluation_target"]["working_color_space"] == "ProPhoto RGB"
    assert asset["evaluation_target"]["working_transfer_function"] == "linear"
    assert backend["depth_edge_score"] is not None
