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
    assert asset["evaluation_target"]["path"] == str(image_path.relative_to(tmp_path))
    assert asset["evaluation_target"]["bit_depth"] == 16
    assert asset["evaluation_target"]["evaluate_at_native_resolution"] is True
    assert backend["depth_edge_score"] is not None
