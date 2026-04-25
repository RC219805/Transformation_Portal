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


def _write_evalset(root: Path, asset_path: Path, *, sha256: str | None = None) -> Path:
    evalset_dir = root / "evalsets" / "apex"
    evalset_dir.mkdir(parents=True)
    payload = {
        "schema_version": APEX_EVALSET_SCHEMA_VERSION,
        "evalset_id": "unit_apex",
        "version": "v1",
        "description": "unit",
        "assets": [
            {
                "asset_id": "unit_image",
                "asset_ref": str(asset_path.relative_to(root)),
                "sha256": sha256 or sha256_file(asset_path),
                "scene_type": "pool_exterior",
                "expected_materials": ["water", "stone"],
                "risk_zones": ["water_glass_boundary"],
                "reject_if": ["haloing"],
                "manual_quality_score": None,
            }
        ],
    }
    path = evalset_dir / "evalset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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


def test_apex_eval_report_marks_missing_assets(tmp_path):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    image_path.unlink()

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    assert report["evalset"]["ready_asset_count"] == 0
    assert report["assets"][0]["asset_status"]["status"] == "missing_asset"


def test_run_apex_eval_returns_nonzero_for_non_ready_assets(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)
    image_path.unlink()
    output_dir = tmp_path / "report"

    script_path = Path(__file__).resolve().parents[2] / "tools" / "run_apex_eval.py"
    spec = importlib.util.spec_from_file_location("run_apex_eval_unit", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    evalset_path = _write_evalset(tmp_path, image_path)

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
            provenance={"model": "mock"},
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
    assert backend["assets"][0]["status"] == "success"
    assert backend["assets"][0]["metrics"]["runtime_ms"] == pytest.approx(12.5)
    assert backend["depth_edge_score"] is not None
