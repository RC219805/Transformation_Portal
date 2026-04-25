"""Tests for APEX external asset-root resolution and audit behavior."""

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
    build_apex_eval_report,
    load_apex_evalset,
    sha256_file,
)

pytestmark = pytest.mark.unit


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint16), mode="I;16").save(path)


def _write_evalset(
    repo_root: Path,
    *,
    asset_ref: str,
    sha256: str,
    reference_path: str | None = None,
    source_raw_path: str | None = None,
    dataset_tier: str = "canonical_apex",
) -> Path:
    evalset_dir = repo_root / "evalsets" / "apex_real_estate_v1"
    evalset_dir.mkdir(parents=True, exist_ok=True)
    asset = {
        "asset_id": "unit_canonical",
        "asset_ref": asset_ref,
        "sha256": sha256,
        "asset_role": "canonical_apex_reference",
        "reference_path": reference_path or asset_ref,
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
    if source_raw_path is not None:
        asset["source_raw_path"] = source_raw_path
        asset["source_raw_format"] = "dng"
    payload = {
        "schema_version": APEX_EVALSET_SCHEMA_VERSION,
        "evalset_id": "unit_canonical",
        "version": "v1",
        "description": "unit",
        "dataset_tier": dataset_tier,
        "assets": [asset],
    }
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


def test_asset_ref_resolves_repo_relative_by_default(tmp_path):
    image_path = tmp_path / "corpus" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(tmp_path, asset_ref="corpus/reference16.tif", sha256=sha256_file(image_path))

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["asset_resolution"]["strategy"] == "repo_relative"
    assert "resolved_path" not in asset["asset_resolution"]
    assert "resolved_path" not in asset["asset_status"]


def test_asset_ref_resolves_from_cli_asset_root(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    image_path = asset_root / "apex_real_estate_v1" / "reference_16bit" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(
        repo_root,
        asset_ref="apex_real_estate_v1/reference_16bit/reference16.tif",
        sha256=sha256_file(image_path),
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=repo_root, asset_root=asset_root)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["asset_resolution"]["strategy"] == "cli_asset_root"
    assert asset["canonical_scoring_eligible"] is True


def test_asset_ref_resolves_from_env_asset_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    image_path = asset_root / "apex_real_estate_v1" / "reference_16bit" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(
        repo_root,
        asset_ref="apex_real_estate_v1/reference_16bit/reference16.tif",
        sha256=sha256_file(image_path),
    )
    monkeypatch.setenv("APEX_EVAL_ASSET_ROOT", str(asset_root))

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=repo_root)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["asset_resolution"]["strategy"] == "env_asset_root"


def test_absolute_asset_ref_is_honored(tmp_path):
    image_path = tmp_path / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(tmp_path, asset_ref=str(image_path), sha256=sha256_file(image_path))

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=tmp_path)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["asset_resolution"]["strategy"] == "absolute"
    assert asset["asset_resolution"]["path_was_absolute"] is True


def test_asset_root_traversal_fails_closed_for_asset_ref(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    evalset_path = _write_evalset(repo_root, asset_ref="../outside.tif", sha256="0" * 64)

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=repo_root, asset_root=asset_root)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "missing_asset"
    assert asset["canonical_scoring_eligible"] is False
    assert asset["canonical_scoring_blocked_reason"] == "path_escapes_asset_root"
    assert asset["path_field"] == "asset_ref"
    assert asset["asset_resolution"]["escaped_asset_root"] is True


def test_legacy_resolve_asset_path_rejects_asset_root_escape(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    evalset_path = _write_evalset(repo_root, asset_ref="../outside.tif", sha256="0" * 64)
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root, asset_root=asset_root)

    with pytest.raises(ValueError, match="escapes asset root"):
        evalset.resolve_asset_path(evalset.assets[0])


def test_asset_root_traversal_fails_closed_for_reference_path(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    image_path = asset_root / "apex_real_estate_v1" / "reference_16bit" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(
        repo_root,
        asset_ref="apex_real_estate_v1/reference_16bit/reference16.tif",
        reference_path="../outside.tif",
        sha256=sha256_file(image_path),
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=repo_root, asset_root=asset_root)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "missing_asset"
    assert asset["canonical_scoring_blocked_reason"] == "path_escapes_asset_root"
    assert asset["path_field"] == "reference_path"
    assert asset["reference_resolution"]["escaped_asset_root"] is True


def test_legacy_resolve_reference_path_rejects_asset_root_escape(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    image_path = asset_root / "apex_real_estate_v1" / "reference_16bit" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(
        repo_root,
        asset_ref="apex_real_estate_v1/reference_16bit/reference16.tif",
        reference_path="../outside.tif",
        sha256=sha256_file(image_path),
    )
    evalset = load_apex_evalset(evalset_path, repo_root=repo_root, asset_root=asset_root)

    with pytest.raises(ValueError, match="escapes asset root"):
        evalset.resolve_reference_path(evalset.assets[0])


def test_source_raw_path_is_not_resolved_or_readiness_checked(tmp_path):
    repo_root = tmp_path / "repo"
    asset_root = tmp_path / "assets"
    image_path = asset_root / "apex_real_estate_v1" / "reference_16bit" / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(
        repo_root,
        asset_ref="apex_real_estate_v1/reference_16bit/reference16.tif",
        source_raw_path="../outside/raw.dng",
        sha256=sha256_file(image_path),
    )

    report = build_apex_eval_report(evalset_path, output_dir=tmp_path / "report", repo_root=repo_root, asset_root=asset_root)

    asset = report["assets"][0]
    assert asset["asset_status"]["status"] == "ready"
    assert asset["source_raw_path"] == "../outside/raw.dng"
    assert asset["canonical_scoring_eligible"] is True


def test_audit_exit_codes_are_deterministic(tmp_path, monkeypatch):
    module = _load_tool_module("audit_apex_assets.py", "audit_apex_assets_unit")
    image_path = tmp_path / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(tmp_path, asset_ref=str(image_path), sha256="0" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_apex_assets.py", "--evalset", str(evalset_path), "--output-dir", str(tmp_path / "audit")],
    )
    assert module.main() == 1

    image_path.unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_apex_assets.py", "--evalset", str(evalset_path), "--output-dir", str(tmp_path / "audit2")],
    )
    assert module.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_apex_assets.py",
            "--evalset",
            str(evalset_path),
            "--output-dir",
            str(tmp_path / "audit3"),
            "--require-canonical",
            "on",
        ],
    )
    assert module.main() == 2

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_apex_assets.py", "--evalset", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "audit4")],
    )
    assert module.main() == 3


def test_audit_uses_eval_report_output_directory_for_relative_output(tmp_path, monkeypatch):
    module = _load_tool_module("audit_apex_assets.py", "audit_apex_assets_output_dir_unit")
    repo_root = tmp_path / "repo"
    cwd = tmp_path / "outside_cwd"
    cwd.mkdir()
    image_path = repo_root / "reference16.tif"
    _write_image(image_path)
    evalset_path = _write_evalset(repo_root, asset_ref="reference16.tif", sha256=sha256_file(image_path))
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_apex_assets.py",
            "--evalset",
            str(evalset_path),
            "--output-dir",
            "audit",
            "--repo-root",
            str(repo_root),
        ],
    )

    assert module.main() == 0
    assert (repo_root / "audit" / "apex_asset_audit_report.json").is_file()
    assert not (cwd / "audit" / "apex_asset_audit_report.json").exists()
