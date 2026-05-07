#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portal job artifact catalog extraction contract tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_direct_module_import_does_not_import_app() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        "from transformation_portal.portal import job_artifacts; "
        "raise SystemExit(1 if 'app' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_app_legacy_artifact_helpers_remain_available() -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    orchestrator_app = importlib.import_module("app")

    assert orchestrator_app.JobRunMetadata is module.JobRunMetadata
    assert orchestrator_app.ArtifactPathValidationError is module.ArtifactPathValidationError
    assert orchestrator_app.InvalidArtifactPathError is module.InvalidArtifactPathError
    assert orchestrator_app.AbsoluteArtifactPathError is module.AbsoluteArtifactPathError
    assert orchestrator_app.ArtifactPathOutsideJobOutputDirError is module.ArtifactPathOutsideJobOutputDirError
    assert orchestrator_app._infer_artifact_type is module._infer_artifact_type
    assert orchestrator_app._artifact_content_type is module._artifact_content_type
    assert orchestrator_app._artifact_media_kind is module._artifact_media_kind
    assert orchestrator_app._artifact_is_previewable is module._artifact_is_previewable
    assert orchestrator_app._artifact_is_browser_previewable is module._artifact_is_browser_previewable
    assert orchestrator_app._artifact_preview_proxy_path is module._artifact_preview_proxy_path
    assert orchestrator_app._add_artifact_preview_proxy_lookup is module._add_artifact_preview_proxy_lookup
    assert orchestrator_app._artifact_response_headers is module._artifact_response_headers
    assert orchestrator_app._artifact_display_hint is module._artifact_display_hint
    assert callable(orchestrator_app._artifact_fingerprint)
    assert callable(orchestrator_app._serialize_indexed_artifact)
    assert callable(orchestrator_app._normalize_artifact_relative_path)
    assert callable(orchestrator_app._hydrate_artifact_lookup_from_items)
    assert callable(orchestrator_app._resolve_job_run_metadata)
    assert callable(orchestrator_app._index_job_artifacts)


def test_direct_and_app_index_payloads_match_for_preview_proxy(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    orchestrator_app = importlib.import_module("app")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "render.tif").write_bytes(b"II*\x00")
    (output_dir / "render.tif.preview.png").write_bytes(b"\x89PNG")

    direct = module._index_job_artifacts(job_id="job_artifacts", output_dir=output_dir)
    job = orchestrator_app.Job(
        id="job_artifacts",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    legacy_items = orchestrator_app._index_job_artifacts(job)

    assert legacy_items == direct.items
    assert job.artifacts == direct.artifacts
    assert job.artifact_lookup == direct.artifact_lookup
    tiff_item = next(item for item in legacy_items if item["path"] == "render.tif")
    assert tiff_item["preview_url"] == "/v1/jobs/job_artifacts/artifacts/render.tif.preview.png"
    assert tiff_item["download_url"] == "/v1/jobs/job_artifacts/artifacts/render.tif"
    assert tiff_item["browser_previewable"] is True
    assert job.artifact_lookup["render.tif.preview.png"] == (output_dir / "render.tif.preview.png").resolve()


def test_app_wrapper_injects_current_artifact_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    orchestrator_app = importlib.import_module("app")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    large = output_dir / "large.txt"
    large.write_bytes(b"12345")
    (output_dir / "small.txt").write_text("ok", encoding="utf-8")
    job = orchestrator_app.Job(
        id="job_artifacts_limits",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )

    monkeypatch.setattr(orchestrator_app, "MAX_INDEXED_ARTIFACTS", 1)
    monkeypatch.setattr(orchestrator_app, "ARTIFACT_FINGERPRINT_MAX_BYTES", 4)

    indexed = orchestrator_app._index_job_artifacts(job)

    assert [item["path"] for item in indexed] == ["large.txt"]
    assert indexed[0]["fingerprint_status"] == "skipped_size"
    assert module._artifact_fingerprint(large, large.stat().st_size)[1] == "ok"
    assert job.artifacts["truncated"] is True


def test_app_index_wrapper_honors_job_output_dir_monkeypatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator_app = importlib.import_module("app")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    job = orchestrator_app.Job(
        id="job_artifacts_output_dir_patch",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {}},
    )

    monkeypatch.setattr(orchestrator_app, "_job_output_dir", lambda _: output_dir)

    indexed = orchestrator_app._index_job_artifacts(job)

    assert [item["path"] for item in indexed] == ["manifest.json"]
    assert job.artifacts["output_dir"] == str(output_dir.resolve())


@pytest.mark.parametrize(
    ("artifact_path", "error_type"),
    [
        ("", "InvalidArtifactPathError"),
        ("~/.ssh/id_rsa", "InvalidArtifactPathError"),
        ("/tmp/secret.txt", "AbsoluteArtifactPathError"),
        ("../secret.txt", "ArtifactPathOutsideJobOutputDirError"),
        ("folder\\secret.txt", "InvalidArtifactPathError"),
    ],
)
def test_direct_relative_path_validation_preserves_error_types(
    artifact_path: str,
    error_type: str,
) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")

    with pytest.raises(getattr(module, error_type)):
        module._normalize_artifact_relative_path(artifact_path)


def test_direct_index_skips_symlink_escape(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    link = output_dir / "escape.png"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = module._index_job_artifacts(job_id="job_artifacts_symlink", output_dir=output_dir)

    assert result.items == []
    assert result.artifacts["items"] == []
    assert result.artifact_lookup == {}


def test_direct_hydration_registers_preview_proxy(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    artifact_path = output_dir / "renders" / "hero.tif"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"II*\x00")
    proxy_path = output_dir / "renders" / "hero.tif.preview.png"
    proxy_path.write_bytes(b"\x89PNG")

    lookup = module._hydrate_artifact_lookup_from_items(
        items=[{"path": "renders/hero.tif", "relative_path": "renders/hero.tif"}],
        output_dir=output_dir,
    )

    assert lookup["renders/hero.tif"] == artifact_path.resolve()
    assert lookup["renders/hero.tif.preview.png"] == proxy_path.resolve()


def test_direct_index_prefers_run_card_scoped_artifacts(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    depth_dir = output_dir / "depth"
    manifests_dir = output_dir / "manifests"
    depth_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    current_depth = depth_dir / "current_depth.png"
    current_depth.write_bytes(b"current")
    (depth_dir / "stale_depth.png").write_bytes(b"stale")
    batch_manifest = manifests_dir / "batch_2026-05-06_120000.json"
    batch_manifest.write_text(json.dumps({"batch_id": "2026-05-06_120000", "results": []}), encoding="utf-8")
    run_card = output_dir / "run_card_2026-05-06_120000.json"
    run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-05-06_120000",
                "success_count": 1,
                "error_count": 0,
                "artifact_index": [{"relative_path": "depth/current_depth.png"}],
            }
        ),
        encoding="utf-8",
    )

    result = module._index_job_artifacts(job_id="job_artifacts_scoped", output_dir=output_dir)

    assert {item["path"] for item in result.items} == {
        "depth/current_depth.png",
        "run_card_2026-05-06_120000.json",
    }
    assert "depth/stale_depth.png" not in result.artifact_lookup


def test_direct_index_uses_batch_manifest_when_run_card_has_no_artifact_index(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True)
    batch_manifest = manifests_dir / "batch_2026-05-06_120000.json"
    batch_manifest.write_text(
        json.dumps(
            {
                "batch_id": "2026-05-06_120000",
                "results": [{"status": "error"}],
                "stats": {"total_images": 1},
            }
        ),
        encoding="utf-8",
    )
    run_card = output_dir / "run_card_2026-05-06_120000.json"
    run_card.write_text(
        json.dumps(
            {
                "batch_id": "2026-05-06_120000",
                "success_count": 0,
                "error_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = module._index_job_artifacts(job_id="job_artifacts_manifest", output_dir=output_dir)

    assert [item["path"] for item in result.items] == ["manifests/batch_2026-05-06_120000.json"]
