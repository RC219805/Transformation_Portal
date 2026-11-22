#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portal job artifact catalog extraction contract tests."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

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


def test_direct_module_serializes_artifact_relative_paths_as_posix() -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")

    assert (
        module._relative_artifact_path(
            PureWindowsPath("C:/job-output/renders/hero.png"),
            PureWindowsPath("C:/job-output"),
        )
        == "renders/hero.png"
    )


def test_svg_artifacts_are_not_browser_previewable_and_are_served_as_attachments(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    artifact_path = tmp_path / "active.svg"
    artifact_path.write_text('<svg onload="alert(1)"></svg>', encoding="utf-8")

    payload = module._serialize_indexed_artifact(
        job_id="job_artifacts_svg",
        relative_path="active.svg",
        path=artifact_path,
    )
    headers = module._artifact_response_headers(artifact_path)

    assert module._artifact_content_type(artifact_path) == "image/svg+xml"
    assert module._artifact_is_browser_previewable(artifact_path) is False
    assert payload["browser_previewable"] is False
    assert "attachment" in headers["Content-Disposition"]
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("animated.gif", "image/gif"),
        ("modern.avif", "image/avif"),
    ],
)
def test_browser_previewable_image_formats_are_served_inline(
    tmp_path: Path,
    filename: str,
    content_type: str,
) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    artifact_path = tmp_path / filename
    artifact_path.write_bytes(b"image")

    payload = module._serialize_indexed_artifact(
        job_id="job_artifacts_browser_image",
        relative_path=filename,
        path=artifact_path,
    )
    headers = module._artifact_response_headers(artifact_path)

    assert module._infer_artifact_type(artifact_path) == "image"
    assert module._artifact_content_type(artifact_path) == content_type
    assert module._artifact_is_previewable(artifact_path) is True
    assert module._artifact_is_browser_previewable(artifact_path) is True
    assert payload["previewable"] is True
    assert payload["browser_previewable"] is True
    assert "Content-Disposition" not in headers
    assert headers["X-Content-Type-Options"] == "nosniff"


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
    outside_a = tmp_path / "secret-a.png"
    outside_b = tmp_path / "secret-b.png"
    outside_a.write_bytes(b"secret-a")
    outside_b.write_bytes(b"secret-b")
    link_a = output_dir / "escape-a.png"
    link_b = output_dir / "escape-b.png"
    try:
        link_a.symlink_to(outside_a)
        link_b.symlink_to(outside_b)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = module._index_job_artifacts(
        job_id="job_artifacts_symlink",
        output_dir=output_dir,
        max_indexed_artifacts=1,
    )

    assert result.items == []
    assert result.artifacts["items"] == []
    assert result.artifacts["truncated"] is False
    assert result.artifact_lookup == {}


def test_direct_index_does_not_descend_into_symlinked_directories(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    outside_dir = tmp_path / "outside"
    output_dir.mkdir()
    outside_dir.mkdir()
    for index in range(3):
        (outside_dir / f"secret-{index}.txt").write_text("secret", encoding="utf-8")
    link = output_dir / "linked"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = module._index_job_artifacts(
        job_id="job_artifacts_symlink_dir",
        output_dir=output_dir,
        max_indexed_artifacts=1,
    )

    assert result.items == []
    assert result.artifacts["truncated"] is False
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


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp))


def test_run_window_accepts_only_paths_changed_inside_it(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    untouched = tmp_path / "untouched.txt"
    untouched.write_text("operator file", encoding="utf-8")
    # ``untouched`` keeps its real timestamps, which precede a window opening in
    # the future; the other two are moved explicitly so no clock race is involved.
    window_start = untouched.stat().st_ctime + 100.0
    window_end = window_start + 100.0
    produced = tmp_path / "produced.txt"
    produced.write_text("job output", encoding="utf-8")
    _set_mtime(produced, window_start + 50.0)
    replaced_later = tmp_path / "replaced-later.txt"
    replaced_later.write_text("rewritten by a later run", encoding="utf-8")
    _set_mtime(replaced_later, window_end + 50.0)

    def changed(path: Path, not_before: float | None = None, not_after: float | None = None) -> bool:
        return module._path_changed_within(path, not_before=not_before, not_after=not_after)

    assert changed(untouched) is True
    assert changed(untouched, window_start, window_end) is False
    assert changed(produced, window_start, window_end) is True
    assert changed(replaced_later, window_start, window_end) is False
    assert changed(replaced_later, window_start) is True
    assert changed(tmp_path / "missing.txt", window_start) is False
    assert changed(untouched, float("inf")) is False


def test_run_window_counts_a_copy_that_preserved_an_older_mtime(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    source = tmp_path / "source.tif"
    source.write_bytes(b"payload")
    _set_mtime(source, 1_000_000_000.0)
    time.sleep(0.05)
    copied = tmp_path / "bag" / "data" / "source.tif"
    copied.parent.mkdir(parents=True)
    shutil.copy2(source, copied)
    source_changed_at = source.stat().st_ctime
    copy_landed_at = copied.stat().st_ctime
    if copy_landed_at <= source_changed_at:
        pytest.skip("filesystem timestamps are too coarse to order the copy after its source")
    window_start = (source_changed_at + copy_landed_at) / 2.0

    assert copied.stat().st_mtime == pytest.approx(1_000_000_000.0)
    assert module._path_changed_within(copied, not_before=window_start, not_after=None) is True
    assert module._path_changed_within(source, not_before=window_start, not_after=None) is False


def test_direct_index_walk_claims_only_files_changed_inside_the_run_window(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "shared-out"
    (output_dir / "client_deliverables").mkdir(parents=True)
    operator_index = output_dir / "archive_index_normalized.csv.gz"
    operator_index.write_bytes(b"operator supplied index")
    other_job_output = output_dir / "client_deliverables" / "final_render.png"
    other_job_output.write_bytes(b"\x89PNG other job output")
    window_start = max(operator_index.stat().st_ctime, other_job_output.stat().st_ctime) + 100.0
    report = output_dir / "bag_validate_report.json"
    report.write_text("{}", encoding="utf-8")
    _set_mtime(report, window_start + 5.0)

    unbounded = module._index_job_artifacts(job_id="job_shared", output_dir=output_dir)
    bound = module._index_job_artifacts(
        job_id="job_shared",
        output_dir=output_dir,
        max_indexed_artifacts=1,
        not_before=window_start,
    )

    assert {item["path"] for item in unbounded.items} == {
        "archive_index_normalized.csv.gz",
        "bag_validate_report.json",
        "client_deliverables/final_render.png",
    }
    assert [item["path"] for item in bound.items] == ["bag_validate_report.json"]
    assert set(bound.artifact_lookup) == {"bag_validate_report.json"}
    # Files outside the window are not counted, so they cannot force truncation.
    assert bound.artifacts["truncated"] is False


def test_direct_run_metadata_ignores_run_cards_and_manifests_outside_the_run_window(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.job_artifacts")
    output_dir = tmp_path / "out"
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True)
    stale_card = output_dir / "run_card_2099-01-01_000000.json"
    stale_card.write_text(
        json.dumps({"batch_id": "2099-01-01_000000", "total_images": 4, "success_count": 3, "error_count": 1}),
        encoding="utf-8",
    )
    stale_manifest = manifests_dir / "batch_2099-01-01_000000.json"
    stale_manifest.write_text(json.dumps({"batch_id": "2099-01-01_000000", "results": []}), encoding="utf-8")
    window_start = max(stale_card.stat().st_ctime, stale_manifest.stat().st_ctime) + 100.0

    unbounded = module._resolve_job_run_metadata(output_dir)
    without_own_card = module._resolve_job_run_metadata(output_dir, not_before=window_start)

    assert unbounded is not None and unbounded.run_card_path == stale_card.resolve()
    assert without_own_card == module.JobRunMetadata(output_dir=output_dir.resolve())

    own_card = output_dir / "run_card_2026-03-03_000000.json"
    own_card.write_text(
        json.dumps({"batch_id": "2026-03-03_000000", "total_images": 2, "success_count": 2, "error_count": 0}),
        encoding="utf-8",
    )
    _set_mtime(own_card, window_start + 5.0)
    own_manifest = manifests_dir / "batch_2026-03-03_000000.json"
    own_manifest.write_text(json.dumps({"batch_id": "2026-03-03_000000", "results": []}), encoding="utf-8")
    _set_mtime(own_manifest, window_start + 5.0)

    still_unbounded = module._resolve_job_run_metadata(output_dir)
    with_own_card = module._resolve_job_run_metadata(output_dir, not_before=window_start)
    after_window = module._resolve_job_run_metadata(output_dir, not_before=window_start, not_after=window_start + 1.0)

    # Unbound, the stale card still wins on its newer batch id; bound, it cannot.
    assert still_unbounded is not None and still_unbounded.run_card_path == stale_card.resolve()
    assert with_own_card is not None
    assert with_own_card.run_card_path == own_card.resolve()
    assert with_own_card.run_card_payload is not None
    assert with_own_card.run_card_payload["batch_id"] == "2026-03-03_000000"
    assert with_own_card.batch_manifest_path == own_manifest.resolve()
    assert after_window == module.JobRunMetadata(output_dir=output_dir.resolve())
