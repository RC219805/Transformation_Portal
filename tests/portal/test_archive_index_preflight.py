#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive index preflight extraction contract tests."""

from __future__ import annotations

import csv
import gzip
import importlib
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_archive_index(path: Path, relpaths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["origin_drive", "partition", "relpath"],
            lineterminator="\n",
        )
        writer.writeheader()
        for relpath in relpaths:
            writer.writerow(
                {
                    "origin_drive": "local",
                    "partition": "test",
                    "relpath": relpath,
                }
            )


def _validate_direct(
    module: Any,
    archive_index: Path,
    archive_root: Path,
    *,
    scan_mode: str = "full",
    cache: dict[tuple[str, int, int, str, str], dict[str, Any]] | None = None,
    cache_lock: threading.Lock | None = None,
    cache_max_entries: int | None = None,
    relpath_validator: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "scan_mode": scan_mode,
        "allowed_path_roots": [archive_index.parent],
        "allowed_input_roots": [archive_root.parent],
        "repo_root": REPO_ROOT,
    }
    if cache is not None:
        kwargs["cache"] = cache
    if cache_lock is not None:
        kwargs["cache_lock"] = cache_lock
    if cache_max_entries is not None:
        kwargs["cache_max_entries"] = cache_max_entries
    if relpath_validator is not None:
        kwargs["relpath_validator"] = relpath_validator
    return module._validate_archive_index_against_root(archive_index, archive_root, **kwargs)


def test_direct_module_import_does_not_import_app() -> None:
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        "import transformation_portal.portal.archive_index_preflight; "
        "raise SystemExit(1 if 'app' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_app_legacy_archive_index_helpers_remain_available() -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    orchestrator_app = importlib.import_module("app")

    assert orchestrator_app.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT == module.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT
    assert orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE is module._ARCHIVE_INDEX_PREFLIGHT_CACHE
    assert orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK is module._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK
    assert orchestrator_app._archive_index_preflight_message is module._archive_index_preflight_message
    assert orchestrator_app._archive_index_preflight_root_reason is module._archive_index_preflight_root_reason
    assert orchestrator_app._validate_archive_index_relpath is module._validate_archive_index_relpath
    assert callable(orchestrator_app._validate_archive_index_against_root)


def test_app_wrapper_matches_direct_module_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    orchestrator_app = importlib.import_module("app")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])

    monkeypatch.setattr(orchestrator_app, "ALLOWED_PATH_ROOTS", [tmp_path])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    with orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK:
        orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE.clear()

    direct = _validate_direct(module, archive_index, archive_root)
    with orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK:
        orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE.clear()
    legacy = orchestrator_app._validate_archive_index_against_root(archive_index, archive_root)

    assert legacy == direct
    assert legacy == {
        "ok": True,
        "rows_total": 1,
        "blocked_rows": 0,
        "examples": [],
        "scan_mode": "full",
        "truncated": False,
    }


@pytest.mark.parametrize(
    ("relpath", "reason"),
    [
        ("../escape.dng", "parent_traversal"),
        ("/absolute.dng", "absolute_relpath"),
        ("C:/absolute.dng", "drive_prefixed_relpath"),
        ("", "empty_relpath"),
        ("missing.dng", "missing"),
        ("folder", "directory"),
    ],
)
def test_direct_validation_preserves_invalid_relpath_reasons(
    tmp_path: Path,
    relpath: str,
    reason: str,
) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "folder").mkdir()
    archive_index = tmp_path / f"archive_index_{reason}.csv.gz"
    _write_archive_index(archive_index, [relpath])

    result = _validate_direct(module, archive_index, archive_root)

    assert result["ok"] is False
    assert result["blocked_rows"] == 1
    assert result["examples"][0]["reason"] == reason


def test_direct_validation_rejects_symlink_escape(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    outside = tmp_path / "outside.dng"
    outside.write_bytes(b"outside")
    link_path = archive_root / "asset-link.dng"
    try:
        link_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    archive_index = tmp_path / "archive_index_symlink.csv.gz"
    _write_archive_index(archive_index, ["asset-link.dng"])

    result = _validate_direct(module, archive_index, archive_root)

    assert result["ok"] is False
    assert result["examples"][0]["reason"] == "symlink_traversal"


def test_direct_validation_rejects_intermediate_symlink_archive_root(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    archive_index = allowed_root / "archive_index.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])
    outside = tmp_path / "outside"
    outside_archive_root = outside / "archive_root"
    outside_archive_root.mkdir(parents=True)
    (outside_archive_root / "asset-001.dng").write_bytes(b"raw")
    link_path = allowed_root / "link-outside"
    try:
        link_path.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = module._validate_archive_index_against_root(
        archive_index,
        link_path / "archive_root",
        allowed_path_roots=[allowed_root],
        allowed_input_roots=[allowed_root],
        repo_root=REPO_ROOT,
    )

    assert result["ok"] is False
    assert result["examples"][0]["reason"] == "archive_root_not_directory"


def test_direct_validation_preserves_final_symlink_archive_root_reason(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_index = tmp_path / "archive_index.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    (real_root / "asset-001.dng").write_bytes(b"raw")
    link_root = tmp_path / "link_root"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    result = _validate_direct(module, archive_index, link_root)

    assert result["ok"] is False
    assert result["examples"][0]["reason"] == "archive_root_symlink"


def test_direct_validation_rejects_missing_columns_and_bad_gzip(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    missing_columns = tmp_path / "archive_index_missing_columns.csv"
    missing_columns.write_text("relpath\nasset-001.dng\n", encoding="utf-8")
    bad_gzip = tmp_path / "archive_index_bad.csv.gz"
    bad_gzip.write_bytes(b"not a gzip stream")

    missing_columns_result = _validate_direct(module, missing_columns, archive_root)
    bad_gzip_result = _validate_direct(module, bad_gzip, archive_root)

    assert missing_columns_result["ok"] is False
    assert missing_columns_result["examples"][0]["reason"].startswith("missing_columns:")
    assert bad_gzip_result["ok"] is False
    assert bad_gzip_result["examples"][0]["reason"] == "archive_index_unreadable:BadGzipFile"


def test_direct_preview_scan_is_bounded_deep_copied_and_cached(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    existing_relpaths = [f"asset-{idx:03d}.dng" for idx in range(module.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT)]
    for relpath in existing_relpaths:
        (archive_root / relpath).write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_bounded.csv.gz"
    _write_archive_index(archive_index, [*existing_relpaths, "late-missing.dng"])
    cache: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    cache_lock = threading.Lock()

    preview_result = _validate_direct(
        module,
        archive_index,
        archive_root,
        scan_mode="preview",
        cache=cache,
        cache_lock=cache_lock,
    )
    preview_result["ok"] = False

    def fail_if_rescanned(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        raise AssertionError("cached preview result should avoid rescanning rows")

    cached_preview = _validate_direct(
        module,
        archive_index,
        archive_root,
        scan_mode="preview",
        cache=cache,
        cache_lock=cache_lock,
        relpath_validator=fail_if_rescanned,
    )

    assert cached_preview["ok"] is True
    assert cached_preview["truncated"] is True
    assert cached_preview["rows_total"] == module.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT


def test_direct_cache_uses_fifo_eviction(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    cache: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    cache_lock = threading.Lock()
    indexes = []
    for idx in range(3):
        archive_index = tmp_path / f"archive_index_{idx}.csv.gz"
        _write_archive_index(archive_index, ["asset-001.dng"])
        indexes.append(archive_index)
        _validate_direct(
            module,
            archive_index,
            archive_root,
            cache=cache,
            cache_lock=cache_lock,
            cache_max_entries=2,
        )

    cached_paths = {key[0] for key in cache}
    assert len(cache) == 2
    assert str(indexes[0]) not in cached_paths
    assert str(indexes[1]) in cached_paths
    assert str(indexes[2]) in cached_paths


def test_direct_validation_rejects_invalid_cache_max_entries(tmp_path: Path) -> None:
    module = importlib.import_module("transformation_portal.portal.archive_index_preflight")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])

    with pytest.raises(ValueError, match="cache_max_entries must be at least 1"):
        _validate_direct(
            module,
            archive_index,
            archive_root,
            cache={},
            cache_lock=threading.Lock(),
            cache_max_entries=0,
        )


def test_app_wrapper_uses_monkeypatched_relpath_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator_app = importlib.import_module("app")
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    (archive_root / "asset-001.dng").write_bytes(b"raw")
    archive_index = tmp_path / "archive_index_normalized.csv.gz"
    _write_archive_index(archive_index, ["asset-001.dng"])

    monkeypatch.setattr(orchestrator_app, "ALLOWED_PATH_ROOTS", [tmp_path])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [tmp_path])

    def forced_failure(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str]:
        return False, "forced.dng", "forced_failure"

    monkeypatch.setattr(orchestrator_app, "_validate_archive_index_relpath", forced_failure)
    with orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK:
        orchestrator_app._ARCHIVE_INDEX_PREFLIGHT_CACHE.clear()

    result = orchestrator_app._validate_archive_index_against_root(archive_index, archive_root)

    assert result["ok"] is False
    assert result["examples"][0] == {
        "row": 2,
        "relpath": "forced.dng",
        "reason": "forced_failure",
    }
