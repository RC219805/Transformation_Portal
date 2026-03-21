"""Tests for EvalSuite schema lock updater tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "update_schema_locks.py"
SPEC = importlib.util.spec_from_file_location("update_schema_locks", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_locks_only_schema_json_in_canonical_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    schema_root = repo_root / "docs" / "schemas" / "evalsuite"
    lockfile = repo_root / "docs" / "contracts" / "SCHEMA_LOCKS.sha256"

    first = schema_root / "z.v0" / "z.schema.json"
    second = schema_root / "a.v0" / "a.schema.json"
    ignored = schema_root / "b.v0" / "ignored.json"
    _write_json(first, {"schema": "z"})
    _write_json(second, {"schema": "a"})
    _write_json(ignored, {"schema": "ignored"})

    monkeypatch.setattr(updater, "REPO_ROOT", repo_root)
    monkeypatch.setattr(updater, "SCHEMA_ROOT", schema_root)
    monkeypatch.setattr(updater, "LOCKFILE", lockfile)

    updater.main()

    entries = []
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _, rel = line.split()
        entries.append(rel)

    assert entries == [
        "docs/schemas/evalsuite/a.v0/a.schema.json",
        "docs/schemas/evalsuite/z.v0/z.schema.json",
    ]


def test_atomic_write_text_writes_and_cleans_tmp(tmp_path: Path) -> None:
    target = tmp_path / "SCHEMA_LOCKS.sha256"

    updater.atomic_write_text(target, "hello\n")

    assert target.read_text(encoding="utf-8") == "hello\n"
    tmp_candidates = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert tmp_candidates == []


def test_atomic_write_text_cleans_tmp_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "SCHEMA_LOCKS.sha256"

    def _explode_replace(self: Path, _target: Path) -> None:
        del self, _target
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", _explode_replace)

    with pytest.raises(OSError, match="replace failed"):
        updater.atomic_write_text(target, "hello\n")

    tmp_candidates = list(target.parent.glob(f".{target.name}.*.tmp"))
    assert tmp_candidates == []
