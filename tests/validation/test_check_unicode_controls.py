"""Tests for Unicode control-character validation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_unicode_controls.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_unicode_controls", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_file_allows_ordinary_unicode_text(tmp_path: Path) -> None:
    module = _load_module()
    good_file = tmp_path / "good.py"
    good_file.write_text("# cafe: \u00e9\nVALUE = 'ok'\n", encoding="utf-8")

    assert module.check_file(good_file) == []


def test_check_file_reports_bidirectional_controls(tmp_path: Path) -> None:
    module = _load_module()
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("VALUE = 'safe'\u202e\n", encoding="utf-8")

    violations = module.check_file(bad_file)

    assert len(violations) == 1
    assert "Bidirectional Unicode U+202E" in violations[0]
    assert str(bad_file) in violations[0]


def test_check_file_reports_other_format_controls(tmp_path: Path) -> None:
    module = _load_module()
    bad_file = tmp_path / "bad.md"
    bad_file.write_text("hidden\u200bmarker\n", encoding="utf-8")

    violations = module.check_file(bad_file)

    assert len(violations) == 1
    assert "Format control character U+200B" in violations[0]
    assert "ZERO WIDTH SPACE" in violations[0]


def test_main_scans_explicit_supported_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    bad_file = tmp_path / "explicit.yaml"
    bad_file.write_text("key: value\u2069\n", encoding="utf-8")

    exit_code = module.main([str(bad_file)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Found dangerous Unicode control characters:" in captured.err
    assert "Bidirectional Unicode U+2069" in captured.err


def test_main_ignores_unsupported_explicit_paths(tmp_path: Path) -> None:
    module = _load_module()
    ignored_file = tmp_path / "ignored.txt"
    ignored_file.write_text("hidden\u200bmarker\n", encoding="utf-8")

    assert module.main([str(ignored_file)]) == 0


def test_main_uses_staged_supported_files_when_paths_are_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    bad_file = tmp_path / "staged.py"
    bad_file.write_text("VALUE = 'safe'\u202d\n", encoding="utf-8")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=f"{bad_file}\n{tmp_path / 'ignored.txt'}\n",
            stderr="",
        ),
    )

    exit_code = module.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Bidirectional Unicode U+202D" in captured.err
