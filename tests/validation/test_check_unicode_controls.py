"""Tests for Unicode control-character validation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

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
            stdout=f"{bad_file}\0{tmp_path / 'ignored.txt'}\0".encode(),
            stderr=b"",
        ),
    )

    exit_code = module.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Bidirectional Unicode U+202D" in captured.err


@pytest.mark.parametrize("filename", ["with spaces.sh", "with\nnewline.py", "caf\u00e9.yaml", "deprecated/file.md"])
@pytest.mark.parametrize("all_tracked", [False, True])
def test_git_inventory_preserves_filenames_and_scans_shell_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, all_tracked: bool
) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "--quiet"], check=True, cwd=tmp_path)
    bad_file = tmp_path / filename
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("# hidden\u202e\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", filename], check=True, cwd=tmp_path)

    assert module.main(["--all"] if all_tracked else []) == 1


def test_staged_inventory_includes_renames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "--quiet"], check=True, cwd=tmp_path)
    (tmp_path / "before.py").write_text("# hidden\u202e\n", encoding="utf-8")
    subprocess.run(["git", "add", "before.py"], check=True, cwd=tmp_path)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture"],
        check=True,
        cwd=tmp_path,
    )
    subprocess.run(["git", "mv", "before.py", "after.py"], check=True, cwd=tmp_path)

    assert module.main([]) == 1


@pytest.mark.parametrize("argv", [[], ["--all"]])
def test_git_inventory_failure_is_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)

    assert module.main(argv) == 1
    assert "Error getting Git files:" in capsys.readouterr().err


def test_unavailable_git_is_blocking(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module()

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(module.subprocess, "run", missing_git)

    assert module.main(["--all"]) == 1
    assert "Could not run git:" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["missing", "directory", "invalid_utf8"])
def test_explicit_unreadable_files_are_blocking(tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str) -> None:
    module = _load_module()
    bad_file = tmp_path / "unreadable.py"
    if failure == "directory":
        bad_file.mkdir()
    elif failure == "invalid_utf8":
        bad_file.write_bytes(b"# invalid \xff\n")

    assert module.main([str(bad_file)]) == 1
    assert "Error reading file:" in capsys.readouterr().err


def test_explicit_permission_errors_are_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    def denied_read(*args, **kwargs):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(Path, "read_text", denied_read)

    assert module.main([str(tmp_path / "denied.py")]) == 1


def test_all_cannot_silently_override_explicit_paths(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--all", str(tmp_path / "file.py")])
    assert exc_info.value.code == 2


def test_all_ignores_untracked_generated_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "--quiet"], check=True, cwd=tmp_path)
    (tmp_path / "generated.py").write_text("# hidden\u202e\n", encoding="utf-8")

    assert module.main(["--all"]) == 0
