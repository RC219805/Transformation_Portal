#!/usr/bin/env python3
"""Tests for the tautological-assert lint itself.

The lint at ``scripts/ci/check_no_tautological_tests.py`` is what guards every
other test file from quietly drifting into ``assert True`` placeholders. If the
lint stops detecting offenders, the guard rusts. This file pins the contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_LINT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_no_tautological_tests.py"
_spec = importlib.util.spec_from_file_location("check_no_tautological_tests", _LINT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
find_tautological_asserts = _module.find_tautological_asserts
main = _module.main


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Detection: positive cases
# ---------------------------------------------------------------------------


def test_detects_assert_true(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert True\n")
    offenders = find_tautological_asserts(path)
    assert [line for line, _ in offenders] == [2]


def test_detects_assert_truthy_int(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert 1\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_nonempty_string(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert 'hi'\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_not_false(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert not False\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_not_zero(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert not 0\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


# ---------------------------------------------------------------------------
# Detection: negative cases (must NOT trigger)
# ---------------------------------------------------------------------------


def test_ignores_real_assert(tmp_path):
    path = _write(tmp_path, "test_real.py", "def test_x():\n    assert 1 + 1 == 2\n")
    assert find_tautological_asserts(path) == []


def test_ignores_assert_false(tmp_path):
    """`assert False` is a real failure marker (e.g., 'unreachable'); not banned."""
    path = _write(tmp_path, "test_real.py", "def test_x():\n    assert False, 'unreachable'\n")
    assert find_tautological_asserts(path) == []


def test_ignores_assert_none(tmp_path):
    """`assert None` is also always-false; not a tautology."""
    path = _write(tmp_path, "test_real.py", "def test_x():\n    assert None\n")
    assert find_tautological_asserts(path) == []


def test_ignores_assert_inside_string_literal(tmp_path):
    """Source code embedded in fixtures must not trigger the lint.

    This is the realistic case in tests/test_retrofit_test_markers.py.
    """
    body = "def test_x():\n" "    fixture = 'def test_inner():\\n    assert True\\n'\n" "    assert len(fixture) > 0\n"
    path = _write(tmp_path, "test_fixture_string.py", body)
    assert find_tautological_asserts(path) == []


def test_respects_escape_hatch_comment(tmp_path):
    body = "def test_x():\n    assert True  # tautology-ok: smoke check\n"
    path = _write(tmp_path, "test_smoke.py", body)
    assert find_tautological_asserts(path) == []


# ---------------------------------------------------------------------------
# CLI: exit code contract
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean_dir(tmp_path):
    _write(tmp_path, "test_clean.py", "def test_x():\n    assert 1 + 1 == 2\n")
    assert main(["check_no_tautological_tests.py", str(tmp_path)]) == 0


def test_main_returns_one_on_offender(tmp_path, capsys):
    _write(tmp_path, "test_dirty.py", "def test_x():\n    assert True\n")
    code = main(["check_no_tautological_tests.py", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "test_dirty.py" in err


def test_main_only_inspects_test_files(tmp_path):
    """A non-test-named file with `assert True` must not trip the lint."""
    _write(tmp_path, "helper.py", "def helper():\n    assert True\n")
    _write(tmp_path, "test_clean.py", "def test_x():\n    assert 1 == 1\n")
    assert main(["check_no_tautological_tests.py", str(tmp_path)]) == 0


def test_main_handles_syntax_error_without_crashing(tmp_path):
    _write(tmp_path, "test_broken.py", "def test_x(:\n    invalid syntax\n")
    # Syntax errors are not the lint's job; should pass clean here.
    assert main(["check_no_tautological_tests.py", str(tmp_path)]) == 0
