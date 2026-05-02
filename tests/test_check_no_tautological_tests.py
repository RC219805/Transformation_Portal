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


def test_detects_assert_nonempty_list(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert [1]\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_nonempty_tuple(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert (1,)\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_nonempty_dict(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert {'k': 'v'}\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_detects_assert_nonempty_set(tmp_path):
    path = _write(tmp_path, "test_offender.py", "def test_x():\n    assert {1}\n")
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_ignores_container_with_call_element(tmp_path):
    """``assert [compute()]`` evaluates ``compute()`` for its side effect, so
    it could legitimately raise — the assertion is not a tautology even
    though the resulting non-empty list is always truthy."""
    body = "def test_x():\n    def compute():\n        return 1\n    assert [compute()]\n"
    path = _write(tmp_path, "test_real.py", body)
    assert find_tautological_asserts(path) == []


def test_ignores_container_with_name_element(tmp_path):
    body = "def test_x():\n    value = 1\n    assert [value]\n"
    path = _write(tmp_path, "test_real.py", body)
    assert find_tautological_asserts(path) == []


def test_ignores_dict_with_dynamic_value(tmp_path):
    body = "def test_x():\n    value = 1\n    assert {'k': value}\n"
    path = _write(tmp_path, "test_real.py", body)
    assert find_tautological_asserts(path) == []


def test_ignores_dict_with_dynamic_key(tmp_path):
    body = "def test_x():\n    key = 'k'\n    assert {key: 1}\n"
    path = _write(tmp_path, "test_real.py", body)
    assert find_tautological_asserts(path) == []


# ---------------------------------------------------------------------------
# Detection: negative cases (must NOT trigger)
# ---------------------------------------------------------------------------


def test_ignores_empty_list(tmp_path):
    """`assert []` is always falsy, not a tautology."""
    path = _write(tmp_path, "test_real.py", "def test_x():\n    assert []\n")
    assert find_tautological_asserts(path) == []


def test_ignores_empty_dict(tmp_path):
    path = _write(tmp_path, "test_real.py", "def test_x():\n    assert {}\n")
    assert find_tautological_asserts(path) == []


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
    body = "def test_x():\n" + "    fixture = 'def test_inner():\\n    assert True\\n'\n" + "    assert len(fixture) > 0\n"
    path = _write(tmp_path, "test_fixture_string.py", body)
    assert not find_tautological_asserts(path)


def test_respects_escape_hatch_comment(tmp_path):
    body = "def test_x():\n    assert True  # tautology-ok: smoke check\n"
    path = _write(tmp_path, "test_smoke.py", body)
    assert find_tautological_asserts(path) == []


def test_escape_hatch_in_string_literal_does_not_bypass(tmp_path):
    """The tag must be in an actual ``#`` comment. A same-line string that
    happens to contain the substring must NOT silence the lint."""
    body = 'def test_x():\n    assert True, "tautology-ok"\n'
    path = _write(tmp_path, "test_dirty.py", body)
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_escape_hatch_on_different_line_does_not_bypass(tmp_path):
    """The tag must be on the SAME line as the offender."""
    body = "def test_x():\n    # tautology-ok\n    assert True\n"
    path = _write(tmp_path, "test_dirty.py", body)
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_escape_hatch_negated_phrase_does_not_bypass(tmp_path):
    """A comment like ``# not-tautology-ok`` must NOT bypass the lint.

    The tag is recognized only when it stands as its own word — the regex
    rejects ``-`` or alphanumerics on either side.
    """
    body = "def test_x():\n    assert True  # not-tautology-ok\n"
    path = _write(tmp_path, "test_dirty.py", body)
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_escape_hatch_with_suffix_does_not_bypass(tmp_path):
    """``# tautology-okay`` must NOT bypass — the trailing letter disqualifies it."""
    body = "def test_x():\n    assert True  # tautology-okay\n"
    path = _write(tmp_path, "test_dirty.py", body)
    offenders = find_tautological_asserts(path)
    assert len(offenders) == 1


def test_escape_hatch_in_prose_comment_works(tmp_path):
    """The tag inside a longer comment is fine as long as it stands as a word."""
    body = "def test_x():\n    assert True  # tautology-ok: see issue #123 for context\n"
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
