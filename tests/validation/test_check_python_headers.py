"""Tests for Python header validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_python_headers.py"
MA_STATE_PATH = PROJECT_ROOT / "src" / "transformation_portal" / "rl" / "ma_state.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_python_headers", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docstring_header_cookie_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    bad_file = tmp_path / "bad_header.py"
    bad_file.write_text(
        '"""Multi-agent state encoding: Local + global state features."""\n' "VALUE = 1\n",
        encoding="utf-8",
    )

    violations = module.find_violations([bad_file])

    assert len(violations) == 1
    assert "encoding: Local" in violations[0]
    assert "valid PEP 263" in violations[0]


def test_invalid_comment_encoding_cookie_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    bad_file = tmp_path / "invalid_cookie.py"
    bad_file.write_text("# encoding: Local\nVALUE = 1\n", encoding="utf-8")

    violations = module.find_violations([bad_file])

    assert len(violations) == 1
    assert "encoding: Local" in violations[0]


def test_valid_encoding_cookie_on_first_line_is_allowed(tmp_path: Path) -> None:
    module = _load_module()
    good_file = tmp_path / "utf8_header.py"
    good_file.write_text("# -*- coding: utf-8 -*-\nVALUE = 1\n", encoding="utf-8")

    assert module.find_violations([good_file]) == []


def test_valid_encoding_cookie_on_second_line_is_allowed(tmp_path: Path) -> None:
    module = _load_module()
    good_file = tmp_path / "utf8_second_line.py"
    good_file.write_text(
        "#!/usr/bin/env python3\n# coding: utf-8\nVALUE = 1\n",
        encoding="utf-8",
    )

    assert module.find_violations([good_file]) == []


def test_ma_state_header_is_now_safe() -> None:
    module = _load_module()

    assert module.find_violations([MA_STATE_PATH]) == []
