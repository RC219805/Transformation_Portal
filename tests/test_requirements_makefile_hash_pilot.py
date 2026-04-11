from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE_PATH = REPO_ROOT / "requirements" / "Makefile"


def _read_makefile() -> str:
    return MAKEFILE_PATH.read_text(encoding="utf-8")


def _target_body(name: str) -> str:
    text = _read_makefile()
    match = re.search(rf"^{re.escape(name)}:(?:[^\n]*)\n(?P<body>(?:\t.*\n)+)", text, flags=re.MULTILINE)
    assert match is not None, f"Makefile target {name} not found"
    return match.group("body")


def test_makefile_declares_hash_pilot_targets_and_default_output_dir() -> None:
    text = _read_makefile()
    phony_line = next(line for line in text.splitlines() if line.startswith(".PHONY:"))

    assert "HASH_PILOT_OUT_DIR ?= $(CURDIR)/.hash-pilot" in text
    assert "compile-accel" in phony_line
    assert "compile-hash-pilot" in phony_line
    assert "check-hash-pilot" in phony_line
    assert "compile-hash-pilot:" in text
    assert "check-hash-pilot:" in text


def test_makefile_uses_generate_hashes_only_for_the_pilot_lane() -> None:
    text = _read_makefile()

    assert "PIP_COMPILE_HASHED := " in text
    assert "--generate-hashes" in text
    pip_compile_line = next(line for line in text.splitlines() if line.startswith("PIP_COMPILE :="))
    assert "--generate-hashes" not in pip_compile_line


def test_compile_hash_pilot_scope_matches_non_ml_checked_in_contract() -> None:
    body = _target_body("compile-hash-pilot")

    for lock_name in ("all.txt", "base.txt", "dev.txt", "ci.txt", "security.txt", "tools-archive.txt"):
        assert f'"$(HASH_PILOT_OUT_DIR)/{lock_name}"' in body

    for lock_name in ("ml-core-darwin-x86_64.txt", "ml-core-darwin-arm64.txt", "ml-core-linux.txt"):
        assert lock_name not in body


def test_check_hash_pilot_enforces_hash_validation_for_generated_outputs() -> None:
    body = _target_body("check-hash-pilot")

    assert "for lock in all.txt base.txt dev.txt ci.txt security.txt tools-archive.txt; do \\" in body
    assert 'python -m pip install --dry-run --require-hashes -r "$$lock_path" >/dev/null; \\' in body
