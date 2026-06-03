from __future__ import annotations

from pathlib import Path

import pytest

from scripts.governance.check_script_topology import COMPATIBILITY_WRAPPERS, validate_script_topology

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reader(contents: dict[str, str]):
    def read_text(path: str) -> str:
        return contents[path]

    return read_text


def test_script_topology_accepts_compatibility_wrappers() -> None:
    tracked_paths = set()
    contents = {}

    for wrapper, (canonical, marker) in COMPATIBILITY_WRAPPERS.items():
        tracked_paths.add(wrapper)
        tracked_paths.add(canonical)
        contents[wrapper] = f"#!/usr/bin/env python3\n{marker} main\n"

    assert validate_script_topology(tracked_paths, read_text=_reader(contents)) == []


def test_script_topology_rejects_retired_organizer_paths() -> None:
    violations = validate_script_topology(
        {"scripts/organize_outputs.sh"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/organize_outputs.sh"
    assert "retired broad-mutating" in violations[0].reason


def test_script_topology_rejects_script_root_historical_reports() -> None:
    violations = validate_script_topology(
        {"scripts/PIPELINE_OPTIMIZATION_REPORT.md"},
        read_text=_reader({}),
    )

    assert len(violations) == 1
    assert violations[0].suggestion == "move historical evidence to docs/historical/script-audits/"


def test_script_topology_requires_wrapper_to_delegate_to_canonical_path() -> None:
    violations = validate_script_topology(
        {
            "scripts/install_models.py",
            "scripts/setup/install_models.py",
        },
        read_text=_reader({"scripts/install_models.py": "print('not a wrapper')\n"}),
    )

    assert len(violations) == 1
    assert violations[0].path == "scripts/install_models.py"
    assert "does not delegate" in violations[0].reason


def test_repository_compatibility_wrappers_reference_canonical_modules() -> None:
    for wrapper, (_canonical, marker) in COMPATIBILITY_WRAPPERS.items():
        wrapper_path = REPO_ROOT / wrapper
        assert wrapper_path.exists(), f"Missing compatibility wrapper: {wrapper}"
        assert marker in wrapper_path.read_text(encoding="utf-8")
