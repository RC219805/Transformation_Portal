"""Enforce core-vs-ML dependency tier boundaries and CI lane selectors."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalize_requirement_name(requirement: str) -> str:
    """Extract canonical package name from requirement text."""
    token = requirement.split(";", maxsplit=1)[0].strip()
    token = re.split(r"[<>=!~\[\]]", token, maxsplit=1)[0]
    return token.strip().lower().replace("_", "-")


def _load_optional_dependency_groups() -> dict[str, list[str]]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]["optional-dependencies"]


def _load_core_dependencies() -> list[str]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]["dependencies"]


def test_core_dependencies_exclude_torch_stack():
    """Core install tier must remain torch-free."""
    core_dependency_names = {_normalize_requirement_name(entry) for entry in _load_core_dependencies()}
    forbidden = {"torch", "torchvision", "sam2"}
    leaked = sorted(core_dependency_names.intersection(forbidden))
    assert not leaked, f"Core dependency tier leaked ML packages: {leaked}"


def test_ml_optional_dependencies_include_torch_stack():
    """ML optional tier must explicitly carry torch dependencies."""
    optional_groups = _load_optional_dependency_groups()
    ml_names = {_normalize_requirement_name(entry) for entry in optional_groups["ml"]}
    required = {"torch", "torchvision"}
    missing = sorted(required.difference(ml_names))
    assert not missing, f"ML dependency tier missing required packages: {missing}"


def test_ci_marker_split_between_fast_and_slow_ml_lanes():
    """CI should keep fast-ML and slow-ML marker selectors distinct."""
    ci_workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    slow_suite_workflow = (REPO_ROOT / ".github/workflows/ml-slow-suite.yml").read_text(encoding="utf-8")

    assert '-m "ml and not slow and not benchmark and not stress"' in ci_workflow
    assert '-m "ml and slow"' in slow_suite_workflow
