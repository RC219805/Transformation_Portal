"""Coverage configuration honesty contracts."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
COVERAGE_PLAN_PATH = REPO_ROOT / "docs" / "testing" / "test_coverage_improvement_plan.md"
ZERO_COVERAGE_RATCHET_PACKAGES = (
    "depth_intelligence/",
    "diffusion/",
    "dwm/",
    "interfaces/",
    "pfm/",
)


def _coverage_config() -> dict:
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    return pyproject["tool"]["coverage"]


def test_coverage_source_keeps_production_packages_in_scope() -> None:
    run_config = _coverage_config()["run"]

    assert run_config["source"] == ["src"]

    omitted = "\n".join(run_config.get("omit", []))
    forbidden_omits = (
        "src/transformation_portal",
        "transformation_portal/*",
        "*/src/*",
        "*/src/transformation_portal/*",
    )
    for pattern in forbidden_omits:
        assert pattern not in omitted


def test_zero_coverage_production_packages_are_explicit_ratchet_targets() -> None:
    coverage_plan = COVERAGE_PLAN_PATH.read_text(encoding="utf-8")

    assert "### Zero-Coverage Production Packages (ratchet targets)" in coverage_plan
    assert "NOT excluded from coverage measurement" in coverage_plan
    for package in ZERO_COVERAGE_RATCHET_PACKAGES:
        assert f"`{package}`" in coverage_plan
