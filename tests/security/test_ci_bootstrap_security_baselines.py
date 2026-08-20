from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DETERMINISM_TOOLCHAIN = 'python -m pip install --upgrade "pip==26.1.2" ' '"setuptools==83.0.0" "wheel==0.46.2"'
PYPDF_SECURITY_FLOOR = "pypdf>=6.15.0"
PYPDF_LOCK_PIN = "pypdf==6.16.1"


@pytest.mark.parametrize(
    ("workflow_path", "expected_occurrences"),
    [
        (".github/workflows/determinism-gate.yml", 2),
        (".github/workflows/determinism-cross-isa.yml", 2),
        (".github/workflows/diagnostic-trial.yml", 1),
    ],
)
def test_determinism_workflows_pin_non_vulnerable_bootstrap_tools(
    workflow_path: str,
    expected_occurrences: int,
) -> None:
    workflow = (REPO_ROOT / workflow_path).read_text(encoding="utf-8")

    assert workflow.count(SAFE_DETERMINISM_TOOLCHAIN) == expected_occurrences
    assert '"pip==24.0"' not in workflow
    assert '"setuptools==69.0.3"' not in workflow
    assert '"wheel==0.42.0"' not in workflow


@pytest.mark.parametrize(
    ("workflow_path", "pip_command"),
    [
        (".github/workflows/secure-install-pilot.yml", "python -m pip"),
        (
            ".github/workflows/dependency-update.yml",
            '"${{ steps.setup-python.outputs.python-path }}" -I -m pip --isolated',
        ),
        (".github/workflows/ci-quality-firewall.yml", "python -m pip"),
    ],
)
def test_lock_generation_workflows_use_pip_tools_with_pip_26_support(
    workflow_path: str,
    pip_command: str,
) -> None:
    workflow = (REPO_ROOT / workflow_path).read_text(encoding="utf-8")

    assert f'{pip_command} install --upgrade "pip==26.1.2"' in workflow
    assert f'{pip_command} install "pip-tools==7.6.0"' in workflow
    assert '"pip-tools==7.5.3"' not in workflow
    assert '"pip-tools==7.5.2"' not in workflow


def test_pypdf_governed_surfaces_require_non_vulnerable_release() -> None:
    ci_input = (REPO_ROOT / "requirements/ci.in").read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    all_lock = (REPO_ROOT / "requirements/all.txt").read_text(encoding="utf-8")
    ci_lock = (REPO_ROOT / "requirements/ci.txt").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    dependency_adr = (REPO_ROOT / "docs/architecture/ADR-032-dependency-pinning-strategy.md").read_text(encoding="utf-8")

    assert PYPDF_SECURITY_FLOOR in ci_input
    assert f'"{PYPDF_SECURITY_FLOOR}"' in pyproject
    assert PYPDF_LOCK_PIN in all_lock
    assert PYPDF_LOCK_PIN in ci_lock
    assert ">=6.15.0" in contributing
    assert "GHSA-fwg2-594c-jp42" in contributing
    assert "GHSA-fp3f-mc75-235c" in contributing
    assert "`>=6.15.0`" in dependency_adr

    governed_surfaces = "\n".join((ci_input, pyproject, all_lock, ci_lock, contributing, dependency_adr))
    assert "pypdf>=6.13.3" not in governed_surfaces
    assert "pypdf==6.14.2" not in governed_surfaces
