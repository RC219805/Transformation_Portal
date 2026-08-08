from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DETERMINISM_TOOLCHAIN = 'python -m pip install --upgrade "pip==26.1.2" ' '"setuptools==83.0.0" "wheel==0.46.2"'


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
    "workflow_path",
    [
        ".github/workflows/secure-install-pilot.yml",
        ".github/workflows/dependency-update.yml",
        ".github/workflows/ci-quality-firewall.yml",
    ],
)
def test_lock_generation_workflows_use_pip_tools_with_pip_26_support(
    workflow_path: str,
) -> None:
    workflow = (REPO_ROOT / workflow_path).read_text(encoding="utf-8")

    assert '"pip-tools==7.5.3"' in workflow
    assert '"pip-tools==7.5.2"' not in workflow
