from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNED_PIP_PIN = "pip==26.2.1"
GOVERNED_LOCK_CLICK_PIN = "click==8.4.2"
SAFE_DETERMINISM_TOOLCHAIN = f'python -m pip install --upgrade "{GOVERNED_PIP_PIN}" ' '"setuptools==83.0.0" "wheel==0.46.2"'
PYPDF_SECURITY_FLOOR = "pypdf>=6.15.0"
PYPDF_LOCK_PIN = "pypdf==6.16.2"


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

    assert f'{pip_command} install --upgrade "{GOVERNED_PIP_PIN}"' in workflow
    assert f'{pip_command} install "pip-tools==7.6.1" "{GOVERNED_LOCK_CLICK_PIN}"' in workflow
    assert '"pip-tools==7.6.0"' not in workflow
    assert '"pip-tools==7.5.3"' not in workflow
    assert '"pip-tools==7.5.2"' not in workflow


def _workflow_run_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() != "run: |":
            continue
        run_indent = len(line) - len(line.lstrip(" "))
        block_lines: list[str] = []
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            indent = len(candidate) - len(candidate.lstrip(" "))
            if stripped and indent <= run_indent:
                break
            block_lines.append(candidate)
        blocks.append("\n".join(block_lines))
    return blocks


def _make_recipe_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith((" ", "\t")) or ":" not in line:
            continue
        recipe_lines: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith("\t"):
                recipe_lines.append(candidate)
                continue
            if candidate.strip():
                break
        if recipe_lines:
            blocks.append("\n".join(recipe_lines))
    return blocks


def _build_constraint_contract_errors(block: str, pin_needle: str) -> list[str]:
    commands = [line.strip() for line in block.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    errors: list[str] = []
    if any("PIP_CONSTRAINT" in command for command in commands):
        errors.append("PIP_CONSTRAINT cannot replace an isolated-build constraint")

    for index, command in enumerate(commands):
        if "pip install" not in command or "-c requirements/constraints.txt" not in command:
            continue
        if "--build-constraint requirements/constraints.txt" not in command:
            errors.append(f"missing paired build constraint: {command}")
        if not any(pin_needle in prior for prior in commands[:index]):
            errors.append(f"governed pip bootstrap must precede constrained install: {command}")
    return errors


@pytest.mark.parametrize(
    ("relative_path", "pin_needle", "expected_constraint_count"),
    [
        ("Makefile", "pip==$(PIP_VERSION)", 2),
        (".github/workflows/enforcement.yml", GOVERNED_PIP_PIN, 4),
        (".github/workflows/security-unified.yml", GOVERNED_PIP_PIN, 2),
    ],
)
def test_pip_26_2_runtime_constraints_are_paired_after_bootstrap(
    relative_path: str,
    pin_needle: str,
    expected_constraint_count: int,
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    blocks = _make_recipe_blocks(text) if relative_path == "Makefile" else _workflow_run_blocks(text)
    constrained_blocks = [block for block in blocks if "-c requirements/constraints.txt" in block]

    assert "PIP_CONSTRAINT" not in text
    assert sum(block.count("-c requirements/constraints.txt") for block in constrained_blocks) == expected_constraint_count
    assert [error for block in constrained_blocks for error in _build_constraint_contract_errors(block, pin_needle)] == []


@pytest.mark.parametrize(
    "broken_block",
    [
        'python -m pip install --upgrade "pip==26.2.1"\npython -m pip install -c requirements/constraints.txt -r requirements-ci.txt',
        'python -m pip install --upgrade "pip==26.2.1"\nPIP_CONSTRAINT=requirements/constraints.txt python -m pip install -c requirements/constraints.txt -r requirements-ci.txt',
        'python -m pip install -c requirements/constraints.txt --build-constraint requirements/constraints.txt -r requirements-ci.txt\npython -m pip install --upgrade "pip==26.2.1"',
        'python -m pip install --upgrade "pip==26.2.1"\n# --build-constraint requirements/constraints.txt\npython -m pip install -c requirements/constraints.txt -r requirements-ci.txt',
    ],
)
def test_build_constraint_contract_rejects_unpaired_or_stale_configuration(broken_block: str) -> None:
    assert _build_constraint_contract_errors(broken_block, GOVERNED_PIP_PIN)


def test_core_make_targets_use_the_declared_pip_pin() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "PIP_VERSION := 26.2.1" in makefile
    assert makefile.count('pip install --upgrade "pip==$(PIP_VERSION)"') == 3


def test_firewall_resolver_matrix_uses_governed_pip_on_supported_python_versions() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci-quality-firewall.yml").read_text(encoding="utf-8")
    resolver_step = workflow.split("- name: Resolver simulation for requirements-ci", 1)[1].split("- name:", 1)[0]

    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert f'python -m pip install --upgrade "{GOVERNED_PIP_PIN}"' in resolver_step


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
